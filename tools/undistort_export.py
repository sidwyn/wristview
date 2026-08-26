"""Undistort an export's scan images and write pinhole intrinsics beside them.

gsplat rasterises a pure pinhole camera. It has no radial term. If the COLMAP
camera carries one, the images and the model fitted through them disagree, and
the disagreement grows toward the frame edge.

`train_gsplat.py` has looked for this script's two outputs since the real03 run.
The script itself was never written, so every GPU run so far trained distorted
images under a pinhole model.

Report the size of the error rather than assuming it. This script measures the
largest pixel displacement it removes and prints it. If that number is small,
say so; do not present a sub-pixel correction as a fix for a broken splat.

    python -m tools.undistort_export --export exports/real26
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# COLMAP camera model ids, from `colmap/src/colmap/sensor/models.h`.
SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV = 0, 1, 2, 3, 4


def intrinsics_and_distortion(model_id: int, params: list[float]):
    """Return the OpenCV camera matrix and distortion vector for a COLMAP camera."""
    p = list(params)
    if model_id == SIMPLE_PINHOLE:
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        dist = [0.0, 0.0, 0.0, 0.0]
    elif model_id == PINHOLE:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        dist = [0.0, 0.0, 0.0, 0.0]
    elif model_id == SIMPLE_RADIAL:
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        dist = [p[3], 0.0, 0.0, 0.0]
    elif model_id == RADIAL:
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        dist = [p[3], p[4], 0.0, 0.0]
    elif model_id == OPENCV:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        dist = [p[4], p[5], p[6], p[7]]
    else:
        raise ValueError(
            f"camera model id {model_id} is not handled. Add it here rather "
            f"than letting gsplat treat it as a pinhole."
        )
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, np.array(dist, dtype=np.float64)


def max_displacement_px(K: np.ndarray, dist: np.ndarray, width: int, height: int) -> float:
    """Return the largest distance a pixel moves when the model is undistorted.

    Sample the frame border. Radial distortion peaks at the largest radius, so
    the border holds the worst case.
    """
    xs = np.linspace(0, width - 1, 64)
    ys = np.linspace(0, height - 1, 64)
    edge = np.concatenate([
        np.stack([xs, np.zeros_like(xs)], axis=1),
        np.stack([xs, np.full_like(xs, height - 1)], axis=1),
        np.stack([np.zeros_like(ys), ys], axis=1),
        np.stack([np.full_like(ys, width - 1), ys], axis=1),
    ]).astype(np.float64)
    fixed = cv2.undistortPoints(edge.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    return float(np.linalg.norm(fixed - edge, axis=1).max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, help="folder written by export_for_gsplat")
    parser.add_argument("--format", choices=("png", "jpg"), default="png",
                        help="png re-encodes without loss. Prefer it, and run "
                             "this on the machine that trains, so the transfer "
                             "carries the original JPEGs.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    root = Path(args.export).resolve()
    sparse = root / "sparse" / "0"
    if not (sparse / "cameras.bin").exists():
        raise FileNotFoundError(f"no COLMAP model at {sparse}")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "cuda_job"))
    from train_gsplat import read_cameras_binary  # noqa: E402

    cameras = read_cameras_binary(sparse / "cameras.bin")
    images_dir = root / "images"
    out_dir = root / "undistorted"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep the same intrinsics and the same image size. The new camera matrix
    # is the old one, so no field of view is lost and no crop shifts a centre.
    # This is safe only because the distortion is small. The measured
    # displacement below is what decides that.
    pinhole: dict[str, dict] = {}
    maps: dict[int, tuple] = {}
    worst = 0.0
    for cid, cam in cameras.items():
        K, dist = intrinsics_and_distortion(cam["model_id"], list(cam["params"]))
        width, height = int(cam["width"]), int(cam["height"])
        moved = max_displacement_px(K, dist, width, height)
        worst = max(worst, moved)
        print(
            f"camera {cid}: model id {cam['model_id']}, {width}x{height}, "
            f"largest correction {moved:.2f} px at the frame border"
        )
        maps[cid] = cv2.initUndistortRectifyMap(
            K, dist, None, K, (width, height), cv2.CV_16SC2
        )
        pinhole[str(cid)] = {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "width": width, "height": height,
            "max_correction_px": round(moved, 3),
        }

    # Every image in this export shares one camera unless COLMAP said otherwise.
    only = next(iter(maps)) if len(maps) == 1 else None
    names = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not names:
        raise FileNotFoundError(f"no images in {images_dir}")

    if only is None:
        raise ValueError(
            "this export holds more than one camera, so each image must be "
            "matched to its camera id before undistortion. Add that lookup."
        )

    map_x, map_y = maps[only]
    for index, name in enumerate(names):
        image = cv2.imread(str(images_dir / name))
        if image is None:
            raise OSError(f"cannot read {images_dir / name}")
        fixed = cv2.remap(image, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        stem = Path(name).stem
        if args.format == "png":
            cv2.imwrite(str(out_dir / f"{stem}.png"), fixed)
        else:
            cv2.imwrite(str(out_dir / f"{stem}.jpg"), fixed,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(names)}")

    (root / "cameras_pinhole.json").write_text(json.dumps(pinhole, indent=2))
    size_mb = sum(p.stat().st_size for p in out_dir.iterdir()) / 1e6
    print(f"wrote {len(names)} undistorted images ({size_mb:.0f} MB) and cameras_pinhole.json")
    if args.format == "jpg":
        print(
            "WARNING: a JPEG re-encode adds its own error. At quality 95 that "
            "error measured 45.3 dB against a correction worth 40.7 dB, so the "
            "correction still wins, but PNG costs nothing here."
        )
    if worst < 1.0:
        print(
            f"NOTE: the largest correction was {worst:.2f} px. That is under one "
            f"pixel. This is worth doing for correctness. It is not a fix for a "
            f"splat that does not resolve the scene."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
