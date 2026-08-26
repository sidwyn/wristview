"""Render a splat from its own registered scan poses and score it.

This is the gate. Run it on any splat.pt before that splat is used for
anything, whether it was trained here or on a rented GPU.

The pose comes straight from the COLMAP reconstruction. No retarget, no mount
transform, no wrist camera, no viewpoint policy. If the splat cannot redraw the
photograph the camera took from that exact pose, nothing downstream can help.

Session real26 shipped a splat that scored 15.7 to 17.3 dB here and produced
featureless discs in the wrist view. The render stage had reported healthy
alpha coverage, because Gaussians covered every pixel. They were the wrong
Gaussians. Coverage is not correctness.

    python -m tools.check_splat --splat runs/real26/01_scene/splat.pt \
        --run runs/real26 --views 12 --stills qc/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.backends.splat_mps import GaussianModel, render  # noqa: E402
from wristview.device import resolve as resolve_device  # noqa: E402
from wristview.splatqc import evaluate  # noqa: E402


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Return PSNR in dB for two float images in [0, 1]."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splat", help="score this splat with the LOCAL MPS renderer, "
                                       "which is not authoritative. See --measurements.")
    parser.add_argument("--measurements", help="apply the gate to a measurements.json "
                                              "written by tools/cuda_job/measure_splat.py. "
                                              "This is the authoritative path.")
    parser.add_argument("--run", help="run root, for the COLMAP model and frames")
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--stills", default=None, help="write side-by-side comparisons here")
    parser.add_argument("--report", default=None, help="write the gate result as JSON here")
    parser.add_argument("--max-per-tile", type=int, default=128,
                        help="hard cap on Gaussians composited per tile. "
                             "Scoring refuses to proceed if anything is dropped.")
    parser.add_argument("--max-side", type=int, default=960,
                        help="render at this long side; the gate compares like with like")
    args = parser.parse_args()

    # The authoritative path. gsplat rendered the views on a CUDA box and wrote
    # the numbers out; this applies the thresholds. The verdict has one
    # implementation and it is `wristview.splatqc`.
    if args.measurements:
        data = json.loads(Path(args.measurements).read_text())
        report = evaluate(
            data["psnr_db"],
            gaussian_count=int(data["gaussian_count"]),
            view_count=int(data["view_count"]),
        )
        report["renderer"] = data.get("renderer", "unknown")
        report["splat"] = data.get("splat")
        report["authoritative"] = report["renderer"] == "gsplat"
        print(json.dumps(report, indent=2))
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2))
        if report["passed"] is not True:
            print("\nGATE FAILED. Do not render from this splat.")
            for line in report.get("failures", []):
                print(f"  - {line}")
            return 1
        print("\nGATE PASSED.")
        return 0

    if not (args.splat and args.run):
        raise SystemExit("give --measurements, or --splat with --run")

    print(
        "WARNING: this is the local MPS renderer and it is NOT authoritative.\n"
        "  Asked to score the real26 GPU splat it returned 7.21 dB where gsplat\n"
        "  returned 30.46 dB on the same splat, pose and photograph. Use it to\n"
        "  preview, not to judge. The gate is --measurements.\n"
    )

    import pycolmap

    run = Path(args.run).resolve()
    rec = pycolmap.Reconstruction(str(run / "01_scene" / "colmap" / "sparse"))
    frames_dir = run / "00_ingest" / "scan" / "frames"

    device = resolve_device()
    model = GaussianModel.load(args.splat, device=device)
    count = int(model.count)

    image_ids = sorted(rec.images.keys())
    if not image_ids:
        raise ValueError("the reconstruction registered no images")
    picks = np.linspace(0, len(image_ids) - 1, min(args.views, len(image_ids))).astype(int)

    stills_dir = Path(args.stills).resolve() if args.stills else None
    if stills_dir:
        stills_dir.mkdir(parents=True, exist_ok=True)

    scores: list[float] = []
    for slot, index in enumerate(np.unique(picks)):
        image = rec.images[image_ids[int(index)]]
        camera = rec.cameras[image.camera_id]
        photo_path = frames_dir / image.name
        photo = cv2.imread(str(photo_path))
        if photo is None:
            raise OSError(f"cannot read the scan frame {photo_path}")

        # Take the pose exactly as the reconstruction stores it.
        world_to_cam = np.eye(4, dtype=np.float64)
        world_to_cam[:3, :4] = image.cam_from_world().matrix()

        scale = min(1.0, args.max_side / max(camera.width, camera.height))
        width, height = int(round(camera.width * scale)), int(round(camera.height * scale))
        params = list(camera.params)
        fx = fy = params[0] * scale
        cx, cy = params[1] * scale, params[2] * scale
        if camera.model.name in ("PINHOLE", "OPENCV"):
            fx, fy = params[0] * scale, params[1] * scale
            cx, cy = params[2] * scale, params[3] * scale

        with torch.no_grad():
            result = render(
                model, torch.tensor(world_to_cam, dtype=torch.float32, device=device),
                fx, fy, cx, cy, width, height,
                max_per_tile=args.max_per_tile,
            )
        # Never score a truncated render. The rasteriser pads each tile to
        # max_per_tile and drops the rest, and it already counts what it drops.
        # At the default of 128 the real26 GPU splat lost 11,076,570 pairs and
        # scored 6.39 dB. That number described the cap, not the splat.
        dropped = int(result.__dict__.get("_overflow", 0))
        if dropped:
            raise RuntimeError(
                f"the MPS rasteriser dropped {dropped:,} Gaussian-tile pairs "
                f"rendering {image.name}, so this image is not the splat and "
                f"scoring it would be meaningless. Raise max_per_tile above "
                f"{args.max_per_tile}, or measure on a GPU with "
                f"tools/cuda_job/measure_splat.py."
            )
        drawn = result.rgb.detach().cpu().numpy()
        drawn = np.clip(drawn, 0.0, 1.0)

        target = cv2.cvtColor(cv2.resize(photo, (width, height), interpolation=cv2.INTER_AREA),
                              cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        value = psnr(drawn, target)
        scores.append(value)
        print(f"  {image.name}: {value:.2f} dB")

        if stills_dir:
            left = (drawn[:, :, ::-1] * 255).astype(np.uint8)
            right = (target[:, :, ::-1] * 255).astype(np.uint8)
            pair = np.concatenate([left, right], axis=1)
            cv2.putText(pair, f"splat {value:.1f} dB", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(pair, "photograph", (width + 10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(stills_dir / f"gate_{slot:02d}_{image.name}.png"), pair)

    report = evaluate(scores, gaussian_count=count, view_count=rec.num_images())
    report["splat"] = str(Path(args.splat).resolve())
    print()
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))

    if report["passed"] is not True:
        print()
        print("GATE FAILED. Do not render from this splat.")
        for line in report.get("failures", []):
            print(f"  - {line}")
        return 1
    print()
    print("GATE PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
