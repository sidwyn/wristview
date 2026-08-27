"""Measure a splat against its own training views, using gsplat.

This produces the numbers the splat gate judges. It runs on the CUDA box
because that is where the renderer is known to be correct.

The MPS rasteriser is not trusted for this. Asked to score the real26 GPU
splat it returned 7.21 dB while gsplat returned 30.46 dB on the same splat,
the same pose and the same photograph. Three faults in the MPS path explain
part of that and one is still unexplained. A gate that judges a splat through
an unverified renderer cannot separate a bad splat from a renderer that cannot
draw it, so it is not a gate.

This script decides nothing. It renders, measures, and writes the measurements
out. `tools/check_splat.py --measurements` applies the thresholds, so the
verdict has one implementation and it lives in `wristview.splatqc`.

    python measure_splat.py --data /root/real26 --splat /root/result/splat.pt \
        --out /root/measurements.json --views 12
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization
from train_gsplat import qvec_to_rotmat, read_images_binary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--splat", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--all-views", action="store_true",
                        help="measure every registered view. Needed for the "
                             "per-height breakdown, because the low bands hold "
                             "only a handful of frames each.")
    parser.add_argument("--heights", default=None,
                        help="JSON of {image name: {height_m, lapvar, low_pass}}. "
                             "Groups PSNR by camera height above the desk.")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--stills", default=None)
    args = parser.parse_args()

    data = Path(args.data)
    params = torch.load(args.splat, map_location="cuda", weights_only=True)
    sparse = data / "sparse" / "0"
    images = read_images_binary(sparse / "images.bin")

    pinhole = json.loads((data / "cameras_pinhole.json").read_text())
    image_dir = data / "undistorted"
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"{image_dir} does not exist. Run undistort_export.py first: the "
            f"gate must compare against the same frames training saw."
        )
    on_disk = {p.stem: p for p in sorted(image_dir.iterdir()) if p.is_file()}

    ordered = sorted(images.values(), key=lambda v: v["name"])
    if args.all_views:
        picks = np.arange(len(ordered))
    else:
        picks = np.linspace(0, len(ordered) - 1, min(args.views, len(ordered))).astype(int)
    heights = json.loads(Path(args.heights).read_text()) if args.heights else {}

    stills = Path(args.stills) if args.stills else None
    if stills:
        stills.mkdir(parents=True, exist_ok=True)

    scores: list[float] = []
    names: list[str] = []
    for slot, index in enumerate(sorted(set(int(i) for i in picks))):
        view = ordered[index]
        override = pinhole[str(view["camera_id"])]
        width, height = override["width"], override["height"]
        K = np.array([
            [override["fx"], 0.0, override["cx"]],
            [0.0, override["fy"], override["cy"]],
            [0.0, 0.0, 1.0],
        ])
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = qvec_to_rotmat(view["qvec"])
        world_to_cam[:3, 3] = view["tvec"]

        with torch.no_grad():
            drawn, _, _ = rasterization(
                means=params["means"],
                quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=params["sh"],
                viewmats=torch.tensor(world_to_cam, dtype=torch.float32, device="cuda")[None],
                Ks=torch.tensor(K, dtype=torch.float32, device="cuda")[None],
                width=width, height=height, sh_degree=args.sh_degree, packed=True,
            )
        image = drawn[0].clamp(0, 1).cpu().numpy()

        path = on_disk.get(Path(view["name"]).stem)
        photo = cv2.imread(str(path)) if path else None
        if photo is None:
            raise FileNotFoundError(f"no undistorted frame for {view['name']}")
        target = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mse = float(np.mean((image - target) ** 2))
        value = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
        scores.append(value)
        names.append(view["name"])
        if not args.all_views or (slot + 1) % 100 == 0:
            print(f"  {view['name']}: {value:.2f} dB", flush=True)

        if stills:
            left = (image[:, :, ::-1] * 255).astype(np.uint8)
            right = (target[:, :, ::-1] * 255).astype(np.uint8)
            pair = np.concatenate([left, right], axis=1)
            cv2.putText(pair, f"splat {value:.1f} dB", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
            cv2.putText(pair, "photograph", (width + 20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
            cv2.imwrite(str(stills / f"gate_{slot:02d}_{Path(view['name']).stem}.jpg"),
                        pair, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Break the score down by the height the training view was taken from.
    #
    # A blurry view does not only damage the splat where it was taken. The
    # Gaussians it pulls on are shared, so soft low-pass frames can smear
    # detail in views that were themselves sharp. Grouping by height is how
    # that shows up: if the low bands score badly and the high bands are also
    # worse than a scan without them, the coverage cost more than it bought.
    bands = [(0.00, 0.10), (0.10, 0.12), (0.12, 0.15), (0.15, 0.20),
             (0.20, 0.30), (0.30, 0.45), (0.45, 10.0)]
    by_height = []
    if heights:
        h = np.array([heights.get(nm, {}).get("height_m", np.nan) for nm in names])
        lap = np.array([heights.get(nm, {}).get("lapvar") or np.nan for nm in names])
        low = np.array([bool(heights.get(nm, {}).get("low_pass")) for nm in names])
        arr = np.array(scores)
        for lo, hi in bands:
            m = (h >= lo) & (h < hi) & np.isfinite(h)
            if not m.any():
                continue
            by_height.append({
                "band_cm": [lo * 100, min(hi, 1.0) * 100],
                "views": int(m.sum()),
                "psnr_median_db": round(float(np.median(arr[m])), 2),
                "psnr_p10_db": round(float(np.percentile(arr[m], 10)), 2),
                "psnr_min_db": round(float(arr[m].min()), 2),
                "lapvar_median": round(float(np.nanmedian(lap[m])), 1),
                "from_low_pass": int(low[m].sum()),
            })
        for entry in by_height:
            print(f"  {entry['band_cm'][0]:5.0f}-{entry['band_cm'][1]:3.0f} cm  "
                  f"{entry['views']:4d} views  PSNR median {entry['psnr_median_db']:6.2f} dB  "
                  f"p10 {entry['psnr_p10_db']:6.2f}  lapvar {entry['lapvar_median']:7.1f}  "
                  f"low-pass {entry['from_low_pass']:4d}", flush=True)

    payload = {
        "by_height": by_height,
        "renderer": "gsplat",
        "splat": str(Path(args.splat).resolve()),
        "gaussian_count": int(params["means"].shape[0]),
        "view_count": len(images),
        "views_measured": names,
        "psnr_db": [round(v, 3) for v in scores],
        "resolution": [width, height],
        "note": (
            "Measured against the undistorted frames, which are the frames "
            "training saw. This file holds measurements only. "
            "tools/check_splat.py applies the thresholds."
        ),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}: median {float(np.median(scores)):.2f} dB over "
          f"{len(scores)} views, {payload['gaussian_count']:,} Gaussians")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
