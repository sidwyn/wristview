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
    picks = np.linspace(0, len(ordered) - 1, min(args.views, len(ordered))).astype(int)

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

    payload = {
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
