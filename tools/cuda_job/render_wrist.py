"""Rasterise the wrist camera views of a splat, on CUDA.

This draws the cameras `tools/export_wrist_cameras.py` computed. It does not
compute a mount, choose a viewpoint, or apply a lens. Stage 5 owns all three,
and a second copy here is a second thing to disagree.

It returns colour and depth. Stage 5 needs the depth to decide which pixels of
the gripper mesh sit in front of the scene.

    python render_wrist.py --splat /root/result/splat.pt \
        --cameras /root/cameras --out /root/wrist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

# Depth is stored as a 16 bit PNG in millimetres. That covers 65.5 m at 1 mm,
# against a far plane of 12 m, and it compresses. float32 would be 4x the
# transfer for precision the compositor cannot use.
DEPTH_SCALE_MM = 1000.0
DEPTH_MAX_MM = 65535


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splat", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    args = parser.parse_args()

    cam_dir = Path(args.cameras)
    meta = json.loads((cam_dir / "cameras.json").read_text())
    data = np.load(cam_dir / "wrist_cameras.npz")
    view_matrices = data["view_matrices"]

    out = Path(args.out)
    (out / "color").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "alpha").mkdir(parents=True, exist_ok=True)

    params = torch.load(args.splat, map_location="cuda", weights_only=True)
    quats = params["quats"] / params["quats"].norm(dim=-1, keepdim=True)
    scales = torch.exp(params["scales"])
    opacities = torch.sigmoid(params["opacities"])

    width, height = meta["width"], meta["height"]
    K = torch.tensor([
        [meta["fx"], 0.0, meta["cx"]],
        [0.0, meta["fy"], meta["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device="cuda")[None]

    alphas: list[float] = []
    for index, world_to_cam in enumerate(view_matrices):
        with torch.no_grad():
            image, alpha, _ = rasterization(
                means=params["means"], quats=quats, scales=scales,
                opacities=opacities, colors=params["sh"],
                viewmats=torch.tensor(world_to_cam, dtype=torch.float32, device="cuda")[None],
                Ks=K, width=width, height=height,
                sh_degree=args.sh_degree, packed=True,
                near_plane=meta["near_plane_m"], far_plane=meta["far_plane_m"],
                render_mode="RGB+ED",
            )
        rgb = image[0, :, :, :3].clamp(0, 1).cpu().numpy()
        depth = image[0, :, :, 3].cpu().numpy()
        alphas.append(float(alpha[0].mean()))

        cv2.imwrite(str(out / "color" / f"{index:05d}.png"),
                    (rgb[:, :, ::-1] * 255).astype(np.uint8))
        millimetres = np.clip(depth * DEPTH_SCALE_MM, 0, DEPTH_MAX_MM).astype(np.uint16)
        cv2.imwrite(str(out / "depth" / f"{index:05d}.png"), millimetres)
        # Alpha decides which pixels hold a surface. Depth alone does not: the
        # expected-depth channel is non-zero wherever any Gaussian contributes
        # at all, however faintly, so a coverage figure taken from it reads
        # 100 per cent on a view that is largely empty.
        coverage = (alpha[0, :, :, 0].cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(out / "alpha" / f"{index:05d}.png"), coverage)
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(view_matrices)}", flush=True)

    report = dict(meta)
    report["renderer"] = "gsplat"
    report["splat_alpha_mean"] = round(float(np.mean(alphas)), 4)
    report["splat_alpha_min"] = round(float(np.min(alphas)), 4)
    report["depth_scale_mm"] = DEPTH_SCALE_MM
    (out / "render.json").write_text(json.dumps(report, indent=2))

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"rendered {len(view_matrices)} frames at {width}x{height}, "
          f"{size / 1e6:.0f} MB, mean splat alpha {report['splat_alpha_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
