"""Measure MPS allocator memory across a short splat training run.

The Metal path ran out of memory once and the cause was guessed at rather than
measured. This runs a fixed number of steps and records the allocator every
`--every` steps, so the curve can be read instead of argued about.

    python -m tools.allocator_curve --run runs/real03 --steps 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.logging_setup import get, setup  # noqa: E402

log = get(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--empty-cache-every", type=int, default=0,
                        help="call torch.mps.empty_cache every N steps, 0 to never")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup(None, verbose=False)

    import torch

    from wristview.backends import dense_cloud as dc
    from wristview.backends.splat_trainer import TrainCamera, train_splat
    from wristview.camera import Intrinsics
    from wristview.config import Config
    from wristview.geometry import invert_pose
    from wristview.runctx import read_json

    root_path = Path(args.run)
    config = Config.load(root_path / "config.yaml")
    cfg = dict(config.section("scene").get("splat", {}))
    if not cfg:
        raise SystemExit("no scene.splat section in the config")
    cfg["iterations"] = args.steps

    root = root_path
    points, colors = dc.read_ply(root / "01_scene" / "dense.ply")
    scene = read_json(root / "01_scene" / "cameras.json")
    intrinsics = Intrinsics.from_dict(scene["intrinsics"])
    manifest = read_json(root / "00_ingest" / "manifest.json")["clips"]["scan"]
    frames_dir = root / manifest["frames_dir"]

    import cv2

    long_side = int(cfg.get("train_resolution", 800))
    cameras = []
    for frame in scene["frames"]:
        image = cv2.imread(str(frames_dir / frame["name"]))
        if image is None:
            continue
        scale = min(1.0, long_side / max(image.shape[1], image.shape[0]))
        width = int(round(image.shape[1] * scale))
        height = int(round(image.shape[0] * scale))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        scaled = intrinsics.scaled(width, height)
        view = invert_pose(np.asarray(frame["pose_world_from_cam"], dtype=np.float64))
        cameras.append(TrainCamera(
            name=frame["name"],
            view_matrix=torch.tensor(view, dtype=torch.float32, device="mps"),
            fx=scaled.fx, fy=scaled.fy, cx=scaled.cx, cy=scaled.cy,
            width=width, height=height,
            image=torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0,
                               dtype=torch.float32, device="mps"),
        ))
    log.info("%d training cameras at %dx%d, %d seed points",
             len(cameras), cameras[0].width, cameras[0].height, len(points))

    rows = []

    def probe(step: int, gaussians: int) -> None:
        if args.empty_cache_every and step % args.empty_cache_every == 0:
            torch.mps.empty_cache()
        if step % args.every and step != 1:
            return
        torch.mps.synchronize()
        rows.append({
            "step": step,
            "gaussians": int(gaussians),
            "allocated_gib": torch.mps.current_allocated_memory() / 2**30,
            "driver_gib": torch.mps.driver_allocated_memory() / 2**30,
            "recommended_max_gib": torch.mps.recommended_max_memory() / 2**30,
            "seconds": time.time() - start,
        })
        log.info("step %5d  gaussians %7d  allocated %5.2f GiB  driver %5.2f GiB",
                 step, gaussians, rows[-1]["allocated_gib"], rows[-1]["driver_gib"])

    start = time.time()
    train_splat(points, colors, cameras, cfg, "mps", probe=probe)

    out = Path(args.out) if args.out else Path(args.run) / "allocator_curve.json"
    out.write_text(json.dumps(rows, indent=2))

    log.info("")
    log.info("%6s %10s %12s %12s", "step", "gaussians", "allocated", "driver")
    for r in rows:
        log.info("%6d %10d %10.2f GiB %8.2f GiB", r["step"], r["gaussians"],
                 r["allocated_gib"], r["driver_gib"])
    if len(rows) > 1:
        growth = rows[-1]["allocated_gib"] - rows[0]["allocated_gib"]
        per = growth / max(rows[-1]["gaussians"] - rows[0]["gaussians"], 1) * 1e6
        log.info("")
        log.info("allocated grew %.2f GiB over %d steps, %.1f MiB per 1000 Gaussians added",
                 growth, rows[-1]["step"] - rows[0]["step"], per * 1024)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
