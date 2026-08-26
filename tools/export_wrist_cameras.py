"""Export the wrist camera poses Stage 5 would render, for a GPU rasteriser.

The pod must draw the same cameras this repo draws. `train_gsplat.py` used to
recompute the mount itself from four constants, with a comment warning that the
two copies had to be kept in step. They were two implementations of one thing,
which is the arrangement that eventually disagrees.

So compute the cameras here, once, with `mount.wrist_camera_offset`, and ship
the matrices. The pod rasterises and decides nothing.

The pod returns the splat background and its depth. Compositing stays here: the
gripper mesh, the object points and the fisheye remap are Stage 5's, and moving
them would fork a second renderer to disagree with.

    python -m tools.export_wrist_cameras --run runs/real26 --clip demo_0 \
        --out exports/real26-cameras --width 640 --height 360
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.camera import Intrinsics  # noqa: E402
from wristview.geometry import invert_pose  # noqa: E402
from wristview.mount import wrist_camera_offset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--clip", default="demo_0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())["render"]
    run = Path(args.run).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    traj = np.load(run / "04_retarget" / args.clip / "ee_trajectory.npz")

    # `control` is one frame per action sample, which is what Stage 5 renders
    # by default and what pairs with a policy step.
    if cfg.get("sample_rate", "control") == "control":
        poses, source = traj["poses"], "control"
        valid = np.ones(len(poses), dtype=bool)
    else:
        poses, source = traj["poses_video_rate"], "video"
        valid = traj["hand_valid"].astype(bool)

    intrinsics = Intrinsics.from_fov(
        args.width, args.height, float(cfg["wrist_camera"].get("fov_deg", 90.0))
    )
    offset = wrist_camera_offset(cfg["wrist_camera"])

    view_matrices = np.stack([invert_pose(pose @ offset) for pose in poses])
    camera_positions = np.stack([(pose @ offset)[:3, 3] for pose in poses])

    np.savez_compressed(
        out / "wrist_cameras.npz",
        view_matrices=view_matrices.astype(np.float64),
        camera_positions=camera_positions.astype(np.float64),
        valid=valid,
    )
    meta = {
        "run": run.name,
        "clip": args.clip,
        "frames": int(len(poses)),
        "sample_rate": source,
        "width": args.width,
        "height": args.height,
        "fx": intrinsics.fx, "fy": intrinsics.fy,
        "cx": intrinsics.cx, "cy": intrinsics.cy,
        "near_plane_m": float(cfg.get("near_plane_m", 0.02)),
        "far_plane_m": float(cfg.get("far_plane_m", 12.0)),
        "background_rgb": list(cfg.get("background_rgb", [0.05, 0.05, 0.06])),
        "note": (
            "view_matrices are world to camera, already including the mount "
            "offset. The pod rasterises these and nothing else. The fisheye "
            "remap, the gripper and the object are applied on the laptop, "
            "because Stage 5 owns them."
        ),
    }
    (out / "cameras.json").write_text(json.dumps(meta, indent=2))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out}: {len(poses)} cameras at {args.width}x{args.height}, "
          f"{size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
