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

**The viewpoint gate runs here, before anything is uploaded.** It used to run
only in Stage 5, which happens after the pod has trained a splat and drawn
every frame. real27full therefore exported 4,673 cameras, rented a 4090,
trained for 40 minutes, rendered 14,019 frames and downloaded 1.9 GB before
reaching the code that says 20.8 per cent of those cameras are further than
15 cm from any scan view. The gate needs only the scan poses and the wrist
poses, and both are on the laptop when this script runs. A gate that runs after
the money is spent is not a gate.

The thresholds still live in `wristview.coverage`, so this and Stage 5 read one
set of numbers and cannot drift apart.

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
from wristview.coverage import render_viewpoint_coverage  # noqa: E402
from wristview.geometry import invert_pose  # noqa: E402
from wristview.mount import assert_above_plane, wrist_camera_offset  # noqa: E402


def scan_centres(run: Path) -> np.ndarray:
    """Camera centres of every registered scan view, in metres.

    Read from `01_scene/cameras.json`, which holds the poses Stage 1 itself
    used. Stage 1 transforms the reconstruction in place, so the COLMAP model
    on disk is already metric; reading it back and applying the scale factor a
    second time shrinks every camera by 12x and turns this gate into fiction.
    """
    frames = json.loads((run / "01_scene" / "cameras.json").read_text())["frames"]
    return np.array([np.asarray(f["pose_world_from_cam"])[:3, 3] for f in frames])


def desk_plane(run: Path) -> tuple[np.ndarray, float]:
    plane = json.loads(
        (run / "01_scene" / "scale.json").read_text()
    )["diagnostics"].get("desk_plane")
    if not plane or plane.get("offset_m") is None:
        raise SystemExit(
            f"{run.name}: 01_scene/scale.json has no desk plane, so the "
            "viewpoint gate cannot run. Stage 1 fits the plane; rerun it "
            "rather than exporting cameras that nothing has checked."
        )
    return np.asarray(plane["normal"], dtype=float), float(plane["offset_m"])


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
        # This used to be `np.ones(len(poses))`: every control-rate camera
        # declared valid, whatever it rested on. real27 shipped 1,901 of 4,673
        # cameras built on held poses that way, and 290 of them were below the
        # desk. Stage 4 knows which control samples are supported by a
        # measurement on both sides, so read it, exactly as the video-rate
        # branch below reads `hand_valid`.
        if "hand_valid_control" not in traj:
            raise SystemExit(
                f"{args.clip}: 04_retarget/ee_trajectory.npz has no "
                f"`hand_valid_control`, so this run predates the control-rate "
                f"validity fix and every camera in it would be exported as "
                f"valid whether or not a hand was ever measured there. Re-run "
                f"Stage 4 for this run."
            )
        valid = traj["hand_valid_control"].astype(bool)
    else:
        poses, source = traj["poses_video_rate"], "video"
        valid = traj["hand_valid"].astype(bool)

    intrinsics = Intrinsics.from_fov(
        args.width, args.height, float(cfg["wrist_camera"].get("fov_deg", 90.0))
    )
    offset = wrist_camera_offset(cfg["wrist_camera"])

    view_matrices = np.stack([invert_pose(pose @ offset) for pose in poses])
    camera_positions = np.stack([(pose @ offset)[:3, 3] for pose in poses])

    # A camera below the work surface is not a viewpoint. That is geometry,
    # not a threshold, so those frames are invalid in the same sense as the
    # unmeasured ones and are dropped here. `assert_above_plane` below is then
    # a post-condition: it must never fire, and if it does the drop is wrong.
    normal, offset = desk_plane(run)
    above = (camera_positions @ (normal / np.linalg.norm(normal))) - offset >= 0.0
    below_desk = int((valid & ~above).sum())
    valid = valid & above

    # Drop rather than mark. A camera that rests on no measurement is not a
    # view of anything, and shipping it costs GPU time to render a held frame
    # that must then be filtered again downstream. `source_index` keeps the
    # mapping explicit so nothing has to reconstruct it.
    source_index = np.nonzero(valid)[0]
    dropped = int((~valid).sum())
    view_matrices = view_matrices[source_index]
    camera_positions = camera_positions[source_index]
    valid = np.ones(len(source_index), dtype=bool)

    # ---- the gate, before a single byte is uploaded ---------------------
    #
    # A post-condition on the drop above, not a filter. If it fires, the drop
    # is wrong, and `wrist_camera_offset` validating only its standoff
    # parameter is why nothing caught this for four sessions.
    assert_above_plane(camera_positions, normal, offset, clip_id=args.clip)

    coverage = render_viewpoint_coverage(
        scan_centres(run), camera_positions, normal, offset, render_valid=valid,
    )
    (out / "viewpoint_coverage.json").write_text(json.dumps(coverage, indent=2))
    print(
        f"{args.clip}: nearest scan view, median "
        f"{coverage['nearest_scan_view_median_m'] * 100:.1f} cm, p90 "
        f"{coverage['nearest_scan_view_p90_m'] * 100:.1f} cm; "
        f"{coverage['fraction_below_scan_floor'] * 100:.1f} per cent of frames "
        f"below the scan floor at {coverage['scan_floor_m'] * 100:.1f} cm"
    )
    if coverage["passed"] is not True:
        for failure in coverage.get("failures", []):
            print(f"  VIEWPOINT COVERAGE: {failure}")
        if not bool(cfg.get("allow_extrapolation", False)):
            print(
                f"\nGATE FAILED for {args.clip}. No cameras written, nothing to "
                f"upload. A splat is an interpolator and this is the one render "
                f"defect only a capture change can fix: shoot the scan from "
                f"where the wrist camera goes. The report is at "
                f"{out / 'viewpoint_coverage.json'}."
            )
            return 1

    np.savez_compressed(
        out / "wrist_cameras.npz",
        view_matrices=view_matrices.astype(np.float64),
        camera_positions=camera_positions.astype(np.float64),
        valid=valid,
        source_index=source_index.astype(np.int64),
    )
    meta = {
        "run": run.name,
        "clip": args.clip,
        "frames": int(len(source_index)),
        "frames_before_validity_filter": int(len(poses)),
        "frames_dropped_unmeasured": dropped,
        "frames_dropped_below_desk": below_desk,
        # Which control frames these cameras are. Stage 5 reads this so it
        # renders exactly the frames the layer holds, instead of re-deriving
        # the selection from validity and geometry and disagreeing by two.
        "source_index": [int(i) for i in source_index],
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
    print(f"wrote {out}: {len(source_index)} cameras at {args.width}x{args.height}, "
          f"{size / 1e3:.0f} kB, {dropped} dropped as unmeasured, "
          f"{below_desk} dropped below the desk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
