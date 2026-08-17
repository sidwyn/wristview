"""Render wrist-view videos beside their source egocentric frames.

Renders from the dense point cloud rather than the Gaussian splat, because the
splat needs a long training run and, on Apple Silicon, a hand-written
rasterizer. The point cloud is metrically correct where the reconstruction is,
which is what a geometry review needs.

Each output frame is: source egocentric frame on the left, synthesised wrist
view on the right, with a caption that says what is measured and what is not.

    python -m tools.render_wrist_videos --run runs/real03 --out deliverables/wrist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.backends import dense_cloud as dc  # noqa: E402
from wristview.backends import gripper as gripper_backend  # noqa: E402
from wristview.backends import mesh_render as mr  # noqa: E402
from wristview.camera import Intrinsics, fisheye_maps  # noqa: E402
from wristview.geometry import invert_pose, transform_points  # noqa: E402
from wristview.logging_setup import get, setup  # noqa: E402
from wristview.runctx import read_json  # noqa: E402
from wristview.stages.s05_render import wrist_camera_offset  # noqa: E402
from wristview.videoio import write_video  # noqa: E402

log = get(__name__)

PANEL_W, PANEL_H = 640, 480


def label(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], origin=(10, 26)):
    """Caption with a dark plate behind it, so text stays readable on any frame."""
    x, y = origin
    for text, colour in lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(image, (x - 5, y - th - 6), (x + tw + 5, y + 6), (0, 0, 0), -1)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        y += th + 12
    return image


def render_clip(
    run_root: Path,
    clip_id: str,
    points: np.ndarray,
    colors: np.ndarray,
    out_dir: Path,
    fps: float,
    splat_px: int,
    fisheye: bool,
    clean: bool = False,
) -> dict:
    """Render one clip's side-by-side sequence. Returns a small report."""
    manifest = read_json(run_root / "00_ingest" / "manifest.json")["clips"][clip_id]
    frames_dir = run_root / manifest["frames_dir"]
    frame_names = manifest["frame_names"]

    traj = np.load(run_root / "04_retarget" / clip_id / "ee_trajectory.npz")
    ee = traj["poses_video_rate"]
    widths = traj["width_video_rate"]
    closed = traj["closed_video_rate"]
    hand_valid = traj["hand_valid"]

    # The object, segmented from the scan. It sits where the scan saw it until
    # the grasp, then it is carried rigidly by the gripper. That is an
    # assumption, not a measurement: it holds while the grasp is firm and the
    # object does not slip or rotate in the hand, and it is captioned as such
    # on every frame so nobody mistakes it for tracking.
    object_points = object_colors = None
    grasp_onset = None
    object_path = run_root / "01_scene" / "object.ply"
    if object_path.exists():
        object_points, object_colors = dc.read_ply(object_path)
        onsets = np.nonzero(closed)[0]
        grasp_onset = int(onsets[0]) if len(onsets) else None

    import yaml

    cfg = yaml.safe_load((run_root / "config.yaml").read_text())
    wrist_cfg = cfg["render"]["wrist_camera"]
    offset = wrist_camera_offset(wrist_cfg)
    intrinsics = Intrinsics.from_fov(PANEL_W, PANEL_H, float(wrist_cfg.get("fov_deg", 90.0)))
    spec = gripper_backend.load(cfg["retarget"])

    maps = fisheye_maps(intrinsics, list(cfg["render"]["fisheye_coeffs"])) if fisheye else None

    seq_dir = out_dir / clip_id
    seq_dir.mkdir(parents=True, exist_ok=True)
    for stale in seq_dir.glob("*.png"):
        stale.unlink()

    count = min(len(ee), len(frame_names))
    coverage = np.zeros(count)

    for index in range(count):
        # ---- right panel: the synthesised wrist view ----
        view = invert_pose(ee[index] @ offset)
        scene_rgb, scene_depth = mr.rasterize_points_fast(
            points, colors, view,
            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            PANEL_W, PANEL_H, near=0.02, far=6.0, splat_px=splat_px,
        )
        coverage[index] = float(np.isfinite(scene_depth).mean())

        layers = [(scene_rgb, scene_depth)]

        # ---- the object, static before the grasp and carried after ----
        object_state = "none"
        if object_points is not None:
            if grasp_onset is not None and index >= grasp_onset and closed[index]:
                carried = ee[index] @ invert_pose(ee[grasp_onset])
                posed = transform_points(carried, object_points)
                object_state = "carried"
            else:
                posed = object_points
                object_state = "static"
            o_rgb, o_depth = mr.rasterize_points_fast(
                posed, object_colors, view,
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                PANEL_W, PANEL_H, near=0.02, far=6.0, splat_px=splat_px,
            )
            layers.append((o_rgb, o_depth))

        boxes = gripper_backend.jaw_boxes(spec, float(widths[index]))
        meshes = [mr.box_mesh(c, h) for c, h in boxes]
        verts, faces = mr.combine(meshes)
        verts = transform_points(ee[index], verts)
        shade = (0.25, 0.28, 0.34) if closed[index] else (0.45, 0.48, 0.55)
        g_rgb, g_depth = mr.rasterize_mesh(
            verts, faces, view, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            PANEL_W, PANEL_H, color=shade, near=0.02,
        )
        layers.append((g_rgb, g_depth))

        composed, _ = mr.composite(layers, np.array([0.04, 0.04, 0.05]))
        right = (np.clip(composed, 0, 1) * 255).astype(np.uint8)
        right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
        if maps is not None:
            right = cv2.remap(right, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        right = np.ascontiguousarray(right)

        # ---- left panel: what the operator actually filmed ----
        source = cv2.imread(str(frames_dir / frame_names[index]))
        left = np.ascontiguousarray(cv2.resize(source, (PANEL_W, PANEL_H)))

        if not clean:
            label(left, [("SOURCE  egocentric camera", (255, 255, 255))])
        object_caption = {
            "carried": ("object CARRIED by gripper (rigid, from grasp)", (140, 220, 255)),
            "static": ("object STATIC at scan position", (180, 180, 180)),
            "none": ("no object segmented", (140, 140, 140)),
        }[object_state]
        if not clean:
            label(right, [
                ("RENDERED  virtual wrist camera", (255, 255, 255)),
                (f"scene coverage {coverage[index] * 100:.0f}%", (180, 220, 180)),
                ("hand tracked" if hand_valid[index] else "hand LOST, pose held",
                 (180, 220, 180) if hand_valid[index] else (120, 160, 255)),
                (f"gripper proxy {'CLOSED' if closed[index] else 'open'} "
                 f"{widths[index] * 100:.1f} cm", (200, 200, 255)),
                object_caption,
            ])

        pair = np.hstack([left, np.full((PANEL_H, 4, 3), 40, np.uint8), right])
        cv2.imwrite(str(seq_dir / f"{index:05d}.png"), pair)

        if (index + 1) % 50 == 0:
            log.info("  %s: %d/%d frames", clip_id, index + 1, count)

    video = write_video(seq_dir, out_dir / f"{clip_id}.mp4", fps=fps, pattern="%05d.png")
    return {
        "clip": clip_id,
        "frames": count,
        "video": str(video) if video else None,
        "coverage_mean": round(float(coverage.mean()), 4),
        "coverage_min": round(float(coverage.min()), 4),
        "hand_tracked_fraction": round(float(hand_valid[:count].mean()), 4),
        "gripper_closed_frames": int(closed[:count].sum()),
        "grasp_onset_frame": grasp_onset,
        "object_points": int(len(object_points)) if object_points is not None else 0,
        "frames_object_carried": int(sum(
            1 for i in range(count)
            if grasp_onset is not None and i >= grasp_onset and closed[i]
        )),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default="deliverables/wrist")
    parser.add_argument("--cloud", default=None, help="defaults to <run>/01_scene/dense.ply")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--splat-px", type=int, default=1)
    parser.add_argument("--fisheye", action="store_true")
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--clean", action="store_true",
                        help="no captions and no overlay: source on the left, wrist view "
                             "on the right, nothing else. This is the website asset.")
    args = parser.parse_args()

    setup(None, verbose=False)
    run_root = Path(args.run).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cloud_path = Path(args.cloud) if args.cloud else run_root / "01_scene" / "dense.ply"
    log.info("reading point cloud %s", cloud_path)
    points, colors = dc.read_ply(cloud_path)
    log.info("cloud: %d points", len(points))

    retarget = read_json(run_root / "04_retarget" / "summary.json")
    clips = args.clips or [c for c, v in retarget.items() if not v.get("rejected")]

    reports = []
    for clip_id in clips:
        log.info("rendering %s", clip_id)
        reports.append(
            render_clip(run_root, clip_id, points, colors, out_dir,
                        args.fps, args.splat_px, args.fisheye, clean=args.clean)
        )

    (out_dir / "render_report.json").write_text(json.dumps(reports, indent=2))
    log.info("wrote %d videos to %s", len(reports), out_dir)
    for r in reports:
        log.info("  %s: %d frames, coverage %.0f%%, hand %.0f%%",
                 r["clip"], r["frames"], r["coverage_mean"] * 100,
                 r["hand_tracked_fraction"] * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
