"""Stage 5 · Wrist-view rendering.

In:  splat from Stage 1, end-effector trajectory from Stage 4, object pose and
     object model from Stage 3, gripper spec.
Out: a wrist-camera image sequence.

The property this stage rests on, and the reason the approach works at all:
the splat was trained on the scan, and the scan has no person in it. So the
human arm is simply absent from the background. No inpainting, no matting, no
video diffusion. The arm disappears for free.

Composited per frame, in depth order:
  splat      the static scene, rendered from the wrist pose
  object     the point model from Stage 3, at its per-frame pose
  gripper    boxes from the URDF, opened to the retargeted width

The pinhole render is then warped into the deployment camera model, so the
images match the lens the policy will actually see through.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ..backends import gripper as gripper_backend
from ..backends import mesh_render
from ..backends.splat_mps import render as splat_render
from ..mount import wrist_camera_offset
from ..camera import Intrinsics, fisheye_maps
from ..device import resolve as resolve_device
from ..geometry import invert_pose, make_pose, transform_points
from ..logging_setup import get
from ..runctx import RunContext, StageRecorder, read_json, write_json
from ..videoio import write_video

log = get(__name__)

STAGE = 5
NAME = "render"



def _render_episode(
    ctx: RunContext,
    rec: StageRecorder,
    clip_id: str,
    splat,
    spec: gripper_backend.GripperSpec,
    device: str,
    cfg: dict,
) -> dict:
    """Render one episode's wrist views."""
    out_dir = ctx.episode_dir(STAGE, clip_id)
    wrist_dir = out_dir / "wrist"
    wrist_dir.mkdir(parents=True, exist_ok=True)

    trajectory = np.load(ctx.episode_dir(4, clip_id, create=False) / "ee_trajectory.npz")

    # Render the control-rate trajectory, not the video-rate one. A wrist
    # frame is only useful paired with the action taken at that instant, and
    # the action stream is at the control rate. At 60 fps input and 15 Hz
    # control that is a quarter of the frames for the same dataset.
    rate = str(cfg.get("sample_rate", "control"))
    if rate == "video":
        ee_poses = trajectory["poses_video_rate"]
        widths = trajectory["width_video_rate"]
        closed = trajectory["closed_video_rate"]
        timestamps = np.arange(len(ee_poses)) / float(trajectory["source_fps"])
    else:
        ee_poses = trajectory["poses"]
        widths = trajectory["width_m"]
        closed = trajectory["closed"]
        timestamps = trajectory["timestamps_s"]

    limit = int(cfg.get("max_frames", 0) or 0)
    if limit and len(ee_poses) > limit:
        log.warning(
            "%s: capping the render at %d of %d frames (render.max_frames)",
            clip_id, limit, len(ee_poses),
        )
        keep = np.linspace(0, len(ee_poses) - 1, limit).astype(int)
        ee_poses, widths, closed = ee_poses[keep], widths[keep], closed[keep]
        timestamps = timestamps[keep]

    estimate_dir = ctx.episode_dir(3, clip_id, create=False)
    object_poses_video = np.load(estimate_dir / "object_pose.npy")
    object_valid_video = np.load(estimate_dir / "object_valid.npy")

    # Object poses are indexed by demo frame. Map each render sample onto the
    # nearest one rather than assuming the two sequences line up.
    source_fps = float(trajectory["source_fps"])
    frame_index = np.clip(
        np.round(timestamps * source_fps).astype(int), 0, len(object_poses_video) - 1
    )
    object_poses = object_poses_video[frame_index]
    object_valid = object_valid_video[frame_index]
    model_path = estimate_dir / "object_model.npy"
    object_model = np.load(model_path) if model_path.exists() else None

    width = int(cfg.get("width", 640))
    height = int(cfg.get("height", 480))
    wrist_cfg = cfg.get("wrist_camera", {})
    intrinsics = Intrinsics.from_fov(width, height, float(wrist_cfg.get("fov_deg", 90.0)))
    offset = wrist_camera_offset(wrist_cfg)

    near = float(cfg.get("near_plane_m", 0.02))
    far = float(cfg.get("far_plane_m", 12.0))
    background = np.asarray(cfg.get("background_rgb", [0.05, 0.05, 0.06]), dtype=np.float64)
    background_tensor = torch.tensor(background, device=device, dtype=torch.float32)

    # The fisheye maps depend only on the intrinsics, so build them once.
    maps = None
    if cfg.get("camera_model", "pinhole") == "fisheye":
        maps = fisheye_maps(intrinsics, list(cfg.get("fisheye_coeffs", [0.0, 0.0, 0.0, 0.0])))

    draw_gripper = bool(cfg.get("draw_gripper", True))
    draw_object = bool(cfg.get("draw_object", True)) and object_model is not None

    count = len(ee_poses)
    splat_coverage = np.zeros(count)

    for index in range(count):
        wrist_pose = ee_poses[index] @ offset
        view_matrix = invert_pose(wrist_pose)

        # --- splat: the scene, with no human in it ---
        splat_color = np.tile(background, (height, width, 1))
        splat_depth = np.full((height, width), np.inf)
        if splat is not None:
            view = torch.tensor(view_matrix, device=device, dtype=torch.float32)
            with torch.no_grad():
                result = splat_render(
                    splat, view, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                    width, height, background=background_tensor,
                    tile_size=16, max_per_tile=256, near=near, far=far,
                )
            splat_color = result.rgb.clamp(0, 1).cpu().numpy().astype(np.float64)
            alpha = result.alpha.cpu().numpy()
            raw_depth = result.depth.cpu().numpy().astype(np.float64)
            # Where the splat is transparent there is no surface, so it must
            # not win the depth test against the gripper.
            splat_depth = np.where(alpha > 0.35, raw_depth, np.inf)
            splat_coverage[index] = float((alpha > 0.35).mean())

        layers = [(splat_color, splat_depth)]

        # --- object ---
        if draw_object and object_valid[index]:
            posed = transform_points(object_poses[index], object_model)
            object_color, object_depth = mesh_render.rasterize_points(
                posed, None, view_matrix,
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                width, height, point_radius_m=0.005, near=near,
            )
            layers.append((object_color, object_depth))

        # --- gripper ---
        if draw_gripper:
            boxes = gripper_backend.jaw_boxes(spec, float(widths[index]))
            meshes = [mesh_render.box_mesh(center, half) for center, half in boxes]
            vertices, faces = mesh_render.combine(meshes)
            # The boxes are in the gripper frame; move them into the world.
            vertices = transform_points(ee_poses[index], vertices)
            shade = (0.28, 0.30, 0.34) if closed[index] else (0.42, 0.44, 0.48)
            gripper_color, gripper_depth = mesh_render.rasterize_mesh(
                vertices, faces, view_matrix,
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                width, height, color=shade, near=near,
            )
            layers.append((gripper_color, gripper_depth))

        image, _ = mesh_render.composite(layers, background)
        frame = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if maps is not None:
            frame = cv2.remap(
                frame, maps[0], maps[1],
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )

        cv2.imwrite(str(wrist_dir / f"{index:05d}.png"), frame)
        if (index + 1) % 50 == 0 or index == count - 1:
            log.info("  %s: rendered %d/%d wrist frames", clip_id, index + 1, count)

    preview = None
    if bool(cfg.get("write_preview_video", True)):
        preview = write_video(
            wrist_dir, out_dir / "wrist.mp4",
            fps=float(cfg.get("preview_fps", 15)), pattern="%05d.png",
        )

    _write_contact_sheet(wrist_dir, out_dir / "contact_sheet.png", count)

    status = {
        "clip_id": clip_id,
        "frames": count,
        "resolution": [width, height],
        "camera_model": cfg.get("camera_model", "pinhole"),
        "fov_deg": float(wrist_cfg.get("fov_deg", 90.0)),
        "wrist_offset_m": list(np.asarray(wrist_cfg.get("translation_m", []), dtype=float)),
        "wrist_rotation_rpy_deg": list(
            np.asarray(wrist_cfg.get("rotation_rpy_deg", []), dtype=float)
        ),
        "splat_coverage_mean": round(float(splat_coverage.mean()), 4),
        "splat_coverage_min": round(float(splat_coverage.min()), 4),
        "gripper_drawn": draw_gripper,
        "object_drawn": bool(draw_object),
        "closed_frames": int(closed.sum()),
        "frames_dir": ctx.rel(wrist_dir),
        "preview": ctx.rel(preview) if preview else None,
    }

    # Splat quality falls off away from the scanned viewpoints, and the wrist
    # camera looks from angles nobody walked through. Low coverage is the
    # signal for that, so surface it rather than shipping empty frames quietly.
    if status["splat_coverage_mean"] < 0.5:
        status["warning"] = (
            f"the splat covers only {status['splat_coverage_mean'] * 100:.0f}% of the "
            f"average wrist frame. The wrist camera is looking at angles the scan "
            f"never saw. Re-scan closer to the workspace and from lower viewpoints."
        )
        log.warning("%s: %s", clip_id, status["warning"])

    # ---- did the scan ever see these viewpoints -------------------------
    #
    # The coverage warning above reads the splat's alpha, which is not the same
    # question and can be reassuring while the render is worthless: session 6
    # filled 87 to 94 per cent of every frame from viewpoints 24 to 44 cm away
    # from anything that ever saw the scene, and 100 per cent of its frames sat
    # below the scan's lowest view. Alpha says a Gaussian was drawn there. It
    # does not say a camera was ever there to constrain it.
    scale_path = ctx.stage_dir(1, create=False) / "scale.json"
    cameras_path = ctx.stage_dir(1, create=False) / "cameras.json"
    if scale_path.exists() and cameras_path.exists():
        plane = (read_json(scale_path).get("diagnostics") or {}).get("desk_plane")
        frames = read_json(cameras_path).get("frames") or []
        if plane and plane.get("offset_m") is not None and frames:
            from .. import coverage as coverage_module

            scan_centres = np.array([
                np.asarray(f["pose_world_from_cam"])[:3, 3] for f in frames
            ])
            eyes = np.array([(ee_poses[i] @ offset)[:3, 3] for i in range(count)])
            report = coverage_module.render_viewpoint_coverage(
                scan_centres, eyes,
                np.asarray(plane["normal"], dtype=float), float(plane["offset_m"]),
            )
            status["viewpoint_coverage"] = report
            log.info(
                "%s: nearest scan view, median %.1f cm, p90 %.1f cm; %.0f%% of "
                "frames below the scan floor at %.1f cm",
                clip_id, report["nearest_scan_view_median_m"] * 100,
                report["nearest_scan_view_p90_m"] * 100,
                report["fraction_below_scan_floor"] * 100,
                report["scan_floor_m"] * 100,
            )
            for failure in report.get("failures", []):
                log.error("%s: VIEWPOINT COVERAGE: %s", clip_id, failure)
            if report.get("failures") and not bool(
                (ctx.config.get("render", {}) or {}).get("allow_extrapolation", False)
            ):
                raise ValueError(
                    f"{clip_id}: the wrist camera renders from viewpoints the scan "
                    f"never covered, so this render is extrapolation rather than "
                    f"interpolation. "
                    + "; ".join(report["failures"])
                    + ". Set render.allow_extrapolation to render anyway, and say "
                      "so wherever the video is shown."
                )

    write_json(out_dir / "status.json", status)
    log.info(
        "%s: %d wrist frames at %dx%d, splat coverage %.0f%%, preview %s",
        clip_id, count, width, height, status["splat_coverage_mean"] * 100,
        status["preview"],
    )
    return status


def _write_contact_sheet(frames_dir: Path, out_path: Path, count: int, columns: int = 4) -> None:
    """A single image of frames spread across the episode.

    This is the artifact that goes to a buyer. Three frames in an email beat a
    directory of a hundred and eighty.
    """
    if count == 0:
        return
    picks = np.linspace(0, count - 1, min(8, count)).astype(int)
    tiles = []
    for index in picks:
        image = cv2.imread(str(frames_dir / f"{index:05d}.png"))
        if image is None:
            continue
        cv2.putText(
            image, f"f{index}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        tiles.append(image)
    if not tiles:
        return

    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(out_path), np.vstack(rows))


def run(ctx: RunContext) -> dict:
    """Run Stage 5 over the run directory."""
    rec = StageRecorder(ctx, STAGE, NAME)
    cfg = ctx.config.section("render")

    try:
        retarget_summary = read_json(ctx.stage_dir(4, create=False) / "summary.json")
        rec.meta.inputs = {
            "retarget_summary": ctx.rel(ctx.stage_dir(4, create=False) / "summary.json")
        }

        device = resolve_device(
            ctx.config.get("device.preferred", "auto"),
            bool(ctx.config.get("device.allow_cpu_fallback", True)),
        )
        rec.backend("device", device)
        rec.backend("splat_renderer", "wristview_mps_rasterizer")
        rec.backend("camera_model", cfg.get("camera_model", "pinhole"))

        from .s01_scene import load_splat

        splat = load_splat(ctx, device)
        if splat is None:
            rec.note(
                "no splat from Stage 1. The wrist views will show only the gripper "
                "and the object against a flat background."
            )
            log.warning("no splat available; rendering geometry only")
        else:
            log.info("splat loaded: %d gaussians", splat.count)

        spec = gripper_backend.load(ctx.config.section("retarget"))
        rec.backend("gripper", spec.source)

        statuses = {}
        for clip_id, entry in retarget_summary.items():
            if entry.get("rejected"):
                log.warning("%s: skipped, rejected by Stage 4", clip_id)
                continue
            log.info("--- Stage 5: %s ---", clip_id)
            with rec.timed(f"render.{clip_id}"):
                statuses[clip_id] = _render_episode(
                    ctx, rec, clip_id, splat, spec, device, cfg
                )
            rec.output(f"{clip_id}_wrist", ctx.episode_dir(STAGE, clip_id) / "wrist")
            rec.output(
                f"{clip_id}_preview", ctx.episode_dir(STAGE, clip_id) / "wrist.mp4"
            )
            rec.output(
                f"{clip_id}_contact_sheet",
                ctx.episode_dir(STAGE, clip_id) / "contact_sheet.png",
            )

        summary_path = write_json(ctx.stage_dir(STAGE) / "summary.json", statuses)
        rec.output("summary", summary_path)
        rec.metric("episodes", len(statuses))
        rec.metric("frames", sum(s["frames"] for s in statuses.values()))
        rec.metric(
            "splat_coverage", {k: v["splat_coverage_mean"] for k, v in statuses.items()}
        )

        rec.write("ok" if statuses else "failed")
        return statuses

    except Exception as exc:
        rec.note(f"{type(exc).__name__}: {exc}")
        rec.write("failed")
        raise
