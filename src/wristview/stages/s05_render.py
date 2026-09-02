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

from .. import objectbox
from ..backends import gripper as gripper_backend
from ..backends import mesh_render
from ..backends.splat_mps import render as splat_render
from ..camera import Intrinsics, fisheye_maps
from ..device import resolve as resolve_device
from ..geometry import invert_pose, transform_points
from ..logging_setup import get
from ..mount import wrist_camera_offset
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
    source_fps_value = float(trajectory["source_fps"])
    if rate == "video":
        ee_poses = trajectory["poses_video_rate"]
        widths = trajectory["width_video_rate"]
        closed = trajectory["closed_video_rate"]
        timestamps = np.arange(len(ee_poses)) / source_fps_value
        renderable = trajectory["hand_valid"].astype(bool)
    else:
        ee_poses = trajectory["poses"]
        widths = trajectory["width_m"]
        closed = trajectory["closed"]
        timestamps = trajectory["timestamps_s"]
        # Control-rate validity is computed where the resampling happens, in
        # Stage 4, and read here. It used to be re-derived by rounding each
        # control timestamp to the nearest video frame, which is a second
        # implementation of one thing: Stage 4 marks a control sample valid
        # only when the source frames on BOTH sides were measured, and rounding
        # to the nearest accepts a sample whose other neighbour was not. The
        # exporter reads the same array, so the two agree by construction and
        # the external splat layer's frame count matches this episode's.
        if "hand_valid_control" in trajectory:
            renderable = trajectory["hand_valid_control"].astype(bool)
        else:
            log.warning(
                "%s: 04_retarget has no hand_valid_control, so this run "
                "predates the control-rate validity fix. Falling back to "
                "rounding, which can disagree with the exporter by a few "
                "frames. Re-run Stage 4 to remove the guess.",
                clip_id,
            )
            video_valid = trajectory["hand_valid"].astype(bool)
            renderable = video_valid[
                np.clip(np.round(timestamps * source_fps_value).astype(int),
                        0, len(video_valid) - 1)
            ]

    # Render measured frames only.
    #
    # Neither branch used to read `hand_valid`. Stage 4 held the last measured
    # pose across every frame after tracking stopped, and Stage 5 drew all of
    # them: real26 produced 181 control-rate frames in which the camera did not
    # move by a single bit, and nothing in the output said so. A frame with no
    # measurement behind it is not data, and rendering it makes it look like
    # data.
    excluded = int((~renderable).sum())
    if not renderable.any():
        raise RuntimeError(
            f"{clip_id}: no frame has a hand measurement behind it, so there "
            f"is nothing to render."
        )
    if excluded:
        log.warning(
            "%s: rendering %d of %d frames. %d have no hand measurement and "
            "are excluded, not drawn.",
            clip_id, int(renderable.sum()), len(renderable), excluded,
        )
    # Keep the unfiltered arrays: an external splat layer may name a smaller
    # set of frames still, and re-slicing an already-sliced array with a mask
    # built against the original is how off-by-N bugs get in.
    full_poses, full_widths = ee_poses, widths
    full_closed, full_times = closed, timestamps

    render_index = np.nonzero(renderable)[0]
    ee_poses = ee_poses[renderable]
    widths = widths[renderable]
    closed = closed[renderable]
    timestamps = timestamps[renderable]

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

    # A splat layer rendered elsewhere. The local MPS rasteriser pads each tile
    # to `max_per_tile` and silently drops the rest, which at 256 loses tens of
    # millions of Gaussian-tile pairs on a splat of this size and draws
    # something that is not the scene. gsplat on a CUDA box draws it correctly,
    # so that render is loaded here and composited by the code below.
    #
    # Compositing, the lens, the gripper and the object stay in this file. Only
    # the splat layer moves.
    layer_dir = cfg.get("splat_layer_dir")
    external = Path(layer_dir) if layer_dir else None
    # One layer directory per clip. If the configured path holds a
    # subdirectory named for this clip, use that.
    #
    # Without this the same directory is composited into every episode. On
    # real28 a single `splat_layer_dir` was passed for all 30 clips and only
    # demo_0 was valid; the other 29 were that clip's scene composited under a
    # different clip's gripper and object, which looks plausible frame by
    # frame and is wrong in every one. A per-clip path cannot be expressed by
    # the config alone, because Stage 5 loops over clips internally.
    if external is not None and (external / clip_id).is_dir():
        external = external / clip_id
    external_meta = None
    if external is not None:
        external_meta = read_json(external / "render.json")
        if [external_meta["width"], external_meta["height"]] != [width, height]:
            raise ValueError(
                f"the external splat layer is "
                f"{external_meta['width']}x{external_meta['height']} and this "
                f"render is {width}x{height}. They must match, or the depth "
                f"test compares different pixels."
            )
        layer_index = external_meta.get("source_index")
        if layer_index is not None:
            # The layer names the control frames it holds, so render exactly
            # those. Re-deriving the selection here produced a two-frame
            # disagreement on demo_5, because the exporter also drops cameras
            # below the desk and this stage does not know that rule.
            layer_index = np.asarray(layer_index, dtype=int)
            keep_layer = np.zeros(len(full_poses), dtype=bool)
            keep_layer[layer_index[layer_index < len(full_poses)]] = True
            dropped_here = int((renderable & ~keep_layer).sum())
            if dropped_here:
                log.info(
                    "%s: the splat layer holds %d of the %d frames this stage "
                    "would render; following the layer.",
                    clip_id, int(keep_layer.sum()), int(renderable.sum()),
                )
            renderable = renderable & keep_layer
            render_index = np.nonzero(renderable)[0]
            ee_poses = full_poses[renderable]
            widths = full_widths[renderable]
            closed = full_closed[renderable]
            timestamps = full_times[renderable]
        elif int(external_meta["frames"]) != len(ee_poses):
            raise ValueError(
                f"the external splat layer holds {external_meta['frames']} "
                f"frames and this episode has {len(ee_poses)}."
            )
        log.info("splat layer from %s, rendered by %s",
                 external, external_meta.get("renderer", "unknown"))

    # The fisheye maps depend only on the intrinsics, so build them once.
    maps = None
    if cfg.get("camera_model", "pinhole") == "fisheye":
        maps = fisheye_maps(intrinsics, list(cfg.get("fisheye_coeffs", [0.0, 0.0, 0.0, 0.0])))

    # The object's body, written by Stage 3 next to its pose.
    object_box = None
    object_colour = (0.75, 0.65, 0.15)
    box_path = estimate_dir / "object_box.json"
    if box_path.exists():
        box = read_json(box_path)
        object_box = objectbox.box_mesh(box["dimensions_m"])
        object_colour = tuple(box["colour_rgb"])
        log.info(
            "%s: object body %s mm, colour from %s",
            clip_id,
            " x ".join(f"{v * 1000:.1f}" for v in box["dimensions_m"]),
            box.get("colour_source", "unknown"),
        )
    elif object_model is not None and len(object_model) <= 1:
        # A one point model renders as a dot and looks like a tracked object.
        # Refuse rather than draw a marker and call it manipulation data.
        raise RuntimeError(
            f"{clip_id}: the object model holds {len(object_model)} point and "
            f"there is no object_box.json beside it, so the object would "
            f"render as a dot. Re-run Stage 3, which now writes a body."
        )

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
        if external is not None:
            colour_png = cv2.imread(str(external / "color" / f"{index:05d}.png"))
            depth_png = cv2.imread(str(external / "depth" / f"{index:05d}.png"),
                                   cv2.IMREAD_UNCHANGED)
            if colour_png is None or depth_png is None:
                raise FileNotFoundError(
                    f"the external splat layer has no frame {index:05d}"
                )
            splat_color = cv2.cvtColor(colour_png, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            metres = depth_png.astype(np.float64) / float(external_meta["depth_scale_mm"])
            alpha_png = cv2.imread(str(external / "alpha" / f"{index:05d}.png"),
                                   cv2.IMREAD_UNCHANGED)
            if alpha_png is None:
                raise FileNotFoundError(
                    f"the external splat layer has no alpha for frame {index:05d}. "
                    f"Coverage taken from depth alone reads 100 per cent on a "
                    f"view that is largely empty, so it is not a substitute."
                )
            alpha = alpha_png.astype(np.float64) / 255.0
            # The same rule the local path uses: below this the pixel holds no
            # surface, so it must not win the depth test against the gripper.
            splat_depth = np.where(alpha > 0.35, metres, np.inf)
            splat_coverage[index] = float((alpha > 0.35).mean())
        elif splat is not None:
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
        #
        # Draw a body, not a marker. This used to call `rasterize_points` on
        # `object_model.npy`, which the plane-solve branch of Stage 3 filled
        # with a single point at the origin, so the object appeared as a 5 mm
        # dot on every frame of every clip. The cube that looked like the
        # object was the splat's static copy of it.
        #
        # As a mesh it occludes correctly: `composite` already resolves the
        # layers by depth, so the box hides the splat behind it and the gripper
        # jaws hide the box when they pass in front.
        if draw_object and object_valid[index]:
            if object_box is not None:
                vertices, faces = object_box
                posed = transform_points(object_poses[index], vertices)
                object_color, object_depth = mesh_render.rasterize_mesh(
                    posed, faces, view_matrix,
                    intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                    width, height, color=object_colour, near=near,
                )
            else:
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
        "frames_excluded_no_measurement": excluded,
        "source_frame_index": [int(i) for i in render_index],
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
