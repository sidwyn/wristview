"""Stage 3 · Hand and object estimation.

In:  demo frames, camera poses from Stage 2, the splat from Stage 1.
Out: hand pose per frame in world coordinates, object mask and object pose.

This is the hardest stage, and the build plan says so. Three problems stacked:

  hand    HaMeR does not install on Apple Silicon, so MediaPipe runs instead
          and is lifted to metric camera coordinates by PnP.
  object  Grounding DINO proposes a box from the task text, SAM 2 turns it
          into a mask, and optical flow carries the mask through occlusion.
  pose    Depth-Anything gives relative depth. It is made metric by fitting
          against splat-rendered depth on background pixels. The masked
          pixels then back-project to metric 3D, and a trimmed rigid fit
          against a canonical model gives the per-frame object pose.

Rotation of a small, near-symmetric object seen from one side is weakly
constrained. The per-frame fit residual is written out so that weakness is
visible rather than hidden.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

from ..backends import depth as depth_backend
from ..backends import hands as hand_backend
from ..backends import objects as object_backend
from ..backends.splat_mps import render as splat_render
from ..camera import Intrinsics
from ..device import resolve as resolve_device
from ..geometry import invert_pose, make_pose, orthonormalize, smooth_poses, transform_points
from ..handqc import drop_isolated, drop_isolated_labels
from ..logging_setup import get
from ..objectbox import box_mesh, resolve_dimensions, sample_colour
from ..runctx import RunContext, StageRecorder, read_json, verify_frames_present, write_json

log = get(__name__)

STAGE = 3
NAME = "estimate"


def _select_hand_backend(cfg: dict, rec: StageRecorder) -> str:
    """Pick the hand estimator and record why the others were not used."""
    requested = cfg.get("backend", "auto")

    if requested == "synthetic_groundtruth":
        return "synthetic_groundtruth"

    if requested in ("wilor", "auto"):
        ok, reason = hand_backend.wilor_available()
        if ok:
            return "wilor"
        if requested == "wilor":
            raise RuntimeError(f"hand backend wilor requested but unavailable: {reason}")
        log.warning("WiLoR unavailable (%s)", reason)
        rec.note(f"WiLoR unavailable: {reason}")

    if requested in ("hamer", "auto"):
        ok, reason = hand_backend.hamer_available()
        if ok:
            raise NotImplementedError(
                "HaMeR is importable but no adapter is written. It has never been "
                "installable on this platform, so WiLoR was adopted instead."
            )
        if requested == "hamer":
            raise RuntimeError(f"hand backend hamer requested but unavailable: {reason}")
        log.warning("HaMeR unavailable (%s); falling back to MediaPipe", reason)
        rec.note(f"HaMeR unavailable: {reason}")

    return "mediapipe"


def _groundtruth_path(ctx: RunContext) -> Path | None:
    """Find a fixture ground-truth file beside the source videos."""
    sources = read_json(ctx.root / "sources.json")
    candidate = Path(sources["scan"]).parent / "groundtruth.json"
    return candidate if candidate.exists() else None


def _render_splat_depth(
    splat, camera_pose: np.ndarray, intrinsics: Intrinsics, device: str, far: float,
    near: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Metric depth of the static scene, from the splat, at this camera pose.

    `near` culls floaters rather than clipping real geometry. A camera is
    never within 30 cm of the surface it is filming, so anything nearer is a
    Gaussian that parked in front of a training camera. Left in, they blanket
    the frame and the depth comes back pinned at the near plane.
    """
    view = torch.tensor(invert_pose(camera_pose), device=device, dtype=torch.float32)
    with torch.no_grad():
        result = splat_render(
            splat, view, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            intrinsics.width, intrinsics.height,
            tile_size=16, max_per_tile=192, near=near, far=far,
        )
    depth = result.depth.cpu().numpy()
    valid = (result.alpha.cpu().numpy() > 0.5) & (depth < far * 0.99)
    return depth, valid


def _trimmed_rigid_fit(
    source: np.ndarray,
    target_tree: cKDTree,
    target: np.ndarray,
    initial: np.ndarray,
    iterations: int,
    trim_fraction: float,
) -> tuple[np.ndarray, float]:
    """ICP with trimming. Returns the pose and the inlier RMSE.

    Trimming matters because the mask always leaks a few background or finger
    pixels, and a plain least-squares fit lets those drag the whole pose.
    """
    from ..geometry import umeyama_sim3

    pose = initial.copy()
    rmse = float("inf")

    for _ in range(iterations):
        moved = transform_points(pose, source)
        distances, indices = target_tree.query(moved, k=1)

        keep_count = max(4, int(len(distances) * (1.0 - trim_fraction)))
        order = np.argsort(distances)[:keep_count]
        if len(order) < 4:
            break

        matched = target[indices[order]]
        _, rotation, translation = umeyama_sim3(
            transform_points(pose, source[order]), matched, with_scale=False
        )
        delta = make_pose(rotation, translation)
        pose = delta @ pose
        pose[:3, :3] = orthonormalize(pose[:3, :3])
        rmse = float(np.sqrt((distances[order] ** 2).mean()))

    return pose, rmse


def _estimate_episode(
    ctx: RunContext,
    rec: StageRecorder,
    clip_id: str,
    clip: dict,
    intrinsics: Intrinsics,
    camera_poses: np.ndarray,
    splat,
    device: str,
    cfg: dict,
    instruction: str,
    models: dict,
) -> dict:
    """Estimate hand and object for one episode."""
    out_dir = ctx.episode_dir(STAGE, clip_id)
    masks_dir = out_dir / "object_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = ctx.root / clip["frames_dir"]
    frame_names = clip["frame_names"]
    count = len(frame_names)

    verify_frames_present(frames_dir, frame_names, clip_id)
    object_cfg = cfg.get("object", {})
    pose_cfg = cfg.get("pose", {})

    # ---- hand ------------------------------------------------------------
    hand_world = np.zeros((count, hand_backend.NUM_LANDMARKS, 3))
    hand_cam = np.zeros((count, hand_backend.NUM_LANDMARKS, 3))
    hand_px = np.zeros((count, hand_backend.NUM_LANDMARKS, 2))
    hand_valid = np.zeros(count, dtype=bool)
    hand_confidence = np.zeros(count)
    # Which hand was picked, per frame. The selector takes the largest box, and
    # with two hands in shot the nearer one is larger, so the selection can
    # cross from one hand to the other mid-episode without anything failing.
    # The trajectory that comes out is smooth and wrong.
    hand_side = np.full(count, "", dtype=object)
    hands_seen = np.zeros(count, dtype=int)
    # Selecting by label instead of by size makes a two-handed clip usable as a
    # single-arm clip: the wanted hand is followed throughout, and the other is
    # ignored rather than competing for the selection. It costs the frames the
    # detector mislabels, which become gaps, and the gaps are counted below.
    hand_select = str(
        (cfg.get("hand", {}).get("per_clip") or {}).get(clip_id, {}).get("select")
        or cfg.get("hand", {}).get("select", "largest")
    )
    hand_selection = np.full(count, "", dtype=object)

    hand_kind = models["hand_kind"]
    groundtruth_hands = None
    if hand_kind == "synthetic_groundtruth":
        groundtruth_hands = hand_backend.GroundTruthHands(
            models["groundtruth_path"], clip_id,
            run_fps=float(clip.get("effective_fps") or clip["video_info"]["fps"]),
        )

    # The object and depth path runs at a reduced resolution. Rendering splat
    # depth at the demo's native 2160x1214 is 2.6 million pixels a frame, and
    # nothing downstream needs that: the mask, the monocular depth, and the
    # back-projected points all feed a rigid fit whose accuracy is set by the
    # depth model, not by pixel count. Hand estimation stays at full
    # resolution, because WiLoR crops around the hand and detail matters there.
    work_long_side = int(cfg.get("work_resolution", 640))
    work_scale = min(1.0, work_long_side / max(intrinsics.width, intrinsics.height))
    work_width = max(32, int(round(intrinsics.width * work_scale)))
    work_height = max(32, int(round(intrinsics.height * work_scale)))
    work_intrinsics = intrinsics.scaled(work_width, work_height)
    if work_scale < 1.0:
        log.info(
            "%s: object and depth path runs at %dx%d, hand at %dx%d",
            clip_id, work_width, work_height, intrinsics.width, intrinsics.height,
        )

    def _record(index: int, frame) -> None:
        hand_cam[index] = frame.landmarks_cam
        hand_px[index] = frame.landmarks_px
        hand_world[index] = transform_points(camera_poses[index], frame.landmarks_cam)
        hand_valid[index] = True
        hand_confidence[index] = frame.confidence
        hand_side[index] = frame.handedness
        hands_seen[index] = getattr(frame, "hands_in_frame", 0)

    with rec.timed(f"hand.{clip_id}"):
        if groundtruth_hands is not None:
            for index in range(count):
                frame = groundtruth_hands.frame(index, camera_poses[index], intrinsics)
                if frame.detected:
                    _record(index, frame)
        else:
            estimator = models["hand_estimator"]
            if hasattr(estimator, "select"):
                estimator.select = hand_select
            elif hand_select != "largest":
                raise ValueError(
                    f"{clip_id}: hand.select={hand_select!r} needs the WiLoR "
                    f"backend, but {hand_kind} is running"
                )
            for index, name in enumerate(frame_names):
                image = cv2.imread(str(frames_dir / name))
                if image is None:
                    continue
                frame = estimator.process(image, intrinsics)
                hand_selection[index] = frame.selection
                if frame.detected:
                    _record(index, frame)
                else:
                    hands_seen[index] = getattr(frame, "hands_in_frame", 0)
                if (index + 1) % 100 == 0:
                    log.info(
                        "  %s: hand %d/%d frames, %d detected",
                        clip_id, index + 1, count, int(hand_valid.sum()),
                    )

    # A detection no neighbouring frame supports is a false positive, and the
    # detector's confidence cannot say so: WiLoR returns 1.000 on every frame it
    # accepts. See handqc and defect 37.
    hand_valid, isolated_report = drop_isolated(hand_valid)
    if isolated_report["frames_dropped"]:
        log.warning(
            "%s: dropped %d isolated hand frame(s) %s, no neighbouring frame "
            "supports them",
            clip_id, isolated_report["frames_dropped"],
            isolated_report["dropped_frames"],
        )

    hand_rate = float(hand_valid.mean())
    log.info("%s: hand detected on %d/%d frames (%.0f%%) via %s",
             clip_id, int(hand_valid.sum()), count, hand_rate * 100, hand_kind)

    # ---- depth -----------------------------------------------------------
    depth_model = models.get("depth_model")
    depth_kind = models["depth_kind"]
    far = float(ctx.config.get("render.far_plane_m", 12.0))

    # ---- object ----------------------------------------------------------
    object_poses = np.repeat(np.eye(4)[None], count, axis=0)
    object_valid = np.zeros(count, dtype=bool)
    object_rmse = np.full(count, np.nan)
    mask_areas = np.zeros(count, dtype=int)

    canonical: np.ndarray | None = None
    canonical_centroid = np.zeros(3)

    # Resolve the object's size before the loop. If it is not known, that is a
    # capture problem and it should stop the stage, not produce a bodiless
    # object that renders as a dot.
    box_dimensions, box_dimensions_source = resolve_dimensions(
        {**pose_cfg, **((pose_cfg.get("per_clip") or {}).get(clip_id, {}))}
    )
    log.info("object body: %.1f x %.1f x %.1f mm, from %s",
             *(box_dimensions * 1000), box_dimensions_source)

    # The work surface, from Stage 1, and the object's measured height. Both
    # are needed to place a resting object without a depth estimate.
    from .. import carry, reprojection
    from .. import plane as plane_module

    desk_normal = desk_offset = None
    # A run with no marker has no scale.json at all. Reading it unconditionally
    # is a missing-capability error dressed up as a crash, and this same line
    # already broke Stage 4 once.
    scale_path = ctx.stage_dir(1, create=False) / "scale.json"
    scene_meta = read_json(scale_path) if scale_path.exists() else {}
    plane_meta = (scene_meta.get("diagnostics") or {}).get("desk_plane")
    if plane_meta and plane_meta.get("offset_m") is not None:
        desk_normal = np.asarray(plane_meta["normal"], dtype=np.float64)
        desk_offset = float(plane_meta["offset_m"])
        log.info("%s: desk plane from Stage 1, offset %.4f m, normal %s",
                 clip_id, desk_offset, np.round(desk_normal, 4).tolist())
    # Which object this clip manipulates, and how tall it is. Session 6 shot
    # four clips on one set holding a cube, a bin and a tape measure, and three
    # clips manipulate the cube while the fourth manipulates the tape measure.
    # With one prompt for the whole run the detector took the most salient
    # object, the bin, which never moves, and every object pose described a
    # thing nobody touched.
    # What to hand the detector, most specific first:
    #   1. estimate.pose.per_clip[clip].prompt   this clip's object
    #   2. estimate.object.prompt                this session's object
    #   3. the CLI --instruction                 a sentence about the task
    #
    # Three was the only one wired up. `estimate.object.prompt` was declared in
    # the config, documented, and read by no code, so real27 passed the whole
    # task sentence to Grounding DINO: "pick up the tea box and place it on the
    # mat". A sentence is not a noun phrase and the detector had nothing to
    # lock onto, which surfaced as a box covering 93 per cent of the frame.
    #
    # A declared key that nothing reads is the same defect as an undeclared one
    # and is harder to see, because it survives the unknown-key check.
    per_clip = (pose_cfg.get("per_clip") or {}).get(clip_id, {})
    object_prompt = str((cfg.get("object") or {}).get("prompt") or "").strip()
    if per_clip.get("prompt"):
        instruction = str(per_clip["prompt"])
        prompt_source = "estimate.pose.per_clip"
    elif object_prompt:
        instruction = object_prompt
        prompt_source = "estimate.object.prompt"
    else:
        prompt_source = "the --instruction sentence, which names a task not an object"
    object_height_m = float(per_clip.get("object_height_m",
                                         pose_cfg.get("object_height_m", 0.0)))
    log.info("%s: tracking '%s' (from %s), height %.1f mm",
             clip_id, instruction, prompt_source, object_height_m * 1000)
    if desk_normal is not None and object_height_m <= 0:
        log.warning(
            "%s: a desk plane exists but retarget.object_height_m is unset, so the "
            "depth path is used instead. Measure the object and set it.", clip_id,
        )
    object_source = np.full(count, "", dtype=object)
    # The silhouette ray per frame, kept so the carry solve can place the object
    # along it once the hand lifts it off the plane.
    object_rays = np.zeros((count, 3))
    object_ray_origins = np.zeros((count, 3))
    object_ray_valid = np.zeros(count, dtype=bool)
    # Silhouette centroids, in WORK pixels. The object path runs at a reduced
    # resolution, so the reprojection gate has to use the scaled intrinsics or
    # it will report a fixed fraction of the frame as error.
    object_centroid_px = np.zeros((count, 2))
    object_centroid_valid = np.zeros(count, dtype=bool)
    target_pose = np.eye(4)
    previous_gray: np.ndarray | None = None
    previous_mask: np.ndarray | None = None
    depth_fits: list[dict] = []

    detector = models.get("detector")
    segmenter = models.get("segmenter")
    object_kind = models["object_kind"]

    with rec.timed(f"object.{clip_id}"):
        for index, name in enumerate(frame_names):
            full = cv2.imread(str(frames_dir / name))
            if full is None:
                continue
            image = (
                cv2.resize(full, (work_width, work_height), interpolation=cv2.INTER_AREA)
                if work_scale < 1.0
                else full
            )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # --- mask ---
            # Grounding DINO re-detects only every `detect_interval` frames,
            # or whenever the track has been lost. Running a text-conditioned
            # detector on every frame is both the slowest part of the stage
            # and the least stable: the box jitters between frames and drags
            # the mask with it. Between detections SAM 2 is prompted with the
            # previous mask's centroid, which tracks smoothly.
            mask = None
            interval = max(1, int(object_cfg.get("detect_interval", 15)))
            redetect = (index % interval == 0) or previous_mask is None

            if detector is not None and segmenter is not None and redetect:
                box = detector.detect(
                    image, instruction,
                    float(object_cfg.get("box_threshold", 0.3)),
                    float(object_cfg.get("text_threshold", 0.25)),
                )
                if box is not None:
                    mask = segmenter.segment(image, box=box)

            if mask is None and segmenter is not None:
                point = None
                if previous_mask is not None and previous_mask.any():
                    ys, xs = np.nonzero(previous_mask)
                    point = np.array([xs.mean(), ys.mean()], dtype=np.float64)
                elif hand_valid[index]:
                    point = hand_backend.grasp_center(hand_px[index]) * work_scale
                if point is not None:
                    mask = segmenter.segment(image, point=point)

            if mask is None and segmenter is None and hand_valid[index]:
                point = hand_backend.grasp_center(hand_px[index]) * work_scale
                mask = object_backend.seed_mask_from_point(image, point)

            mask = object_backend.clean_mask(
                mask,
                int(object_cfg.get("min_mask_pixels", 400)),
                float(object_cfg.get("max_mask_fraction", 0.4)),
            )
            if mask is None and previous_mask is not None and previous_gray is not None:
                # A brief loss is normal when the hand covers the object.
                mask = object_backend.propagate_mask(previous_gray, gray, previous_mask)
                mask = object_backend.clean_mask(
                    mask,
                    int(object_cfg.get("min_mask_pixels", 400)),
                    float(object_cfg.get("max_mask_fraction", 0.4)),
                )

            previous_gray = gray
            if mask is None:
                continue
            previous_mask = mask
            mask_areas[index] = int(mask.sum())
            cv2.imwrite(str(masks_dir / f"{index:05d}.png"), (mask * 255).astype(np.uint8))

            # --- pose on the desk plane, no depth anywhere ---
            #
            # Preferred whenever Stage 1 fitted a plane and the object's real
            # height is known. The depth path below stays only as a fallback,
            # because it is what put session 4's cube 85 mm into the air while
            # reporting a 1.1 mm residual.
            if desk_normal is not None and object_height_m > 0:
                solved = plane_module.object_pose_on_plane(
                    mask, camera_poses[index], work_intrinsics,
                    desk_normal, desk_offset, object_height_m,
                    reference_rotation=(target_pose[:3, :3] if target_pose is not None else None),
                )
                if solved is not None:
                    pose, plane_report = solved
                    object_poses[index] = pose
                    object_valid[index] = True
                    object_rmse[index] = 0.0
                    object_source[index] = "plane"
                    object_rays[index] = plane_report["ray_world"]
                    object_ray_origins[index] = plane_report["camera_position"]
                    object_ray_valid[index] = True
                    object_centroid_px[index] = plane_report["centroid_px"]
                    object_centroid_valid[index] = True
                    target_pose = pose
                    if canonical is None:
                        canonical_centroid = pose[:3, 3]
                        # The plane solve returns a pose and no shape. This
                        # used to record np.zeros((1, 3)), a single point, and
                        # Stage 5 drew the object as a 5 mm dot for the whole
                        # clip. The cube visible in those renders was the
                        # splat's static copy, which does not move when the
                        # operator picks the object up.
                        #
                        # The size is known: `object_height_m` is measured with
                        # a ruler and the solver already uses it to place this
                        # centre. Use the same number for the body.
                        corners, _ = box_mesh(box_dimensions)
                        canonical = corners
                    continue

            # --- metric depth (fallback) ---
            metric = None
            if splat is not None:
                reference, reference_valid = _render_splat_depth(
                    splat, camera_poses[index], work_intrinsics, device, far,
                    near=float(cfg.get("depth_near_m", 0.3)),
                )
                if depth_model is not None:
                    relative = depth_model.predict(image)
                    exclude = mask.copy()
                    if hand_valid[index]:
                        exclude |= _hand_mask(hand_px[index] * work_scale, mask.shape)
                    metric, fit = depth_backend.fit_metric_depth(
                        relative, reference, reference_valid, exclude=exclude
                    )
                    if index % 30 == 0:
                        depth_fits.append({"frame": index, **fit})
                if metric is None:
                    # No monocular model, or the fit failed. The splat's own
                    # depth is metric but has the object where the scan had
                    # it, so this is only right before the object is moved.
                    metric = np.where(reference_valid, reference, 0.0)

            if metric is None:
                continue

            points_cam = depth_backend.backproject(metric, mask, work_intrinsics)
            if len(points_cam) < 50:
                continue
            points_world = transform_points(camera_poses[index], points_cam)

            # Reject a cloud that is far too big to be one graspable object.
            spread = float(np.linalg.norm(points_world.std(axis=0)))
            if spread > 0.35:
                continue

            # --- rigid fit ---
            if canonical is None:
                canonical_centroid = points_world.mean(axis=0)
                canonical = points_world - canonical_centroid
                target_pose = make_pose(np.eye(3), canonical_centroid)
                object_poses[index] = target_pose
                object_valid[index] = True
                object_rmse[index] = 0.0
                continue

            tree = cKDTree(points_world)
            initial = target_pose.copy()
            initial[:3, 3] = points_world.mean(axis=0) - initial[:3, :3] @ np.zeros(3)
            pose, rmse = _trimmed_rigid_fit(
                canonical, tree, points_world, initial,
                int(pose_cfg.get("icp_iterations", 20)),
                float(pose_cfg.get("icp_trim_fraction", 0.2)),
            )
            object_poses[index] = pose
            object_valid[index] = True
            object_rmse[index] = rmse
            target_pose = pose

    # ---- the object leaves the plane -------------------------------------
    #
    # The plane solve places the object's centre at `offset + height / 2` on
    # every frame, so the tracked object cannot rise. Session 6 clip 5 measured
    # 2.000 cm above the desk on all 185 valid frames, to a spread of 0.0000 mm,
    # while the hand carrying it reached 24.7 cm. That is correct before contact
    # and wrong for the whole carry, and it surfaced downstream as grasp
    # detection finding contact on 0 of 186 frames: fingertips cannot reach an
    # object left behind on the table.
    carry_report: dict = {"frames_carried": 0}
    carry_cfg = pose_cfg.get("carry", {})
    solved_on_plane = int(sum(1 for x in object_source if x == "plane"))
    if bool(carry_cfg.get("enabled", True)) and solved_on_plane and hand_valid.any():
        rest = carry.resting_pose(
            object_poses, object_valid,
            stillness_m=float(carry_cfg.get("rest_stillness_m", 0.01)),
            min_rest_frames=int(carry_cfg.get("min_rest_frames", 5)),
        )
        if rest is None:
            log.warning(
                "%s: no frame had the object visible with the hand clear of it, "
                "so the resting pose is unknown and the carry solve cannot run",
                clip_id,
            )
        else:
            rest_pose, rest_report = rest
            radius = float(carry_cfg.get("object_radius_m", max(object_height_m, 0.02) / 2))
            onsets = carry.contact_onsets(
                hand_world, hand_valid, rest_pose[:3, 3], radius,
                enter_m=float(carry_cfg.get("contact_margin_m", 0.03)),
                min_frames=int(carry_cfg.get("min_contact_frames", 3)),
            )
            # A resting pose measured after the release describes where the
            # object ended up, not where it started. Solving the carry from it
            # produces a complete, plausible, wrong trajectory. Leave the
            # carried frames unsolved instead and say so.
            # Trim the still run to the part before the grasp, and take the
            # resting position from that. A run that continues past contact is
            # normal: the hand arrives before it lifts, and until the object
            # moves the plane solve keeps returning the same point.
            rest_run = tuple(rest_report["rest_run"])
            window, problem = carry.rest_window_before_contact(
                rest_run, onsets,
                min_rest_frames=int(carry_cfg.get("min_rest_frames", 5)),
            )
            if window is not None and window != rest_run:
                inside = np.zeros(len(object_valid), dtype=bool)
                inside[window[0]:window[1]] = True
                inside &= object_valid
                if inside.any():
                    rest_pose = rest_pose.copy()
                    rest_pose[:3, 3] = np.median(object_poses[inside][:, :3, 3], axis=0)
                    rest_report = dict(rest_report)
                    rest_report["rest_run_trimmed"] = [int(window[0]), int(window[1])]
                    rest_report["trimmed_at_first_contact"] = int(min(onsets))
                    log.info(
                        "%s: resting pose trimmed to frames %d-%d, before contact "
                        "at %d, from %d frames",
                        clip_id, window[0], window[1], min(onsets), int(inside.sum()),
                    )
                    rest_run = window
            if problem is not None:
                log.error("%s: CARRY NOT SOLVED. %s", clip_id, problem)
                carry_report = {
                    "frames_carried": 0,
                    "contact_runs": [],
                    "resting_pose": rest_report,
                    "rejected": problem,
                    "onsets": [int(o) for o in onsets],
                }
            else:
                object_poses, object_valid, carry_report = carry.solve_carried(
                    object_poses, object_valid, object_rays, object_ray_origins,
                    hand_world, hand_valid, rest_pose, onsets, rest_run,
                    release_ray_m=float(carry_cfg.get("release_ray_m", 0.06)),
                    release_frames=int(carry_cfg.get("release_frames", 3)),
                )
                carry_report["resting_pose"] = rest_report
            for index in range(count):
                if object_source[index] != "plane" and object_valid[index]:
                    object_source[index] = "carried"
            for start, stop in carry_report["contact_runs"]:
                for index in range(start, stop):
                    if object_valid[index]:
                        object_source[index] = "carried"
            log.info(
                "%s: contact on %d frame(s) across %d run(s) %s; resting pose from "
                "%d clear frame(s), scatter %.2f cm",
                clip_id, carry_report["frames_carried"],
                len(carry_report["contact_runs"]), carry_report["contact_runs"],
                rest_report["frames_used"], rest_report["scatter_median_cm"],
            )

    object_rate = float(object_valid.mean())
    by_plane = int(sum(1 for x in object_source if x == "plane"))
    if by_plane:
        log.info("%s: %d of %d object poses solved on the desk plane, no depth used",
                 clip_id, by_plane, int(object_valid.sum()))
    log.info(
        "%s: object tracked on %d/%d frames (%.0f%%) via %s, median ICP residual %.4f m",
        clip_id, int(object_valid.sum()), count, object_rate * 100, object_kind,
        float(np.nanmedian(object_rmse)) if object_valid.any() else float("nan"),
    )

    if object_valid.sum() >= 3:
        window = int(pose_cfg.get("smoothing_window", 5))
        object_poses[object_valid] = smooth_poses(object_poses[object_valid], window)

    # ---- write ------------------------------------------------------------
    # ---- hand identity guard ---------------------------------------------
    #
    # Count a switch only where two hands were actually detected on one side of
    # it. The defect this guards against is the largest-box selector crossing
    # between two hands, which can only happen when two hands are in frame. A
    # label that flips while a single hand is present is the detector
    # mislabelling that one hand, and rejecting a clip for it is a false alarm.
    #
    # Session 6 made the difference concrete. The first version counted label
    # changes alone and rejected all four clips, including the single-hand
    # control, which flipped once in 149 frames with no second hand anywhere in
    # the clip. Measuring the proxy rather than the defect is its own failure.
    # A person has two hands. A frame reporting three is the detector failing,
    # and it cannot be used as evidence that two hands were present: session 6's
    # single-hand control was rejected on exactly one such frame, where WiLoR
    # returned three detections while 148 other frames returned one.
    order = [
        i for i in np.nonzero(hand_valid)[0]
        if hand_side[i] and int(hands_seen[i]) <= 2
    ]
    impossible = int(sum(
        1 for i in np.nonzero(hand_valid)[0] if int(hands_seen[i]) > 2
    ))
    if impossible:
        log.info(
            "%s: %d frame(s) reported more than two hands and were left out of "
            "the identity check, because a person has two hands and the count "
            "is the detector's error, not evidence", clip_id, impossible,
        )
    sides = [str(hand_side[i]) for i in order]
    kept = [int(hands_seen[i]) for i in order]

    # Remove labels no neighbour supports, BEFORE counting crossings. One
    # mislabelled frame otherwise scores two crossings and invalidates a clip
    # whose hand never changed. See `drop_isolated_labels`.
    trusted, label_report = drop_isolated_labels(sides)
    if label_report["labels_dropped"]:
        log.info(
            "%s: %d isolated hand label(s) dropped before the identity check, "
            "at %s. Neither neighbour agreed with them, so they are detector "
            "mislabels rather than the selector crossing hands.",
            clip_id, label_report["labels_dropped"],
            label_report["dropped_indices"][:10],
        )
    sides = [s for s, ok in zip(sides, trusted, strict=True) if ok]
    kept = [k for k, ok in zip(kept, trusted, strict=True) if ok]

    switches = 0
    mislabels = 0
    for a, b, na, nb in zip(sides, sides[1:], kept, kept[1:], strict=False):
        if a == b:
            continue
        if max(na, nb) >= 2:
            switches += 1
        else:
            mislabels += 1
    side_counts = {s: sides.count(s) for s in set(sides)}
    if mislabels:
        log.info(
            "%s: %d hand label flip(s) ignored, only one hand was in frame at "
            "the time, so the detector mislabelled one hand rather than the "
            "selector crossing between two", clip_id, mislabels,
        )
    if switches:
        log.error(
            "%s: the selected hand changed side %d times (%s). This clip's "
            "single-hand trajectory is INVALID: the largest-box selector "
            "crossed between two hands mid-episode, and the result will look "
            "smooth while describing neither hand.",
            clip_id, switches, side_counts,
        )
    elif sides:
        log.info("%s: hand identity stable, %s throughout", clip_id, sides[0])

    # Selecting by label makes the counter above vacuous: every accepted frame
    # carries the wanted label because that is the acceptance test, so `switches`
    # is zero by construction and proves nothing. The defect it was built to
    # catch, the trajectory jumping from one hand to the other, is still
    # possible whenever the detector mislabels a hand, so it needs a measure
    # that does not depend on the label. Displacement is that measure: two
    # hands are tens of centimetres apart, and a crossing has to travel that
    # distance in one frame step.
    ambiguous = int(sum(1 for s in hand_selection if s == "ambiguous"))
    absent = int(sum(1 for s in hand_selection if s == "absent"))
    steps: list[float] = []
    valid_indices = np.nonzero(hand_valid)[0]
    for a, b in zip(valid_indices, valid_indices[1:], strict=False):
        if b - a != 1:
            continue    # a gap can hide real motion, so it is not a jump
        steps.append(float(np.linalg.norm(hand_world[b, 0] - hand_world[a, 0])))
    jump_max_cm = round(max(steps) * 100, 2) if steps else None
    jump_p99_cm = (
        round(float(np.percentile(steps, 99)) * 100, 2) if steps else None
    )
    if hand_select != "largest":
        log.info(
            "%s: selected by label (%s). %d frame(s) ambiguous, %d absent. "
            "Largest single-frame wrist move %s cm, p99 %s cm. The label-switch "
            "counter reads %d but is vacuous under label selection; the "
            "displacement figures are what would show a crossing.",
            clip_id, hand_select, ambiguous, absent,
            jump_max_cm, jump_p99_cm, switches,
        )

    # ---- reprojection gate ------------------------------------------------
    #
    # Every 3D quantity is projected back into the frame it came from and
    # compared with what was observed there. This is the check that would have
    # caught the worst defect in this project on its first run: the hand's 3D
    # position was collapsing toward the optical axis by a factor of 25, the 2D
    # overlays stayed correct because they come from the detector, and grasp
    # detection failing across three sessions looked like a grasp problem.
    # Median reprojection error was 614 px and nothing was measuring it.
    reports = [
        reprojection.hand_check(hand_cam, hand_px, hand_valid, intrinsics),
        reprojection.object_check(
            object_poses, object_valid, object_centroid_px, object_centroid_valid,
            camera_poses, work_intrinsics, sources=object_source,
        ),
    ]
    for report in reports:
        if report.get("frames"):
            log.info(
                "%s: %s, median %s px, p90 %s px over %d frames (%s)",
                clip_id, report["check"], report["median_px"], report["p90_px"],
                report["frames"], report["evidence"],
            )
    failures = reprojection.gate(reports)
    for failure in failures:
        log.error("%s: REPROJECTION GATE FAILED: %s", clip_id, failure)

    write_json(out_dir / "qc.json", {
        "clip_id": clip_id,
        "reprojection": reports,
        "passed": not failures,
        "failures": failures,
    })
    if failures and bool(cfg.get("fail_on_reprojection", True)):
        raise ValueError(
            f"{clip_id}: reprojection gate failed, so no 3D quantity from this "
            f"clip can be trusted. " + "; ".join(failures)
        )

    hand_path = out_dir / "hand.npz"
    np.savez_compressed(
        hand_path,
        landmarks_world=hand_world,
        landmarks_cam=hand_cam,
        landmarks_px=hand_px,
        valid=hand_valid,
        confidence=hand_confidence,
        hand_side=np.array([str(s) for s in hand_side]),
        hands_in_frame=hands_seen,
        hand_side_switches=np.array(switches),
        hand_label_mislabels=np.array(mislabels),
        hand_frames_impossible=np.array(impossible),
        hand_select=np.array(hand_select),
        hand_selection=np.array([str(s) for s in hand_selection]),
        hand_frames_ambiguous=np.array(ambiguous),
        hand_frames_absent=np.array(absent),
    )
    # Stage 4 needs the contact runs, not just a report of them. Its own grasp
    # detector measures fingertip distance to the tracked object, which cannot
    # work while the object is solved on the plane: session 6 had the object on
    # the desk and the hand 20 cm above it, so the detector found 0 frames on
    # every clip. Contact decided here, against the resting pose, is the signal
    # that survives.
    write_json(out_dir / "contact_runs.json", {
        "runs": carry_report.get("contact_runs", []),
        "frames_carried": carry_report.get("frames_carried", 0),
        "resting_pose": carry_report.get("resting_pose"),
        "source": "carry.contact_onsets against the resting pose",
    })
    object_pose_path = out_dir / "object_pose.npy"
    np.save(object_pose_path, object_poses)
    np.save(out_dir / "object_valid.npy", object_valid)
    np.save(out_dir / "object_source.npy", np.array([str(x) for x in object_source]))

    box_colour, sampled_pixels = _sample_object_colour(
        frames_dir, frame_names, masks_dir, object_valid
    )
    box_colour_source = (
        f"median of {sampled_pixels} object mask pixels" if sampled_pixels
        else "default, because no object pixels could be sampled"
    )
    if not sampled_pixels:
        log.warning("%s: the object box keeps its default colour, nothing sampled", clip_id)
    log.info("object body colour %s, from %s",
             tuple(round(c, 3) for c in box_colour), box_colour_source)

    # Record the body separately from the point cloud. Stage 5 renders this as
    # a mesh, so the depth test occludes it against the splat and the gripper.
    write_json(out_dir / "object_box.json", {
        "dimensions_m": [round(float(v), 5) for v in box_dimensions],
        "dimensions_source": box_dimensions_source,
        "colour_rgb": [round(float(v), 4) for v in box_colour],
        "colour_source": box_colour_source,
        "note": (
            "A box at the measured size, not the object's true shape. It is "
            "the difference between an object and a marker. Replace it with a "
            "reconstruction when one exists; Stage 5 only wants vertices and "
            "faces."
        ),
    })

    canonical_path = None
    if canonical is not None:
        canonical_path = out_dir / "object_model.npy"
        np.save(canonical_path, canonical)

    _write_overlay(ctx, out_dir, frames_dir, frame_names, hand_px, hand_valid, masks_dir)

    status = {
        "clip_id": clip_id,
        "frames": count,
        "hand_backend": hand_kind,
        "hand_detection_rate": round(hand_rate, 4),
        "isolated_hand_frames": isolated_report,
        "hand_select": hand_select,
        "hand_frames_ambiguous": ambiguous,
        "hand_frames_absent": absent,
        "hand_jump_max_cm": jump_max_cm,
        "hand_jump_p99_cm": jump_p99_cm,
        "hand_side_switches": switches,
        "hand_side_switch_counter_vacuous": hand_select != "largest",
        "object_backend": object_kind,
        "depth_backend": depth_kind,
        "object_track_rate": round(object_rate, 4),
        "carry": carry_report,
        "object_icp_rmse_median_m": (
            round(float(np.nanmedian(object_rmse)), 5) if object_valid.any() else None
        ),
        "object_model_points": int(len(canonical)) if canonical is not None else 0,
        "mask_area_median_px": int(np.median(mask_areas[mask_areas > 0])) if (mask_areas > 0).any() else 0,
        "depth_fits": depth_fits,
        "hand": ctx.rel(hand_path),
        "object_pose": ctx.rel(object_pose_path),
        "object_masks": ctx.rel(masks_dir),
        "object_model": ctx.rel(canonical_path) if canonical_path else None,
    }
    write_json(out_dir / "status.json", status)
    return status


def _hand_mask(landmarks_px: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A coarse hand region, used to keep the hand out of the depth fit."""
    mask = np.zeros(shape, dtype=bool)
    points = landmarks_px.astype(np.int32)
    hull = cv2.convexHull(points)
    filled = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    return cv2.dilate(filled, kernel).astype(bool) | mask


def _write_overlay(
    ctx: RunContext,
    out_dir: Path,
    frames_dir: Path,
    frame_names: list[str],
    hand_px: np.ndarray,
    hand_valid: np.ndarray,
    masks_dir: Path,
) -> None:
    """Draw the hand skeleton and object mask on the video.

    This is the M3 check from the milestone table: hand and mask overlaid on
    the demo video, tracking through the grasp.
    """
    overlay_dir = out_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
    ]
    for index, name in enumerate(frame_names):
        image = cv2.imread(str(frames_dir / name))
        if image is None:
            continue
        mask_path = masks_dir / f"{index:05d}.png"
        if mask_path.exists():
            raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            # Masks are stored at the object path's working resolution, which
            # is smaller than the frame.
            if raw.shape[:2] != image.shape[:2]:
                raw = cv2.resize(
                    raw, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            mask = raw > 127
            tint = image.copy()
            tint[mask] = (0.45 * tint[mask] + 0.55 * np.array([60, 220, 60])).astype(np.uint8)
            image = tint
        if hand_valid[index]:
            points = hand_px[index].astype(int)
            for a, b in bones:
                cv2.line(image, tuple(points[a]), tuple(points[b]), (0, 200, 255), 2)
            for point in points:
                cv2.circle(image, tuple(point), 3, (30, 30, 240), -1)
        cv2.imwrite(str(overlay_dir / f"{index:05d}.png"), image)

    from ..videoio import write_video

    write_video(overlay_dir, out_dir / "overlay.mp4", fps=15.0, pattern="%05d.png")


def run(ctx: RunContext) -> dict:
    """Run Stage 3 over the run directory."""
    rec = StageRecorder(ctx, STAGE, NAME)
    cfg = ctx.config.section("estimate")

    try:
        ingest_dir = ctx.stage_dir(0, create=False)
        manifest = read_json(ingest_dir / "manifest.json")
        intrinsics_all = read_json(ingest_dir / "intrinsics.json")

        # Stage 1 self-calibrates the camera, and every metric measurement
        # downstream must use that result rather than Stage 0's prior. On this
        # capture the prior was 1836 px against a refined 2819, a factor of
        # 1.535, and lifting the hand with the prior put it roughly 20 cm
        # below the desk: the end effector's whole z range sat outside the
        # reconstructed scene, with no cloud point within 10 cm of it.
        refined_intrinsics = None
        cameras_path = ctx.stage_dir(1, create=False) / "cameras.json"
        if cameras_path.exists():
            refined_intrinsics = Intrinsics.from_dict(read_json(cameras_path)["intrinsics"])
            log.info("using the refined camera from Stage 1: f=%.1f px", refined_intrinsics.fx)
        else:
            log.warning("no refined camera from Stage 1; falling back to the Stage 0 prior")
        localize_summary = read_json(ctx.stage_dir(2, create=False) / "summary.json")
        sources = read_json(ctx.root / "sources.json")
        instruction = sources.get("instruction") or "the object"

        rec.meta.inputs = {
            "manifest": ctx.rel(ingest_dir / "manifest.json"),
            "localize_summary": ctx.rel(ctx.stage_dir(2, create=False) / "summary.json"),
            "instruction": instruction,
        }

        device = resolve_device(
            ctx.config.get("device.preferred", "auto"),
            bool(ctx.config.get("device.allow_cpu_fallback", True)),
        )
        rec.backend("device", device)

        # ---- pick backends once, for every episode ----------------------
        models: dict = {}
        hand_cfg = cfg.get("hand", {})
        hand_kind = _select_hand_backend(hand_cfg, rec)
        # Ground truth exists only for the synthetic fixture, which is a unit
        # test asset. Real footage has none, so this stays None and the
        # fallback below can never fire on a real run.
        groundtruth_path = _groundtruth_path(ctx)

        with rec.timed("load.hand"):
            if hand_kind == "wilor":
                models["hand_estimator"] = hand_backend.WiLoRHands(device=device)
            elif hand_kind == "mediapipe":
                models["hand_estimator"] = hand_backend.MediaPipeHands(
                    max_hands=int(hand_cfg.get("max_hands", 1)),
                    min_detection_confidence=float(
                        hand_cfg.get("min_detection_confidence", 0.5)
                    ),
                    min_tracking_confidence=float(
                        hand_cfg.get("min_tracking_confidence", 0.5)
                    ),
                )
            else:
                models["hand_estimator"] = None

        models["groundtruth_path"] = groundtruth_path
        models["hand_kind"] = hand_kind
        rec.backend("hand", hand_kind)
        log.info("hand backend: %s", hand_kind)

        object_requested = cfg.get("object", {}).get("backend", "auto")
        detector = segmenter = None
        object_kind = "grasp_point_grabcut"
        if object_requested in ("auto", "groundingdino_sam2"):
            dino_ok, dino_reason = object_backend.grounding_dino_available()
            sam_ok, sam_reason = object_backend.sam2_available()
            if sam_ok:
                try:
                    with rec.timed("load.sam2"):
                        segmenter = object_backend.Sam2Segmenter(device)
                    object_kind = "sam2_grasp_point"
                except Exception as exc:  # noqa: BLE001 - fall back, do not crash
                    log.warning("SAM 2 failed to load: %s", exc)
                    rec.note(f"SAM 2 load failed: {exc}")
                    segmenter = None
            else:
                log.warning("SAM 2 unavailable: %s", sam_reason)
                rec.note(f"SAM 2 unavailable: {sam_reason}")

            if dino_ok and segmenter is not None:
                try:
                    with rec.timed("load.groundingdino"):
                        detector = object_backend.GroundingDinoDetector(device)
                    object_kind = "groundingdino_sam2"
                except Exception as exc:  # noqa: BLE001
                    log.warning("Grounding DINO failed to load: %s", exc)
                    rec.note(f"Grounding DINO load failed: {exc}")
                    detector = None
            elif not dino_ok:
                log.warning("Grounding DINO unavailable: %s", dino_reason)

        models["detector"] = detector
        models["segmenter"] = segmenter
        models["object_kind"] = object_kind
        rec.backend("object", object_kind)

        depth_model = None
        depth_kind = "splat_only"
        if cfg.get("depth", {}).get("backend", "auto") in ("auto", "depth_anything_v2"):
            ok, reason = depth_backend.depth_anything_available()
            if ok:
                try:
                    with rec.timed("load.depth"):
                        depth_model = depth_backend.DepthAnythingV2(
                            device, cfg["depth"].get("model_id")
                        )
                    depth_kind = "depth_anything_v2_splat_aligned"
                except Exception as exc:  # noqa: BLE001
                    log.warning("Depth-Anything V2 failed to load: %s", exc)
                    rec.note(f"Depth-Anything V2 load failed: {exc}")
            else:
                log.warning("Depth-Anything V2 unavailable: %s", reason)
                rec.note(f"Depth-Anything V2 unavailable: {reason}")
        models["depth_model"] = depth_model
        models["depth_kind"] = depth_kind
        rec.backend("depth", depth_kind)

        from .s01_scene import load_splat

        splat = load_splat(ctx, device)
        if splat is None:
            log.warning("no splat from Stage 1; depth cannot be made metric")
            rec.note("no splat available, so monocular depth stays relative")

        # ---- run every accepted episode ----------------------------------
        statuses = {}
        for clip_id, clip in manifest["clips"].items():
            if clip["kind"] != "demo":
                continue
            episode_status = localize_summary.get(clip_id, {})
            if episode_status.get("rejected", True):
                log.warning("%s: skipped, rejected by Stage 2", clip_id)
                continue

            log.info("--- Stage 3: %s ---", clip_id)
            intrinsics = refined_intrinsics or Intrinsics.from_dict(intrinsics_all[clip_id])
            camera_poses = np.load(
                ctx.episode_dir(2, clip_id, create=False) / "camera_poses.npy"
            )

            episode_models = dict(models)
            statuses[clip_id] = _estimate_episode(
                ctx, rec, clip_id, clip, intrinsics, camera_poses, splat,
                device, cfg, instruction, episode_models,
            )

            # Fixture-only escape hatch. The synthetic hand is a smooth capsule
            # render, which is out of distribution for detectors trained on
            # photographs, so a fixture run would otherwise never exercise
            # Stages 4 and 5. It cannot fire on real footage, because real
            # footage has no groundtruth.json. Recorded loudly when it does.
            if (
                statuses[clip_id]["hand_detection_rate"] < 0.5
                and groundtruth_path is not None
                and episode_models["hand_kind"] != "synthetic_groundtruth"
            ):
                log.warning(
                    "%s: hand detection rate %.0f%%. Fixture ground truth is "
                    "available, so re-running with it.",
                    clip_id, statuses[clip_id]["hand_detection_rate"] * 100,
                )
                rec.note(
                    f"{clip_id}: {episode_models['hand_kind']} detected the hand on "
                    f"{statuses[clip_id]['hand_detection_rate'] * 100:.0f}% of frames. "
                    f"Fell back to SYNTHETIC GROUND TRUTH landmarks from the fixture. "
                    f"This is not an estimate and does not apply to real footage."
                )
                episode_models["hand_kind"] = "synthetic_groundtruth"
                statuses[clip_id] = _estimate_episode(
                    ctx, rec, clip_id, clip, intrinsics, camera_poses, splat,
                    device, cfg, instruction, episode_models,
                )

            rec.output(f"{clip_id}_hand", ctx.episode_dir(STAGE, clip_id) / "hand.npz")
            rec.output(
                f"{clip_id}_object_pose", ctx.episode_dir(STAGE, clip_id) / "object_pose.npy"
            )
            rec.output(f"{clip_id}_overlay", ctx.episode_dir(STAGE, clip_id) / "overlay.mp4")

        summary_path = write_json(ctx.stage_dir(STAGE) / "summary.json", statuses)
        rec.output("summary", summary_path)
        rec.metric("episodes", len(statuses))
        rec.metric(
            "hand_detection_rates",
            {k: v["hand_detection_rate"] for k, v in statuses.items()},
        )
        rec.metric(
            "object_track_rates", {k: v["object_track_rate"] for k, v in statuses.items()}
        )

        rec.write("ok" if statuses else "failed")
        return statuses

    except Exception as exc:
        rec.note(f"{type(exc).__name__}: {exc}")
        rec.write("failed")
        raise


def load_summary(ctx: RunContext) -> dict:
    return read_json(ctx.stage_dir(STAGE, create=False) / "summary.json")


def _sample_object_colour(frames_dir, frame_names, masks_dir, object_valid, samples: int = 12):
    """Take the object's colour from its own pixels on a spread of frames.

    Returns the colour and the number of pixels behind it. A caller that gets
    zero must not describe the result as a measurement.
    """
    import cv2

    indices = np.nonzero(np.asarray(object_valid))[0]
    if not len(indices):
        return (0.75, 0.65, 0.15), 0
    picks = indices[np.linspace(0, len(indices) - 1, min(samples, len(indices))).astype(int)]
    images, masks = [], []
    for index in sorted(set(int(i) for i in picks)):
        mask_path = Path(masks_dir) / f"{index:05d}.png"
        frame_path = Path(frames_dir) / frame_names[index]
        if not mask_path.exists() or not frame_path.exists():
            continue
        image = cv2.imread(str(frame_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks.append(mask)
    if not images:
        return (0.75, 0.65, 0.15), 0
    return sample_colour(images, masks)
