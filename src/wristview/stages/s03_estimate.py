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
from ..logging_setup import get
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
    splat, camera_pose: np.ndarray, intrinsics: Intrinsics, device: str, far: float
) -> tuple[np.ndarray, np.ndarray]:
    """Metric depth of the static scene, from the splat, at this camera pose."""
    view = torch.tensor(invert_pose(camera_pose), device=device, dtype=torch.float32)
    with torch.no_grad():
        result = splat_render(
            splat, view, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            intrinsics.width, intrinsics.height,
            tile_size=16, max_per_tile=192, near=0.02, far=far,
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

    hand_kind = models["hand_kind"]
    groundtruth_hands = None
    if hand_kind == "synthetic_groundtruth":
        groundtruth_hands = hand_backend.GroundTruthHands(models["groundtruth_path"], clip_id)

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

    with rec.timed(f"hand.{clip_id}"):
        if groundtruth_hands is not None:
            for index in range(count):
                frame = groundtruth_hands.frame(index, camera_poses[index], intrinsics)
                if frame.detected:
                    _record(index, frame)
        else:
            estimator = models["hand_estimator"]
            for index, name in enumerate(frame_names):
                image = cv2.imread(str(frames_dir / name))
                if image is None:
                    continue
                frame = estimator.process(image, intrinsics)
                if frame.detected:
                    _record(index, frame)
                if (index + 1) % 100 == 0:
                    log.info(
                        "  %s: hand %d/%d frames, %d detected",
                        clip_id, index + 1, count, int(hand_valid.sum()),
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

            # --- metric depth ---
            metric = None
            if splat is not None:
                reference, reference_valid = _render_splat_depth(
                    splat, camera_poses[index], work_intrinsics, device, far
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

    object_rate = float(object_valid.mean())
    log.info(
        "%s: object tracked on %d/%d frames (%.0f%%) via %s, median ICP residual %.4f m",
        clip_id, int(object_valid.sum()), count, object_rate * 100, object_kind,
        float(np.nanmedian(object_rmse)) if object_valid.any() else float("nan"),
    )

    if object_valid.sum() >= 3:
        window = int(pose_cfg.get("smoothing_window", 5))
        object_poses[object_valid] = smooth_poses(object_poses[object_valid], window)

    # ---- write ------------------------------------------------------------
    hand_path = out_dir / "hand.npz"
    np.savez_compressed(
        hand_path,
        landmarks_world=hand_world,
        landmarks_cam=hand_cam,
        landmarks_px=hand_px,
        valid=hand_valid,
        confidence=hand_confidence,
    )
    object_pose_path = out_dir / "object_pose.npy"
    np.save(object_pose_path, object_poses)
    np.save(out_dir / "object_valid.npy", object_valid)

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
        "object_backend": object_kind,
        "depth_backend": depth_kind,
        "object_track_rate": round(object_rate, 4),
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
            intrinsics = Intrinsics.from_dict(intrinsics_all[clip_id])
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
