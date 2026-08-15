"""Stage 2 · Demo localization.

In:  demo frames from Stage 0, the scan reconstruction from Stage 1.
Out: a 6DoF camera pose per demo frame, in metric scene coordinates.

Each demo frame is registered against the scan reconstruction: match its
features to the most similar scan frames, turn those 2D-2D matches into 2D-3D
correspondences through the scan's triangulated points, then solve absolute
pose with RANSAC.

When too few frames register, the episode falls back to its ARKit trajectory
mapped through the Stage 1 Sim(3). Which path ran is recorded per episode,
because a fallback episode and a registered episode are not the same evidence.
"""

from __future__ import annotations

import numpy as np

from ..backends import sfm
from ..camera import Intrinsics
from ..device import resolve as resolve_device
from ..geometry import slerp_fill, smooth_poses
from ..logging_setup import get
from ..runctx import RunContext, StageRecorder, read_json, write_json

log = get(__name__)

STAGE = 2
NAME = "localize"


def _localize_episode(
    ctx: RunContext,
    rec: StageRecorder,
    clip_id: str,
    clip: dict,
    intrinsics: Intrinsics,
    reconstruction,
    scan_names: list[str],
    scan_frames_dir,
    scan_descriptors: np.ndarray,
    device: str,
    cfg: dict,
) -> dict:
    """Register one demo episode against the scan reconstruction."""
    import pycolmap

    out_dir = ctx.episode_dir(STAGE, clip_id)
    frames_dir = ctx.root / clip["frames_dir"]
    frame_names = clip["frame_names"]

    features_path = out_dir / "features.h5"
    matches_path = out_dir / "matches.h5"

    with rec.timed(f"features.{clip_id}"):
        sfm.extract_features(frames_dir, frame_names, features_path, device)

    with rec.timed(f"pairs.{clip_id}"):
        query_descriptors = sfm.global_descriptors(frames_dir, frame_names)
        pairs = sfm.build_query_pairs(
            frame_names, scan_names, query_descriptors, scan_descriptors,
            top_k=int(cfg.get("retrieval_top_k", 20)),
        )
    log.info("%s: matching %d query pairs against the scan", clip_id, len(pairs))

    with rec.timed(f"matching.{clip_id}"):
        sfm.match_pairs(
            pairs, features_path, matches_path, device,
            features_ref_path=ctx.stage_dir(1, create=False) / "features.h5",
        )

    lookup = sfm.build_keypoint_to_point3d(reconstruction)

    # Use the camera Stage 1 converged on, not the prior. The demo clips come
    # from the same lens in the same session, and Stage 1 self-calibrates a
    # SIMPLE_RADIAL model whose focal and distortion differ from the guess.
    # Solving PnP against the guess would bake that error into every pose.
    scan_cameras = list(reconstruction.cameras.values())
    if scan_cameras:
        camera = scan_cameras[0]
        log.info(
            "%s: PnP against the refined scan camera (%s, params %s)",
            clip_id, camera.model.name if hasattr(camera.model, "name") else camera.model,
            [round(float(p), 3) for p in camera.params],
        )
    else:
        camera = pycolmap.Camera.create(
            camera_id=1, model="PINHOLE", focal_length=intrinsics.fx,
            width=intrinsics.width, height=intrinsics.height,
        )
        camera.params = [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]
        log.warning("%s: scan reconstruction has no camera; using the Stage 0 prior", clip_id)

    pair_lookup: dict[str, list[str]] = {}
    for query, reference in pairs:
        pair_lookup.setdefault(query, []).append(reference)

    poses = np.repeat(np.eye(4)[None], len(frame_names), axis=0)
    valid = np.zeros(len(frame_names), dtype=bool)
    inlier_counts = np.zeros(len(frame_names), dtype=int)

    with rec.timed(f"pnp.{clip_id}"):
        for index, name in enumerate(frame_names):
            keypoints = sfm.load_keypoints(features_path, name)
            pose, diagnostics = sfm.localize_frame(
                name, keypoints, pair_lookup.get(name, []), matches_path, lookup,
                reconstruction, camera,
                float(cfg.get("ransac_max_error_px", 12.0)),
                int(cfg.get("ransac_min_inliers", 12)),
            )
            inlier_counts[index] = diagnostics["inliers"]
            if pose is not None:
                poses[index] = pose
                valid[index] = True

    register_rate = float(valid.mean())
    log.info(
        "%s: %d/%d frames registered (%.1f%%), median inliers %d",
        clip_id, int(valid.sum()), len(frame_names), register_rate * 100,
        int(np.median(inlier_counts[valid])) if valid.any() else 0,
    )

    source = "sfm_registration"
    min_rate = float(cfg.get("min_register_rate", 0.7))
    arkit_meta = clip.get("arkit")

    if register_rate < min_rate:
        scale_payload = read_json(ctx.stage_dir(1, create=False) / "scale.json")
        scale_method = scale_payload.get("diagnostics", {}).get("method")

        if arkit_meta and scale_method == "arkit_sim3":
            log.warning(
                "%s: registration rate %.1f%% is below the %.0f%% threshold. "
                "Falling back to the ARKit trajectory.",
                clip_id, register_rate * 100, min_rate * 100,
            )
            # Stage 1 transformed the whole reconstruction into the ARKit
            # metric frame, so an ARKit trajectory is already in world
            # coordinates and needs no further transform.
            arkit = np.load(ctx.root / arkit_meta["path"])["poses"]
            count = min(len(arkit), len(frame_names))
            poses = np.repeat(np.eye(4)[None], len(frame_names), axis=0)
            poses[:count] = arkit[:count]
            valid = np.zeros(len(frame_names), dtype=bool)
            valid[:count] = True
            source = "arkit_fallback"
        elif arkit_meta:
            # Scale came from ArUco or a manual measurement, so the world is
            # still in COLMAP's arbitrary frame. An ARKit trajectory lives in
            # a different frame entirely and cannot be dropped in without an
            # alignment nobody has computed.
            log.error(
                "%s: registration rate %.1f%% is below threshold. An ARKit "
                "trajectory exists, but Stage 1 scaled the world with '%s', so "
                "the two are in different frames and the fallback is unsafe.",
                clip_id, register_rate * 100, scale_method,
            )
            source = "failed"
        else:
            log.error(
                "%s: registration rate %.1f%% is below threshold and no ARKit "
                "trajectory is available for this clip.",
                clip_id, register_rate * 100,
            )
            source = "failed"

    tracking_loss = 1.0 - float(valid.mean())
    max_loss = float(cfg.get("max_tracking_loss", 0.02))
    rejected = source == "failed" or tracking_loss > max_loss

    smoothed = poses
    if valid.any() and source != "failed":
        smoothed = slerp_fill(poses, valid)
        smoothed = smooth_poses(smoothed, int(cfg.get("smoothing_window", 5)))

    poses_path = out_dir / "camera_poses.npy"
    np.save(poses_path, smoothed)
    np.save(out_dir / "pose_valid.npy", valid)

    status = {
        "clip_id": clip_id,
        "frame_count": len(frame_names),
        "registered": int(valid.sum()),
        "register_rate": round(register_rate, 4),
        "tracking_loss": round(tracking_loss, 4),
        "max_tracking_loss": max_loss,
        "pose_source": source,
        "rejected": rejected,
        "median_inliers": int(np.median(inlier_counts[valid])) if valid.any() else 0,
        "poses": ctx.rel(poses_path),
    }
    if rejected:
        reason = (
            f"tracking lost on {tracking_loss * 100:.1f}% of frames, "
            f"above the {max_loss * 100:.1f}% limit"
            if source != "failed"
            else "localization failed and no ARKit fallback was available"
        )
        status["reject_reason"] = reason
        log.error("%s REJECTED: %s", clip_id, reason)
    else:
        log.info(
            "%s accepted: tracking loss %.2f%%, pose source %s",
            clip_id, tracking_loss * 100, source,
        )

    write_json(out_dir / "status.json", status)
    return status


def run(ctx: RunContext) -> dict:
    """Run Stage 2 over the run directory."""
    rec = StageRecorder(ctx, STAGE, NAME)
    cfg = ctx.config.section("localize")

    try:
        ingest_dir = ctx.stage_dir(0, create=False)
        scene_dir = ctx.stage_dir(1, create=False)
        manifest = read_json(ingest_dir / "manifest.json")
        intrinsics_all = read_json(ingest_dir / "intrinsics.json")
        rec.meta.inputs = {
            "manifest": ctx.rel(ingest_dir / "manifest.json"),
            "colmap_model": ctx.rel(scene_dir / "colmap" / "sparse"),
        }

        device = resolve_device(
            ctx.config.get("device.preferred", "auto"),
            bool(ctx.config.get("device.allow_cpu_fallback", True)),
        )
        rec.backend("device", device)
        rec.backend("localizer", "lightglue_pnp_ransac")

        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(scene_dir / "colmap" / "sparse"))
        log.info(
            "loaded scan reconstruction: %d images, %d points",
            reconstruction.num_reg_images(), reconstruction.num_points3D(),
        )

        scan = manifest["clips"]["scan"]
        scan_frames_dir = ctx.root / scan["frames_dir"]
        scan_names = scan["frame_names"]
        with rec.timed("scan_descriptors"):
            scan_descriptors = sfm.global_descriptors(scan_frames_dir, scan_names)

        statuses = {}
        for clip_id, clip in manifest["clips"].items():
            if clip["kind"] != "demo":
                continue
            log.info("--- Stage 2: %s (%d frames) ---", clip_id, clip["frame_count"])
            intrinsics = Intrinsics.from_dict(intrinsics_all[clip_id])
            statuses[clip_id] = _localize_episode(
                ctx, rec, clip_id, clip, intrinsics, reconstruction,
                scan_names, scan_frames_dir, scan_descriptors, device, cfg,
            )
            rec.output(f"{clip_id}_poses", ctx.episode_dir(STAGE, clip_id) / "camera_poses.npy")

        summary_path = write_json(ctx.stage_dir(STAGE) / "summary.json", statuses)
        rec.output("summary", summary_path)
        rec.metric("episodes", len(statuses))
        rec.metric("accepted", sum(1 for s in statuses.values() if not s["rejected"]))
        rec.metric(
            "register_rates", {k: v["register_rate"] for k, v in statuses.items()}
        )
        rec.metric("pose_sources", {k: v["pose_source"] for k, v in statuses.items()})

        accepted = [k for k, v in statuses.items() if not v["rejected"]]
        if not accepted:
            rec.note("every episode was rejected; nothing downstream can run")
            log.error("Stage 2: every episode was rejected")

        rec.write("ok" if accepted else "failed")
        return statuses

    except Exception as exc:
        rec.note(f"{type(exc).__name__}: {exc}")
        rec.write("failed")
        raise


def load_summary(ctx: RunContext) -> dict:
    return read_json(ctx.stage_dir(STAGE, create=False) / "summary.json")


def load_poses(ctx: RunContext, clip_id: str) -> np.ndarray:
    return np.load(ctx.episode_dir(STAGE, clip_id, create=False) / "camera_poses.npy")
