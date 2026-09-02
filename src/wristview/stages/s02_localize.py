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
from ..qc import (
    SELF_MATCH_BASELINE_S,
    estimate_marker_world_pose,
    marker_agreement,
    marker_gates,
    preflight_gate,
)
from ..runctx import RunContext, StageRecorder, read_json, verify_frames_present, write_json

log = get(__name__)

STAGE = 2
NAME = "localize"


def _motion_diagnostics(
    poses: np.ndarray, valid: np.ndarray, fps: float,
    frame_times_s: np.ndarray | None = None,
) -> dict | None:
    """Speed and path length implied by a pose sequence.

    Used to sanity-check localization against physics rather than against its
    own inlier count.
    """
    if int(valid.sum()) < 3:
        return None
    centres = poses[valid][:, :3, 3]
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    if len(steps) == 0:
        return None
    # Divide by the real interval between kept frames. Stage 0 removes blurred
    # frames, so a frame index is not a clock. See `qc.hand_velocity_outliers`
    # for the session where that error rejected a good clip.
    if frame_times_s is not None:
        times = np.asarray(frame_times_s, dtype=float)[valid]
        seconds = np.maximum(np.diff(times), 1e-6)
    else:
        seconds = np.full(len(steps), 1.0 / max(fps, 1e-6))
    speeds = steps / seconds
    return {
        "path_length_m": round(float(steps.sum()), 4),
        "median_speed_m_s": round(float(np.median(speeds)), 4),
        "max_speed_m_s": round(float(speeds.max()), 4),
        "centre_spread_m": [round(float(v), 4) for v in np.ptp(centres, axis=0)],
    }


def _match_counts(path, pairs: list[tuple[str, str]]) -> list[int]:
    """Raw match counts for a list of pairs, skipping any that were not run."""
    import h5py

    counts: list[int] = []
    with h5py.File(str(path), "r", libver="latest") as handle:
        for name0, name1 in pairs:
            key = sfm.names_to_pair(name0, name1)
            if key in handle:
                counts.append(int((handle[key]["matches0"][()] != -1).sum()))
    return counts


def _scan_self_match_ceiling(ctx: RunContext, scan_names: list[str], scan_fps: float) -> float:
    """How well the scan matches itself across a real viewpoint change.

    The ceiling this footage supports. Comparing a demo against it cancels
    texture, resolution and keypoint budget, which an absolute count does not.

    The baseline is a time gap, not one frame. Adjacent frames at 6 fps are
    0.17 s apart and match almost perfectly, which inflates the ceiling: the
    same scan measured 770 that way against 572 across a 1 second gap, and the
    difference moved a clip from pass to fail with nothing about the footage
    having changed.
    """
    matches_path = ctx.stage_dir(1, create=False) / "matches.h5"
    if not matches_path.exists() or len(scan_names) < 2:
        return 0.0

    gap = max(1, int(round(SELF_MATCH_BASELINE_S * scan_fps)))
    pairs = list(zip(scan_names[:-gap], scan_names[gap:], strict=False))
    counts = _match_counts(matches_path, pairs)
    if not counts:
        # Stage 1 pairs frames by retrieval, so a wide gap may not have been
        # matched at all. Fall back to adjacent and say so.
        log.warning(
            "no scan pairs %d frames apart were matched; falling back to "
            "adjacent frames, which inflates the ceiling", gap,
        )
        pairs = list(zip(scan_names[:-1], scan_names[1:], strict=False))
        counts = _match_counts(matches_path, pairs)
    return float(np.median(counts)) if counts else 0.0


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
    scan_self_matches: float = 0.0,
    marker_world: np.ndarray | None = None,
    refined_intrinsics: Intrinsics | None = None,
) -> dict:
    """Register one demo episode against the scan reconstruction."""
    import pycolmap

    out_dir = ctx.episode_dir(STAGE, clip_id)
    frames_dir = ctx.root / clip["frames_dir"]
    frame_names = clip["frame_names"]

    features_path = out_dir / "features.h5"
    matches_path = out_dir / "matches.h5"

    verify_frames_present(frames_dir, frame_names, clip_id)
    refined_intrinsics = refined_intrinsics or intrinsics
    force = bool(cfg.get("force_rematch", False))

    with rec.timed(f"features.{clip_id}"):
        if not force and sfm.features_cover(features_path, frame_names):
            log.info("%s: reusing existing features for %d frames", clip_id, len(frame_names))
        else:
            sfm.extract_features(
                frames_dir, frame_names, features_path, device,
                max_keypoints=int(cfg.get("max_keypoints", 1024)),
            )

    with rec.timed(f"pairs.{clip_id}"):
        query_descriptors = sfm.global_descriptors(frames_dir, frame_names)
        pairs = sfm.build_query_pairs(
            frame_names, scan_names, query_descriptors, scan_descriptors,
            top_k=int(cfg.get("retrieval_top_k", 20)),
        )

    with rec.timed(f"matching.{clip_id}"):
        # Matching a demo against the scan is the expensive part of this
        # stage, so a failure after it must not pay for it twice.
        if not force and sfm.matches_cover(matches_path, pairs):
            log.info("%s: reusing existing matches for %d pairs", clip_id, len(pairs))
        else:
            log.info("%s: matching %d query pairs against the scan", clip_id, len(pairs))
            sfm.match_pairs(
                pairs, features_path, matches_path, device,
                features_ref_path=ctx.stage_dir(1, create=False) / "features.h5",
            )

    # Viewpoint overlap, measured from the matches just computed. Same
    # quantity the standalone preflight reports, so a batch cannot reach
    # Stage 3 without the check that would have caught a bad capture.
    median_demo_matches = None
    per_frame_best: list[int] = []
    for name in frame_names:
        refs = [r for q, r in pairs if q == name]
        if refs:
            counts = _match_counts(matches_path, [(name, r) for r in refs])
            if counts:
                per_frame_best.append(max(counts))
    if per_frame_best:
        median_demo_matches = float(np.median(per_frame_best))

    lookup = sfm.build_keypoint_to_point3d(reconstruction)

    # Scene size, for judging whether a recovered trajectory is possible.
    scene_span_m = None
    try:
        scene_meta = read_json(ctx.stage_dir(1, create=False) / "meta.json")
        if scene_meta.get("metrics", {}).get("scene_extent_unit") == "m":
            scene_span_m = float(np.linalg.norm(scene_meta["metrics"]["scene_extent"]))
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        pass

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
    correspondence_counts = np.zeros(len(frame_names), dtype=int)

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
            correspondence_counts[index] = diagnostics["correspondences"]
            if pose is not None:
                poses[index] = pose
                valid[index] = True

    register_rate = float(valid.mean())
    inlier_ratio = (
        float(np.median(inlier_counts[valid] / np.maximum(correspondence_counts[valid], 1)))
        if valid.any()
        else 0.0
    )
    log.info(
        "%s: %d/%d frames registered (%.1f%%), median inliers %d of %d (%.0f%%)",
        clip_id, int(valid.sum()), len(frame_names), register_rate * 100,
        int(np.median(inlier_counts[valid])) if valid.any() else 0,
        int(np.median(correspondence_counts[valid])) if valid.any() else 0,
        inlier_ratio * 100,
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

    # Physical plausibility. A registration rate alone proves nothing: PnP
    # returns a pose whenever it finds enough inliers, and with weak geometry
    # those poses can be individually "valid" and collectively nonsense. On
    # this footage every frame registered while the camera appeared to travel
    # 19 m in 7.8 s across a 0.53 m scene, because the demo close-ups share
    # too little with the wide scan. Orientation was stable throughout, so
    # nothing upstream looked wrong.
    _times = clip.get("frame_times_s")
    motion = _motion_diagnostics(
        poses, valid, fps=float(clip.get("effective_fps") or 30.0),
        frame_times_s=np.asarray(_times, dtype=float) if _times else None,
    )
    max_speed = float(cfg.get("max_camera_speed_m_s", 2.0))
    min_ratio = float(cfg.get("min_inlier_ratio", 0.35))

    implausible: list[str] = []
    if motion and motion["median_speed_m_s"] > max_speed:
        implausible.append(
            f"camera moves at a median {motion['median_speed_m_s']:.2f} m/s, "
            f"above the {max_speed:.2f} m/s limit"
        )
    if motion and scene_span_m and motion["path_length_m"] > 8.0 * scene_span_m:
        implausible.append(
            f"camera path is {motion['path_length_m']:.2f} m across a "
            f"{scene_span_m:.2f} m scene, {motion['path_length_m'] / scene_span_m:.0f} "
            f"times its size"
        )
    if source == "sfm_registration" and inlier_ratio < min_ratio:
        implausible.append(
            f"only {inlier_ratio * 100:.0f}% of correspondences are inliers, "
            f"below the {min_ratio * 100:.0f}% floor, so the demo views do not "
            f"overlap the scan well enough to trust the pose"
        )

    if implausible:
        if bool(cfg.get("accept_implausible", False)):
            # Escape hatch for exercising the downstream stages on real data
            # when the capture cannot support localization. The geometry is
            # wrong and every later artifact inherits that, so it is recorded
            # on the episode rather than merely logged.
            log.error(
                "%s: trajectory is not physically possible, but accept_implausible "
                "is set, so it will be passed downstream. EVERY LATER ARTIFACT FOR "
                "THIS EPISODE IS GEOMETRICALLY INVALID.", clip_id,
            )
            source = "implausible_accepted"
        elif not rejected:
            rejected = True
            source = "implausible"

    smoothed = poses
    if valid.any() and source != "failed":
        smoothed = slerp_fill(poses, valid)
        smoothed = smooth_poses(smoothed, int(cfg.get("smoothing_window", 5)))

    poses_path = out_dir / "camera_poses.npy"
    np.save(poses_path, smoothed)
    np.save(out_dir / "pose_valid.npy", valid)

    # Independent check. The marker solves the camera pose from four coplanar
    # corners whose spacing is a physical measurement; Stage 2 solves it from
    # hundreds of triangulated scene points. They share only the intrinsics,
    # so agreement is evidence from two directions and disagreement means at
    # least one is wrong.
    gates: list = []
    agreement: dict = {}
    scale_cfg = ctx.config.section("scene").get("scale", {})
    marker_length = scale_cfg.get("aruco_marker_length_m")
    if marker_length and valid.any():
        with rec.timed(f"marker_check.{clip_id}"):
            agreement = marker_agreement(
                # The refined camera, never the Stage 0 prior. PnP solves
                # against the refined focal, so checking with the guess
                # compares two different cameras: on this capture the prior
                # was 1836 px against a refined 2819, and the check reported a
                # flat 21 cm and 7.2 degrees of "disagreement" on every clip.
                frames_dir, frame_names, smoothed, valid, refined_intrinsics,
                side_m=float(marker_length),
                dictionary_name=scale_cfg.get("aruco_dict", "DICT_4X4_50"),
                marker_id=int(scale_cfg.get("aruco_marker_id") or 0),
                # From the scan, never from the demo poses under test.
                marker_world=marker_world,
            )
        gates.extend(marker_gates(agreement))
        if agreement.get("frames"):
            log.info(
                "%s: marker agrees with the recovered pose to %.2f cm and %.2f deg "
                "median over %d frames (p90 %.2f cm, %.2f deg)",
                clip_id, agreement["position_median_cm"], agreement["rotation_median_deg"],
                agreement["frames"], agreement["position_p90_cm"],
                agreement["rotation_p90_deg"],
            )
        else:
            log.warning("%s: no independent marker check available (%s)",
                        clip_id, agreement.get("reason", "unknown"))

    # Viewpoint overlap, from the matches this stage already computed.
    if median_demo_matches is not None and scan_self_matches:
        gates.append(preflight_gate(median_demo_matches, scan_self_matches))

    failed_gates = [g.name for g in gates if not g.passed]
    if failed_gates:
        log.warning("%s: QC gates failed: %s", clip_id, ", ".join(failed_gates))

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
        "median_correspondences": int(np.median(correspondence_counts[valid])) if valid.any() else 0,
        "inlier_ratio": round(inlier_ratio, 4),
        "motion": motion,
        "marker_agreement": agreement,
        "qc_gates": [g.to_dict() for g in gates],
        "qc_failed": failed_gates,
        "implausible": implausible,
        "geometrically_valid": not implausible,
        "scene_span_m": scene_span_m,
        "poses": ctx.rel(poses_path),
    }
    if rejected:
        if implausible:
            reason = "the recovered trajectory is not physically possible: " + "; ".join(implausible)
        elif source == "failed":
            reason = "localization failed and no ARKit fallback was available"
        else:
            reason = (
                f"tracking lost on {tracking_loss * 100:.1f}% of frames, "
                f"above the {max_loss * 100:.1f}% limit"
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

        # Stage 1 self-calibrates the camera, so its refined intrinsics are
        # what every later measurement must use.
        cameras_payload = read_json(ctx.stage_dir(1, create=False) / "cameras.json")
        refined_intrinsics = Intrinsics.from_dict(cameras_payload["intrinsics"])
        log.info("refined camera from Stage 1: f=%.1f px", refined_intrinsics.fx)
        with rec.timed("scan_descriptors"):
            scan_descriptors = sfm.global_descriptors(scan_frames_dir, scan_names)
        with rec.timed("scan_self_match"):
            scan_self_matches = _scan_self_match_ceiling(
                ctx, scan_names,
                float(scan.get("effective_fps") or scan["video_info"]["fps"]),
            )
        log.info("scan self-match ceiling: %.0f features", scan_self_matches)
        rec.metric("scan_self_match_ceiling", scan_self_matches)

        # The marker's world pose, fixed by the scan. Every demo is checked
        # against this rather than against itself.
        marker_world = None
        scale_cfg_top = ctx.config.section("scene").get("scale", {})
        if scale_cfg_top.get("aruco_marker_length_m"):
            names = [f["name"] for f in cameras_payload["frames"]]
            poses = np.array([f["pose_world_from_cam"] for f in cameras_payload["frames"]])
            with rec.timed("marker_world"):
                marker_world = estimate_marker_world_pose(
                    scan_frames_dir, names, poses, np.ones(len(names), bool),
                    refined_intrinsics,
                    float(scale_cfg_top["aruco_marker_length_m"]),
                    scale_cfg_top.get("aruco_dict", "DICT_4X4_50"),
                    int(scale_cfg_top.get("aruco_marker_id") or 0),
                )
            if marker_world is None:
                log.warning("marker not solvable from the scan; demo checks will be weaker")
            else:
                log.info("marker world position from scan: %s",
                         marker_world[:3, 3].round(4).tolist())
                rec.metric("marker_world_position", marker_world[:3, 3].round(5).tolist())

        # Which demos to localize. null is every demo, which is the norm.
        # Naming a subset exists for the smoke test: Stage 2 is the expensive
        # stage, about 12 minutes a clip on this Mac, and proving the path on
        # one clip before committing to fifty is worth a config key.
        #
        # A run that localizes a subset writes a summary describing ONLY that
        # subset, so the skipped clips are recorded rather than silently
        # absent.
        only = cfg.get("clips")
        if only:
            only = [str(c) for c in only]
            known = [k for k, v in manifest["clips"].items() if v["kind"] == "demo"]
            missing = [c for c in only if c not in known]
            if missing:
                raise ValueError(
                    f"localize.clips names {missing}, which Stage 0 did not ingest. "
                    f"Known demos: {known}"
                )
            log.warning("localize.clips restricts this run to %d of %d demos: %s",
                        len(only), len(known), ", ".join(only))

        statuses = {}
        skipped = []
        for clip_id, clip in manifest["clips"].items():
            if clip["kind"] != "demo":
                continue
            if only and clip_id not in only:
                skipped.append(clip_id)
                continue
            log.info("--- Stage 2: %s (%d frames) ---", clip_id, clip["frame_count"])
            intrinsics = Intrinsics.from_dict(intrinsics_all[clip_id])
            statuses[clip_id] = _localize_episode(
                ctx, rec, clip_id, clip, intrinsics, reconstruction,
                scan_names, scan_frames_dir, scan_descriptors, device, cfg,
                scan_self_matches=scan_self_matches, marker_world=marker_world,
                refined_intrinsics=refined_intrinsics,
            )
            rec.output(f"{clip_id}_poses", ctx.episode_dir(STAGE, clip_id) / "camera_poses.npy")

        if skipped:
            rec.metric("skipped_by_config", skipped)
            rec.note(f"localize.clips skipped {len(skipped)} demos: {', '.join(skipped)}")

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
