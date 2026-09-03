"""Stage 1 · Scene reconstruction.

In:  scan frames from Stage 0.
Out: camera poses, sparse cloud, Gaussian splat, metric scale factor.

The metric scale problem is the part not to skip. A COLMAP reconstruction is
scale-ambiguous, and every downstream number, gripper width, approach
distance, trajectory speed, is meaningless without real units. Four sources,
tried in order:

  ARKit         fit a Sim(3) between the ARKit trajectory, which is already
                in metres, and the COLMAP trajectory. Scale falls out of it.
  ArUco         triangulate a printed marker of known size, measure its side.
  known object  name something already in the shot and give one real
                dimension. Detect it, triangulate it, measure it.
  manual        state the factor, or a real distance and its reconstructed
                counterpart.

Prefer ARKit or ArUco when the capture allows: both are decided before the
shutter closes. The other two recover a shoot that did not plan for scale.

The whole reconstruction is then transformed into metric world coordinates,
so no later stage has to remember which frame it is in.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..backends import sfm
from ..backends.splat_mps import GaussianModel
from ..backends.splat_trainer import load_cameras, train_splat
from ..camera import Intrinsics
from ..device import resolve as resolve_device
from ..geometry import invert_pose, orthonormalize, sim3_matrix, transform_points, umeyama_sim3
from ..logging_setup import get
from ..runctx import RunContext, StageRecorder, read_json, verify_frames_present, write_json

log = get(__name__)

STAGE = 1
NAME = "scene"


def _fit_arkit_scale(
    colmap_poses: dict[str, np.ndarray],
    frame_names: list[str],
    arkit_poses: np.ndarray,
    frame_times: list[float] | None = None,
    arkit_times: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict] | None:
    """Fit a similarity transform from COLMAP to the ARKit metric frame.

    Frames are matched to ARKit poses by timestamp, not by index. Stage 0
    resamples the clip and then drops duplicates, so kept frame i is not
    ARKit pose i: on the fixture, 260 ARKit poses against 390 extracted and
    258 kept made the fit 0.12 m RMSE with a 1.9 m worst case, on a
    trajectory that is exact by construction.
    """
    if frame_times is not None and arkit_times is not None and len(arkit_times):
        pairs = []
        for i, name in enumerate(frame_names):
            if name not in colmap_poses or i >= len(frame_times):
                continue
            nearest = int(np.argmin(np.abs(arkit_times - frame_times[i])))
            # Refuse a match that is more than half a frame away.
            if abs(arkit_times[nearest] - frame_times[i]) <= 0.5 / max(
                len(arkit_times) / max(arkit_times[-1] - arkit_times[0], 1e-6), 1e-6
            ):
                pairs.append((nearest, name))
        usable = pairs
    else:
        registered = [(i, name) for i, name in enumerate(frame_names) if name in colmap_poses]
        usable = [(i, name) for i, name in registered if i < len(arkit_poses)]

    if len(usable) < 8:
        log.warning(
            "only %d frames have both a COLMAP pose and an ARKit pose; "
            "that is too few for a stable Sim(3) fit",
            len(usable),
        )
        return None

    source = np.array([colmap_poses[name][:3, 3] for _, name in usable])
    target = np.array([arkit_poses[i][:3, 3] for i, _ in usable])

    scale, rotation, translation = umeyama_sim3(source, target, with_scale=True)
    transform = sim3_matrix(scale, rotation, translation)

    residual = np.linalg.norm(transform_points(transform, source) - target, axis=1)
    diagnostics = {
        "method": "arkit_sim3",
        "correspondences": len(usable),
        "rmse_m": float(np.sqrt((residual**2).mean())),
        "max_error_m": float(residual.max()),
        "median_error_m": float(np.median(residual)),
    }
    log.info(
        "ARKit Sim(3): scale %.5f over %d frames, RMSE %.4f m, max %.4f m",
        scale, len(usable), diagnostics["rmse_m"], diagnostics["max_error_m"],
    )
    return scale, transform, diagnostics



def _required_marker_length(scale_cfg: dict) -> float:
    """Read the marker size. Refuse to guess it.

    A default for a physical quantity fails silently. `aruco_marker_length_m`
    held 0.15 from the first commit. No marker this project printed was 150 mm.
    A run that forgot the override built a reconstruction 1.5x too large, and
    every metric number downstream inherited that error.
    """
    value = scale_cfg.get("aruco_marker_length_m")
    if value is None:
        raise ValueError(
            "scene.scale.aruco_marker_length_m is required and has no default. "
            "Measure the marker's black square with a tape and pass it, for "
            "example --set scene.scale.aruco_marker_length_m=0.100"
        )
    length = float(value)
    if not 0.01 <= length <= 1.0:
        raise ValueError(
            f"scene.scale.aruco_marker_length_m is {length} m, outside 0.01 to "
            f"1.0 m. Check the units: the value is metres, not millimetres."
        )
    return length


def _fit_aruco_scale(
    frames_dir: Path,
    frame_names: list[str],
    colmap_poses: dict[str, np.ndarray],
    intrinsics: Intrinsics,
    dictionary_name: str,
    marker_length_m: float,
    marker_id: int | None = None,
    min_detections: int = 10,
    scene_points: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict] | None:
    """Recover scale from a printed marker of known size.

    Detect the marker in registered frames, triangulate its corners with the
    COLMAP poses, then compare the measured side length against the printed
    one.

    Only one marker sets the scale. Averaging across marker ids is wrong: a
    real capture produced a spurious id 17 in 8 frames at 29 px alongside the
    genuine id 0 in 114 frames at 300 px, and taking the median of both gave
    0.0857 m per unit against a true 0.0596, a 44 percent error. Every extra
    id is either a different marker of unknown size or a misdetection, and
    neither belongs in the estimate.

    `marker_id` pins which one to trust. Without it the most-detected id
    wins, and anything seen in fewer than `min_detections` frames is dropped.
    """
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    except (AttributeError, cv2.error) as exc:
        log.warning("ArUco unavailable in this OpenCV build: %s", exc)
        return None

    observations: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for name in frame_names:
        if name not in colmap_poses:
            continue
        image = cv2.imread(str(frames_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        corners, ids, _ = detector.detectMarkers(image)
        if ids is None:
            continue
        # Named `detected_id`, not `marker_id`: reusing the parameter name
        # here shadowed the caller's argument, so the configured id was
        # silently replaced by whichever id happened to be seen last.
        for detected_id, corner in zip(ids.ravel(), corners, strict=True):
            observations.setdefault(int(detected_id), []).append(
                (corner.reshape(4, 2), colmap_poses[name])
            )

    if not observations:
        log.info("no ArUco marker found in the scan frames")
        return None

    counts = {mid: len(v) for mid, v in sorted(observations.items())}
    log.info("ArUco ids seen in registered scan frames: %s", counts)

    if marker_id is not None:
        if marker_id not in observations:
            log.warning(
                "configured ArUco id %d was never detected; ids seen were %s",
                marker_id, sorted(observations),
            )
            return None
        chosen = marker_id
    else:
        chosen = max(observations, key=lambda m: len(observations[m]))

    rejected = {m: c for m, c in counts.items() if m != chosen}
    if rejected:
        log.warning(
            "using ArUco id %d (%d detections) for scale and ignoring %s. "
            "Only one marker of known size may set the scale.",
            chosen, counts[chosen], rejected,
        )
    if counts[chosen] < min_detections:
        log.warning(
            "ArUco id %d was detected in only %d frames, below the %d needed "
            "to trust it",
            chosen, counts[chosen], min_detections,
        )
        return None

    matrix = intrinsics.matrix
    lengths: list[float] = []
    marker_centre = None
    world_up: np.ndarray | None = None
    for marker_id_seen, views in observations.items():
        if marker_id_seen != chosen or len(views) < 2:
            continue
        corners3d = []
        for corner_index in range(4):
            projections, pixels = [], []
            for corner, pose in views[:12]:
                cam_from_world = invert_pose(pose)
                projections.append(matrix @ cam_from_world[:3, :4])
                pixels.append(corner[corner_index])
            point = _triangulate(projections, pixels)
            if point is None:
                break
            corners3d.append(point)
        if len(corners3d) != 4:
            continue
        sides = [
            np.linalg.norm(corners3d[i] - corners3d[(i + 1) % 4]) for i in range(4)
        ]
        lengths.append(float(np.mean(sides)))

        # The marker lies flat on the work surface, so its normal is the
        # surface normal. That is the only source of "up" in this pipeline:
        # COLMAP's world orientation is arbitrary, and ArUco scaling only
        # scales, so nothing else knows which way the desk faces.
        normal = np.cross(corners3d[1] - corners3d[0], corners3d[3] - corners3d[0])
        norm = np.linalg.norm(normal)
        # Kept for the close-range coverage check, which needs somewhere to
        # measure "the working area" from. The marker sits beside the object
        # by procedure, so it stands in for it.
        marker_centre = np.mean(corners3d, axis=0)
        if norm > 1e-9:
            normal = normal / norm
            # Point it toward the cameras, which are above the desk.
            centre = marker_centre
            votes = sum(
                1 for _, pose in views[:20]
                if float(np.dot(normal, pose[:3, 3] - centre)) > 0
            )
            if votes < len(views[:20]) / 2:
                normal = -normal
            world_up = normal
        log.info("marker %d: mean side %.5f in COLMAP units", marker_id_seen, lengths[-1])

    if not lengths:
        return None

    # Fit the surface the marker is lying on, in the same units the marker is
    # about to scale. Done before the transform so the caller can scale it.
    desk_plane = None
    if world_up is not None and scene_points is not None and len(scene_points) >= 16:
        try:
            from ..plane import fit_plane

            normal, offset, plane_report = fit_plane(scene_points, world_up)
            desk_plane = {
                "normal": [round(float(v), 6) for v in normal],
                "offset_colmap_units": round(float(offset), 6),
                "report": plane_report,
            }
            log.info(
                "desk plane: %d of %d points within 6 mm, median residual %.2f mm, "
                "%.1f deg from the marker normal",
                plane_report["inliers"], plane_report["points"],
                plane_report["residual_median_mm"], plane_report["angle_to_hint_deg"],
            )
        except ValueError as exc:
            log.warning("could not fit a desk plane: %s", exc)

    measured = float(np.median(lengths))
    if measured < 1e-9:
        return None
    scale = marker_length_m / measured

    # The plane was fitted before scaling. A pure scaling leaves the normal
    # alone and multiplies the offset, so record the metric value here rather
    # than leaving every caller to remember the conversion.
    if desk_plane is not None:
        desk_plane["offset_m"] = round(desk_plane["offset_colmap_units"] * scale, 6)

    transform = sim3_matrix(scale, np.eye(3), np.zeros(3))
    diagnostics = {
        "method": "aruco",
        "marker_id": chosen,
        "detections": counts[chosen],
        "ids_seen": counts,
        "ids_ignored": rejected,
        "measured_side_colmap_units": measured,
        "known_side_m": marker_length_m,
        # In metres: the corners were triangulated before scaling.
        "marker_world_position": (
            [round(float(v) * scale, 5) for v in marker_centre]
            if marker_centre is not None else None
        ),
        "world_up": world_up.round(6).tolist() if world_up is not None else None,
        # The work surface itself. Fitted once per session, because it does not
        # move, and used by Stage 3 to place a resting object without any
        # depth estimate. See src/wristview/plane.py.
        "desk_plane": desk_plane,
        "note": "Scale only. ArUco fixes size but not the world orientation or origin.",
    }
    log.info("ArUco scale: %.5f from %d markers", scale, len(lengths))
    return scale, transform, diagnostics


def _fit_known_object_scale(
    frames_dir: Path,
    frame_names: list[str],
    colmap_poses: dict[str, np.ndarray],
    intrinsics: Intrinsics,
    config: dict,
    device: str,
    max_views: int = 24,
) -> tuple[float, np.ndarray, dict] | None:
    """Metric scale from an object of known size already in the scene.

    The practical alternative to a printed marker: name something in the shot
    and give one real dimension. A drinking glass is ideal because it is a
    vertical cylinder, so its silhouette width is its diameter from every
    viewpoint, while its silhouette height foreshortens with viewing angle.

    Method: detect the object in registered scan frames, triangulate the mask
    centroid to get its position, then for each view convert the mask's pixel
    width into scene units using that view's distance. Width does not need
    features on the object itself, which matters for glass.
    """
    from ..backends import objects as object_backend

    prompt = str(config.get("prompt") or "").strip()
    diameter_m = config.get("diameter_m")
    height_m = config.get("height_m")
    if not prompt or not diameter_m:
        return None

    dino_ok, dino_reason = object_backend.grounding_dino_available()
    sam_ok, sam_reason = object_backend.sam2_available()
    if not (dino_ok and sam_ok):
        log.warning("known-object scale needs Grounding DINO and SAM 2: %s %s",
                    dino_reason, sam_reason)
        return None

    try:
        detector = object_backend.GroundingDinoDetector(device)
        segmenter = object_backend.Sam2Segmenter(device)
    except Exception as exc:  # noqa: BLE001 - fall through to the next source
        log.warning("known-object scale could not load its models: %s", exc)
        return None

    registered = [n for n in frame_names if n in colmap_poses]
    if len(registered) < 3:
        return None
    step = max(1, len(registered) // max_views)
    sampled = registered[::step][:max_views]

    observations: list[tuple[np.ndarray, np.ndarray]] = []
    for name in sampled:
        image = cv2.imread(str(frames_dir / name))
        if image is None:
            continue
        box = detector.detect(image, prompt, 0.35, 0.25)
        if box is None:
            continue
        mask = segmenter.segment(image, box=box)
        mask = object_backend.clean_mask(mask, 200, 0.25)
        if mask is None:
            continue
        observations.append((mask, colmap_poses[name]))

    if len(observations) < 3:
        log.warning(
            "known-object scale: found '%s' in only %d of %d sampled scan frames",
            prompt, len(observations), len(sampled),
        )
        return None

    matrix = intrinsics.matrix
    projections, pixels = [], []
    for mask, pose in observations:
        ys, xs = np.nonzero(mask)
        projections.append(matrix @ invert_pose(pose)[:3, :4])
        pixels.append(np.array([xs.mean(), ys.mean()]))

    centre = _triangulate(projections, pixels)
    if centre is None:
        log.warning("known-object scale: could not triangulate the object centre")
        return None

    widths, heights = [], []
    for mask, pose in observations:
        distance = float(np.linalg.norm(pose[:3, 3] - centre))
        ys, xs = np.nonzero(mask)
        widths.append(float(xs.max() - xs.min()) * distance / intrinsics.fx)
        heights.append(float(ys.max() - ys.min()) * distance / intrinsics.fy)

    measured_width = float(np.median(widths))
    if measured_width < 1e-9:
        return None
    scale = float(diameter_m) / measured_width

    diagnostics = {
        "method": "known_object",
        "prompt": prompt,
        "views_used": len(observations),
        "views_sampled": len(sampled),
        "known_diameter_m": float(diameter_m),
        "measured_diameter_units": round(measured_width, 6),
        "width_spread_units": round(float(np.percentile(widths, 84) - np.percentile(widths, 16)), 6),
    }

    if height_m:
        # Height foreshortens with viewing angle, so the largest observation is
        # the least foreshortened. A cross-check, never the primary measure.
        measured_height = float(np.percentile(heights, 90))
        diagnostics["known_height_m"] = float(height_m)
        diagnostics["measured_height_units"] = round(measured_height, 6)
        diagnostics["height_implied_scale"] = round(float(height_m) / max(measured_height, 1e-9), 6)
        diagnostics["height_agreement"] = round(
            diagnostics["height_implied_scale"] / scale, 4
        )
        log.info(
            "known-object cross-check: height implies scale %.5f against %.5f from "
            "diameter, agreement %.2f",
            diagnostics["height_implied_scale"], scale, diagnostics["height_agreement"],
        )

    log.info(
        "known-object scale: '%s' measured %.5f units across in %d views, "
        "known %.3f m, scale %.5f",
        prompt, measured_width, len(observations), diameter_m, scale,
    )
    return scale, sim3_matrix(scale, np.eye(3), np.zeros(3)), diagnostics


def _manual_scale(config: dict) -> tuple[float, np.ndarray, dict] | None:
    """Metric scale from one measured distance.

    The fallback when there is no ARKit trajectory and no marker in the scene.
    Either give the factor directly, or give a real distance in metres and the
    same distance read off the unscaled reconstruction.

    This only fixes size. It leaves the world's orientation and origin in
    COLMAP's arbitrary frame, which is fine: every later stage works in
    relative geometry, and only lengths need to be real.
    """
    if not config:
        return None

    factor = config.get("scale_factor")
    known = config.get("known_distance_m")
    measured = config.get("reconstruction_distance_units")

    if factor is None:
        if known is None or measured is None:
            return None
        if float(measured) < 1e-9:
            log.warning("manual scale: reconstruction_distance_units is zero")
            return None
        factor = float(known) / float(measured)

    factor = float(factor)
    if not np.isfinite(factor) or factor <= 0:
        log.warning("manual scale: factor %s is not usable", factor)
        return None

    diagnostics = {
        "method": "manual",
        "known_distance_m": known,
        "reconstruction_distance_units": measured,
        "note": config.get("note", ""),
        "caveat": (
            "Manual scale fixes size only. World orientation and origin stay in "
            "COLMAP's arbitrary frame. Accuracy is the accuracy of the measurement."
        ),
    }
    log.info("manual scale: factor %.6f", factor)
    return factor, sim3_matrix(factor, np.eye(3), np.zeros(3)), diagnostics


def _triangulate(projections: list[np.ndarray], pixels: list[np.ndarray]) -> np.ndarray | None:
    """Linear triangulation of one point from several views. DLT."""
    rows = []
    for projection, pixel in zip(projections, pixels, strict=True):
        rows.append(pixel[0] * projection[2] - projection[0])
        rows.append(pixel[1] * projection[2] - projection[1])
    matrix = np.asarray(rows)
    if matrix.shape[0] < 4:
        return None
    _, _, vt_mat = np.linalg.svd(matrix)
    homogeneous = vt_mat[-1]
    if abs(homogeneous[3]) < 1e-12:
        return None
    return homogeneous[:3] / homogeneous[3]


def run(ctx: RunContext) -> dict:
    """Run Stage 1 over the run directory."""
    rec = StageRecorder(ctx, STAGE, NAME)
    cfg = ctx.config.section("scene")
    out_dir = ctx.stage_dir(STAGE)

    try:
        ingest_dir = ctx.stage_dir(0, create=False)
        manifest = read_json(ingest_dir / "manifest.json")
        intrinsics_all = read_json(ingest_dir / "intrinsics.json")
        rec.meta.inputs = {
            "manifest": ctx.rel(ingest_dir / "manifest.json"),
            "intrinsics": ctx.rel(ingest_dir / "intrinsics.json"),
        }

        scan = manifest["clips"]["scan"]
        frames_dir = ctx.root / scan["frames_dir"]
        frame_names = scan["frame_names"]
        intrinsics_raw = intrinsics_all["scan"]
        intrinsics = Intrinsics.from_dict(intrinsics_raw)
        device = resolve_device(
            ctx.config.get("device.preferred", "auto"),
            bool(ctx.config.get("device.allow_cpu_fallback", True)),
        )
        rec.backend("device", device)

        verify_frames_present(frames_dir, frame_names, "scan")
        log.info("Stage 1: reconstructing from %d scan frames", len(frame_names))

        # ---- feature extraction and matching -----------------------------
        features_path = out_dir / "features.h5"
        matches_path = out_dir / "matches.h5"

        if cfg.get("matcher", "hloc") == "hloc":
            rec.backend("matcher", "superpoint+lightglue_mps")
            force = bool(cfg.get("force_rematch", False))

            with rec.timed("features"):
                if not force and sfm.features_cover(features_path, frame_names):
                    log.info("reusing existing features for %d frames", len(frame_names))
                    rec.note("features reused from a previous run")
                else:
                    sfm.extract_features(
                        frames_dir, frame_names, features_path, device,
                        max_keypoints=int(cfg.get("max_keypoints", 1024)),
                    )

            with rec.timed("pairs"):
                descriptors = sfm.global_descriptors(frames_dir, frame_names)
                pairs = sfm.build_pairs(
                    frame_names,
                    descriptors,
                    mode=cfg.get("pair_mode", "exhaustive"),
                    seq_window=int(cfg.get("sequential_window", 10)),
                    retrieval_k=int(cfg.get("retrieval_num_matched", 15)),
                )
                pairs_path = sfm.write_pairs_file(pairs, out_dir / "pairs.txt")

            with rec.timed("matching"):
                if not force and sfm.matches_cover(matches_path, pairs):
                    log.info("reusing existing matches for %d pairs", len(pairs))
                    rec.note("matches reused from a previous run")
                else:
                    log.info("matching %d image pairs", len(pairs))
                    sfm.match_pairs(pairs, features_path, matches_path, device)
        else:
            raise NotImplementedError(
                "Only the hloc matcher path is implemented. Set scene.matcher to hloc."
            )

        rec.metric("pairs", len(pairs))

        # ---- incremental mapping -----------------------------------------
        sfm_dir = out_dir / "colmap"
        sparse_dir = sfm_dir / "sparse"
        # The raw reconstruction is cached separately from the metric one.
        # `sparse` is written after the Sim(3) is applied, so reusing it as
        # the mapping cache re-scales an already-scaled model: the marker
        # measured 1.678 units on the first run and 0.144 on the second,
        # which is exactly 1.678 times the 0.0857 scale the first run
        # applied. Caching the raw model keeps Stage 1 idempotent.
        raw_dir = sfm_dir / "sparse_raw"
        with rec.timed("mapping"):
            if not force and (raw_dir / "cameras.bin").exists():
                import pycolmap

                reconstruction = pycolmap.Reconstruction(str(raw_dir))
                log.info(
                    "reusing existing reconstruction: %d registered images",
                    reconstruction.num_reg_images(),
                )
                rec.note("reconstruction reused from a previous run")
            else:
                reconstruction = sfm.run_reconstruction(
                    sfm_dir, frames_dir, pairs_path, features_path, matches_path,
                    intrinsics=intrinsics_raw, image_list=frame_names,
                )
                raw_dir.mkdir(parents=True, exist_ok=True)
                reconstruction.write(str(raw_dir))
        if reconstruction is None:
            raise RuntimeError(
                "COLMAP mapping produced no model. The scan did not reconstruct. "
                "This is Gate 0 failing: re-shoot with more viewpoint overlap and "
                "with iPhone stabilization switched off."
            )

        poses = sfm.reconstruction_poses(reconstruction)
        registration_rate = len(poses) / max(len(frame_names), 1)
        mean_reproj = float(reconstruction.compute_mean_reprojection_error())

        # What COLMAP converged on, which is what every later stage must use.
        refined = sfm.refined_camera(reconstruction)
        if refined:
            log.info(
                "camera refined: %s f=%.1f px (prior %.1f, %+.1f%%), c=(%.1f, %.1f), distortion %s",
                refined["model"], refined["fx"], intrinsics.fx,
                (refined["fx"] / intrinsics.fx - 1.0) * 100,
                refined["cx"], refined["cy"],
                {k: round(v, 5) for k, v in refined["distortion"].items()},
            )
            intrinsics = Intrinsics(
                width=refined["width"], height=refined["height"],
                fx=refined["fx"], fy=refined["fy"],
                cx=refined["cx"], cy=refined["cy"],
            )
            write_json(out_dir / "camera_refined.json", refined)
            rec.output("camera_refined", out_dir / "camera_refined.json")
            rec.metric("refined_focal_px", round(refined["fx"], 2))
            rec.metric("refined_distortion", {k: round(v, 6) for k, v in refined["distortion"].items()})
            rec.metric("refined_fov_deg", round(intrinsics.horizontal_fov_deg, 2))

        mean_track_length = float(reconstruction.compute_mean_track_length())
        rec.metric("mean_track_length", round(mean_track_length, 3))
        rec.metric(
            "mean_observations_per_image",
            round(float(reconstruction.compute_mean_observations_per_reg_image()), 2),
        )
        log.info(
            "COLMAP: %d/%d frames registered (%.1f%%), %d points, "
            "mean reprojection error %.3f px, mean track length %.2f",
            len(poses), len(frame_names), registration_rate * 100,
            reconstruction.num_points3D(), mean_reproj, mean_track_length,
        )
        rec.metric("registered_frames", len(poses))
        rec.metric("registration_rate", round(registration_rate, 4))
        rec.metric("sparse_points", int(reconstruction.num_points3D()))
        rec.metric("mean_reprojection_error_px", round(mean_reproj, 4))

        # Both reconstruction gates, evaluated together and recorded. The
        # reprojection threshold existed in the config for the whole project
        # and was never read here, so it never rejected anything.
        from ..qc import reconstruction_gates

        for gate in reconstruction_gates(
            {
                "registration_rate": registration_rate,
                "mean_reprojection_error_px": mean_reproj,
            },
            float(cfg.get("min_registration_rate", 0.8)),
            float(cfg.get("max_reproj_error_px", 1.5)),
        ):
            rec.metric(f"gate_{gate.name}", {
                "passed": gate.passed, "value": gate.value,
                "threshold": gate.threshold, "unit": gate.unit,
            })
            if not gate.passed:
                log.error("GATE FAILED %s: %s %s against a limit of %s. %s",
                          gate.name, gate.value, gate.unit, gate.threshold,
                          gate.detail)
                rec.note(f"gate {gate.name} failed: {gate.value} {gate.unit} "
                         f"against {gate.threshold}")

        min_rate = float(cfg.get("min_registration_rate", 0.8))
        if registration_rate < min_rate:
            message = (
                f"Gate 0 fails: only {registration_rate * 100:.1f}% of scan frames "
                f"registered, against a {min_rate * 100:.0f}% threshold. "
                f"Fix the capture before anything downstream is worth running."
            )
            log.error(message)
            rec.note(message)
            raise RuntimeError(message)

        # ---- metric scale -------------------------------------------------
        # The sparse cloud, in COLMAP units, for the desk-plane fit below.
        sparse_xyz = np.array(
            [p.xyz for p in reconstruction.points3D.values()], dtype=np.float64
        )

        scale_cfg = cfg.get("scale", {})
        source = scale_cfg.get("source", "auto")
        scale_result = None

        arkit_meta = manifest.get("arkit")
        if source in ("auto", "arkit") and arkit_meta:
            with rec.timed("scale.arkit"):
                arkit = np.load(ctx.root / arkit_meta["path"])
                scale_result = _fit_arkit_scale(
                    poses, frame_names, arkit["poses"],
                    frame_times=scan.get("frame_times_s"),
                    arkit_times=arkit.get("timestamps"),
                )

        if scale_result is None and source in ("auto", "aruco"):
            with rec.timed("scale.aruco"):
                scale_result = _fit_aruco_scale(
                    frames_dir, frame_names, poses, intrinsics,
                    scale_cfg.get("aruco_dict", "DICT_4X4_50"),
                    _required_marker_length(scale_cfg),
                    marker_id=scale_cfg.get("aruco_marker_id"),
                    min_detections=int(scale_cfg.get("aruco_min_detections", 10)),
                    scene_points=sparse_xyz,
                )

        if scale_result is None and source in ("auto", "known_object"):
            with rec.timed("scale.known_object"):
                scale_result = _fit_known_object_scale(
                    frames_dir, frame_names, poses, intrinsics,
                    scale_cfg.get("known_object", {}), device,
                )

        if scale_result is None and source in ("auto", "manual"):
            scale_result = _manual_scale(scale_cfg.get("manual", {}))

        if scale_result is None:
            # Give the operator the number they need to fix this, rather than
            # only the complaint. The camera path length is easy to compare
            # against a real distance walked, or against a measured object.
            centres = np.array([poses[name][:3, 3] for name in frame_names if name in poses])
            path_units = float(
                np.sum(np.linalg.norm(np.diff(centres, axis=0), axis=1))
            ) if len(centres) > 1 else 0.0
            span_units = float(np.linalg.norm(centres.max(axis=0) - centres.min(axis=0))) \
                if len(centres) else 0.0

            message = (
                "No metric scale could be recovered. Neither an ARKit trajectory nor "
                "an ArUco marker was usable. Every downstream distance would be in "
                "arbitrary units, and a wrong scale fails silently."
            )
            hint = (
                f"To fix it, measure one real distance in the scene and set "
                f"scene.scale.manual.known_distance_m together with "
                f"reconstruction_distance_units. For reference this reconstruction "
                f"spans {span_units:.4f} units corner to corner and the camera path "
                f"is {path_units:.4f} units long."
            )
            if not bool(scale_cfg.get("allow_unscaled", False)):
                log.error("%s %s", message, hint)
                rec.note(message)
                rec.note(hint)
                raise RuntimeError(
                    f"{message} {hint} Set scene.scale.allow_unscaled to true to override."
                )
            log.warning("%s Continuing unscaled because allow_unscaled is set. %s", message, hint)
            rec.note(message + " Continued unscaled by config.")
            rec.note(hint)
            rec.metric("unscaled_camera_path_units", round(path_units, 5))
            rec.metric("unscaled_scene_span_units", round(span_units, 5))
            scale, transform, diagnostics = 1.0, np.eye(4), {"method": "none", "verified": False}
        else:
            scale, transform, diagnostics = scale_result

        rec.backend("scale_source", diagnostics["method"])

        # Move the whole reconstruction into metric world coordinates, so no
        # later stage has to track which frame it is looking at.
        #
        # The reconstruction itself is transformed, not just a copy of its
        # poses. Stage 2 localizes demo frames against the model written to
        # disk, so leaving that model in COLMAP's arbitrary frame puts every
        # demo camera in a different space from the splat, while each stage
        # still looks internally consistent. The fixture caught it: camera
        # error came out at 2.5 m and 147 degrees against ground truth.
        with rec.timed("transform"):
            if diagnostics["method"] != "none":
                import pycolmap

                rotation = transform[:3, :3] / scale
                reconstruction.transform(
                    pycolmap.Sim3d(
                        scale,
                        pycolmap.Rotation3d(orthonormalize(rotation)),
                        transform[:3, 3],
                    )
                )

            metric_poses = sfm.reconstruction_poses(reconstruction)
            points, colors = sfm.reconstruction_points(reconstruction)
            metric_points = points

        scale_payload = {
            "scale_factor": float(scale),
            "sim3_colmap_to_metric": transform.tolist(),
            "units": "metres",
            "verified": diagnostics["method"] != "none",
            "diagnostics": diagnostics,
        }
        scale_path = write_json(out_dir / "scale.json", scale_payload)

        # ---- can this scan constrain geometry at all ----------------------
        #
        # Registration rate and reprojection error both measure the scan
        # against itself. A capture taken from a single spot scores perfectly
        # on both while carrying no depth information at all: session 6
        # registered 327 of 327 at 1.3565 px from a 2.96 cm baseline at 43 cm,
        # then rendered every wrist frame from below its lowest view. Reported
        # here, where a reshoot is still cheap.
        plane = diagnostics.get("desk_plane")
        if plane and plane.get("offset_m") is not None and len(metric_poses) >= 2:
            from .. import coverage as coverage_module

            centres = np.array([pose[:3, 3] for pose in metric_poses.values()])
            directions = np.array([
                pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
                for pose in metric_poses.values()
            ])
            scan_report = coverage_module.scan_geometry(
                centres, np.asarray(plane["normal"], dtype=float),
                float(plane["offset_m"]), view_directions=directions,
            )
            rec.metric("scan_geometry", scan_report)
            log.info(
                "scan geometry: %d views, baseline %.1f cm, height band %.1f cm, "
                "baseline over subject distance %.3f",
                scan_report["views"], scan_report["max_baseline_m"] * 100,
                scan_report["height_span_m"] * 100,
                scan_report.get("baseline_over_distance", float("nan")),
            )
            for failure in scan_report.get("failures", []):
                log.error("SCAN GEOMETRY: %s", failure)
                rec.note(f"scan geometry: {failure}")
            if scan_report.get("failures"):
                log.error(
                    "This scan cannot support a wrist render. Localization may "
                    "still succeed and the splat may still score well on the "
                    "scan's own views; both measure the scan against itself. "
                    "See CAPTURE-SOP.md phase 3."
                )
        rec.output("scale", scale_path)
        rec.metric("scale_factor", float(scale))

        # A sanity check a human can read: how big is the reconstructed room.
        # Reported at the 2nd and 98th percentile, because a sparse cloud
        # always has a few far outliers and the raw bounding box tracks those
        # rather than the room. Labelled by whether the scale is real.
        if len(metric_points):
            low = np.percentile(metric_points, 2, axis=0)
            high = np.percentile(metric_points, 98, axis=0)
            extent = high - low
            unit = "m" if diagnostics["method"] != "none" else "units (UNSCALED)"
            log.info(
                "scene extent (2nd to 98th percentile): %.2f x %.2f x %.2f %s",
                extent[0], extent[1], extent[2], unit,
            )
            rec.metric("scene_extent", [round(float(v), 3) for v in extent])
            rec.metric("scene_extent_unit", "m" if diagnostics["method"] != "none" else "colmap")

        # ---- artifacts ----------------------------------------------------
        cameras_payload = {
            "intrinsics": intrinsics.to_dict(),
            "frames": [
                {"name": name, "pose_world_from_cam": metric_poses[name].tolist()}
                for name in frame_names
                if name in metric_poses
            ],
        }
        cameras_path = write_json(out_dir / "cameras.json", cameras_payload)
        rec.output("cameras", cameras_path)

        ply_path = out_dir / "scene.ply"
        _write_ply(ply_path, metric_points, colors)
        rec.output("scene_ply", ply_path)

        # pycolmap refuses to write into a directory that does not exist.
        sparse_dir = sfm_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        reconstruction.write(str(sparse_dir))
        rec.output("colmap_model", sfm_dir / "sparse")
        rec.output("features", features_path)
        rec.output("matches", matches_path)

        # ---- Gaussian splat ------------------------------------------------
        splat_cfg = cfg.get("splat", {})
        if bool(splat_cfg.get("enabled", True)):
            rec.backend("splat", "wristview_mps_rasterizer")
            rec.note(
                "Gaussian splatting uses a native PyTorch tile rasterizer on MPS. "
                "gsplat is CUDA-only. gsplat-mlx fails to build its Metal extension "
                "against MLX 0.32 on this machine. Brush needs a Rust toolchain and "
                "cannot render from an arbitrary pose through a Python call, which "
                "Stage 5 requires."
            )
            entries = [
                {
                    "name": name,
                    "view_matrix": invert_pose(metric_poses[name]).tolist(),
                    "fx": intrinsics.fx, "fy": intrinsics.fy,
                    "cx": intrinsics.cx, "cy": intrinsics.cy,
                    "width": intrinsics.width, "height": intrinsics.height,
                }
                for name in frame_names
                if name in metric_poses
            ]
            with rec.timed("splat.load"):
                cameras = load_cameras(
                    entries, frames_dir, device, int(splat_cfg.get("resolution", 720))
                )
            with rec.timed("splat.train"):
                model, splat_metrics = train_splat(
                    metric_points, colors, cameras, splat_cfg, device,
                    progress_dir=out_dir / "splat_progress",
                )
            splat_path = model.save(out_dir / "splat.pt")
            model.export_ply(out_dir / "splat_points.ply", max_points=200000)
            rec.output("splat", splat_path)
            rec.output("splat_ply", out_dir / "splat_points.ply")
            rec.metric("splat", {k: v for k, v in splat_metrics.items() if k != "history"})
            write_json(out_dir / "splat_history.json", splat_metrics["history"])

            # The splat gate. A splat that cannot redraw its own training views
            # cannot draw a new one either.
            #
            # real26 trained to 17.51 dB over 10455 Gaussians and shipped. The
            # wrist render then produced featureless discs. The numbers that
            # would have caught it were already measured here and nothing read
            # them. `tools/check_splat.py` applies the same gate to a splat
            # trained on a GPU.
            from ..splatqc import evaluate as evaluate_splat

            report = evaluate_splat(
                splat_metrics.get("train_psnr_per_view_db", []),
                gaussian_count=int(splat_metrics["gaussians"]),
                view_count=int(splat_metrics.get("training_views", len(cameras))),
            )
            rec.metric("gate_splat_quality", report)
            if report["passed"] is False:
                for line in report["failures"]:
                    log.error("GATE FAILED splat_quality: %s", line)
                message = (
                    "Gate fails: the splat cannot reproduce its own training "
                    "views. " + " ".join(report["failures"])
                )
                rec.note(message)
                raise RuntimeError(message)
        else:
            rec.backend("splat", "disabled")
            log.warning("splat training disabled by config; Stage 5 will have no background")

        rec.write("ok")
        return {"registration_rate": registration_rate, "scale": scale}

    except Exception as exc:
        rec.note(f"{type(exc).__name__}: {exc}")
        rec.write("failed")
        raise


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    """Write the sparse cloud so a human can open it and see the room."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    with open(path, "w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, rgb, strict=True):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )
    return path


def load_scale(ctx: RunContext) -> dict:
    return read_json(ctx.stage_dir(STAGE, create=False) / "scale.json")


def load_splat(ctx: RunContext, device: str) -> GaussianModel | None:
    path = ctx.stage_dir(STAGE, create=False) / "splat.pt"
    if not path.exists():
        return None
    model = GaussianModel.load(path, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_reconstruction(ctx: RunContext):
    import pycolmap

    path = ctx.stage_dir(STAGE, create=False) / "colmap" / "sparse"
    if not path.exists():
        raise FileNotFoundError(f"Stage 1 reconstruction missing at {path}")
    return pycolmap.Reconstruction(str(path))


def load_metric_transform(ctx: RunContext) -> tuple[float, np.ndarray]:
    """The Sim(3) that takes COLMAP coordinates to metres."""
    payload = load_scale(ctx)
    return float(payload["scale_factor"]), np.asarray(payload["sim3_colmap_to_metric"])
