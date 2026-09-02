"""Stage 0 · Ingest.

In:  scan video, demo videos, optional ARKit trajectory JSON.
Out: deduplicated frames and camera intrinsics.

Reads `sources.json` from the run root, which the driver writes.
Writes `00_ingest/` containing per-clip frames, `intrinsics.json`,
`manifest.json`, and `meta.json`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..camera import Intrinsics, estimate_from_exif
from ..logging_setup import get
from ..runctx import RunContext, StageRecorder, read_json, write_json
from ..videoio import (
    best_tile_normalised_variance,
    extract_frames,
    hamming,
    laplacian_variance,
    phash,
    probe,
)

log = get(__name__)

STAGE = 0
NAME = "ingest"


def _floor_measure_name(tiles: int) -> str:
    """Name the region the absolute floor judges, so a log line cannot mislead.

    The whole-frame and best-tile scales differ by roughly 3x on close footage.
    A log that printed only the number would read identically for a gate that
    had become three times stricter.
    """
    if tiles <= 1:
        return "whole-frame"
    return f"best-of-{tiles}x{tiles}"


def _filter_frames(
    frame_paths: list[Path],
    blur_relative_threshold: float,
    blur_absolute_threshold: float,
    blur_max_drop_fraction: float,
    phash_min_distance: int,
    blur_normalised_floor: float = 0.0,
    blur_floor_tiles: int = 1,
    segment_of: np.ndarray | None = None,
) -> tuple[list[Path], dict]:
    """Drop blurred and near-duplicate frames.

    Blur is judged against a median sharpness, because variance of the
    Laplacian scales with scene texture and an absolute cut tuned on one room
    behaves differently in the next. The absolute threshold stays only as a
    floor for frames that carry no signal at all.

    **Which median matters.** A scan shot as several passes at different
    heights has genuinely different sharpness per pass: a close pass is soft
    because the lens cannot focus that near, not because those frames are bad.
    Judging every pass against one clip median therefore penalises exactly the
    pass the capture SOP asks for.

    Measured on real26. Two passes gave a clip median of 132.8 and a threshold
    of 39.9, which kept 225 of 299 low-pass frames. Appending a third, sharper
    pass raised the clip median to 149.6 and the threshold to 44.9, which kept
    207. The extra pass silently discarded 18 frames of a different pass, and
    the ones it discarded were the lowest, because those are the blurriest.
    The reconstructed floor rose from 8.47 cm to 12.15 cm on identical footage.

    So `segment_of` labels each frame with the pass it came from, and each pass
    is judged against its own median. The drop cap applies per pass too, for
    the same reason.

    Rejection is capped either way: dropping frames a reconstructor would have
    registered is worse than keeping a few soft ones.
    """
    if not frame_paths:
        return [], {"kept": 0, "dropped_blur": 0, "dropped_duplicate": 0}

    sharpness: list[float] = []
    normalised: list[float] = []
    hashes: list[np.uint64] = []
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is None:
            sharpness.append(-1.0)
            normalised.append(-1.0)
            hashes.append(np.uint64(0))
            continue
        sharpness.append(laplacian_variance(image))
        normalised.append(best_tile_normalised_variance(image, blur_floor_tiles))
        hashes.append(phash(image) if phash_min_distance > 0 else np.uint64(0))

    sharpness_arr = np.array(sharpness)
    normalised_arr = np.array(normalised)
    readable = sharpness_arr >= 0
    median = float(np.median(sharpness_arr[readable])) if readable.any() else 0.0

    if segment_of is None:
        segment_of = np.zeros(len(frame_paths), dtype=int)
    segment_of = np.asarray(segment_of, dtype=int)
    if len(segment_of) != len(frame_paths):
        raise ValueError(
            f"segment_of labels {len(segment_of)} frames and there are "
            f"{len(frame_paths)}"
        )

    blur_mask = np.zeros(len(frame_paths), dtype=bool)
    per_segment: list[dict] = []
    for label in sorted(set(segment_of.tolist())):
        block = segment_of == label
        usable = block & readable
        if not usable.any():
            continue
        block_median = float(np.median(sharpness_arr[usable]))
        threshold = max(blur_relative_threshold * block_median, blur_absolute_threshold)
        keep = usable & (sharpness_arr >= threshold)
        dropped = 1.0 - (keep.sum() / block.sum())
        capped = False
        if dropped > blur_max_drop_fraction:
            # Keep the sharpest allowed share rather than trusting a threshold
            # that clearly does not fit this pass.
            keep_count = max(1, int(round(block.sum() * (1.0 - blur_max_drop_fraction))))
            order = np.argsort(-np.where(block, sharpness_arr, -np.inf))
            keep = np.zeros(len(frame_paths), dtype=bool)
            keep[order[:keep_count]] = True
            capped = True
            log.warning(
                "pass %d: blur threshold %.1f (%.2f x its own median %.1f) would "
                "drop %.0f%% of its frames; capped to %.0f%%",
                label, threshold, blur_relative_threshold, block_median,
                dropped * 100, blur_max_drop_fraction * 100,
            )
        kept_relative = int(keep.sum())

        # The absolute floor, and the reason it exists.
        #
        # The relative rule fixed the case where a scan-wide threshold deleted
        # a soft pass. It opened the opposite one: because the threshold moves
        # down with the pass, a pass that is uniformly soft PROTECTS itself.
        # real27's mat_pass kept 547 of 587 at a threshold of 3.0, and about 11
        # of those frames were worth having.
        #
        # The floor is contrast-normalised, so it is not fooled by exposure,
        # and it is NOT subject to the drop cap above: a cap that rescues a
        # pass from an absolute floor would reintroduce exactly the fault.
        #
        # `blur_floor_tiles` selects WHICH region the floor judges. At 1 it is
        # the whole frame. Above 1 it is the sharpest tile of a square grid,
        # which is the right measure for a pass shot close enough that depth of
        # field, not blur, is what darkens the average. The two are on
        # different scales and are NOT interchangeable: the whole-frame floor
        # calibrated on real26/d is 0.0323 and the best-ninth floor calibrated
        # on the same frames to keep the same 54.2 per cent is 0.1015. Changing
        # one without the other silently changes the gate's strictness.
        if blur_normalised_floor > 0:
            keep = keep & (normalised_arr >= blur_normalised_floor)
        blur_mask |= keep
        per_segment.append({
            "pass": label,
            "frames": int(block.sum()),
            "sharpness_median": round(block_median, 2),
            "normalised_median": round(float(np.median(normalised_arr[usable])), 4),
            "threshold": round(threshold, 2),
            "normalised_floor": blur_normalised_floor,
            "floor_tiles": blur_floor_tiles,
            "kept_relative_only": kept_relative,
            "kept": int(keep.sum()),
            "dropped_by_floor": kept_relative - int(keep.sum()),
            "capped": capped,
        })
    if len(per_segment) > 1:
        for entry in per_segment:
            log.info(
                "  pass %d: %d frames, median sharpness %.1f (%s median %.4f), "
                "relative threshold %.1f kept %d, then the %.4f %s floor kept %d "
                "(dropped %d)",
                entry["pass"], entry["frames"], entry["sharpness_median"],
                _floor_measure_name(entry["floor_tiles"]),
                entry["normalised_median"], entry["threshold"],
                entry["kept_relative_only"], entry["normalised_floor"],
                _floor_measure_name(entry["floor_tiles"]),
                entry["kept"], entry["dropped_by_floor"],
            )

    kept: list[Path] = []
    kept_hashes: list[np.uint64] = []
    dropped_duplicate = 0
    for index, path in enumerate(frame_paths):
        if not blur_mask[index]:
            continue
        # Compare against the last kept frame only. A full pairwise sweep is
        # quadratic and adjacent frames are where duplicates actually appear.
        if (
            phash_min_distance > 0
            and kept_hashes
            and hamming(hashes[index], kept_hashes[-1]) < phash_min_distance
        ):
            dropped_duplicate += 1
            continue
        kept.append(path)
        kept_hashes.append(hashes[index])

    stats = {
        "kept": len(kept),
        "dropped_blur": int((~blur_mask).sum()),
        "dropped_duplicate": dropped_duplicate,
        "blur_per_pass": per_segment,
        "sharpness_median": round(median, 3),
        "sharpness_mean": round(float(sharpness_arr[readable].mean()), 3) if readable.any() else 0.0,
        "sharpness_p10": round(float(np.percentile(sharpness_arr[readable], 10)), 3)
        if readable.any()
        else 0.0,
    }
    return kept, stats


def _renumber(kept: list[Path], out_dir: Path, prefix: str) -> list[str]:
    """Move the kept frames into a contiguous sequence and clear the rest."""
    staging = out_dir.parent / f".{out_dir.name}_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    names: list[str] = []
    for index, path in enumerate(kept):
        name = f"{prefix}_{index:05d}.jpg"
        shutil.move(str(path), staging / name)
        names.append(name)

    shutil.rmtree(out_dir)
    staging.rename(out_dir)
    return names


def _load_arkit(path: Path | None) -> dict | None:
    """Load an ARKit trajectory JSON if the user supplied one.

    Expected shape: `{"frames": [{"timestamp": float, "transform": [16 floats]}]}`
    where transform is a row-major camera-to-world matrix in metres. A list of
    the same objects at the top level is also accepted.
    """
    if path is None or not Path(path).exists():
        return None
    payload = read_json(path)
    frames = payload.get("frames", payload) if isinstance(payload, dict) else payload
    if not isinstance(frames, list) or not frames:
        log.warning("ARKit file %s has no frames; ignoring", path)
        return None

    timestamps = []
    transforms = []
    for entry in frames:
        matrix = entry.get("transform") or entry.get("matrix")
        if matrix is None:
            continue
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.size != 16:
            continue
        transforms.append(arr.reshape(4, 4))
        timestamps.append(float(entry.get("timestamp", len(timestamps))))

    if len(transforms) < 3:
        log.warning("ARKit file %s has fewer than 3 usable poses; ignoring", path)
        return None

    return {
        "timestamps": np.array(timestamps),
        "poses": np.stack(transforms),
    }




def _scan_match_counts(scan_dir, scan_names, demo_dir, demo_names, device, samples):
    """Best scan-match count for each sampled demo frame.

    This measures the question Stage 2 asks: can this frame localize against the
    scan? Marker visibility cannot answer it. See `workspace` for the numbers
    that rejected the marker.
    """
    import tempfile
    from pathlib import Path

    from ..backends import sfm

    pick = np.linspace(0, len(demo_names) - 1, min(samples, len(demo_names)))
    pick = sorted(set(int(v) for v in pick))
    scan_pick = np.linspace(0, len(scan_names) - 1, min(40, len(scan_names)))
    scan_pick = sorted(set(int(v) for v in scan_pick))

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        demo_features = work / "demo.h5"
        scan_features = work / "scan.h5"
        sfm.extract_features(demo_dir, [demo_names[i] for i in pick], demo_features, device, 1024)
        sfm.extract_features(scan_dir, [scan_names[i] for i in scan_pick], scan_features, device, 1024)
        pairs = [(demo_names[i], scan_names[j]) for i in pick for j in scan_pick]
        matches = work / "m.h5"
        sfm.match_pairs(pairs, demo_features, matches, device, features_ref_path=scan_features)

        import h5py

        best = {}
        with h5py.File(str(matches), "r") as handle:
            for a in handle.keys():
                counts = [
                    int((np.array(handle[a][b]["matches0"]).ravel() > -1).sum())
                    for b in handle[a].keys()
                ]
                best[a] = max(counts) if counts else 0

    sampled = np.array([best.get(demo_names[i], 0) for i in pick], dtype=float)
    # Carry each sample forward to the frames around it.
    return np.interp(np.arange(len(demo_names)), pick, sampled)


def run(ctx: RunContext) -> dict:
    """Run Stage 0 over the run directory."""
    rec = StageRecorder(ctx, STAGE, NAME)
    cfg = ctx.config.section("ingest")
    out_dir = ctx.stage_dir(STAGE)

    sources = read_json(ctx.root / "sources.json")
    rec.meta.inputs = sources

    try:
        manifest: dict = {"clips": {}}
        intrinsics_out: dict = {}

        clips: list[tuple[str, str, float, int]] = [
            ("scan", sources["scan"], float(cfg["scan_fps"]), int(cfg["max_scan_frames"]))
        ]
        for index, demo_path in enumerate(sources.get("demos", [])):
            demo_fps = float(cfg.get("demo_fps") or 0.0)
            clips.append(
                (f"demo_{index}", demo_path, demo_fps, int(cfg["max_demo_frames"]))
            )

        for clip_id, video_path, fps, max_frames in clips:
            with rec.timed(f"probe.{clip_id}"):
                info = probe(video_path)
            log.info(
                "%s: %dx%d @ %.2f fps, %d frames, %.1fs, codec %s",
                clip_id, info.width, info.height, info.fps, info.frame_count,
                info.duration_s, info.codec,
            )

            intrinsics, source_label = estimate_from_exif(
                info.width,
                info.height,
                info.focal_35mm,
                float(cfg["fallback_focal_ratio"]),
                info.focal_source,
            )

            # The pipeline assumes a pinhole model. Ultra-wide footage breaks
            # it, so reject the clip rather than reconstruct it wrongly.
            max_fov = float(cfg["max_horizontal_fov_deg"])
            if intrinsics.horizontal_fov_deg > max_fov:
                message = (
                    f"{clip_id} rejected: horizontal FOV {intrinsics.horizontal_fov_deg:.1f} deg "
                    f"exceeds {max_fov:.1f} deg. This looks like ultra-wide footage. "
                    f"Re-shoot on the main wide lens."
                )
                log.error(message)
                rec.note(message)
                raise ValueError(message)

            frames_dir = out_dir / clip_id / "frames"
            with rec.timed(f"extract.{clip_id}"):
                extracted = extract_frames(
                    video_path,
                    frames_dir,
                    fps=fps if fps > 0 else None,
                    quality=int(cfg["jpeg_quality"]),
                    max_frames=max_frames,
                    prefix=clip_id,
                )
            if not extracted:
                raise ValueError(f"{clip_id}: ffmpeg extracted no frames from {video_path}")

            is_scan = clip_id == "scan"
            dedup_distance = int(
                cfg["phash_min_distance"] if is_scan else cfg.get("demo_phash_min_distance", 0)
            )
            # A scan shot as several passes is judged pass by pass. The seams
            # are given, not guessed: whoever concatenated the passes knows
            # where they are, and a detector that guessed would be one more
            # thing to be wrong.
            boundaries = cfg.get("scan_pass_boundaries_s") if is_scan else None
            segment_of = None
            if boundaries:
                extraction_fps_guess = fps if fps > 0 else info.fps
                seconds = np.arange(len(extracted)) / max(extraction_fps_guess, 1e-9)
                segment_of = np.searchsorted(
                    np.asarray(sorted(float(b) for b in boundaries)), seconds, side="right"
                )
                log.info(
                    "%s: %d pass(es) from boundaries %s, judged on their own medians",
                    clip_id, len(set(segment_of.tolist())), list(boundaries),
                )

            with rec.timed(f"filter.{clip_id}"):
                kept, stats = _filter_frames(
                    extracted,
                    float(cfg["blur_relative_threshold"]),
                    float(cfg.get("blur_absolute_threshold", 0.0)),
                    float(cfg["blur_max_drop_fraction"]),
                    dedup_distance,
                    segment_of=segment_of,
                    blur_normalised_floor=float(
                        cfg.get("blur_normalised_floor", 0.0)
                        if clip_id == "scan" else 0.0
                    ),
                    blur_floor_tiles=int(cfg.get("blur_floor_tiles", 1)),
                )
            if not kept:
                raise ValueError(f"{clip_id}: every frame was rejected as blurred or duplicate")

            # Record when each kept frame happened, in source-video seconds.
            # Frame index is not a usable clock: extraction resamples, and
            # deduplication then removes an uneven subset. Anything matching
            # frames against an external trajectory has to match on time.
            extraction_fps = fps if fps > 0 else info.fps
            position = {path: idx for idx, path in enumerate(extracted)}
            frame_times = [position[path] / max(extraction_fps, 1e-9) for path in kept]

            names = _renumber(kept, frames_dir, clip_id)

            # The effective frame rate after filtering. Stage 4 needs it to
            # convert frame indices into seconds.
            effective_fps = (fps if fps > 0 else info.fps) * (len(names) / max(len(extracted), 1))

            # ---- workspace segment ------------------------------------
            #
            # A take holds more than the demonstration. Session real25 opened
            # and closed with a sync clock on a monitor. Stage 2 rejected that
            # clip: a few clock frames matched room geometry, returned wrong
            # poses, and turned a 60 cm camera path into 25.78 m.
            #
            # Measure the scan-match count for each frame. Marker visibility
            # was tried first and cannot separate the two cases. See the
            # `workspace` module for the numbers.
            segment = None
            if clip_id != "scan" and bool(cfg.get("detect_workspace_segment", True)):
                from ..qc import PREFLIGHT_WARN_MATCHES
                from ..workspace import detect_workspace_segment, matched_frames

                scan_clip = manifest["clips"].get("scan")
                if scan_clip is None:
                    raise ValueError(
                        f"{clip_id}: the workspace segment needs the scan, and "
                        f"the scan was not ingested first"
                    )
                from ..device import resolve as resolve_device

                device = resolve_device(
                    ctx.config.get("device.preferred", "auto"),
                    ctx.config.get("device.allow_cpu_fallback", True),
                )
                counts = _scan_match_counts(
                    ctx.root / scan_clip["frames_dir"], scan_clip["frame_names"],
                    frames_dir, names, device,
                    int(cfg.get("workspace_samples", 24)),
                )
                try:
                    lo, hi, segment = detect_workspace_segment(
                        matched_frames(counts, PREFLIGHT_WARN_MATCHES),
                        np.asarray(frame_times, dtype=float),
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{clip_id}: cannot find the workspace segment. {exc}"
                    ) from exc
                segment["match_count_median"] = round(float(np.median(counts)), 1)
                segment["warn_matches"] = float(PREFLIGHT_WARN_MATCHES)
                log.info(
                    "%s: workspace segment t=%.2f-%.2fs, %d of %d frames (%.0f%%), "
                    "median scan matches %.0f",
                    clip_id, segment["start_s"], segment["end_s"], segment["frames"],
                    len(names), segment["fraction_of_clip"] * 100,
                    segment["match_count_median"],
                )

            manifest["clips"][clip_id] = {
                "kind": "scan" if clip_id == "scan" else "demo",
                "source_video": str(video_path),
                "video_info": info.to_dict(),
                "frames_dir": ctx.rel(frames_dir),
                "frame_names": names,
                "frame_times_s": [round(t, 6) for t in frame_times],
                "frame_count": len(names),
                "extracted_count": len(extracted),
                "requested_fps": fps if fps > 0 else info.fps,
                "effective_fps": round(effective_fps, 4),
                "quality": stats,
                "workspace_segment": segment,
            }
            # With no EXIF focal length the guess is only a starting point.
            # Say so, and let Stage 1 hand the camera to COLMAP to refine
            # rather than freezing a number nobody measured.
            # Only a measured EXIF focal length is trusted enough to freeze.
            # A fallback guess and a lens label are both starting points: the
            # label states the lens's nominal focal length, and video crops the
            # sensor away from it, so real27's 24mm label refined to the
            # equivalent of 27mm. Hand both to COLMAP to refine rather than
            # freezing a number nobody measured on this footage.
            self_calibrate = source_label != "exif_focal35"
            intrinsics_out[clip_id] = {
                **intrinsics.to_dict(),
                "source": source_label,
                "self_calibrate": self_calibrate,
                # SIMPLE_RADIAL carries ONE distortion term, which cannot
                # describe the ~110 degrees of barrel an iPhone 0.5x ultra-wide
                # produces. `colmap_camera_model` overrides it so an ultra-wide
                # scan can be reconstructed with OPENCV (k1,k2,p1,p2) or
                # OPENCV_FISHEYE (k1..k4) instead of being fitted with a model
                # that cannot represent the lens.
                "colmap_camera_model": str(
                    cfg.get("colmap_camera_model")
                    or ("SIMPLE_RADIAL" if self_calibrate else "PINHOLE")
                ),
            }
            if self_calibrate:
                log.info(
                    "%s: focal length is a %s, not a measurement. Stage 1 will "
                    "self-calibrate a %s camera starting from f=%.0f px.",
                    clip_id,
                    "lens label" if source_label == "exif_lens_label" else "guess",
                    intrinsics_out[clip_id]["colmap_camera_model"],
                    intrinsics.fx,
                )
            log.info(
                "%s: kept %d of %d frames (blur %d, duplicate %d), fov %.1f deg from %s",
                clip_id, stats["kept"], len(extracted), stats["dropped_blur"],
                stats["dropped_duplicate"], intrinsics.horizontal_fov_deg, source_label,
            )

        # ARKit is the metric-scale reference for the scan and the Stage 2 pose
        # fallback for each demo. `sources.arkit` is a map of clip id to path.
        # A bare string is accepted and read as the scan's trajectory.
        arkit_sources = sources.get("arkit") or {}
        if isinstance(arkit_sources, str):
            arkit_sources = {"scan": arkit_sources}

        any_arkit = False
        for clip_id in list(manifest["clips"]):
            arkit_source = arkit_sources.get(clip_id)
            arkit = _load_arkit(Path(arkit_source) if arkit_source else None)
            if arkit is None:
                manifest["clips"][clip_id]["arkit"] = None
                continue

            arkit_path = out_dir / clip_id / "arkit.npz"
            arkit_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                arkit_path, timestamps=arkit["timestamps"], poses=arkit["poses"]
            )
            manifest["clips"][clip_id]["arkit"] = {
                "path": ctx.rel(arkit_path),
                "frame_count": int(len(arkit["poses"])),
                "source": str(arkit_source),
            }
            rec.output(f"{clip_id}_arkit", arkit_path)
            any_arkit = True
            log.info("%s: ARKit trajectory with %d poses in metres", clip_id, len(arkit["poses"]))

        # Stage 1 reads this one for metric scale.
        manifest["arkit"] = manifest["clips"]["scan"].get("arkit")
        rec.backend("scale_reference", "arkit" if manifest["arkit"] else "none")

        if not any_arkit:
            log.warning(
                "no ARKit trajectory supplied. Stage 1 will try an ArUco marker for "
                "metric scale, and will refuse to continue unscaled unless "
                "scene.scale.allow_unscaled is true."
            )
        elif manifest["arkit"] is None:
            log.warning(
                "ARKit trajectories exist for demos but not for the scan. Stage 1 "
                "will fall back to ArUco for metric scale."
            )

        manifest_path = write_json(out_dir / "manifest.json", manifest)
        intrinsics_path = write_json(out_dir / "intrinsics.json", intrinsics_out)

        rec.output("manifest", manifest_path)
        rec.output("intrinsics", intrinsics_path)
        rec.metric("clips", len(manifest["clips"]))
        rec.metric("scan_frames", manifest["clips"]["scan"]["frame_count"])
        rec.metric(
            "demo_frames",
            {k: v["frame_count"] for k, v in manifest["clips"].items() if v["kind"] == "demo"},
        )
        rec.metric(
            "arkit_available",
            {k: v.get("arkit") is not None for k, v in manifest["clips"].items()},
        )

        rec.write("ok")
        return manifest

    except Exception as exc:
        rec.note(f"{type(exc).__name__}: {exc}")
        rec.write("failed")
        raise


def load_manifest(ctx: RunContext) -> dict:
    """Read Stage 0's manifest. Used by later stages, which never call Stage 0."""
    path = ctx.stage_dir(STAGE, create=False) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 0 has not run: {path} is missing")
    with open(path) as handle:
        return json.load(handle)


def load_intrinsics(ctx: RunContext, clip_id: str) -> Intrinsics:
    path = ctx.stage_dir(STAGE, create=False) / "intrinsics.json"
    with open(path) as handle:
        payload = json.load(handle)
    if clip_id not in payload:
        raise KeyError(f"no intrinsics for clip {clip_id} in {path}")
    return Intrinsics.from_dict(payload[clip_id])
