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
from ..videoio import extract_frames, hamming, laplacian_variance, phash, probe

log = get(__name__)

STAGE = 0
NAME = "ingest"


def _filter_frames(
    frame_paths: list[Path],
    blur_relative_threshold: float,
    blur_absolute_threshold: float,
    blur_max_drop_fraction: float,
    phash_min_distance: int,
) -> tuple[list[Path], dict]:
    """Drop blurred and near-duplicate frames.

    Blur is judged against the clip's own median sharpness. Variance of the
    Laplacian scales with scene texture, so an absolute cut tuned on one room
    behaves differently in the next. The absolute threshold stays only as a
    floor for frames that carry no signal at all.

    Rejection is capped either way: dropping frames a reconstructor would have
    registered is worse than keeping a few soft ones.
    """
    if not frame_paths:
        return [], {"kept": 0, "dropped_blur": 0, "dropped_duplicate": 0}

    sharpness: list[float] = []
    hashes: list[np.uint64] = []
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is None:
            sharpness.append(-1.0)
            hashes.append(np.uint64(0))
            continue
        sharpness.append(laplacian_variance(image))
        hashes.append(phash(image) if phash_min_distance > 0 else np.uint64(0))

    sharpness_arr = np.array(sharpness)
    readable = sharpness_arr >= 0
    median = float(np.median(sharpness_arr[readable])) if readable.any() else 0.0
    threshold = max(blur_relative_threshold * median, blur_absolute_threshold)

    blur_mask = readable & (sharpness_arr >= threshold)
    drop_fraction = 1.0 - blur_mask.mean()

    if drop_fraction > blur_max_drop_fraction:
        # Keep the sharpest allowed share rather than trusting a threshold
        # that clearly does not fit this clip.
        keep_count = max(1, int(round(len(frame_paths) * (1.0 - blur_max_drop_fraction))))
        order = np.argsort(-sharpness_arr)
        blur_mask = np.zeros(len(frame_paths), dtype=bool)
        blur_mask[order[:keep_count]] = True
        log.warning(
            "blur threshold %.1f (%.2f x median %.1f) would drop %.0f%% of frames; "
            "capped to %.0f%%",
            threshold, blur_relative_threshold, median,
            drop_fraction * 100, blur_max_drop_fraction * 100,
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
        "blur_threshold_used": round(threshold, 3),
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
                info.width, info.height, info.focal_35mm, float(cfg["fallback_focal_ratio"])
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
            with rec.timed(f"filter.{clip_id}"):
                kept, stats = _filter_frames(
                    extracted,
                    float(cfg["blur_relative_threshold"]),
                    float(cfg.get("blur_absolute_threshold", 0.0)),
                    float(cfg["blur_max_drop_fraction"]),
                    dedup_distance,
                )
            if not kept:
                raise ValueError(f"{clip_id}: every frame was rejected as blurred or duplicate")

            names = _renumber(kept, frames_dir, clip_id)

            # The effective frame rate after filtering. Stage 4 needs it to
            # convert frame indices into seconds.
            effective_fps = (fps if fps > 0 else info.fps) * (len(names) / max(len(extracted), 1))

            manifest["clips"][clip_id] = {
                "kind": "scan" if clip_id == "scan" else "demo",
                "source_video": str(video_path),
                "video_info": info.to_dict(),
                "frames_dir": ctx.rel(frames_dir),
                "frame_names": names,
                "frame_count": len(names),
                "extracted_count": len(extracted),
                "requested_fps": fps if fps > 0 else info.fps,
                "effective_fps": round(effective_fps, 4),
                "quality": stats,
            }
            # With no EXIF focal length the guess is only a starting point.
            # Say so, and let Stage 1 hand the camera to COLMAP to refine
            # rather than freezing a number nobody measured.
            self_calibrate = source_label == "fallback_guess"
            intrinsics_out[clip_id] = {
                **intrinsics.to_dict(),
                "source": source_label,
                "self_calibrate": self_calibrate,
                "colmap_camera_model": "SIMPLE_RADIAL" if self_calibrate else "PINHOLE",
            }
            if self_calibrate:
                log.info(
                    "%s: no EXIF focal length. Stage 1 will self-calibrate a "
                    "SIMPLE_RADIAL camera starting from f=%.0f px.",
                    clip_id, intrinsics.fx,
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
