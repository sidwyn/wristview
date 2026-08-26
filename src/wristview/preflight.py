"""Stage 2 pre-flight: will these clips localize?

Answers in under a minute the question a full pipeline run takes hours to
answer. Stage 2 registers each demo frame against the scan reconstruction, and
that only works when the two share enough viewpoint that feature matching
succeeds. On the first real capture it did not: the demos were tight top-down
close-ups, the scan was a wide oblique orbit, and the best demo-to-scan match
gave 126 features where scan-to-scan neighbours gave over 700.

The measurement is that comparison, made directly:

  reference   how well the scan matches itself, neighbour to neighbour.
              This is the ceiling the footage can support.
  demo        how well the best scan frame matches a demo frame.
  ratio       demo over reference. The scale-free number that decides it.

A ratio is used rather than an absolute count because the ceiling depends on
texture, resolution and keypoint budget. Comparing a clip against its own
scan's self-match cancels all three.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backends import sfm
from .logging_setup import get

# Thresholds come from qc.py so there is exactly one definition. Two copies
# drift, and a preflight that disagrees with the gate it predicts is worse
# than no preflight: an inflated ceiling in the Stage 2 copy failed a clip the
# standalone tool had passed.
from .qc import (
    PREFLIGHT_MAX_WEAK_FRACTION,
    PREFLIGHT_MAX_WEAK_RUN_S,
    PREFLIGHT_PASS_MATCHES,
    PREFLIGHT_WARN_MATCHES,
)
from .qc import PREFLIGHT_PASS_RATIO as PASS_RATIO
from .qc import PREFLIGHT_WARN_RATIO as WARN_RATIO
from .videoio import extract_frames, probe

log = get(__name__)


@dataclass
class PreflightResult:
    scan_frames: int
    demo_frames: int
    reference_matches: float
    demo_matches: float
    ratio: float
    verdict: str
    per_demo_best: list[int]
    baseline_px: float = float("nan")
    reference_pairs_used: int = 0
    # Worst-case statistics. The median hides a minority of failing frames.
    # Session real25 passed on a median of 389 matches. Stage 2 then registered
    # 73.4 per cent of the clip. Two runs of frames failed at the two ends. A
    # median over 6 samples cannot express that failure.
    weak_fraction: float = 0.0
    longest_weak_run_s: float = 0.0
    longest_weak_run_frames: int = 0
    sample_interval_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


def _sample(names: list[str], count: int) -> list[str]:
    if len(names) <= count:
        return names
    idx = np.linspace(0, len(names) - 1, count).astype(int)
    return [names[i] for i in idx]


def run_preflight(
    scan_video: Path,
    demo_video: Path,
    device: str,
    scan_frames: int = 30,
    demo_frames: int = 24,
    max_keypoints: int = 1024,
    workdir: Path | None = None,
) -> PreflightResult:
    """Measure demo-to-scan matchability against the scan's own ceiling."""
    owned = workdir is None
    workdir = Path(workdir or tempfile.mkdtemp(prefix="wristview_preflight_"))
    try:
        scan_dir = workdir / "scan"
        demo_dir = workdir / "demo"

        scan_info = probe(scan_video)
        demo_info = probe(demo_video)

        # Sample across the whole of each clip rather than a burst from one
        # moment: a scan's usefulness is in its viewpoint coverage.
        scan_fps = max(0.5, scan_frames / max(scan_info.duration_s, 1e-3))
        demo_fps = max(0.5, demo_frames / max(demo_info.duration_s, 1e-3))

        scan_names = [
            p.name for p in extract_frames(scan_video, scan_dir, fps=scan_fps, prefix="scan")
        ]
        demo_names = [
            p.name for p in extract_frames(demo_video, demo_dir, fps=demo_fps, prefix="demo")
        ]
        scan_names = _sample(scan_names, scan_frames)
        demo_names = _sample(demo_names, demo_frames)
        if len(scan_names) < 4 or len(demo_names) < 2:
            raise ValueError(
                f"too few frames to judge: {len(scan_names)} scan, {len(demo_names)} demo"
            )

        scan_features = workdir / "scan.h5"
        demo_features = workdir / "demo.h5"
        sfm.extract_features(scan_dir, scan_names, scan_features, device, max_keypoints)
        sfm.extract_features(demo_dir, demo_names, demo_features, device, max_keypoints)

        # Reference ceiling at a matched baseline, not a matched time interval.
        #
        # The old version took scan pairs a fixed number of seconds apart. That
        # is a proxy for baseline that only holds if the camera moves at one
        # speed. Session 4 sweeps three passes at three speeds in 69 s, so its
        # one-second pairs span a much larger baseline than session 3's, the
        # ceiling collapsed from 572 to 232, and the ratio rose above 1.0 while
        # the absolute matchability fell. The denominator moved with the thing
        # it was measuring.
        #
        # So sample scan pairs across many gaps, measure how far features
        # actually travel between them, and keep the ones that move as far as
        # the demo-to-scan pairs do.
        gaps = sorted({max(1, int(round(g))) for g in np.linspace(1, len(scan_names) // 3, 8)})
        reference_pairs = []
        for gap in gaps:
            reference_pairs += list(zip(scan_names[:-gap], scan_names[gap:], strict=False))
        reference_pairs = list(dict.fromkeys(reference_pairs))
        reference_path = workdir / "reference.h5"
        sfm.match_pairs(reference_pairs, scan_features, reference_path, device)
        reference_stats = _match_stats(
            reference_path, scan_features, scan_features, reference_pairs
        )

        # Every demo frame against every sampled scan frame. No retrieval step,
        # because retrieval is itself unreliable exactly when this check fails.
        demo_pairs = [(d, s) for d in demo_names for s in scan_names]
        demo_path = workdir / "demo_scan.h5"
        sfm.match_pairs(
            demo_pairs, demo_features, demo_path, device, features_ref_path=scan_features
        )

        per_demo_best = []
        demo_shifts = []
        for demo_name in demo_names:
            stats = _match_stats(
                demo_path, demo_features, scan_features,
                [(demo_name, s) for s in scan_names],
            )
            if not stats:
                per_demo_best.append(0)
                continue
            best = max(stats, key=lambda item: item[0])
            per_demo_best.append(int(best[0]))
            if np.isfinite(best[1]):
                demo_shifts.append(best[1])

        # The baseline the demo actually demands of the scan.
        baseline_px = float(np.median(demo_shifts)) if demo_shifts else float("nan")

        usable = [(c, d) for c, d in reference_stats if np.isfinite(d)]
        if usable and np.isfinite(baseline_px):
            # Scan pairs whose features travel about as far as the demo's do.
            band = [c for c, d in usable if 0.75 * baseline_px <= d <= 1.33 * baseline_px]
            if len(band) < 5:
                # Nothing at that baseline: take the closest pairs instead, and
                # the log says so, because it means the scan never presented
                # the viewpoint change the demo needs.
                nearest = sorted(usable, key=lambda item: abs(item[1] - baseline_px))
                band = [c for c, _ in nearest[: max(5, len(usable) // 6)]]
            reference_counts = band
        else:
            reference_counts = [c for c, _ in reference_stats]

        reference = float(np.median(reference_counts)) if reference_counts else 0.0
        demo = float(np.median(per_demo_best)) if per_demo_best else 0.0
        ratio = demo / reference if reference > 0 else 0.0

        # Measure the worst case, not only the middle of the distribution.
        #
        # Count the frames below the warn bar. Then find the longest run of
        # consecutive weak frames. Stage 2 fails on a run, not on an average: a
        # run of weak frames gives no pose for that part of the clip, and the
        # frames that do register there return wrong poses.
        counts = np.asarray(per_demo_best, dtype=float)
        weak = counts < PREFLIGHT_WARN_MATCHES
        weak_fraction = float(weak.mean()) if len(weak) else 0.0
        longest = run = 0
        for flag in weak:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        interval = (
            float(demo_info.duration_s) / max(len(counts), 1) if len(counts) else 0.0
        )

        # Every test must hold. A good ratio against a poor scan is not a good
        # capture. A high count says nothing without the ceiling. A good median
        # says nothing when a quarter of the clip is unusable.
        strong_enough = ratio >= PASS_RATIO and demo >= PREFLIGHT_PASS_MATCHES
        warn_enough = ratio >= WARN_RATIO and demo >= PREFLIGHT_WARN_MATCHES
        worst_case_ok = (
            weak_fraction <= PREFLIGHT_MAX_WEAK_FRACTION
            and longest * interval <= PREFLIGHT_MAX_WEAK_RUN_S
        )
        if strong_enough and worst_case_ok:
            verdict = "PASS"
        elif warn_enough and worst_case_ok:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"

        return PreflightResult(
            scan_frames=len(scan_names),
            demo_frames=len(demo_names),
            reference_matches=reference,
            demo_matches=demo,
            ratio=ratio,
            verdict=verdict,
            per_demo_best=per_demo_best,
            baseline_px=baseline_px,
            reference_pairs_used=len(reference_counts),
            weak_fraction=weak_fraction,
            longest_weak_run_s=longest * interval,
            longest_weak_run_frames=longest,
            sample_interval_s=interval,
        )
    finally:
        if owned and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


def _match_stats(
    match_path: Path, features0: Path, features1: Path, pairs: list[tuple[str, str]]
) -> list[tuple[int, float]]:
    """Match count and median keypoint displacement, in pixels, per pair.

    Displacement stands in for baseline. A pair of views of the same workspace
    separated by a larger baseline moves its features further across the image,
    so matching on displacement compares like with like without needing a
    reconstruction, which preflight runs before there is one.
    """
    import h5py

    out: list[tuple[int, float]] = []
    with h5py.File(str(match_path), "r", libver="latest") as handle, \
            h5py.File(str(features0), "r", libver="latest") as f0, \
            h5py.File(str(features1), "r", libver="latest") as f1:
        for name0, name1 in pairs:
            key = sfm.names_to_pair(name0, name1)
            if key not in handle:
                continue
            matches = handle[key]["matches0"][()]
            valid = matches != -1
            count = int(valid.sum())
            if count < 8 or name0 not in f0 or name1 not in f1:
                out.append((count, float("nan")))
                continue
            kp0 = f0[name0]["keypoints"][()]
            kp1 = f1[name1]["keypoints"][()]
            shift = np.linalg.norm(kp0[valid] - kp1[matches[valid]], axis=1)
            out.append((count, float(np.median(shift))))
    return out


def _match_counts(path: Path, pairs: list[tuple[str, str]]) -> list[int]:
    import h5py

    counts: list[int] = []
    with h5py.File(str(path), "r", libver="latest") as handle:
        for name0, name1 in pairs:
            key = sfm.names_to_pair(name0, name1)
            if key in handle:
                counts.append(int((handle[key]["matches0"][()] != -1).sum()))
    return counts
