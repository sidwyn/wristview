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
from .qc import PREFLIGHT_PASS_RATIO as PASS_RATIO
from .qc import PREFLIGHT_WARN_RATIO as WARN_RATIO
from .qc import SELF_MATCH_BASELINE_S
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
    demo_frames: int = 6,
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

        # Reference ceiling across a real viewpoint change, not between
        # adjacent frames. See SELF_MATCH_BASELINE_S in qc.py: adjacent frames
        # match almost perfectly and inflate the ceiling.
        sampled_fps = len(scan_names) / max(scan_info.duration_s, 1e-3)
        gap = max(1, int(round(SELF_MATCH_BASELINE_S * sampled_fps)))
        reference_pairs = list(zip(scan_names[:-gap], scan_names[gap:], strict=False))
        reference_path = workdir / "reference.h5"
        sfm.match_pairs(reference_pairs, scan_features, reference_path, device)
        reference_counts = _match_counts(reference_path, reference_pairs)

        # Every demo frame against every sampled scan frame. No retrieval step,
        # because retrieval is itself unreliable exactly when this check fails.
        demo_pairs = [(d, s) for d in demo_names for s in scan_names]
        demo_path = workdir / "demo_scan.h5"
        sfm.match_pairs(
            demo_pairs, demo_features, demo_path, device, features_ref_path=scan_features
        )

        per_demo_best = []
        for demo_name in demo_names:
            counts = _match_counts(
                demo_path, [(demo_name, s) for s in scan_names]
            )
            per_demo_best.append(int(max(counts)) if counts else 0)

        reference = float(np.median(reference_counts)) if reference_counts else 0.0
        demo = float(np.median(per_demo_best)) if per_demo_best else 0.0
        ratio = demo / reference if reference > 0 else 0.0

        if ratio >= PASS_RATIO:
            verdict = "PASS"
        elif ratio >= WARN_RATIO:
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
        )
    finally:
        if owned and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


def _match_counts(path: Path, pairs: list[tuple[str, str]]) -> list[int]:
    import h5py

    counts: list[int] = []
    with h5py.File(str(path), "r", libver="latest") as handle:
        for name0, name1 in pairs:
            key = sfm.names_to_pair(name0, name1)
            if key in handle:
                counts.append(int((handle[key]["matches0"][()] != -1).sum()))
    return counts
