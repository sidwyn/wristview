"""End-to-end accuracy against synthetic ground truth.

The fixture knows the true camera trajectory, hand landmarks and object pose,
so every stage can be scored rather than merely inspected.

**Read the `source` column before the numbers.** A stage scored against an
input that was itself ground truth measures its own arithmetic, not the
perception feeding it. Both are worth knowing and they are not the same
claim, so each metric records which it is:

  estimated   the stage's own output, from images. A real accuracy number.
  derived     computed from a ground-truth input. Measures this stage's
              maths in isolation, with perfect input.

The fixture's hand is a smooth capsule render, out of distribution for
detectors trained on photographs, so hand estimation falls back to ground
truth and everything downstream of it is `derived`. Hand accuracy on real
imagery is measured separately, on the A3 stills.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .backends import hands as hand_backend
from .logging_setup import get
from .runctx import read_json

log = get(__name__)


@dataclass
class Metric:
    name: str
    unit: str
    source: str
    median: float
    p90: float
    worst: float
    count: int
    note: str = ""

    def row(self) -> str:
        return (
            f"  {self.name:<34} {self.median:8.2f} {self.p90:8.2f} {self.worst:8.2f}"
            f"  {self.unit:<6} {self.source:<9} n={self.count}"
        )


@dataclass
class EpisodeReport:
    clip_id: str
    metrics: list[Metric] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _summarize(name: str, errors: np.ndarray, unit: str, source: str, note: str = "") -> Metric:
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return Metric(name, unit, source, float("nan"), float("nan"), float("nan"), 0, note)
    return Metric(
        name=name,
        unit=unit,
        source=source,
        median=float(np.median(errors)),
        p90=float(np.percentile(errors, 90)),
        worst=float(errors.max()),
        count=int(len(errors)),
        note=note,
    )


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle between two stacks of rotation matrices, in degrees.

    The clamp before arccos is load-bearing twice over: without it a trace
    marginally above 1 from floating point gives nan, and one marginally
    below -1 does the same at 180 degrees. Note that arccos is still
    ill-conditioned near identity, so expect noise of order 1e-6 degrees on
    a perfect match. That is far below anything reported here.
    """
    relative = np.einsum("nij,nkj->nik", a, b)
    trace = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def _gt_index(run_index: np.ndarray, run_fps: float, gt_fps: float, gt_count: int) -> np.ndarray:
    """Map run frame indices onto ground-truth frame indices by timestamp.

    Stage 0 resamples demos, so run frame i is at time i / run_fps and the
    matching ground-truth frame is the nearest one at that time.
    """
    # Round half up rather than np.round, which rounds half to even. At a
    # 20 fps run against 30 fps truth every other frame lands on an exact
    # half, so banker's rounding alternates the direction and the mapping
    # stops being predictable.
    times = run_index / max(run_fps, 1e-9)
    return np.clip(np.floor(times * gt_fps + 0.5).astype(int), 0, gt_count - 1)


def evaluate_episode(
    run_root: Path, clip_id: str, groundtruth: dict, manifest: dict
) -> EpisodeReport:
    """Score one episode against its ground truth."""
    report = EpisodeReport(clip_id=clip_id)
    truth = groundtruth["demos"][clip_id]
    gt_camera = np.asarray(truth["camera_poses"])
    gt_hand = np.asarray(truth["hand_landmarks"])
    gt_object = np.asarray(truth["object_poses"])
    gt_held = np.asarray(truth["held"], dtype=bool)
    gt_width = np.asarray(truth["grasp_width_m"])
    gt_fps = float(truth["fps"])

    clip = manifest["clips"][clip_id]
    run_fps = float(clip.get("effective_fps") or clip["video_info"]["fps"])
    frame_count = clip["frame_count"]
    index = np.arange(frame_count)
    gt_idx = _gt_index(index, run_fps, gt_fps, len(gt_camera))

    # ---- Stage 2, camera pose ------------------------------------------
    pose_path = run_root / "02_localize" / clip_id / "camera_poses.npy"
    status_path = run_root / "02_localize" / clip_id / "status.json"
    if pose_path.exists():
        estimated = np.load(pose_path)
        valid = np.load(run_root / "02_localize" / clip_id / "pose_valid.npy")
        status = read_json(status_path) if status_path.exists() else {}
        if status.get("rejected"):
            report.notes.append(
                f"Stage 2 rejected this episode: {status.get('reject_reason', 'unknown')}"
            )
        use = valid & (np.arange(len(estimated)) < frame_count)
        if use.any():
            truth_poses = gt_camera[gt_idx[use[:frame_count]]]
            got = estimated[: frame_count][use[:frame_count]]
            position = np.linalg.norm(got[:, :3, 3] - truth_poses[:, :3, 3], axis=1) * 100
            rotation = _rotation_error_deg(got[:, :3, :3], truth_poses[:, :3, :3])
            report.metrics.append(
                _summarize("camera position", position, "cm", "estimated")
            )
            report.metrics.append(
                _summarize("camera rotation", rotation, "deg", "estimated")
            )

    # ---- Stage 3, hand and object ---------------------------------------
    estimate_dir = run_root / "03_estimate" / clip_id
    hand_source = "estimated"
    summary_path = run_root / "03_estimate" / "summary.json"
    if summary_path.exists():
        entry = read_json(summary_path).get(clip_id, {})
        if entry.get("hand_backend") == "synthetic_groundtruth":
            hand_source = "derived"
            report.notes.append(
                "hand landmarks came from fixture ground truth, so hand and "
                "everything downstream measure arithmetic, not perception"
            )

    hand_npz = estimate_dir / "hand.npz"
    if hand_npz.exists():
        payload = np.load(hand_npz)
        got_hand = payload["landmarks_world"]
        hand_valid = payload["valid"]
        if hand_valid.any():
            sel = hand_valid[:frame_count]
            errors = (
                np.linalg.norm(
                    got_hand[:frame_count][sel] - gt_hand[gt_idx[: frame_count][sel]], axis=2
                ).mean(axis=1)
                * 100
            )
            report.metrics.append(
                _summarize("hand landmarks, mean", errors, "cm", hand_source)
            )

    object_pose_path = estimate_dir / "object_pose.npy"
    object_valid_path = estimate_dir / "object_valid.npy"
    if object_pose_path.exists() and object_valid_path.exists():
        got_object = np.load(object_pose_path)
        obj_valid = np.load(object_valid_path)
        if obj_valid.any():
            sel = obj_valid[:frame_count]
            errors = (
                np.linalg.norm(
                    got_object[:frame_count][sel, :3, 3]
                    - gt_object[gt_idx[:frame_count][sel], :3, 3],
                    axis=1,
                )
                * 100
            )
            report.metrics.append(
                _summarize("object position", errors, "cm", "estimated")
            )

    # ---- Stage 4, end effector -------------------------------------------
    traj_path = run_root / "04_retarget" / clip_id / "ee_trajectory.npz"
    if traj_path.exists():
        traj = np.load(traj_path)
        ee = traj["poses_video_rate"]
        width = traj["width_video_rate"]
        closed = traj["closed_video_rate"]
        n = min(len(ee), frame_count)
        gt_slice = gt_idx[:n]

        # The end effector is defined at the thumb-index midpoint pulled back
        # along the approach direction, so score it against the same point
        # built from the true hand.
        gt_ee = np.stack(
            [hand_backend.grasp_center(gt_hand[i]) for i in gt_slice]
        )
        gt_approach = np.stack(
            [hand_backend.approach_direction(gt_hand[i]) for i in gt_slice]
        )
        offset = 0.02
        gt_origin = gt_ee - gt_approach * offset
        position = np.linalg.norm(ee[:n, :3, 3] - gt_origin, axis=1) * 100
        report.metrics.append(
            _summarize("end effector position", position, "cm", hand_source)
        )

        width_err = np.abs(width[:n] - gt_width[gt_slice]) * 100
        report.metrics.append(
            _summarize("gripper width", width_err, "cm", hand_source)
        )

        # Grasp timing is a genuinely independent check: the detector has to
        # decide open or closed from geometry and motion, and the fixture
        # knows when the object was actually held.
        agreement = (closed[:n] == gt_held[gt_slice]).mean() * 100
        report.metrics.append(
            Metric("grasp state agreement", "%", "estimated", agreement, agreement,
                   agreement, n, "fraction of frames where open/closed matches truth")
        )

    return report


def evaluate_run(run_root: Path, groundtruth_path: Path) -> list[EpisodeReport]:
    groundtruth = json.loads(Path(groundtruth_path).read_text())
    manifest = read_json(run_root / "00_ingest" / "manifest.json")
    reports = []
    for clip_id, clip in manifest["clips"].items():
        if clip["kind"] != "demo" or clip_id not in groundtruth.get("demos", {}):
            continue
        reports.append(evaluate_episode(run_root, clip_id, groundtruth, manifest))
    return reports


def format_reports(reports: list[EpisodeReport]) -> str:
    lines = [
        "",
        f"  {'metric':<34} {'median':>8} {'p90':>8} {'worst':>8}  {'unit':<6} {'source':<9}",
        f"  {'-' * 34} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 6} {'-' * 9}",
    ]
    for report in reports:
        lines.append(f"\n{report.clip_id}")
        for metric in report.metrics:
            lines.append(metric.row())
        for note in report.notes:
            lines.append(f"    note: {note}")

    # Pooled across episodes, per metric name.
    pooled: dict[tuple[str, str, str], list[float]] = {}
    for report in reports:
        for metric in report.metrics:
            if np.isfinite(metric.median):
                pooled.setdefault((metric.name, metric.unit, metric.source), []).append(
                    metric.median
                )
    if pooled:
        lines.append("\nall episodes, median of per-episode medians")
        for (name, unit, source), values in pooled.items():
            lines.append(
                f"  {name:<34} {np.median(values):8.2f} {'':>8} {'':>8}  {unit:<6} {source:<9}"
            )
    return "\n".join(lines)
