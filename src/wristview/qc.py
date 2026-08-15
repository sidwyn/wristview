"""Quality gates that run on every capture.

Two checks live here, both learned the expensive way from real captures.

**Pre-flight ratio.** Whether a demo clip shares enough viewpoint with the scan
to localize at all. Three sessions measured 126, 212 and 436 raw matches, and
the pipeline reported 100 percent of frames registered on the one that could
not possibly have worked.

**Marker agreement.** Whether Stage 2's camera pose agrees with the pose implied
by an ArUco marker of known size. This is a second opinion from different
mathematics: Stage 2 solves PnP against hundreds of triangulated scene points,
the marker solves PnP against four coplanar corners whose spacing is a physical
measurement. They share the camera intrinsics and nothing else. When both agree
the pose is right for two independent reasons; when they disagree, at least one
is wrong and neither should be trusted.

Thresholds carry the measurement that justifies them. A threshold with no
recorded evidence is a guess, and guesses get tuned until things pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import Intrinsics
from .geometry import invert_pose, make_pose, orthonormalize
from .logging_setup import get

log = get(__name__)


# --------------------------------------------------------------------------
# Thresholds, with the evidence behind each one
# --------------------------------------------------------------------------

PREFLIGHT_PASS_RATIO = 0.40
PREFLIGHT_WARN_RATIO = 0.25
PREFLIGHT_EVIDENCE = (
    "session 1 wide-orbit-only scan: 126 raw matches, Stage 2 recovered an "
    "85 m trajectory across a 0.69 m desk while reporting 100 percent "
    "registered. session 2 with a 15 s close pass in 40 s: 212 raw, ratio "
    "0.39. session 3 with texture added and an 18 s close pass in 30 s: 436 "
    "raw, ratio 0.76, and five clips localized."
)

MARKER_POSITION_MAX_CM = 3.0
MARKER_ROTATION_MAX_DEG = 5.0
MARKER_MIN_COVERAGE = 0.50
MARKER_EVIDENCE = "first measured on session 3; see BUILD-PLAN.md for the values."


@dataclass
class GateResult:
    name: str
    passed: bool
    value: float | None
    threshold: float
    unit: str
    detail: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "unit": self.unit,
            "detail": self.detail,
            **({"extra": self.extra} if self.extra else {}),
        }


# --------------------------------------------------------------------------
# Marker agreement
# --------------------------------------------------------------------------

def marker_corners_local(side_m: float) -> np.ndarray:
    """The four marker corners in the marker's own frame, OpenCV order.

    Top-left, top-right, bottom-right, bottom-left, seen from the front, lying
    on z = 0.
    """
    half = side_m / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def detect_marker(image, dictionary_name: str, marker_id: int) -> np.ndarray | None:
    """Return the marker's four image corners, or None."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return None
    ids = [int(v) for v in ids.ravel()]
    if marker_id not in ids:
        return None
    return corners[ids.index(marker_id)].reshape(4, 2).astype(np.float64)


def marker_pose_in_camera(
    corners_px: np.ndarray, side_m: float, intrinsics: Intrinsics
) -> np.ndarray | None:
    """Solve the marker's pose in the camera frame from its four corners.

    IPPE_SQUARE is the planar-specific solver. A general PnP on four coplanar
    points is ill-conditioned and flips between two solutions.
    """
    try:
        ok, rvec, tvec = cv2.solvePnP(
            marker_corners_local(side_m), corners_px, intrinsics.matrix,
            np.zeros((4, 1)), flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return None
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return make_pose(rotation, tvec.reshape(3))


def estimate_marker_world_pose(
    frames_dir: Path,
    frame_names: list[str],
    camera_poses: np.ndarray,
    valid: np.ndarray,
    intrinsics: Intrinsics,
    side_m: float,
    dictionary_name: str,
    marker_id: int,
    max_frames: int = 40,
) -> np.ndarray | None:
    """Where the marker sits in world coordinates.

    The marker does not move, so every frame that sees it gives an independent
    estimate of one fixed pose. Their median is the reference the per-frame
    check is measured against.
    """
    translations, rotations = [], []
    indices = [i for i in range(len(frame_names)) if i < len(valid) and valid[i]]
    if len(indices) > max_frames:
        indices = list(np.linspace(0, len(indices) - 1, max_frames).astype(int).tolist())
        indices = [i for i in indices]

    for index in indices:
        image = cv2.imread(str(frames_dir / frame_names[index]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        corners = detect_marker(image, dictionary_name, marker_id)
        if corners is None:
            continue
        marker_cam = marker_pose_in_camera(corners, side_m, intrinsics)
        if marker_cam is None:
            continue
        marker_world = camera_poses[index] @ marker_cam
        translations.append(marker_world[:3, 3])
        rotations.append(marker_world[:3, :3])

    if len(translations) < 3:
        return None
    return make_pose(orthonormalize(np.mean(rotations, axis=0)), np.median(translations, axis=0))


def marker_agreement(
    frames_dir: Path,
    frame_names: list[str],
    camera_poses: np.ndarray,
    valid: np.ndarray,
    intrinsics: Intrinsics,
    side_m: float,
    dictionary_name: str = "DICT_4X4_50",
    marker_id: int = 0,
    marker_world: np.ndarray | None = None,
) -> dict:
    """Compare Stage 2's camera pose against the marker-implied camera pose.

    For each frame that sees the marker, the marker's fixed world pose plus its
    measured pose in the camera gives a camera pose that owes nothing to the
    scene reconstruction. The disagreement is the number reported.

    **`marker_world` must come from somewhere other than `camera_poses`.**
    Deriving it from the poses under test makes the check self-referential and
    blind to exactly the failure it exists to catch: displacing every camera by
    25 cm moved the inferred marker by the same 25 cm and the reported error
    stayed at 1.5 cm. Stage 2 passes the pose estimated from the scan, whose
    own registration and reprojection error are independently checked.

    When no reference is supplied the marker pose is estimated from these
    poses, and only `position_spread_cm` is meaningful: a rigid error cancels,
    but a wandering one still shows.
    """
    self_referential = marker_world is None
    if marker_world is None:
        marker_world = estimate_marker_world_pose(
            frames_dir, frame_names, camera_poses, valid, intrinsics,
            side_m, dictionary_name, marker_id,
        )
    if marker_world is None:
        return {"frames": 0, "coverage": 0.0, "reason": "marker never solved in this clip"}

    position_errors, rotation_errors, implied_marker = [], [], []
    for index, name in enumerate(frame_names):
        if index >= len(valid) or not valid[index]:
            continue
        image = cv2.imread(str(frames_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        corners = detect_marker(image, dictionary_name, marker_id)
        if corners is None:
            continue
        marker_cam = marker_pose_in_camera(corners, side_m, intrinsics)
        if marker_cam is None:
            continue

        # Camera pose implied by the marker alone.
        implied = marker_world @ invert_pose(marker_cam)
        actual = camera_poses[index]

        position_errors.append(float(np.linalg.norm(implied[:3, 3] - actual[:3, 3])) * 100)
        # Where this frame thinks the marker is. It does not move, so the
        # spread across frames is an error signal that survives even a
        # self-referential reference.
        implied_marker.append((actual @ marker_cam)[:3, 3])
        relative = implied[:3, :3] @ actual[:3, :3].T
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(float(np.degrees(np.arccos(cosine))))

    if not position_errors:
        return {"frames": 0, "coverage": 0.0, "reason": "marker never solved on a posed frame"}

    position = np.array(position_errors)
    rotation = np.array(rotation_errors)
    spread = float(np.linalg.norm(np.std(np.array(implied_marker), axis=0))) * 100
    return {
        "frames": len(position),
        "coverage": round(len(position) / max(int(np.sum(valid)), 1), 4),
        "reference": "self" if self_referential else "scan",
        "position_spread_cm": round(spread, 3),
        "position_median_cm": round(float(np.median(position)), 3),
        "position_p90_cm": round(float(np.percentile(position, 90)), 3),
        "position_max_cm": round(float(position.max()), 3),
        "rotation_median_deg": round(float(np.median(rotation)), 3),
        "rotation_p90_deg": round(float(np.percentile(rotation, 90)), 3),
        "rotation_max_deg": round(float(rotation.max()), 3),
        "marker_world_position": [round(float(v), 4) for v in marker_world[:3, 3]],
    }


def marker_gates(agreement: dict) -> list[GateResult]:
    """Turn a marker agreement report into pass or fail gates."""
    if not agreement or agreement.get("frames", 0) == 0:
        return [
            GateResult(
                "marker_agreement", False, None, MARKER_POSITION_MAX_CM, "cm",
                agreement.get("reason", "no marker observations"),
            )
        ]
    return [
        GateResult(
            "marker_agreement_position",
            agreement["position_median_cm"] <= MARKER_POSITION_MAX_CM,
            agreement["position_median_cm"], MARKER_POSITION_MAX_CM, "cm",
            f"median over {agreement['frames']} frames. {MARKER_EVIDENCE}",
        ),
        GateResult(
            "marker_agreement_rotation",
            agreement["rotation_median_deg"] <= MARKER_ROTATION_MAX_DEG,
            agreement["rotation_median_deg"], MARKER_ROTATION_MAX_DEG, "deg",
            f"median over {agreement['frames']} frames.",
        ),
        GateResult(
            "marker_coverage",
            agreement["coverage"] >= MARKER_MIN_COVERAGE,
            agreement["coverage"], MARKER_MIN_COVERAGE, "fraction",
            "share of posed frames where the marker gave an independent check.",
        ),
    ]


def preflight_gate(demo_matches: float, scan_matches: float) -> GateResult:
    """The viewpoint-overlap gate, as a ratio against the scan's own ceiling."""
    ratio = demo_matches / scan_matches if scan_matches > 0 else 0.0
    return GateResult(
        "preflight_ratio", ratio >= PREFLIGHT_PASS_RATIO, round(ratio, 4),
        PREFLIGHT_PASS_RATIO, "ratio",
        f"demo-to-scan {demo_matches:.0f} against scan self-match "
        f"{scan_matches:.0f}. {PREFLIGHT_EVIDENCE}",
    )
