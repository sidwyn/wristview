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

# Absolute demo-to-scan match count, judged alongside the ratio. Either alone
# misleads. The ratio is blind to a scan that is uniformly poor, because a weak
# demo against a weak ceiling still scores well. The absolute count is blind to
# texture, resolution and keypoint budget. Measured on four sessions at 1024
# keypoints: 126 failed, 212 was marginal and its demos never localized, 436
# carried five clips end to end, 214 is the current session.
PREFLIGHT_PASS_MATCHES = 400.0
PREFLIGHT_WARN_MATCHES = 250.0

# Superseded. The scan self-match ceiling used to be measured between frames
# this far apart in time. Kept only so the old numbers can be reproduced.
# in time. The number matters: adjacent frames at 6 fps are 0.17 s apart and
# match almost perfectly, which inflates the ceiling and makes every ratio
# look worse. Measured on the same scan, adjacent pairs gave 770 matches and
# 1 second pairs gave 572, which moved one clip from pass to fail purely by
# choice of reference. A ceiling has to represent a real viewpoint change.
SELF_MATCH_BASELINE_S = 1.0

MARKER_POSITION_MAX_CM = 3.0
MARKER_ROTATION_MAX_DEG = 5.0
MARKER_MIN_COVERAGE = 0.50
MARKER_EVIDENCE = "first measured on session 3; see BUILD-PLAN.md for the values."

# A parallel jaw is symmetric about its approach axis, so two orientations
# describe the same grasp. Taking the roll from the hand picks between them by
# the thumb-index axis, which crosses over mid-episode and turns the wrist
# camera upside down. A flip inside an episode is a defect, so the gate is
# zero, not a tolerance.
ROLL_FLIPS_MAX = 0
# How far the camera up axis may sit from world up, in the median over a clip.
# Not zero: the hand tilts, and it should. Before the fix the five demo clips
# measured 73, 161, 161, 153 and 114 degrees, four of them past 90, meaning
# mostly upside down. After, 16 to 35 degrees.
ROLL_UP_ANGLE_MAX_DEG = 60.0
ROLL_EVIDENCE = (
    "session 3, five demo clips. before: 3 to 9 flips per clip and 53 to 135 "
    "frames inverted. after: 0 to 1 flips, 2 to 34 frames inverted."
)


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
        # Thin the *valid* frames, evenly. An earlier version built the
        # linspace over positions and then used those numbers as frame
        # indices, which silently discarded the mask and always read the first
        # `len(indices)` frames of the clip instead of the frames that actually
        # saw the marker. On session 4 that meant the reference came entirely
        # from the wide orbit, and no resampling of the mask changed it.
        picks = np.linspace(0, len(indices) - 1, max_frames).astype(int)
        indices = [indices[i] for i in picks]

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


# A human wrist and forearm rotate at up to roughly 700 to 900 deg/s in fast
# manipulation. The limit here is taken from that range, not fitted to the
# footage, so it stays a physical claim rather than a description of this
# capture. Anything past it is a tracking failure, not motion.
HAND_ANGULAR_RATE_MAX_DEG_S = 900.0
# An isolated bad frame can be filled from its neighbours. A run cannot: two
# consecutive bad frames span 100 ms at 20 fps, and real motion happens inside
# that, so filling it would invent a trajectory rather than recover one.
HAND_OUTLIER_MAX_RUN = 1
# Above this share of the episode, more than one control sample in twenty is
# filled rather than measured, and calling the trajectory a measurement stops
# being honest. Reject instead of repairing.
HAND_OUTLIER_MAX_FRACTION = 0.05
HAND_VELOCITY_EVIDENCE = (
    "measured on the palm frame, not the gripper frame. The gripper roll comes "
    "from the thumb-index axis and so carries finger articulation, which is "
    "faster and noisier than the wrist and does not share its limit. On session "
    "3 the same threshold flagged 5 to 18 steps per clip on the gripper frame "
    "against 2 to 4 on the palm."
)


def palm_rotations(landmarks: np.ndarray) -> np.ndarray:
    """Rotation of the palm for each frame, from the rigid part of the hand.

    Built from the wrist and the index and middle knuckles. These move as one
    piece, unlike the fingertips, so this tracks the wrist rather than the
    grasp.
    """
    landmarks = np.asarray(landmarks, dtype=np.float64)

    def unit(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    forward = unit(landmarks[:, 9] - landmarks[:, 0])
    across = unit(np.cross(landmarks[:, 5] - landmarks[:, 0], forward))
    across = unit(across - np.einsum("ni,ni->n", across, forward)[:, None] * forward)
    return np.stack([across, np.cross(forward, across), forward], axis=-1)


def hand_velocity_outliers(
    landmarks: np.ndarray, valid: np.ndarray, fps: float
) -> dict:
    """Flag frames whose wrist rotation is faster than a human can move.

    Returns a report including `outlier`, a mask over all frames. A step above
    the limit condemns the later of the two frames, because that is the one
    that arrived in the wrong place.
    """
    from scipy.spatial.transform import Rotation

    valid = np.asarray(valid, dtype=bool)
    outlier = np.zeros(len(valid), dtype=bool)
    indices = np.nonzero(valid)[0]
    if len(indices) < 3:
        return {
            "frames_checked": int(len(indices)),
            "frames_flagged": 0,
            "flagged_fraction": 0.0,
            "longest_run": 0,
            "max_rate_deg_s": None,
            "outlier": outlier,
            "flagged_frames": [],
        }

    rotations = palm_rotations(landmarks[indices])
    relative = Rotation.from_matrix(
        np.einsum("nij,njk->nik", rotations[:-1].transpose(0, 2, 1), rotations[1:])
    )
    seconds = np.diff(indices) / max(fps, 1e-6)
    rate = np.degrees(relative.magnitude()) / seconds
    # A step over the limit condemns the later of its two frames. One
    # displaced frame, though, produces two such steps, out and back, and
    # naively blaming the later frame of each marks two frames instead of one.
    # That turns the most repairable case, an isolated spike, into a run of two
    # and rejects the episode. So when a pair of steps goes out and comes back,
    # only the frame between them is at fault.
    over = rate > HAND_ANGULAR_RATE_MAX_DEG_S
    net = np.degrees(
        Rotation.from_matrix(
            np.einsum("nij,njk->nik", rotations[:-2].transpose(0, 2, 1), rotations[2:])
        ).magnitude()
    ) / np.maximum(seconds[:-1] + seconds[1:], 1e-9)

    step = 0
    while step < len(over):
        if not over[step]:
            step += 1
            continue
        spike = (
            step + 1 < len(over)
            and over[step + 1]
            and net[step] <= HAND_ANGULAR_RATE_MAX_DEG_S
        )
        outlier[indices[step + 1]] = True
        # On a spike the frame after the excursion is where the hand really
        # was, so it is not condemned and its step is already accounted for.
        step += 2 if spike else 1

    flagged_count = int(outlier[indices].sum())
    longest = current = 0
    for flagged in outlier[indices]:
        current = current + 1 if flagged else 0
        longest = max(longest, current)

    return {
        "frames_checked": int(len(indices)),
        "frames_flagged": flagged_count,
        "flagged_fraction": round(float(flagged_count / len(indices)), 4),
        "longest_run": int(longest),
        "max_rate_deg_s": round(float(rate.max()), 1),
        "rates_flagged_deg_s": [round(float(v), 1) for v in rate[over]],
        "outlier": outlier,
        "flagged_frames": np.nonzero(outlier)[0].tolist(),
    }


def hand_velocity_gates(report: dict) -> list[GateResult]:
    """Decide whether an episode can be repaired or has to be rejected."""
    return [
        GateResult(
            "hand_velocity_run",
            report["longest_run"] <= HAND_OUTLIER_MAX_RUN,
            report["longest_run"], HAND_OUTLIER_MAX_RUN, "frames",
            "consecutive flagged frames cannot be filled from neighbours. "
            + HAND_VELOCITY_EVIDENCE,
        ),
        GateResult(
            "hand_velocity_fraction",
            report["flagged_fraction"] <= HAND_OUTLIER_MAX_FRACTION,
            report["flagged_fraction"], HAND_OUTLIER_MAX_FRACTION, "fraction",
            f"share of tracked frames above {HAND_ANGULAR_RATE_MAX_DEG_S:.0f} deg/s.",
        ),
    ]


# Mean reprojection error over the sparse reconstruction. This threshold sat
# in the config from the start and was never read by any code, so it protected
# nothing. Session 3 measured 0.909 px and session 4 measured 1.517 px, and
# only the second would ever have been caught. A gate nobody enforces is worse
# than no gate, because the number in the config reads like a guarantee.
MAX_REPROJ_ERROR_PX = 1.5
REPROJ_EVIDENCE = (
    "Gate 0 and session 3 both measured 0.909 px on footage that went on to "
    "localize five clips. Session 4 measured 1.517 px."
)


def reconstruction_gates(metrics: dict, min_registration_rate: float = 0.8,
                         max_reproj_error_px: float = MAX_REPROJ_ERROR_PX
                         ) -> list[GateResult]:
    """Gates on the sparse reconstruction itself."""
    return [
        GateResult(
            "registration_rate",
            metrics.get("registration_rate", 0.0) >= min_registration_rate,
            metrics.get("registration_rate"), min_registration_rate, "fraction",
            "share of scan frames the reconstruction placed. Registration rate "
            "is not a correctness signal on its own: session 1 reported 100 "
            "percent while recovering an 85 m trajectory across a 0.69 m desk.",
        ),
        GateResult(
            "mean_reprojection_error",
            metrics.get("mean_reprojection_error_px", 1e9) <= max_reproj_error_px,
            metrics.get("mean_reprojection_error_px"), max_reproj_error_px, "px",
            REPROJ_EVIDENCE,
        ),
    ]


def roll_gates(roll: dict | None) -> list[GateResult]:
    """Turn a Stage 4 roll report into pass or fail gates."""
    if not roll:
        return [
            GateResult(
                "roll_flips", False, None, ROLL_FLIPS_MAX, "flips",
                "no roll report, so world up was missing and roll was never resolved.",
            )
        ]
    return [
        GateResult(
            "roll_flips",
            roll["flips_after"] <= ROLL_FLIPS_MAX,
            roll["flips_after"], ROLL_FLIPS_MAX, "flips",
            f"was {roll['flips_before']} before canonicalisation. {ROLL_EVIDENCE}",
        ),
        GateResult(
            "roll_up_angle",
            roll["median_up_angle_deg"] <= ROLL_UP_ANGLE_MAX_DEG,
            roll["median_up_angle_deg"], ROLL_UP_ANGLE_MAX_DEG, "deg",
            "median angle between the wrist camera up axis and world up. "
            "Past 90 the camera is upside down.",
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
