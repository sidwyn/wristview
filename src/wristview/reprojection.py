"""Every 3D quantity has to land where it came from, in the frame it came from.

This is the check that would have caught the worst defect in this project on
its first run. The hand's 3D position was collapsing toward the optical axis by
a factor of 25, and nothing noticed for three sessions: the 2D overlays were
drawn from the detector and stayed correct, the reconstruction and the splat
were sound, and grasp detection failing looked like a grasp problem. Projecting
the 3D hand back into its own frame and comparing it with the 2D detection
takes a few lines and would have reported 614 px on day one.

It would also have caught both focal-length bugs, since depth scales with focal
length and a wrong depth shows up as a wrong pixel spread.

**Not every check here is independent evidence, and that difference is
recorded rather than glossed.** An object pose solved by intersecting the ray
through its mask centroid with the desk plane reprojects to that centroid
exactly, by construction, whatever is wrong with the plane. Reporting that zero
as a pass would repeat a mistake already made in this project, where a gate
checked that the object centre sat at half its own height above the desk, a
quantity the solver computes directly, and so passed at zero error on every
frame of a real defect. Each check below is therefore labelled:

  independent      the 3D quantity was derived from something other than the
                   2D observation it is being compared against, so agreement
                   is evidence
  by construction  the 3D quantity was derived from that observation, so
                   agreement is arithmetic and only disagreement is news
"""

from __future__ import annotations

import numpy as np

from .camera import Intrinsics
from .logging_setup import get

log = get(__name__)

# Thresholds, in pixels, on a frame about 1920 across. Written down here rather
# than passed in, so that raising one is a visible edit to this file.
#
# The hand bound is the load-bearing one. A correct lift measures 13 to 16 px
# on session 6, which is the wrist keypoint against the MANO root offset rather
# than error. The defect measured 614. 25 px is about 1.3 per cent of frame
# width: comfortably above the honest residual, far below anything broken.
MAX_HAND_REPROJECTION_PX = 25.0
MAX_OBJECT_REPROJECTION_PX = 40.0
MAX_EFFECTOR_REPROJECTION_PX = 40.0
MAX_WRIST_CAMERA_REPROJECTION_PX = 60.0


def project(points_cam: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """Camera-frame points to pixels. Points behind the camera return NaN."""
    points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    out = np.full((len(points), 2), np.nan)
    ahead = points[:, 2] > 1e-9
    out[ahead, 0] = intrinsics.fx * points[ahead, 0] / points[ahead, 2] + intrinsics.cx
    out[ahead, 1] = intrinsics.fy * points[ahead, 1] / points[ahead, 2] + intrinsics.cy
    return out


def to_camera(points_world: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """World points into the camera frame, given camera-to-world."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(camera_pose)[:3, :3]
    origin = np.asarray(camera_pose)[:3, 3]
    return (points - origin) @ rotation


def errors(
    points_cam: np.ndarray, observed_px: np.ndarray, intrinsics: Intrinsics
) -> np.ndarray:
    """Per-point pixel distance between a projected point and its observation."""
    predicted = project(points_cam, intrinsics)
    observed = np.asarray(observed_px, dtype=np.float64).reshape(-1, 2)
    return np.hypot(predicted[:, 0] - observed[:, 0], predicted[:, 1] - observed[:, 1])


def summarise(values: np.ndarray, label: str, independent: bool, limit: float) -> dict:
    """Median, p90 and worst, plus whether this check is evidence at all."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "check": label,
            "evidence": "independent" if independent else "by construction",
            "frames": 0,
            "median_px": None,
            "p90_px": None,
            "max_px": None,
            "limit_px": limit,
            "passed": None,
            "note": "nothing to compare, so this check says nothing either way",
        }
    median = float(np.median(values))
    return {
        "check": label,
        "evidence": "independent" if independent else "by construction",
        "frames": int(len(values)),
        "median_px": round(median, 2),
        "p90_px": round(float(np.percentile(values, 90)), 2),
        "max_px": round(float(values.max()), 2),
        "limit_px": limit,
        "passed": bool(median <= limit),
        "note": (
            ""
            if independent
            else "derived from the observation it is compared against; only a "
                 "large value is informative, a small one is arithmetic"
        ),
    }


def hand_check(
    landmarks_cam: np.ndarray,
    landmarks_px: np.ndarray,
    valid: np.ndarray,
    intrinsics: Intrinsics,
) -> dict:
    """3D hand joints against the 2D keypoints they were lifted from.

    Independent. The lift takes the detector's translation and its own focal
    length and moves the hand onto our intrinsics; whether the result still
    lands on the detected hand is a real question with a real answer, and for
    three sessions the answer was no.
    """
    per_frame = []
    for index in np.nonzero(valid)[0]:
        values = errors(landmarks_cam[index], landmarks_px[index], intrinsics)
        values = values[np.isfinite(values)]
        if len(values):
            per_frame.append(float(np.median(values)))
    return summarise(
        np.array(per_frame), "hand_joints_vs_keypoints", True, MAX_HAND_REPROJECTION_PX
    )


def object_check(
    object_poses: np.ndarray,
    object_valid: np.ndarray,
    mask_centroids_px: np.ndarray,
    centroid_valid: np.ndarray,
    camera_poses: np.ndarray,
    intrinsics: Intrinsics,
    sources: np.ndarray | None = None,
) -> dict:
    """Object centre against its mask centroid.

    By construction while the object rests: the plane solve places the centre
    on the ray through that very centroid, so it reprojects onto it exactly no
    matter how wrong the plane or the object height is. It becomes independent
    the moment the object is carried, because the carried centre is fixed by
    the hand and only constrained along the ray. So this is split, and the
    carried frames are the ones that carry evidence.
    """
    per_frame = []
    carried = []
    usable = object_valid & centroid_valid
    for index in np.nonzero(usable)[0]:
        cam = to_camera(object_poses[index][:3, 3][None, :], camera_poses[index])
        value = float(errors(cam, mask_centroids_px[index][None, :], intrinsics)[0])
        if not np.isfinite(value):
            continue
        per_frame.append(value)
        if sources is not None and str(sources[index]) == "carried":
            carried.append(value)

    report = summarise(
        np.array(per_frame), "object_centre_vs_mask_centroid", False,
        MAX_OBJECT_REPROJECTION_PX,
    )
    if carried:
        report["carried_frames"] = summarise(
            np.array(carried), "object_centre_vs_mask_centroid_while_carried",
            True, MAX_OBJECT_REPROJECTION_PX,
        )
    return report


def effector_check(
    effector_world: np.ndarray,
    effector_valid: np.ndarray,
    hand_world: np.ndarray,
    hand_valid: np.ndarray,
    expected_pullback_m: float,
    fingertips: tuple[int, ...] = (4, 8),
) -> dict:
    """End effector against the fingertip midpoint, in metres.

    Measured in metres for the same reason as the wrist camera, and the first
    version of this got it wrong in exactly the way this file warns about. The
    effector is *deliberately* not at the fingertip midpoint: Stage 4 pulls it
    back along the approach direction, because a real jaw closes behind the
    fingertips. Comparing pixels and demanding a small number therefore fires
    on the design rather than on any error, and on session 6 it reported 45 to
    95 px on four clips that were doing precisely what they were told.

    What is worth checking is that the offset is the size it was configured to
    be. A pullback of zero means the config never reached the code; a pullback
    of tens of centimetres means a units error. Both are visible here and
    neither is visible in pixels.
    """
    usable = effector_valid & hand_valid
    distances = []
    for index in np.nonzero(usable)[0]:
        target = np.asarray(hand_world[index])[list(fingertips)].mean(axis=0)
        distances.append(float(np.linalg.norm(np.asarray(effector_world[index]) - target)))

    values = np.asarray(distances)
    if not len(values):
        return {
            "check": "effector_pullback_from_fingertips",
            "evidence": "independent",
            "unit": "m",
            "frames": 0,
            "median_m": None,
            "expected_m": expected_pullback_m,
            "passed": None,
            "note": "no frame had both an effector and a hand",
        }

    median = float(np.median(values))
    tolerance = max(0.5 * expected_pullback_m, 0.015)
    return {
        "check": "effector_pullback_from_fingertips",
        "evidence": "independent",
        "unit": "m",
        "frames": int(len(values)),
        "median_m": round(median, 4),
        "p90_m": round(float(np.percentile(values, 90)), 4),
        "expected_m": expected_pullback_m,
        "tolerance_m": round(tolerance, 4),
        "passed": bool(abs(median - expected_pullback_m) <= tolerance),
        "note": (
            "the effector is meant to sit behind the fingertips; this checks "
            "the offset is the configured size, not that it is absent"
        ),
    }


def standoff_check(
    camera_origins_world: np.ndarray,
    grasp_points_world: np.ndarray,
    valid: np.ndarray,
    expected_m: float,
) -> dict:
    """Wrist camera distance from the grasp point, in metres.

    Labelled by construction, and honestly so: the mount offset is applied in
    the gripper frame, so a rigid transform into the world preserves the
    distance exactly and this can only return what the config asked for. It is
    kept because the failure it catches is not a geometry error but a
    plumbing one. real06b rendered its wrist views with
    `render.wrist_camera.standoff_m` unset, the mount fell back to its default,
    and no output said so. A check that can only report the config is still
    worth having when the config silently failing to arrive is the actual bug.

    It is not evidence that the camera is in a sensible place. Nothing here is.
    """
    usable = np.asarray(valid, dtype=bool)
    distances = [
        float(np.linalg.norm(np.asarray(camera_origins_world[i]) - np.asarray(grasp_points_world[i])))
        for i in np.nonzero(usable)[0]
    ]
    values = np.asarray(distances)
    if not len(values):
        return {
            "check": "wrist_camera_standoff",
            "evidence": "by construction",
            "unit": "m",
            "frames": 0,
            "median_m": None,
            "expected_m": expected_m,
            "passed": None,
            "note": "no usable frame",
        }
    median = float(np.median(values))
    tolerance = max(0.05 * expected_m, 0.005)
    return {
        "check": "wrist_camera_standoff",
        "evidence": "by construction",
        "unit": "m",
        "frames": int(len(values)),
        "median_m": round(median, 4),
        "expected_m": expected_m,
        "tolerance_m": round(tolerance, 4),
        "passed": bool(abs(median - expected_m) <= tolerance),
        "note": (
            "catches a standoff that never reached the renderer, not a camera "
            "in the wrong place; the mount is applied in the gripper frame so "
            "this distance is preserved by construction"
        ),
    }


def gate(reports: list[dict]) -> list[str]:
    """Failures worth stopping for. Only independent checks can fail a stage.

    A check that is true by construction cannot be evidence of correctness, so
    it is never allowed to grant a pass, but a large value still means
    something is badly wrong and is reported as a failure.
    """
    failures = []
    for report in reports:
        if report.get("passed") is False:
            if report.get("unit") == "m":
                failures.append(
                    f"{report['check']}: median {report['median_m']} m against an "
                    f"expected {report['expected_m']} m, tolerance "
                    f"{report['tolerance_m']} m, over {report['frames']} frames"
                )
            else:
                failures.append(
                    f"{report['check']}: median {report['median_px']} px exceeds "
                    f"{report['limit_px']} px over {report['frames']} frames "
                    f"({report['evidence']})"
                )
        nested = report.get("carried_frames")
        if nested and nested.get("passed") is False:
            failures.append(
                f"{nested['check']}: median {nested['median_px']} px exceeds "
                f"{nested['limit_px']} px over {nested['frames']} frames"
            )
    return failures
