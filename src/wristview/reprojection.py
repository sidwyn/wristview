"""Project every 3D quantity back into the frame that produced it.

This check would have found the worst defect in this project on its first run.
The hand's 3D position collapsed toward the optical axis by a factor of 25. No
stage noticed for three sessions. The 2D overlays came from the detector and
stayed correct. The reconstruction and the splat were sound. Grasp detection
failed, and that looked like a grasp problem.

Project the 3D hand back into its own frame. Compare it with the 2D detection.
That takes a few lines. It reports 614 px on day one.

The same check finds a wrong focal length. Depth scales with focal length. A
wrong depth gives a wrong pixel spread.

**Not every check here is evidence. This module records the difference.**

An object pose solved from the ray through its mask centroid reprojects onto
that centroid exactly. This holds whatever is wrong with the plane. A zero
result there is arithmetic, not proof.

This project already made that mistake once. A gate checked that the object
centre sat at half its own height above the desk. The solver computes that
quantity directly. The gate therefore passed at zero error on every frame of a
real defect.

Each check below carries a label:

  independent      The 3D quantity came from a source other than the 2D
                   observation. Agreement is then evidence.
  by construction  The 3D quantity came from that observation. Agreement is
                   then arithmetic. Only a large value carries information.
"""

from __future__ import annotations

import numpy as np

from .camera import Intrinsics
from .logging_setup import get

log = get(__name__)

# These thresholds are in pixels. They apply to a frame about 1920 px across.
# Keep them in this file. A change to a threshold is then a visible edit.
#
# The hand bound matters most. A correct lift measures 13 to 16 px on session 6.
# That residual is the wrist keypoint against the MANO root offset. It is not an
# error. The defect measured 614 px. The limit of 25 px is about 1.3 per cent of
# the frame width. It sits above the true residual and far below a fault.
MAX_HAND_REPROJECTION_PX = 25.0
MAX_OBJECT_REPROJECTION_PX = 40.0
MAX_EFFECTOR_REPROJECTION_PX = 40.0
MAX_WRIST_CAMERA_REPROJECTION_PX = 60.0


def project(points_cam: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """Project camera-frame points to pixels. Return NaN for points behind the camera."""
    points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    out = np.full((len(points), 2), np.nan)
    ahead = points[:, 2] > 1e-9
    out[ahead, 0] = intrinsics.fx * points[ahead, 0] / points[ahead, 2] + intrinsics.cx
    out[ahead, 1] = intrinsics.fy * points[ahead, 1] / points[ahead, 2] + intrinsics.cy
    return out


def to_camera(points_world: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """Transform world points into the camera frame. Supply a camera-to-world pose."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(camera_pose)[:3, :3]
    origin = np.asarray(camera_pose)[:3, 3]
    return (points - origin) @ rotation


def errors(
    points_cam: np.ndarray, observed_px: np.ndarray, intrinsics: Intrinsics
) -> np.ndarray:
    """Return the pixel distance from each projected point to its observation."""
    predicted = project(points_cam, intrinsics)
    observed = np.asarray(observed_px, dtype=np.float64).reshape(-1, 2)
    return np.hypot(predicted[:, 0] - observed[:, 0], predicted[:, 1] - observed[:, 1])


def summarise(values: np.ndarray, label: str, independent: bool, limit: float) -> dict:
    """Return the median, the p90 and the worst value. State whether the check
    gives evidence."""
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
    root: int = 0,
) -> dict:
    """Compare the hand position with the position the detector reports.

    The gate uses the root joint. State the reason clearly. A reader can
    otherwise read "measure fewer joints" as a relaxed gate.

    WiLoR infers the hand for a camera at about 12 m. That camera has a focal
    length near 37500 px. It is almost orthographic. The hand spans under 1 per
    cent of its own depth. The projection therefore barely depends on the depth
    of each joint. WiLoR's 3D and 2D outputs agree there to 0.0 px, measured.

    Our camera holds the same hand at about 0.48 m. The hand spans 19 per cent
    of its depth. Perspective is strong. The joint depths now matter. WiLoR
    never had to get them right.

    Session real06 demo_0 gives the numbers. Project each joint at its own
    depth and the error is 80 to 149 px. Project every joint at the root depth
    and the error is 6 to 10 px. The second case matches WiLoR's assumption.

    An all-joint bound therefore measures WiLoR's assumption. It does not
    measure this pipeline. The root gives the hand position. The pipeline uses
    that position. The 25x defect destroyed it and would read 614 px here.

    This function still computes and reports the all-joint figure. A change in
    that figure shows a change in the hand shape or depth. Hide it and the gate
    looks like a choice that it is not.
    """
    per_frame = []
    all_joints = []
    weak = []
    for index in np.nonzero(valid)[0]:
        cam = np.asarray(landmarks_cam[index], dtype=np.float64)
        observed = np.asarray(landmarks_px[index], dtype=np.float64)

        value = float(errors(cam[root][None, :], observed[root][None, :], intrinsics)[0])
        if np.isfinite(value):
            per_frame.append(value)

        full = errors(cam, observed, intrinsics)
        full = full[np.isfinite(full)]
        if len(full):
            all_joints.append(float(np.median(full)))

        # Project the same joints under WiLoR's own assumption.
        depth = cam[root, 2]
        if depth > 1e-9:
            u = intrinsics.fx * cam[:, 0] / depth + intrinsics.cx
            v = intrinsics.fy * cam[:, 1] / depth + intrinsics.cy
            weak.append(float(np.median(np.hypot(u - observed[:, 0], v - observed[:, 1]))))

    report = summarise(
        np.array(per_frame), "hand_position_vs_detected_wrist", True,
        MAX_HAND_REPROJECTION_PX,
    )
    if all_joints:
        report["all_joints_median_px"] = round(float(np.median(all_joints)), 2)
    if weak:
        report["all_joints_weak_perspective_median_px"] = round(float(np.median(weak)), 2)
    report["why_the_root"] = (
        "WiLoR fits the hand under weak perspective at about 12 m; at our "
        "0.48 m the joints' own depths matter and its approximation shows. The "
        "root is the hand's position, which is what the pipeline uses."
    )
    return report


def object_check(
    object_poses: np.ndarray,
    object_valid: np.ndarray,
    mask_centroids_px: np.ndarray,
    centroid_valid: np.ndarray,
    camera_poses: np.ndarray,
    intrinsics: Intrinsics,
    sources: np.ndarray | None = None,
) -> dict:
    """Compare the object centre with its mask centroid.

    This check is by construction while the object rests. The plane solve puts
    the centre on the ray through that centroid. The centre therefore reprojects
    onto the centroid exactly. This holds for any error in the plane or in the
    object height.

    The check becomes independent once a hand carries the object. The hand then
    fixes the carried centre. The ray constrains it in one direction only. This
    function splits the two cases. The carried frames give the evidence.
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
