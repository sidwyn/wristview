"""Track the object pose while a hand holds the object off the table.

`plane.object_pose_on_plane` solves the object from its silhouette and the desk
below it. That method removed the monocular-depth dependency. It brought the
pre-contact resting error from 11-16 cm down to 3.27 cm.

The method has one property that is easy to miss. It fixes the object to the
desk. It places the centre at `offset + height / 2` on every frame. The tracked
object therefore holds one height for the whole clip.

Session 6 clip 5 measured 2.000 cm above the desk on all 185 valid frames. The
spread was 0.0000 mm. The operator's wrist rose to 24.7 cm. No stage reported a
problem.

The consequence appeared two stages later. Grasp detection found contact on 0 of
186 frames. Fingertips cannot reach an object that stays on the table while the
hand carries it away. The median fingertip-to-object gap was 13.8 cm. That gap
is the lift height.

So the plane solve is correct until contact. It is wrong after contact. A
manipulation dataset needs the frames after contact.

This module keeps the correct part and replaces the broken part. The silhouette
ray stays correct throughout. It points at the object on the desk and in the
air. Only the distance along the ray becomes wrong after the lift.

On the desk, the plane fixes that distance. In the hand, the hand fixes it. The
centre is the point on the ray nearest the grasp centre. Neither branch
estimates depth.

Contact is decided while the object still rests. The plane solve is reliable at
that moment. This breaks a circular dependency. Contact needs the object pose.
The carried pose needs contact. The resting pose needs neither.
"""

from __future__ import annotations

import numpy as np

from .logging_setup import get

log = get(__name__)

# These MANO fingertips oppose in a pinch or a wrap.
FINGERTIPS = (4, 8, 12)
WRIST = 0


def point_on_ray_closest_to(
    origin: np.ndarray, direction: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Find the point on the ray that lies nearest to `target`.

    The ray is reliable in a carried frame. It points at the object. This
    function selects the distance along the ray. It also clamps points behind
    the camera. The mask cannot see an object behind the lens.
    """
    direction = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return np.asarray(target, dtype=np.float64)
    unit = direction / length
    t = float((np.asarray(target, dtype=np.float64) - origin) @ unit)
    return origin + max(t, 1e-6) * unit


def grasp_centre(landmarks: np.ndarray) -> np.ndarray:
    """Return the midpoint of the opposing fingertips. A held object sits there."""
    return np.asarray(landmarks, dtype=np.float64)[list(FINGERTIPS)].mean(axis=0)


def resting_pose(
    poses: np.ndarray,
    valid: np.ndarray,
    stillness_m: float = 0.01,
    min_rest_frames: int = 5,
) -> tuple[np.ndarray, dict] | None:
    """Find where the object rested before a hand lifted it.

    This function uses the object alone.

    An earlier version measured the distance from the hand to the object. It
    kept the frames where the hand was far away. That test is backwards. During
    a carry the plane solve leaves the object on the table. The hand is in the
    air. Every carried frame therefore looks clear. Session 6 clip 1 accepted
    336 of 338 frames and gave 7.36 cm of scatter. It found no contact. A filter
    that uses a broken quantity cannot detect the fault in that quantity.

    Stillness avoids the fault. While the object rests, the silhouette ray is
    fixed. The plane solve returns the same point on every frame. While the hand
    carries the object, the ray sweeps. The point then slides across the desk.
    This holds at any true object height.

    Take the first sustained still run. That run gives the object's start
    position. Contact detection needs that position.
    """
    indices = np.nonzero(valid)[0]
    if len(indices) < min_rest_frames:
        return None
    centres = poses[:, :3, 3]

    still = np.zeros(len(indices), dtype=bool)
    for position, (a, b) in enumerate(zip(indices, indices[1:], strict=False)):
        if b - a == 1:
            still[position] = float(np.linalg.norm(centres[b] - centres[a])) < stillness_m

    runs: list[tuple[int, int]] = []
    start = None
    for position, flag in enumerate(still):
        if flag and start is None:
            start = position
        elif not flag and start is not None:
            if position - start + 1 >= min_rest_frames:
                runs.append((start, position + 1))
            start = None
    if start is not None and len(still) - start >= min_rest_frames:
        runs.append((start, len(still) + 1))

    if not runs:
        return None
    first = indices[runs[0][0] : runs[0][1]]

    block = poses[first]
    centre = np.median(block[:, :3, 3], axis=0)
    scatter = np.linalg.norm(block[:, :3, 3] - centre, axis=1)
    pose = np.eye(4)
    pose[:3, :3] = block[len(block) // 2][:3, :3]
    pose[:3, 3] = centre
    return pose, {
        "frames_used": int(len(first)),
        "rest_run": [int(first[0]), int(first[-1]) + 1],
        "still_runs_found": len(runs),
        "scatter_median_cm": round(float(np.median(scatter)) * 100, 3),
        "scatter_p90_cm": round(float(np.percentile(scatter, 90)) * 100, 3),
        "stillness_m": stillness_m,
    }


def contact_onsets(
    landmarks: np.ndarray,
    hand_valid: np.ndarray,
    rest_centre: np.ndarray,
    object_radius_m: float,
    enter_m: float,
    min_frames: int,
) -> list[int]:
    """Find the frames where the hand reached the object's resting place.

    This function decides the onset only. It measures against the resting
    position. That position is the one position known to be correct.

    Do not apply the same test to later frames. The first draft of this module
    did so and failed. After a lift the object is no longer near its resting
    place. A distance-to-rest test then reports a release two frames into every
    pick-up. `solve_carried` decides the release. It uses the silhouette.

    `min_frames` makes the hand stay at the object. A hand that passes over the
    object does not then register as a grasp.
    """
    count = len(landmarks)
    near = np.zeros(count, dtype=bool)
    for index in np.nonzero(hand_valid)[0]:
        tips = landmarks[index][list(FINGERTIPS)]
        gap = float(np.linalg.norm(tips - rest_centre, axis=1).min()) - object_radius_m
        near[index] = gap < enter_m

    onsets: list[int] = []
    index = 0
    while index < count:
        if not near[index]:
            index += 1
            continue
        stop = index
        while stop < count and near[stop]:
            stop += 1
        if stop - index >= min_frames:
            onsets.append(index)
        index = stop
    return onsets


def _distance_to_ray(
    point: np.ndarray, origin: np.ndarray, direction: np.ndarray
) -> float:
    """Return the distance from `point` to the line, in metres."""
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return 0.0
    unit = direction / length
    offset = np.asarray(point, dtype=np.float64) - origin
    return float(np.linalg.norm(offset - float(offset @ unit) * unit))


def rest_window_before_contact(
    rest_run: tuple[int, int], onsets: list[int], min_rest_frames: int = 5
) -> tuple[tuple[int, int] | None, str | None]:
    """Trim a still run to the part that happened BEFORE the grasp.

    A still run may legitimately extend past first contact. The hand reaches
    the object several frames before it lifts it, and while the object has not
    moved the plane solve keeps returning the same point, so the run continues.
    real27 measured a run of frames 0 to 90 with contact at 75 and a scatter of
    0.04 cm: the object genuinely had not moved, and frames 0 to 74 are exactly
    the resting evidence the carry solve needs.

    So the rule is not "the run ends before contact". It is "enough of the run
    happened before contact". Trim at the first onset and check what is left.

    Returns the usable window and, if there is none, why.
    """
    start, stop = int(rest_run[0]), int(rest_run[1])
    if not onsets:
        return (start, stop), None
    first = int(min(onsets))
    trimmed_stop = min(stop, first)
    if trimmed_stop - start < min_rest_frames:
        return None, (
            f"the resting pose was measured on frames {start} to {stop}, and "
            f"the hand first reaches the object at frame {first}, which leaves "
            f"{max(trimmed_stop - start, 0)} frames of rest before contact "
            f"against a minimum of {min_rest_frames}. There is no window in "
            f"which the object was seen at rest before it was touched, so "
            f"where it rested is unknown. On real26/bm demo_1 the run began "
            f"117 frames AFTER the release and the carry solved 7.06 cm below "
            f"the desk."
        )
    return (start, trimmed_stop), None


def rest_precedes_contact(rest_run: tuple[int, int], onsets: list[int]) -> str | None:
    """Return why the resting pose cannot be used, or None if it can.

    `resting_pose` takes the first sustained still run. `solve_carried` then
    attaches the object to the hand at the contact frame, using that run's
    position. The whole arrangement only works if the object was still THERE
    BEFORE the hand arrived. Neither function checked it.

    real26/bm demo_1: the detector found no object until frame 352, so the
    first still run was frames 360 to 510, after the release. `solve_carried`
    used it to attach the object at contact frame 243, 117 frames earlier and
    at a different place on the mat. The carried object solved to a median of
    7.06 cm BELOW the desk across 109 frames, and Stage 4 reported the clip as
    healthy: 510 of 510 registered, a hand on every frame, no velocity
    outliers, no dropped frames.

    This is not circular. `resting_pose` uses the object alone and `onsets`
    are computed from it, so checking the order afterwards tests the
    assumption rather than assuming it.
    """
    start, stop = int(rest_run[0]), int(rest_run[1])
    if not onsets:
        return None
    first = int(min(onsets))
    if stop > first:
        return (
            f"the resting pose was measured on frames {start} to {stop}, but "
            f"the hand first reaches the object at frame {first}. A rest run "
            f"that does not END before contact is not where the object rested: "
            f"it is where the object ended up. Attaching the object to the "
            f"hand from it puts the carry in the wrong place, which on "
            f"real26/bm demo_1 was 7.06 cm below the desk."
        )
    return None


def solve_carried(
    plane_poses: np.ndarray,
    plane_valid: np.ndarray,
    rays: np.ndarray,
    camera_positions: np.ndarray,
    landmarks: np.ndarray,
    hand_valid: np.ndarray,
    rest_pose: np.ndarray,
    onsets: list[int],
    rest_run: tuple[int, int],
    release_ray_m: float = 0.06,
    release_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the object pose per frame. Use the desk before contact and the
    hand after contact.

    At each onset, attach the object to the hand. Measure the attachment once,
    at the onset frame. The object still rests there. The plane solve is still
    correct there. Carry that one transform forward. The motion is then rigid.
    A per-frame estimate would drift instead.

    Decide the release by disagreement, not by distance. While the hand holds
    the object, the silhouette ray and the hand agree. When the hand sets the
    object down and withdraws, the mask stays with the object. The hand moves
    away. The hand-predicted position then leaves the ray.

    Count `release_frames` frames of disagreement. End the run at the first of
    those frames. The object stopped following the hand at that frame.
    """
    # The resting pose must predate the grasp. See `rest_precedes_contact`.
    # This raises rather than returning a flag, because the alternative is a
    # complete, plausible, wrong trajectory that every downstream gate passes.
    _window, problem = rest_window_before_contact(rest_run, onsets)
    if problem is not None:
        raise ValueError(problem)

    count = len(plane_poses)
    poses = plane_poses.copy()
    valid = plane_valid.copy()
    source = np.array(["plane"] * count, dtype=object)
    source[~plane_valid] = ""

    runs: list[list[int]] = []
    carried = 0
    index_after_last_run = 0

    for onset in onsets:
        if onset < index_after_last_run:
            continue      # already inside a run that has not released yet
        if not hand_valid[onset]:
            continue
        hand_to_object = grasp_centre(landmarks[onset]) - rest_pose[:3, 3]
        rotation = rest_pose[:3, :3]

        pending: list[int] = []
        stop = count
        for index in range(onset, count):
            if not hand_valid[index]:
                pending.clear()
                continue
            target = grasp_centre(landmarks[index]) - hand_to_object

            if plane_valid[index]:
                off_ray = _distance_to_ray(
                    target, camera_positions[index], rays[index]
                )
                if off_ray > release_ray_m:
                    pending.append(index)
                    if len(pending) >= release_frames:
                        stop = pending[0]
                        break
                else:
                    pending.clear()
                # The silhouette constrains the position across the image.
                # The hand constrains the position along the ray. Use each one
                # for what it measures. Neither estimates depth.
                centre = point_on_ray_closest_to(
                    camera_positions[index], rays[index], target
                )
            else:
                pending.clear()
                # This frame has no mask. Use the hand only.
                centre = target

            poses[index] = np.eye(4)
            poses[index][:3, :3] = rotation
            poses[index][:3, 3] = centre
            valid[index] = True
            source[index] = "carried"

        # Return the frames after the release point to the plane solve.
        for index in range(stop, count):
            if source[index] != "carried":
                break
            poses[index] = plane_poses[index]
            valid[index] = plane_valid[index]
            source[index] = "plane" if plane_valid[index] else ""

        carried += int((source[onset:stop] == "carried").sum())
        runs.append([onset, stop])
        index_after_last_run = stop

    return poses, valid, {
        "frames_carried": carried,
        "frames_on_plane": int((source == "plane").sum()),
        "contact_runs": [[int(a), int(b)] for a, b in runs],
    }
