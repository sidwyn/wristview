"""Object pose while the object is off the table.

`plane.object_pose_on_plane` solves the object from its silhouette and the desk
it stands on, which removed the monocular-depth dependency and brought the
pre-contact resting error from 11-16 cm down to 3.27 cm. It has one property
that is easy to miss and fatal downstream: it pins the object to the desk. The
centre is placed at `offset + height / 2` on every frame, so the tracked object
sits at exactly the same height for the whole clip.

Session 6 clip 5 measured 2.000 cm above the desk on all 185 valid frames, to a
spread of 0.0000 mm, while the operator's wrist rose to 24.7 cm. Nothing in the
run reported a problem. The consequence surfaced two stages later as grasp
detection finding contact on 0 of 186 frames: the fingertips cannot approach an
object that stays on the table while the hand carries it away, so the median
fingertip-to-object gap was 13.8 cm, which is the lift height.

So the plane solve is right up to the moment of contact and wrong after it, and
those are the frames a manipulation dataset is about.

The fix keeps the part that works and replaces only what breaks. The silhouette
ray is correct throughout: it points at the object whether the object is on the
desk or in the air. Only the distance along it is wrong once the object lifts.
On the desk, the plane fixes that distance. In the hand, the hand fixes it: the
object is where the fingers are, so the centre is the point on the ray nearest
the grasp centre. Neither branch estimates depth.

Contact is decided while the object is still at rest, which is when the plane
solve is trustworthy, and that is what breaks the circularity: contact needs the
object pose, the carried pose needs contact, and the resting pose needs neither.
"""

from __future__ import annotations

import numpy as np

from .logging_setup import get

log = get(__name__)

# MANO fingertips that oppose in a pinch or a wrap.
FINGERTIPS = (4, 8, 12)
WRIST = 0


def point_on_ray_closest_to(
    origin: np.ndarray, direction: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Where along the ray it passes nearest `target`.

    The ray is the one certain thing in a carried frame: it points at the
    object. This picks the distance along it, and clamps behind the camera
    away, because an object behind the lens is not what the mask saw.
    """
    direction = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return np.asarray(target, dtype=np.float64)
    unit = direction / length
    t = float((np.asarray(target, dtype=np.float64) - origin) @ unit)
    return origin + max(t, 1e-6) * unit


def grasp_centre(landmarks: np.ndarray) -> np.ndarray:
    """Middle of the opposing fingertips, which is where a held object sits."""
    return np.asarray(landmarks, dtype=np.float64)[list(FINGERTIPS)].mean(axis=0)


def resting_pose(
    poses: np.ndarray,
    valid: np.ndarray,
    stillness_m: float = 0.01,
    min_rest_frames: int = 5,
) -> tuple[np.ndarray, dict] | None:
    """Where the object sat before anything picked it up.

    Decided from the object alone. An earlier version asked whether the hand
    was far from the object and used the frames where it was, which sounds
    right and is exactly backwards: during a carry the plane solve leaves the
    object on the table while the hand is in the air, so every carried frame
    looks like the hand is clear. Session 6 clip 1 admitted 336 of 338 frames
    that way, with 7.36 cm of scatter, and found no contact at all. Filtering
    on the broken quantity cannot detect that the quantity is broken.

    Stillness does not have that problem. While the object rests, the ray
    through its silhouette is fixed and the plane solve returns the same point
    every frame. While it is carried, the ray sweeps and the point slides
    across the desk, whatever height the object is really at. The first
    sustained still run is the object's starting place, which is the one
    contact detection needs.
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
    """Frames where the hand arrived at the object's resting place.

    Only the onset is decided here, and only against the resting position,
    because that is the one position known to be right. Asking the same
    question about later frames does not work and the first draft of this
    module got it wrong: once the object is carried away it is no longer near
    where it was resting, so a distance-to-rest test reports release two frames
    into every pick-up. Release is a different question, answered in
    `solve_carried` where the silhouette is available to answer it.

    `min_frames` requires the hand to stay there, so that a hand passing over
    the object on its way somewhere else does not register as a grasp.
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
    """How far `point` sits off the line, in metres."""
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return 0.0
    unit = direction / length
    offset = np.asarray(point, dtype=np.float64) - origin
    return float(np.linalg.norm(offset - float(offset @ unit) * unit))


def solve_carried(
    plane_poses: np.ndarray,
    plane_valid: np.ndarray,
    rays: np.ndarray,
    camera_positions: np.ndarray,
    landmarks: np.ndarray,
    hand_valid: np.ndarray,
    rest_pose: np.ndarray,
    onsets: list[int],
    release_ray_m: float = 0.06,
    release_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Object pose per frame: on the desk before contact, in the hand after.

    From each onset the object is rigidly attached to the hand. The attachment
    is measured once, at the onset frame, where the object is still at its
    resting place and the plane solve is still correct. Carrying that one
    transform forward is what makes the motion rigid rather than a fresh guess
    each frame.

    Release is decided by disagreement, not by distance. While the object is
    held, the silhouette ray and the hand point at the same place. When the
    object is set down and the hand withdraws, the mask stays with the object
    and the hand does not, so the hand-predicted position swings off the ray.
    `release_frames` consecutive frames of that is a release, and the run ends
    at the first of them rather than the last, because the object stopped
    following the hand at the start of the disagreement.
    """
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
                # The silhouette constrains the position across the image, the
                # hand constrains it along the ray. Each is used for what it
                # knows, and neither estimates depth.
                centre = point_on_ray_closest_to(
                    camera_positions[index], rays[index], target
                )
            else:
                pending.clear()
                centre = target     # no mask this frame, so the hand is all there is

            poses[index] = np.eye(4)
            poses[index][:3, :3] = rotation
            poses[index][:3, 3] = centre
            valid[index] = True
            source[index] = "carried"

        # Frames provisionally written after the release point belong to the
        # plane solve, not to the hand.
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
