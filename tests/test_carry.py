"""The object leaves the table, and the tracker has to follow it.

These tests are written against the defect that motivated the module. Session 6
clip 5 tracked its object at 2.000 cm above the desk on every one of 185 frames,
spread 0.0000 mm, while the hand carrying it reached 24.7 cm. The first test
below is the one that would have caught it, and it is deliberately the simplest
thing in the file: does the tracked height ever change.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristview.carry import (
    FINGERTIPS,
    contact_onsets,
    grasp_centre,
    point_on_ray_closest_to,
    rest_precedes_contact,
    resting_pose,
    solve_carried,
)

NUM_LANDMARKS = 21
CAMERA = np.array([0.0, 0.0, 1.2])       # looking down the -z axis at the desk
REST = np.array([0.10, 0.05, 0.0])       # where the object sits


def _hand_at(position: np.ndarray) -> np.ndarray:
    """A hand whose fingertips surround `position`."""
    landmarks = np.zeros((NUM_LANDMARKS, 3))
    landmarks[:] = position
    landmarks[WRIST_INDEX] = position + np.array([0.0, -0.09, 0.0])
    for offset, tip in zip(
        ([0.005, 0, 0], [-0.005, 0, 0], [0, 0.005, 0]), FINGERTIPS, strict=True
    ):
        landmarks[tip] = position + np.asarray(offset, dtype=float)
    return landmarks


WRIST_INDEX = 0


def _ray_from_camera_to(point: np.ndarray) -> np.ndarray:
    return np.asarray(point, dtype=float) - CAMERA


# --------------------------------------------------------------------------
# the ray step
# --------------------------------------------------------------------------


def test_closest_point_recovers_the_true_position_from_a_perfect_ray():
    """If the ray points at the object and the hand holds it, both agree."""
    truth = np.array([0.10, 0.05, 0.22])
    found = point_on_ray_closest_to(CAMERA, _ray_from_camera_to(truth), truth)
    assert np.allclose(found, truth, atol=1e-9)


def test_closest_point_stays_on_the_ray_when_the_hand_is_off():
    """The silhouette is trusted across the image, the hand only along it."""
    truth = np.array([0.10, 0.05, 0.22])
    ray = _ray_from_camera_to(truth)
    nudged = truth + np.array([0.03, -0.02, 0.0])
    found = point_on_ray_closest_to(CAMERA, ray, nudged)
    unit = ray / np.linalg.norm(ray)
    residual = (found - CAMERA) - float((found - CAMERA) @ unit) * unit
    assert np.linalg.norm(residual) < 1e-9


def test_a_target_behind_the_camera_is_clamped_in_front():
    behind = CAMERA + np.array([0.0, 0.0, 1.0])
    found = point_on_ray_closest_to(CAMERA, np.array([0.0, 0.0, -1.0]), behind)
    assert found[2] <= CAMERA[2]


def test_a_degenerate_ray_falls_back_to_the_target():
    target = np.array([0.1, 0.1, 0.1])
    found = point_on_ray_closest_to(CAMERA, np.zeros(3), target)
    assert np.allclose(found, target)


# --------------------------------------------------------------------------
# the resting pose
# --------------------------------------------------------------------------


def test_resting_pose_takes_the_still_stretch_not_the_sliding_one():
    """The object sits, then is carried. Only the sitting part is the rest."""
    count = 30
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, :3, 3] = REST
    # From frame 12 the plane solve slides the object across the desk as the
    # silhouette ray sweeps, which is what a carry looks like from the plane.
    for i in range(12, count):
        poses[i, :3, 3] = REST + np.array([0.02 * (i - 11), 0.0, 0.0])

    result = resting_pose(poses, np.ones(count, dtype=bool))
    assert result is not None
    pose, report = result
    assert np.allclose(pose[:3, 3], REST, atol=1e-6)
    assert report["rest_run"][1] <= 13
    assert report["scatter_median_cm"] < 0.1


def test_a_hand_on_the_object_does_not_disturb_the_rest_estimate():
    """The regression this rewrite exists for.

    The first version asked whether the hand was far from the object and kept
    the frames where it was. During a carry the plane solve leaves the object
    on the desk while the hand is in the air, so every carried frame looked
    clear and was admitted: session 6 clip 1 took 336 of 338 frames that way
    and produced 7.36 cm of scatter. Stillness is decided from the object
    alone, so a hand sitting right on it changes nothing.
    """
    count = 30
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, :3, 3] = REST
    for i in range(12, count):
        poses[i, :3, 3] = REST + np.array([0.02 * (i - 11), 0.0, 0.0])

    without_hand = resting_pose(poses, np.ones(count, dtype=bool))
    assert without_hand is not None
    # Nothing about a hand enters the call at all, which is the point.
    assert np.allclose(without_hand[0][:3, 3], REST, atol=1e-6)
    assert without_hand[1]["frames_used"] < count


def test_resting_pose_gives_up_rather_than_guess_from_two_frames():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    assert resting_pose(poses, np.ones(2, dtype=bool)) is None


def test_an_object_that_never_holds_still_has_no_resting_pose():
    """Better to report nothing than to average a moving object."""
    count = 40
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    for i in range(count):
        poses[i, :3, 3] = REST + np.array([0.05 * i, 0.0, 0.0])
    assert resting_pose(poses, np.ones(count, dtype=bool)) is None


# --------------------------------------------------------------------------
# contact
# --------------------------------------------------------------------------


def _approach_then_leave(count: int = 30) -> tuple[np.ndarray, np.ndarray]:
    landmarks = np.zeros((count, NUM_LANDMARKS, 3))
    for i in range(count):
        if 10 <= i < 22:
            landmarks[i] = _hand_at(REST)
        else:
            landmarks[i] = _hand_at(REST + np.array([0.4, 0.0, 0.0]))
    return landmarks, np.ones(count, dtype=bool)


def test_the_onset_is_the_frame_the_hand_arrived():
    landmarks, hand_valid = _approach_then_leave()
    onsets = contact_onsets(
        landmarks, hand_valid, REST,
        object_radius_m=0.02, enter_m=0.03, min_frames=3,
    )
    assert onsets == [10]


def test_a_brief_brush_is_not_a_grasp():
    count = 30
    landmarks = np.zeros((count, NUM_LANDMARKS, 3))
    for i in range(count):
        near = i == 15
        landmarks[i] = _hand_at(REST if near else REST + np.array([0.4, 0, 0]))
    onsets = contact_onsets(
        landmarks, np.ones(count, dtype=bool), REST,
        object_radius_m=0.02, enter_m=0.03, min_frames=3,
    )
    assert onsets == []


def test_a_hand_still_on_the_object_at_the_last_frame_counts():
    count = 20
    landmarks = np.zeros((count, NUM_LANDMARKS, 3))
    for i in range(count):
        landmarks[i] = _hand_at(REST if i >= 12 else REST + np.array([0.4, 0, 0]))
    onsets = contact_onsets(
        landmarks, np.ones(count, dtype=bool), REST,
        object_radius_m=0.02, enter_m=0.03, min_frames=3,
    )
    assert onsets == [12]


def test_a_second_pickup_after_a_release_is_its_own_onset():
    """Two separate grasps must not be merged into one run."""
    count = 60
    landmarks = np.zeros((count, NUM_LANDMARKS, 3))
    for i in range(count):
        on = (8 <= i < 16) or (40 <= i < 50)
        landmarks[i] = _hand_at(REST if on else REST + np.array([0.4, 0, 0]))
    onsets = contact_onsets(
        landmarks, np.ones(count, dtype=bool), REST,
        object_radius_m=0.02, enter_m=0.03, min_frames=3,
    )
    assert onsets == [8, 40]


# --------------------------------------------------------------------------
# the whole solve, against the defect
# --------------------------------------------------------------------------


@pytest.fixture
def lifted_clip():
    """A clip where the object is picked up and moved 22 cm into the air.

    The plane solve is given the answer it would really produce: the object
    pinned to the desk on every frame, because that is what pinning to a plane
    means.
    """
    count = 40
    lift = np.zeros((count, 3))
    for i in range(count):
        if 12 <= i < 30:
            height = 0.22 * np.sin(np.pi * (i - 12) / 18)
            lift[i] = np.array([0.15 * (i - 12) / 18, 0.0, height])

    truth = REST[None, :] + lift

    # The hand reaches IN. It is not on the object at frame 0.
    #
    # The fixture used to put the hand at the object from the first frame,
    # which meant contact began at frame 0 and no resting window existed
    # before it. Real footage does not look like that: real26 saw the hand
    # arrive at frame 57 of 439. Modelling the approach is what lets
    # `rest_precedes_contact` be tested against a clip that should pass it.
    APPROACH = 8
    hand_at = truth.copy()
    for i in range(APPROACH):
        hand_at[i] = truth[i] + np.array([0.0, -0.40 * (APPROACH - i) / APPROACH, 0.10])
    landmarks = np.stack([_hand_at(p) for p in hand_at])
    hand_valid = np.ones(count, dtype=bool)

    plane_poses = np.repeat(np.eye(4)[None], count, axis=0)
    plane_poses[:, :3, 3] = REST          # the defect, exactly as measured
    plane_valid = np.ones(count, dtype=bool)

    rays = np.stack([_ray_from_camera_to(p) for p in truth])
    cameras = np.repeat(CAMERA[None], count, axis=0)
    return truth, plane_poses, plane_valid, rays, cameras, landmarks, hand_valid


def test_the_tracked_object_leaves_the_table(lifted_clip):
    """The test that would have caught session 6, stated as plainly as it is."""
    truth, plane_poses, plane_valid, rays, cameras, landmarks, hand_valid = lifted_clip
    onsets = contact_onsets(
        landmarks, hand_valid, REST,
        object_radius_m=0.02, enter_m=0.03, min_frames=3,
    )
    rest = np.eye(4)
    rest[:3, 3] = REST
    poses, valid, report = solve_carried(
        plane_poses, plane_valid, rays, cameras, landmarks, hand_valid, rest, onsets,
        (0, 8),
    )
    heights = poses[valid][:, 2, 3]
    assert heights.max() - heights.min() > 0.15, (
        "the tracked object never left the desk, which is the session 6 defect"
    )
    assert report["frames_carried"] > 0


def test_the_carried_object_follows_the_truth(lifted_clip):
    truth, plane_poses, plane_valid, rays, cameras, landmarks, hand_valid = lifted_clip
    onsets = contact_onsets(
        landmarks, hand_valid, REST,
        object_radius_m=0.02, enter_m=0.03, min_frames=3,
    )
    rest = np.eye(4)
    rest[:3, 3] = REST
    poses, valid, report = solve_carried(
        plane_poses, plane_valid, rays, cameras, landmarks, hand_valid, rest, onsets,
        (0, 8),
    )
    runs = [tuple(r) for r in report["contact_runs"]]
    carried = [i for run in runs for i in range(*run)]
    error = np.linalg.norm(poses[carried][:, :3, 3] - truth[carried], axis=1)
    assert error.max() < 0.02, f"worst carried error {error.max() * 100:.2f} cm"


def test_frames_outside_contact_keep_the_plane_answer(lifted_clip):
    """The plane solve is right before contact and must not be disturbed."""
    _, plane_poses, plane_valid, rays, cameras, landmarks, hand_valid = lifted_clip
    onsets = contact_onsets(
        landmarks, hand_valid, REST,
        object_radius_m=0.02, enter_m=0.03, min_frames=3,
    )
    rest = np.eye(4)
    rest[:3, 3] = REST
    poses, _, report = solve_carried(
        plane_poses, plane_valid, rays, cameras, landmarks, hand_valid, rest, onsets,
        (0, 8),
    )
    before = min(run[0] for run in report["contact_runs"])
    assert np.allclose(poses[:before, :3, 3], REST)


def test_a_clip_with_no_contact_is_left_entirely_alone(lifted_clip):
    _, plane_poses, plane_valid, rays, cameras, landmarks, hand_valid = lifted_clip
    rest = np.eye(4)
    rest[:3, 3] = REST
    poses, valid, report = solve_carried(
        plane_poses, plane_valid, rays, cameras, landmarks, hand_valid, rest, [], (0, 8)
    )
    assert report["frames_carried"] == 0
    assert np.allclose(poses, plane_poses)


def test_grasp_centre_sits_between_the_fingertips():
    hand = _hand_at(REST)
    assert np.allclose(grasp_centre(hand), hand[list(FINGERTIPS)].mean(axis=0))


class TestRestPrecedesContact:
    """The resting pose must be measured BEFORE the hand arrives.

    real26/bm demo_1: no object was detected until frame 352, so the first
    still run was 360 to 510, after the release. It was used to attach the
    object at contact frame 243. The carried object solved to a median 7.06 cm
    below the desk across 109 frames, and every other metric passed.
    """

    def test_a_rest_run_after_contact_is_refused(self):
        problem = rest_precedes_contact((360, 510), [243, 400])
        assert problem is not None
        assert "360" in problem and "243" in problem

    def test_a_rest_run_before_contact_is_accepted(self):
        assert rest_precedes_contact((0, 90), [100, 300]) is None

    def test_a_rest_run_ending_exactly_at_contact_is_accepted(self):
        assert rest_precedes_contact((0, 100), [100]) is None

    def test_a_rest_run_ending_one_frame_late_is_refused(self):
        assert rest_precedes_contact((0, 101), [100]) is not None

    def test_no_contact_means_nothing_to_check(self):
        assert rest_precedes_contact((360, 510), []) is None

    def test_solve_carried_raises_rather_than_returning_a_wrong_carry(self, lifted_clip):
        """The guard lives in the function, so every caller is protected."""
        _, plane_poses, plane_valid, rays, cameras, landmarks, hand_valid = lifted_clip
        onsets = contact_onsets(
            landmarks, hand_valid, REST,
            object_radius_m=0.02, enter_m=0.03, min_frames=3,
        )
        assert onsets, "fixture must produce a contact for this test to mean anything"
        rest = np.eye(4)
        rest[:3, 3] = REST
        with pytest.raises(ValueError, match="does not END before contact"):
            solve_carried(
                plane_poses, plane_valid, rays, cameras, landmarks, hand_valid,
                rest, onsets, (onsets[0] + 10, onsets[0] + 50),
            )
