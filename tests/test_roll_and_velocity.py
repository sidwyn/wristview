"""Gripper roll canonicalisation and the hand plausibility gate.

Both defects these cover produced plausible output and no crash. The tests
exist because looking at the numbers was the only thing that found them.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wristview.geometry import canonicalise_roll, make_pose
from wristview.qc import (
    HAND_ANGULAR_RATE_MAX_DEG_S,
    hand_velocity_gates,
    hand_velocity_outliers,
    palm_rotations,
    roll_gates,
)

WORLD_UP = np.array([0.0, 0.0, 1.0])
# The wrist camera's up axis in the gripper frame. The mount sits along -y.
LOCAL_UP = np.array([0.0, -1.0, 0.0])


def upright_pose(yaw_deg: float = 0.0) -> np.ndarray:
    """A gripper looking along +x with its camera up along world up."""
    rotation = np.column_stack([np.array([0.0, 1.0, 0.0]), -WORLD_UP, np.array([1.0, 0.0, 0.0])])
    spin = Rotation.from_rotvec(np.radians(yaw_deg) * WORLD_UP).as_matrix()
    return make_pose(spin @ rotation, np.zeros(3))


def rolled(pose: np.ndarray) -> np.ndarray:
    """The other orientation of the same grasp: 180 degrees about approach."""
    out = pose.copy()
    out[:3, 0] = -pose[:3, 0]
    out[:3, 1] = -pose[:3, 1]
    return out


def up_angles(poses: np.ndarray) -> np.ndarray:
    ups = np.array([p[:3, :3] @ LOCAL_UP for p in poses])
    return np.degrees(np.arccos(np.clip(ups @ WORLD_UP, -1.0, 1.0)))


class TestCanonicaliseRoll:
    def test_a_flip_partway_through_is_removed(self):
        poses = np.array([upright_pose(i * 2.0) for i in range(20)])
        poses[10:] = np.array([rolled(p) for p in poses[10:]])

        fixed, report = canonicalise_roll(poses, WORLD_UP)

        assert report["flips_before"] == 1
        assert report["flips_after"] == 0
        assert up_angles(fixed).max() < 90.0

    def test_a_clip_that_is_entirely_upside_down_is_turned_over(self):
        poses = np.array([rolled(upright_pose(i * 2.0)) for i in range(12)])

        fixed, report = canonicalise_roll(poses, WORLD_UP)

        assert report["flips_after"] == 0
        assert report["frames_up_inverted"] == 0
        assert up_angles(fixed).max() < 1e-6

    def test_an_already_correct_clip_is_left_alone(self):
        poses = np.array([upright_pose(i * 3.0) for i in range(15)])

        fixed, report = canonicalise_roll(poses, WORLD_UP)

        assert report["flips_before"] == 0
        assert report["flips_after"] == 0
        np.testing.assert_allclose(fixed, poses, atol=1e-9)

    def test_continuity_does_not_undo_the_canonical_choice(self):
        """The bug that made the obvious pass order wrong.

        Applying the canonical test per frame and then enforcing continuity
        re-inverted frames that canonical had just fixed, because near-vertical
        neighbours can both be canonical and still disagree with each other.
        Continuity runs first now, so no frame ends up inverted.
        """
        # Poses whose up axis sits close to the horizon, where the canonical
        # test carries almost no information.
        poses = []
        for index in range(24):
            tilt = Rotation.from_rotvec(np.radians(88.0) * np.array([1.0, 0.0, 0.0]))
            spin = Rotation.from_rotvec(np.radians(index * 15.0) * WORLD_UP)
            poses.append(make_pose(spin.as_matrix() @ tilt.as_matrix() @ upright_pose()[:3, :3],
                                   np.zeros(3)))
        poses = np.array(poses)
        poses[7:13] = np.array([rolled(p) for p in poses[7:13]])

        _, report = canonicalise_roll(poses, WORLD_UP)

        assert report["flips_after"] == 0

    def test_world_up_is_used_rather_than_a_fixed_convention(self):
        """COLMAP's world orientation is arbitrary, so up is an input."""
        other_up = np.array([0.0, 1.0, 0.0])
        poses = np.array([upright_pose(i * 2.0) for i in range(10)])

        _, against_z = canonicalise_roll(poses, WORLD_UP)
        _, against_y = canonicalise_roll(poses, other_up)

        assert against_z["median_up_angle_deg"] != against_y["median_up_angle_deg"]

    def test_an_empty_sequence_does_not_raise(self):
        fixed, report = canonicalise_roll(np.zeros((0, 4, 4)), WORLD_UP)

        assert len(fixed) == 0
        assert report["flips_after"] == 0

    def test_the_gate_refuses_a_flip_and_reports_a_missing_report(self):
        assert roll_gates({"flips_before": 2, "flips_after": 1,
                           "median_up_angle_deg": 10.0})[0].passed is False
        assert roll_gates(None)[0].passed is False


def straight_reach(count: int, degrees_per_frame: float) -> np.ndarray:
    """Hand landmarks rotating steadily about the palm's across axis."""
    base = np.zeros((21, 3))
    base[0] = [0.0, 0.0, 0.0]
    base[5] = [0.04, 0.0, 0.09]
    base[9] = [0.0, 0.0, 0.10]
    base[17] = [-0.04, 0.0, 0.085]
    frames = []
    for index in range(count):
        spin = Rotation.from_rotvec(
            np.radians(index * degrees_per_frame) * np.array([0.0, 0.0, 1.0])
        ).as_matrix()
        frames.append(base @ spin.T)
    return np.array(frames)


class TestHandVelocityGate:
    def test_a_normal_reach_flags_nothing(self):
        landmarks = straight_reach(40, 10.0)  # 200 deg/s at 20 fps

        report = hand_velocity_outliers(landmarks, np.ones(40, bool), 20.0)

        assert report["frames_flagged"] == 0
        assert all(gate.passed for gate in hand_velocity_gates(report))

    def test_one_impossible_frame_is_flagged_and_repairable(self):
        landmarks = straight_reach(40, 5.0)
        # Displace a single frame far enough to imply a non-human rotation.
        spin = Rotation.from_rotvec(np.radians(120.0) * np.array([1.0, 0.0, 0.0])).as_matrix()
        landmarks[20] = landmarks[20] @ spin.T

        report = hand_velocity_outliers(landmarks, np.ones(40, bool), 20.0)

        assert report["outlier"][20]
        assert report["longest_run"] == 1
        # Isolated, so the episode survives and the frame gets filled.
        assert all(gate.passed for gate in hand_velocity_gates(report))

    def test_consecutive_bad_frames_reject_the_episode(self):
        landmarks = straight_reach(40, 5.0)
        spin = Rotation.from_rotvec(np.radians(120.0) * np.array([1.0, 0.0, 0.0])).as_matrix()
        landmarks[20] = landmarks[20] @ spin.T
        landmarks[21] = landmarks[21] @ spin.T @ spin.T

        report = hand_velocity_outliers(landmarks, np.ones(40, bool), 20.0)

        assert report["longest_run"] >= 2
        run_gate = next(g for g in hand_velocity_gates(report) if g.name == "hand_velocity_run")
        assert run_gate.passed is False

    def test_the_threshold_is_a_human_limit_not_a_fitted_one(self):
        assert HAND_ANGULAR_RATE_MAX_DEG_S == pytest.approx(900.0)

    def test_a_dropout_gap_is_not_mistaken_for_fast_motion(self):
        """A long gap means a large angle over a long time, not a fast wrist."""
        landmarks = straight_reach(40, 5.0)
        valid = np.ones(40, bool)
        valid[10:30] = False  # one second missing at 20 fps

        report = hand_velocity_outliers(landmarks, valid, 20.0)

        assert report["frames_flagged"] == 0

    def test_too_few_frames_returns_an_empty_report(self):
        report = hand_velocity_outliers(straight_reach(2, 5.0), np.ones(2, bool), 20.0)

        assert report["frames_flagged"] == 0
        assert report["outlier"].sum() == 0

    def test_palm_rotations_are_proper_rotations(self):
        rotations = palm_rotations(straight_reach(10, 8.0))

        for rotation in rotations:
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            assert np.linalg.det(rotation) == pytest.approx(1.0)
