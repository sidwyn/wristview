"""Stage 4 retargeting: grasp detection and the hand-to-gripper mapping.

The build plan calls the pose mapping a design choice, not a formula. These
tests pin the choice down so a later change to it is visible rather than
silent.
"""

import numpy as np
import pytest

from wristview.backends import hands as hand_backend
from wristview.backends.gripper import GripperSpec, jaw_boxes
from wristview.stages.s04_retarget import (
    _hysteresis,
    detect_grasp,
    gripper_poses_from_hand,
    resample,
)


def make_hand(
    thumb_tip: np.ndarray,
    index_tip: np.ndarray,
    wrist: np.ndarray | None = None,
) -> np.ndarray:
    """A landmark array with the joints Stage 4 actually reads."""
    hand = np.zeros((21, 3))
    hand[hand_backend.WRIST] = wrist if wrist is not None else np.array([0.0, -0.10, 0.0])
    hand[hand_backend.THUMB_TIP] = thumb_tip
    hand[hand_backend.INDEX_TIP] = index_tip
    hand[hand_backend.INDEX_MCP] = np.array([0.02, -0.02, 0.0])
    hand[hand_backend.MIDDLE_MCP] = np.array([0.0, -0.01, 0.0])
    hand[hand_backend.PINKY_MCP] = np.array([-0.04, -0.02, 0.0])
    return hand


class TestHysteresis:
    def test_removes_a_short_run(self):
        signal = np.array([0, 0, 0, 1, 0, 0, 0], dtype=bool)
        assert not _hysteresis(signal, min_frames=3).any()

    def test_keeps_a_long_run(self):
        signal = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
        assert _hysteresis(signal, min_frames=3).sum() == 4

    def test_does_not_absorb_a_short_final_run(self):
        # A two-frame release at the end of a clip is the terminal state, not
        # chatter. Absorbing it would report the gripper as still closed.
        signal = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
        assert not _hysteresis(signal, min_frames=3)[-2:].any()

    def test_absorbs_chatter_in_the_middle(self):
        signal = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
        assert _hysteresis(signal, min_frames=3).all()

    def test_min_frames_of_one_changes_nothing(self):
        signal = np.array([0, 1, 0, 1, 1, 0], dtype=bool)
        assert np.array_equal(_hysteresis(signal, 1), signal)

    def test_handles_an_empty_signal(self):
        assert len(_hysteresis(np.zeros(0, dtype=bool), 3)) == 0


class TestDetectGrasp:
    def _episode(self, count=40):
        """Fingers close at frame 10 and the object then moves with the hand."""
        landmarks = np.zeros((count, 21, 3))
        object_positions = np.zeros((count, 3))
        for i in range(count):
            carrying = i >= 10
            travel = np.array([0.01 * (i - 10), 0.0, 0.0]) if carrying else np.zeros(3)
            gap = 0.030 if carrying else 0.090
            centre = np.array([0.0, 0.0, 0.80]) + travel
            landmarks[i] = make_hand(
                centre + np.array([gap / 2, 0.0, 0.0]),
                centre - np.array([gap / 2, 0.0, 0.0]),
            )
            object_positions[i] = centre if carrying else np.array([0.0, 0.0, 0.80])
        return landmarks, object_positions

    def test_finds_the_grasp_onset(self):
        landmarks, object_positions = self._episode()
        valid = np.ones(len(landmarks), dtype=bool)
        closed, diagnostics = detect_grasp(
            landmarks, valid, object_positions, valid, fps=30.0,
            cfg={"close_distance_m": 0.045, "open_distance_m": 0.065,
                 "contact_distance_m": 0.05, "motion_coupling_threshold": 0.6,
                 "min_object_speed_m_s": 0.02, "min_state_frames": 3},
        )
        assert closed[:9].sum() == 0
        assert closed[15:].all()
        assert diagnostics["grasp_onset_frame"] == pytest.approx(10, abs=2)

    def test_a_pinch_in_mid_air_is_not_a_grasp(self):
        # Fingers close, but no object is near and nothing moves with them.
        count = 30
        landmarks = np.zeros((count, 21, 3))
        for i in range(count):
            landmarks[i] = make_hand(
                np.array([0.015, 0.0, 1.20]), np.array([-0.015, 0.0, 1.20])
            )
        object_positions = np.tile(np.array([0.0, 0.0, 0.75]), (count, 1))
        valid = np.ones(count, dtype=bool)
        closed, _ = detect_grasp(
            landmarks, valid, object_positions, valid, fps=30.0,
            cfg={"close_distance_m": 0.045, "open_distance_m": 0.065,
                 "contact_distance_m": 0.05, "motion_coupling_threshold": 0.6,
                 "min_object_speed_m_s": 0.02, "min_state_frames": 3},
        )
        assert not closed.any(), "a pinch far from the object must not read as a grasp"

    def test_detects_a_glass_sized_grasp(self):
        """The widths measured by WiLoR on the real clips.

        Open hand 12.2 cm, wrapped on the glass 5.8 cm. A fixed
        close_distance of 4.5 cm sits below the wrapped width, so the trigger
        would never latch and the episode would report no grasp at all.
        """
        count = 60
        landmarks = np.zeros((count, 21, 3))
        object_positions = np.zeros((count, 3))
        for i in range(count):
            carrying = i >= 20
            gap = 0.058 if carrying else 0.122
            travel = np.array([0.008 * (i - 20), 0.0, 0.0]) if carrying else np.zeros(3)
            centre = np.array([0.0, 0.0, 0.80]) + travel
            landmarks[i] = make_hand(
                centre + np.array([gap / 2, 0.0, 0.0]),
                centre - np.array([gap / 2, 0.0, 0.0]),
            )
            object_positions[i] = centre if carrying else np.array([0.0, 0.0, 0.80])

        valid = np.ones(count, dtype=bool)
        closed, diagnostics = detect_grasp(
            landmarks, valid, object_positions, valid, fps=60.0,
            cfg={"auto_threshold": True, "close_distance_m": 0.045,
                 "open_distance_m": 0.065, "contact_distance_m": 0.05,
                 "motion_coupling_threshold": 0.6, "min_object_speed_m_s": 0.02,
                 "min_state_frames": 3},
        )
        assert diagnostics["threshold_source"] == "auto_percentile"
        assert diagnostics["closed_frames"] > 0, "a 5.8 cm wrapped grasp was missed"
        assert closed[30:].all()
        assert not closed[:15].any()

    def test_falls_back_to_config_when_the_hand_never_changes_shape(self):
        count = 40
        landmarks = np.stack(
            [make_hand(np.array([0.05, 0, 0.8]), np.array([-0.05, 0, 0.8]))] * count
        )
        object_positions = np.tile(np.array([0.0, 0.0, 0.80]), (count, 1))
        valid = np.ones(count, dtype=bool)
        _, diagnostics = detect_grasp(
            landmarks, valid, object_positions, valid, fps=60.0,
            cfg={"auto_threshold": True, "close_distance_m": 0.070,
                 "open_distance_m": 0.095},
        )
        assert diagnostics["threshold_source"] == "config"
        assert diagnostics["close_distance_m"] == pytest.approx(0.070)

    def test_reports_no_transitions_when_the_hand_never_closes(self):
        count = 20
        landmarks = np.stack(
            [make_hand(np.array([0.05, 0, 0.8]), np.array([-0.05, 0, 0.8]))] * count
        )
        object_positions = np.tile(np.array([0.0, 0.0, 0.80]), (count, 1))
        valid = np.ones(count, dtype=bool)
        closed, diagnostics = detect_grasp(
            landmarks, valid, object_positions, valid, fps=30.0,
            cfg={"close_distance_m": 0.045, "open_distance_m": 0.065},
        )
        assert diagnostics["closed_frames"] == 0
        assert not closed.any()


class TestGripperPoseMapping:
    def test_origin_sits_behind_the_fingertip_midpoint(self):
        # The plan's rule: do not put the gripper where the wrist was, and pull
        # the origin back because the jaws grip behind the fingertips.
        hand = make_hand(np.array([0.02, 0.05, 0.80]), np.array([-0.02, 0.05, 0.80]))
        offset = 0.02
        poses, _ = gripper_poses_from_hand(hand[None], np.array([True]), offset)

        midpoint = 0.5 * (hand[hand_backend.THUMB_TIP] + hand[hand_backend.INDEX_TIP])
        approach = hand_backend.approach_direction(hand)
        assert np.allclose(poses[0][:3, 3], midpoint - approach * offset, atol=1e-9)

    def test_origin_is_not_the_wrist(self):
        hand = make_hand(np.array([0.02, 0.05, 0.80]), np.array([-0.02, 0.05, 0.80]))
        poses, _ = gripper_poses_from_hand(hand[None], np.array([True]), 0.02)
        assert not np.allclose(poses[0][:3, 3], hand[hand_backend.WRIST], atol=1e-3)

    def test_closing_axis_lies_along_thumb_to_index(self):
        hand = make_hand(np.array([0.03, 0.05, 0.80]), np.array([-0.03, 0.05, 0.80]))
        poses, _ = gripper_poses_from_hand(hand[None], np.array([True]), 0.0)
        axis = hand_backend.grasp_axis(hand)
        # The jaws close along a line, so either sign is correct. The axis is
        # Gram-Schmidt orthogonalized against the approach direction, so it
        # lands near the thumb-index line rather than exactly on it.
        alignment = abs(float(np.dot(poses[0][:3, 0], axis)))
        assert alignment > 0.99, f"closing axis is {np.degrees(np.arccos(alignment)):.1f} deg off"

    def test_width_is_the_thumb_index_distance(self):
        hand = make_hand(np.array([0.035, 0.05, 0.80]), np.array([-0.035, 0.05, 0.80]))
        _, widths = gripper_poses_from_hand(hand[None], np.array([True]), 0.0)
        assert widths[0] == pytest.approx(0.07)

    def test_frames_stay_right_handed(self):
        rng = np.random.default_rng(0)
        hands = np.stack(
            [
                make_hand(rng.normal(0, 0.05, 3), rng.normal(0, 0.05, 3))
                for _ in range(20)
            ]
        )
        poses, _ = gripper_poses_from_hand(hands, np.ones(20, dtype=bool), 0.02)
        for pose in poses:
            assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0, abs=1e-6)

    def test_invalid_frames_are_left_as_identity(self):
        hands = np.zeros((3, 21, 3))
        poses, widths = gripper_poses_from_hand(hands, np.array([False, False, False]), 0.02)
        assert np.allclose(poses, np.eye(4))
        assert np.allclose(widths, 0.0)


class TestResample:
    def test_hits_the_requested_control_rate(self):
        count = 61
        poses = np.repeat(np.eye(4)[None], count, axis=0)
        poses[:, 0, 3] = np.linspace(0, 2, count)
        widths = np.linspace(0.02, 0.08, count)
        closed = np.zeros(count, dtype=bool)

        out_poses, out_widths, out_closed, times = resample(
            poses, widths, closed, source_fps=60.0, target_hz=15.0
        )
        # One second of 60 fps input at 15 Hz gives 16 samples, ends included.
        assert len(out_poses) == 16
        assert times[-1] == pytest.approx(1.0, abs=1e-6)
        assert len(out_widths) == len(out_poses) == len(out_closed)

    def test_translations_interpolate_linearly(self):
        count = 61
        poses = np.repeat(np.eye(4)[None], count, axis=0)
        poses[:, 0, 3] = np.linspace(0, 2, count)
        out_poses, _, _, times = resample(
            poses, np.zeros(count), np.zeros(count, dtype=bool), 60.0, 15.0
        )
        assert np.allclose(out_poses[:, 0, 3], 2.0 * times, atol=1e-6)

    def test_grasp_flag_stays_binary(self):
        count = 61
        closed = np.zeros(count, dtype=bool)
        closed[30:] = True
        poses = np.repeat(np.eye(4)[None], count, axis=0)
        _, _, out_closed, _ = resample(poses, np.zeros(count), closed, 60.0, 15.0)
        assert set(np.unique(out_closed)).issubset({True, False})

    def test_short_input_is_returned_unchanged(self):
        poses = np.repeat(np.eye(4)[None], 1, axis=0)
        out, _, _, _ = resample(poses, np.zeros(1), np.zeros(1, dtype=bool), 60.0, 15.0)
        assert len(out) == 1


class TestGripperSpec:
    def test_clamps_width_to_the_limits(self):
        spec = GripperSpec(0.085, 0.0, 0.055, 0.06, "config")
        assert spec.clamp_width(0.2) == pytest.approx(0.085)
        assert spec.clamp_width(-0.1) == pytest.approx(0.0)
        assert spec.clamp_width(0.04) == pytest.approx(0.04)

    def test_jaw_boxes_open_with_the_width(self):
        spec = GripperSpec(0.085, 0.0, 0.055, 0.06, "config")
        narrow = jaw_boxes(spec, 0.02)
        wide = jaw_boxes(spec, 0.08)
        # Box 1 is a jaw; its x offset tracks half the opening.
        assert abs(wide[1][0][0]) > abs(narrow[1][0][0])

    def test_jaws_straddle_the_origin(self):
        spec = GripperSpec(0.085, 0.0, 0.055, 0.06, "config")
        boxes = jaw_boxes(spec, 0.06)
        assert boxes[1][0][0] * boxes[2][0][0] < 0
