"""Selecting the tracked hand by label rather than by size.

The largest-box selector is correct only when one hand acts. Session 6 shot
three two-handed clips, and in all three the selection crossed between hands as
each came nearer the camera, which produced a smooth trajectory describing
neither hand. Following a named side instead makes those clips usable as
single-arm episodes.

The cases that matter here are the failures, not the happy path: what happens
when the wanted hand is missing, and when two detections both claim it.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristview.backends.hands import NUM_LANDMARKS, HandFrame, WiLoRHands


def _detection(is_right: float, box: tuple[float, float, float, float]):
    """A WiLoR detection, cut down to the fields the selector reads."""
    return {
        "is_right": is_right,
        "hand_bbox": list(box),
        "wilor_preds": {
            "pred_keypoints_3d": np.zeros((1, NUM_LANDMARKS, 3)),
            "pred_cam_t_full": np.array([[0.0, 0.0, 0.5]]),
            "pred_keypoints_2d": np.zeros((1, NUM_LANDMARKS, 2)),
            "scaled_focal_length": 1000.0,
        },
    }


class _FakePipeline:
    def __init__(self, detections):
        self.detections = detections

    def predict(self, _rgb):
        return self.detections


def _estimator(detections, select: str) -> WiLoRHands:
    """A WiLoRHands whose model is a stub, so the selector runs alone."""
    estimator = WiLoRHands.__new__(WiLoRHands)
    estimator.device = "cpu"
    estimator.select = select
    estimator._pipeline = _FakePipeline(detections)
    return estimator


SMALL_RIGHT = _detection(1.0, (0, 0, 40, 40))
BIG_LEFT = _detection(0.0, (0, 0, 200, 200))


def _run(detections, select, intrinsics):
    return _estimator(detections, select).process(
        np.zeros((480, 640, 3), dtype=np.uint8), intrinsics
    )


@pytest.fixture
def intrinsics():
    from wristview.camera import Intrinsics

    return Intrinsics(fx=1000.0, fy=1000.0, cx=320.0, cy=240.0, width=640, height=480)


def test_largest_picks_the_bigger_box_whatever_the_side(intrinsics):
    frame = _run([SMALL_RIGHT, BIG_LEFT], "largest", intrinsics)
    assert frame.detected
    assert frame.handedness == "Left"
    assert frame.selection == "largest"


def test_label_picks_the_named_side_even_when_it_is_smaller(intrinsics):
    """The whole point: size must not decide once a side is named."""
    frame = _run([SMALL_RIGHT, BIG_LEFT], "Right", intrinsics)
    assert frame.detected
    assert frame.handedness == "Right"
    assert frame.selection == "label"


def test_missing_side_is_a_gap_not_the_other_hand(intrinsics):
    """Substituting the visible hand is the defect this exists to prevent."""
    frame = _run([BIG_LEFT], "Right", intrinsics)
    assert not frame.detected
    assert frame.selection == "absent"
    assert frame.hands_in_frame == 1


def test_two_detections_claiming_one_side_is_a_gap(intrinsics):
    """One of them is mislabelled and nothing here says which."""
    both_right = [_detection(1.0, (0, 0, 40, 40)), _detection(1.0, (0, 0, 200, 200))]
    frame = _run(both_right, "Right", intrinsics)
    assert not frame.detected
    assert frame.selection == "ambiguous"
    assert frame.hands_in_frame == 2


def test_no_detections_at_all_is_reported_as_such(intrinsics):
    frame = _run([], "Right", intrinsics)
    assert not frame.detected
    assert frame.selection == ""


def test_an_unknown_side_is_refused_at_construction():
    with pytest.raises(ValueError, match="largest, Left or Right"):
        WiLoRHands(select="either")


def test_hands_in_frame_survives_a_rejected_selection(intrinsics):
    """The identity guard reads this count, so a gap must not zero it."""
    frame = _run([SMALL_RIGHT, BIG_LEFT], "Left", intrinsics)
    assert frame.hands_in_frame == 2
    frame = _run([BIG_LEFT], "Right", intrinsics)
    assert frame.hands_in_frame == 1


def test_default_stays_largest_so_existing_runs_are_unchanged(intrinsics):
    estimator = WiLoRHands.__new__(WiLoRHands)
    estimator._pipeline = _FakePipeline([SMALL_RIGHT, BIG_LEFT])
    estimator.device = "cpu"
    # No select attribute set, as an older pickle or a fresh default would be.
    estimator.select = "largest"
    frame = estimator.process(np.zeros((480, 640, 3), dtype=np.uint8), intrinsics)
    assert frame.handedness == "Left"


def test_handframe_defaults_leave_selection_empty():
    frame = HandFrame(
        detected=False,
        landmarks_px=np.zeros((NUM_LANDMARKS, 2)),
        landmarks_cam=np.zeros((NUM_LANDMARKS, 3)),
        confidence=0.0,
        handedness="",
    )
    assert frame.selection == ""
    assert frame.hands_in_frame == 0
