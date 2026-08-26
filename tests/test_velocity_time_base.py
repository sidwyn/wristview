"""The velocity gate must read a clock, not a frame index.

Stage 0 drops blurred frames. Kept frames are then unevenly spaced in time. A
gate that divides an index gap by a scalar fps overstates the speed wherever
frames were removed.

Session real26 shows the cost. Two steps spanned 1.25 s and 1.55 s of real
time. Read as one frame apart at 17 fps they implied about 4.8 m/s. The true
speeds were 0.23 m/s and 0.12 m/s. The gate rejected a good clip.

A gate that stops rejecting good clips must still reject bad ones. These tests
check both directions.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wristview.qc import HAND_ANGULAR_RATE_MAX_DEG_S, hand_velocity_outliers

NUM_LANDMARKS = 21


def _hand(rotation: np.ndarray) -> np.ndarray:
    """A hand whose palm carries the given rotation.

    `palm_rotations` builds its frame from landmarks 0, 5, 9 and 17.
    """
    base = np.zeros((NUM_LANDMARKS, 3))
    base[0] = [0.0, 0.0, 0.0]
    base[5] = [0.08, 0.0, 0.0]
    base[9] = [0.08, 0.02, 0.0]
    base[17] = [0.06, 0.06, 0.0]
    return base @ rotation.T


def _clip(angles_deg, times_s):
    """Build a hand sequence rotating about z by the given angles."""
    frames = np.stack([
        _hand(Rotation.from_euler("z", a, degrees=True).as_matrix()) for a in angles_deg
    ])
    return frames, np.ones(len(frames), dtype=bool), np.asarray(times_s, dtype=float)


def test_a_slow_turn_across_a_dropped_frame_gap_is_not_flagged():
    """The real26 case. A wide time gap is not a fast hand."""
    # 60 degrees of turn, 1.25 s apart: 48 deg/s, far under the limit. Read as
    # one frame at 17 fps it becomes 1020 deg/s, over the limit.
    angles = [0.0, 60.0, 120.0]
    times = [0.0, 1.25, 2.50]
    landmarks, valid, t = _clip(angles, times)

    fixed = hand_velocity_outliers(landmarks, valid, fps=17.0, frame_times_s=t)
    assert fixed["frames_flagged"] == 0, fixed
    assert fixed["max_rate_deg_s"] < HAND_ANGULAR_RATE_MAX_DEG_S


def test_the_old_frame_index_reading_flags_that_same_slow_turn():
    """Proof the bug was real: same data, index-based time, now flagged."""
    angles = [0.0, 60.0, 120.0]
    times = [0.0, 1.25, 2.50]
    landmarks, valid, _ = _clip(angles, times)

    old = hand_velocity_outliers(landmarks, valid, fps=17.0)
    assert old["max_rate_deg_s"] > HAND_ANGULAR_RATE_MAX_DEG_S
    assert old["frames_flagged"] > 0


def test_the_gate_still_rejects_a_genuinely_impossible_turn():
    """A human wrist reaches about 800 deg/s. This clip does 2400.

    The session that motivated the gate reported 2400 deg/s. Reproduce that
    with honest, evenly spaced times so nothing about the fix can excuse it.
    """
    step = 2400.0 / 20.0            # 120 degrees between frames at 20 fps
    angles = [0.0, step, 2 * step, 3 * step, 4 * step]
    times = np.arange(5) / 20.0
    landmarks, valid, t = _clip(angles, times)

    fixed = hand_velocity_outliers(landmarks, valid, fps=20.0, frame_times_s=t)
    assert fixed["max_rate_deg_s"] > 2000, fixed
    assert fixed["frames_flagged"] > 0
    assert fixed["longest_run"] >= 1


def test_an_isolated_spike_is_still_one_frame_not_two():
    """One displaced frame gives two steps, out and back. Blame one frame."""
    angles = [0.0, 5.0, 200.0, 10.0, 15.0]
    times = np.arange(5) / 20.0
    landmarks, valid, t = _clip(angles, times)

    fixed = hand_velocity_outliers(landmarks, valid, fps=20.0, frame_times_s=t)
    assert fixed["frames_flagged"] == 1, fixed["flagged_frames"]
    assert fixed["longest_run"] == 1


def test_uneven_times_do_not_hide_a_fast_turn():
    """An uneven gap must not excuse motion that is fast across that gap.

    The steps here are 0.05 s and 0.07 s, not equal, and both turn at
    1200 deg/s. The gate must flag them on the true intervals.
    """
    angles = [0.0, 60.0, 144.0]
    times = [0.0, 0.05, 0.12]
    landmarks, valid, t = _clip(angles, times)
    fixed = hand_velocity_outliers(landmarks, valid, fps=20.0, frame_times_s=t)
    assert fixed["max_rate_deg_s"] > HAND_ANGULAR_RATE_MAX_DEG_S
    assert fixed["frames_flagged"] > 0


def test_a_turn_beyond_180_degrees_between_frames_aliases_downward():
    """A known limit of the gate. Record it rather than let it surprise.

    `Rotation.magnitude` returns the geodesic angle, which never exceeds 180
    degrees. A larger turn between two frames therefore reads as its shorter
    equivalent. The gate under-reports such motion; it never over-reports it.

    A 300 degree turn in 0.05 s is 6000 deg/s. The gate sees 60 degrees, which
    is 1200 deg/s. Still flagged here, but the number is not the true rate.
    """
    angles = [0.0, 300.0, 600.0]
    times = [0.0, 0.05, 0.10]
    landmarks, valid, t = _clip(angles, times)
    fixed = hand_velocity_outliers(landmarks, valid, fps=20.0, frame_times_s=t)
    assert fixed["max_rate_deg_s"] == pytest.approx(1200.0, rel=0.02)
    assert fixed["max_rate_deg_s"] < 6000.0


def test_mismatched_times_raise_rather_than_guess():
    landmarks, valid, _ = _clip([0.0, 10.0, 20.0], [0.0, 0.05, 0.10])
    with pytest.raises(ValueError, match="entries"):
        hand_velocity_outliers(landmarks, valid, fps=20.0, frame_times_s=np.arange(2))


def test_without_times_it_keeps_the_old_behaviour():
    """Callers that have no clock still get a usable answer."""
    angles = [0.0, 5.0, 10.0]
    landmarks, valid, _ = _clip(angles, [0, 0.05, 0.10])
    report = hand_velocity_outliers(landmarks, valid, fps=20.0)
    assert report["frames_checked"] == 3
    assert report["frames_flagged"] == 0
