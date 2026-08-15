"""The accuracy evaluator's own arithmetic.

Frame-rate mapping and rotation error are the two places a scoring script
quietly lies: an off-by-one in the mapping inflates every error, and a
rotation metric that ignores the geodesic reports nonsense near 180 degrees.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wristview.evaluate import Metric, _gt_index, _rotation_error_deg, _summarize


class TestFrameMapping:
    def test_matched_rates_are_identity(self):
        idx = _gt_index(np.arange(10), run_fps=30.0, gt_fps=30.0, gt_count=10)
        assert np.array_equal(idx, np.arange(10))

    def test_run_at_20_against_truth_at_30(self):
        # Stage 0 resamples a 30 fps clip to 20 fps, so run frame i sits at
        # time i/20 and the truth frame is round(i * 1.5).
        idx = _gt_index(np.arange(5), run_fps=20.0, gt_fps=30.0, gt_count=100)
        assert np.array_equal(idx, [0, 2, 3, 5, 6])

    def test_clamps_to_available_truth(self):
        idx = _gt_index(np.arange(10), run_fps=10.0, gt_fps=30.0, gt_count=5)
        assert idx.max() == 4

    def test_never_negative(self):
        assert _gt_index(np.arange(3), 20.0, 30.0, 10).min() >= 0


class TestRotationError:
    def test_identical_rotations_are_zero(self):
        r = Rotation.random(5, random_state=0).as_matrix()
        # arccos is ill-conditioned near identity, so a perfect match still
        # carries about 1e-6 degrees of float noise. Everything reported is
        # to two decimal places, so this tolerance is generous.
        assert np.allclose(_rotation_error_deg(r, r), 0.0, atol=1e-4)

    def test_known_angle(self):
        a = np.repeat(np.eye(3)[None], 1, axis=0)
        b = Rotation.from_euler("z", 30, degrees=True).as_matrix()[None]
        assert _rotation_error_deg(a, b)[0] == pytest.approx(30.0, abs=1e-6)

    def test_is_symmetric(self):
        a = Rotation.random(4, random_state=1).as_matrix()
        b = Rotation.random(4, random_state=2).as_matrix()
        assert np.allclose(_rotation_error_deg(a, b), _rotation_error_deg(b, a), atol=1e-9)

    def test_bounded_at_180(self):
        a = np.repeat(np.eye(3)[None], 1, axis=0)
        b = Rotation.from_euler("x", 180, degrees=True).as_matrix()[None]
        assert _rotation_error_deg(a, b)[0] == pytest.approx(180.0, abs=1e-4)

    def test_stays_finite_for_near_180(self):
        # A naive arccos without clamping returns nan here.
        a = np.repeat(np.eye(3)[None], 1, axis=0)
        b = Rotation.from_euler("x", 179.9999, degrees=True).as_matrix()[None]
        assert np.isfinite(_rotation_error_deg(a, b)).all()


class TestSummarize:
    def test_reports_median_p90_and_worst(self):
        m = _summarize("x", np.arange(101, dtype=float), "cm", "estimated")
        assert m.median == pytest.approx(50)
        assert m.p90 == pytest.approx(90)
        assert m.worst == pytest.approx(100)
        assert m.count == 101

    def test_drops_non_finite(self):
        m = _summarize("x", np.array([1.0, np.nan, 3.0, np.inf]), "cm", "estimated")
        assert m.count == 2

    def test_empty_is_nan_not_a_crash(self):
        m = _summarize("x", np.array([]), "cm", "estimated")
        assert np.isnan(m.median)
        assert m.count == 0

    def test_row_records_the_source(self):
        assert "derived" in Metric("x", "cm", "derived", 1, 2, 3, 4).row()
