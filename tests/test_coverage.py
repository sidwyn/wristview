"""Would this have caught session 6 before the render was ever paid for?

Session 6's scan translated 2.96 cm at 43 cm from its subject, stayed inside a
2.7 cm height band, and every one of its rendered frames sat below the lowest
scan view. The reconstruction reported 327 of 327 registered at 1.3565 px and
the splat reached 30.21 dB, so every existing gate passed. These tests use
those exact numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristview.coverage import (
    MAX_VIEWPOINT_GAP_M,
    MIN_SCAN_BASELINE_M,
    render_viewpoint_coverage,
    scan_geometry,
)

# A desk plane with a convenient normal, so heights read as the z coordinate.
NORMAL = np.array([0.0, 0.0, 1.0])
OFFSET = 0.0


def _session_six_scan(count: int = 327) -> np.ndarray:
    """327 views inside a 3 cm ball, 43 cm above the desk. The real capture."""
    rng = np.random.default_rng(6)
    centres = rng.normal(0, 0.008, (count, 3))
    centres[:, 2] = 0.411 + rng.random(count) * 0.028
    return centres


def _obedient_scan(count: int = 300) -> np.ndarray:
    """What CAPTURE-SOP.md actually asks for: three passes at three heights."""
    rng = np.random.default_rng(7)
    out = []
    for height, radius, n in ((0.60, 0.35, count // 3),      # wide orbit
                              (0.40, 0.20, count // 3),      # close pass
                              (0.13, 0.10, count // 3)):     # contact pass
        angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
        out.append(np.stack([radius * np.cos(angle), radius * np.sin(angle),
                             np.full(n, height) + rng.normal(0, 0.01, n)], axis=1))
    return np.concatenate(out)


def _wrist_trajectory(count: int = 186) -> np.ndarray:
    """The wrist camera: 20 to 38 cm up, sweeping 40 cm across the desk."""
    t = np.linspace(0, 1, count)
    return np.stack([
        0.30 * np.sin(2 * np.pi * t),
        0.20 * np.cos(2 * np.pi * t),
        0.198 + 0.187 * (0.5 + 0.5 * np.sin(3 * np.pi * t)),
    ], axis=1)


# --------------------------------------------------------------------------
# the scan on its own
# --------------------------------------------------------------------------


def test_session_six_scan_is_rejected_for_having_no_parallax():
    report = scan_geometry(_session_six_scan(), NORMAL, OFFSET)
    assert report["passed"] is False
    assert report["max_baseline_m"] < 0.06
    assert any("translated" in f for f in report["failures"])
    assert any("height band" in f for f in report["failures"])


def test_an_obedient_scan_passes():
    report = scan_geometry(_obedient_scan(), NORMAL, OFFSET)
    assert report["passed"] is True, report["failures"]
    assert report["max_baseline_m"] > MIN_SCAN_BASELINE_M
    assert report["height_span_m"] > 0.4


def test_a_rotation_only_scan_is_caught_however_many_views_it_has():
    """Count is not evidence. 5000 views from one spot are one view."""
    rng = np.random.default_rng(1)
    centres = rng.normal(0, 0.002, (5000, 3))
    centres[:, 2] += 0.42
    report = scan_geometry(centres, NORMAL, OFFSET)
    assert report["views"] == 5000
    assert report["passed"] is False


def test_baseline_is_reported_against_subject_distance():
    """The ratio is the readable number; a real orbit sits near 1."""
    report = scan_geometry(_obedient_scan(), NORMAL, OFFSET)
    assert "baseline_over_distance" in report
    assert report["baseline_over_distance"] > 0.5

    poor = scan_geometry(_session_six_scan(), NORMAL, OFFSET)
    assert poor["baseline_over_distance"] < 0.2


def test_view_direction_spread_is_reported_when_directions_are_given():
    centres = _obedient_scan()
    directions = -centres / np.linalg.norm(centres, axis=1, keepdims=True)
    report = scan_geometry(centres, NORMAL, OFFSET, view_directions=directions)
    assert report["view_direction_spread_deg"] > 30


def test_too_few_views_says_so_rather_than_passing():
    report = scan_geometry(np.zeros((1, 3)), NORMAL, OFFSET)
    assert report["passed"] is None


# --------------------------------------------------------------------------
# the render against the scan
# --------------------------------------------------------------------------


def test_every_session_six_frame_is_below_the_scan_floor():
    """The measured result: 100 per cent, which is what shipped."""
    report = render_viewpoint_coverage(
        _session_six_scan(), _wrist_trajectory(), NORMAL, OFFSET
    )
    assert report["passed"] is False
    assert report["fraction_below_scan_floor"] == 1.0
    assert any("below the scan" in f for f in report["failures"])


def test_an_obedient_scan_covers_the_same_trajectory():
    report = render_viewpoint_coverage(
        _obedient_scan(), _wrist_trajectory(), NORMAL, OFFSET
    )
    assert report["passed"] is True, report["failures"]
    assert report["nearest_scan_view_median_m"] < MAX_VIEWPOINT_GAP_M
    assert report["fraction_below_scan_floor"] <= 0.10


def test_the_gap_is_reported_per_frame_not_just_on_average():
    report = render_viewpoint_coverage(
        _session_six_scan(), _wrist_trajectory(), NORMAL, OFFSET
    )
    assert report["nearest_scan_view_max_m"] >= report["nearest_scan_view_p90_m"]
    assert report["nearest_scan_view_p90_m"] >= report["nearest_scan_view_median_m"]


def test_a_few_stray_frames_do_not_fail_the_clip():
    """A reach may briefly dip below the floor; that is not a broken capture."""
    scan = _obedient_scan()
    render = _wrist_trajectory()
    render[:5, 2] = 0.05          # five frames under everything
    report = render_viewpoint_coverage(scan, render, NORMAL, OFFSET)
    assert report["fraction_below_scan_floor"] < 0.10
    assert report["passed"] is True


def test_invalid_frames_are_excluded():
    scan = _obedient_scan()
    render = _wrist_trajectory()
    valid = np.ones(len(render), dtype=bool)
    valid[:50] = False
    report = render_viewpoint_coverage(scan, render, NORMAL, OFFSET, render_valid=valid)
    assert report["frames"] == len(render) - 50


def test_nothing_to_compare_reports_nothing():
    report = render_viewpoint_coverage(
        np.zeros((0, 3)), _wrist_trajectory(), NORMAL, OFFSET
    )
    assert report["passed"] is None
