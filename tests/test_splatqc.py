"""The gate that would have caught the real26 splat.

That splat held 10455 Gaussians over 298 views and rendered its own training
frames at 15.7 to 17.3 dB. Every existing check passed it: alpha coverage read
100 per cent because Gaussians were drawn at every pixel, and they were the
wrong Gaussians.
"""

from __future__ import annotations

from wristview.splatqc import evaluate


def test_the_real26_splat_fails_on_both_counts():
    report = evaluate([17.29, 15.71, 16.86], gaussian_count=10455, view_count=298)
    assert report["passed"] is False
    assert len(report["failures"]) == 2
    assert report["gaussians_per_view"] < 50


def test_a_healthy_splat_passes():
    """real06 figures: 2.73 M Gaussians over 327 views, about 35 dB."""
    report = evaluate([35.2, 34.8, 35.6], gaussian_count=2_734_543, view_count=327)
    assert report["passed"] is True
    assert report["failures"] == []


def test_a_thin_splat_fails_even_with_good_psnr():
    """A splat can score well on a few easy views and still be too thin."""
    report = evaluate([30.0, 31.0], gaussian_count=9000, view_count=300)
    assert report["passed"] is False
    assert any("too thin" in f for f in report["failures"])


def test_low_psnr_fails_even_with_many_gaussians():
    report = evaluate([18.0, 17.5], gaussian_count=3_000_000, view_count=300)
    assert report["passed"] is False
    assert any("training views" in f for f in report["failures"])


def test_no_views_reports_nothing_rather_than_passing():
    report = evaluate([], gaussian_count=1_000_000, view_count=300)
    assert report["passed"] is None
    assert report["views_tested"] == 0
