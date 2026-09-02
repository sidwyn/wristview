"""The lens label is the only focal length an iPhone clip from Blackmagic Camera carries.

ffprobe surfaces no numeric EXIF focal tag on that footage, so the 0.5x and 1x
lenses were indistinguishable and both fell back to the same guess. These tests
pin the label read, the trust rule that follows from it, and the per-model
camera prior that an ultra-wide reconstruction needs.
"""

from __future__ import annotations

import pytest

from wristview.backends.sfm import CAMERA_MODEL_PARAMS, camera_params_prior
from wristview.camera import estimate_from_exif
from wristview.videoio import focal_35mm_from_tags


def test_numeric_tag_wins_and_is_trusted():
    focal, source = focal_35mm_from_tags(
        {"focal_length_in_35mm_film": "27", "com.apple.quicktime.model": "iPhone 16 Pro 24mm"}
    )
    assert focal == 27.0
    assert source == "exif_focal35"


@pytest.mark.parametrize(
    "tag",
    ["com.blackmagic-design.camera.lensType", "com.apple.quicktime.model"],
)
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("iPhone 16 Pro 13mm", 13.0),
        ("Apple iPhone 16 Pro 24mm", 24.0),
        ("iPhone 16 Pro 77 mm", 77.0),
        ("iPhone 16 Pro 6.9mm", 6.9),
    ],
)
def test_lens_label_is_parsed(tag, value, expected):
    focal, source = focal_35mm_from_tags({tag: value})
    assert focal == expected
    assert source == "exif_lens_label"


def test_label_without_millimetres_is_not_a_focal_length():
    assert focal_35mm_from_tags({"com.apple.quicktime.model": "Apple iPhone 15 Pro"}) == (None, "none")


def test_no_tags_at_all():
    assert focal_35mm_from_tags({}) == (None, "none")


def test_the_two_sept02_lenses_separate():
    """The whole point: 13mm and 24mm must not land on the same focal length."""
    wide, _ = focal_35mm_from_tags({"com.blackmagic-design.camera.lensType": "iPhone 16 Pro 13mm"})
    main, _ = focal_35mm_from_tags({"com.blackmagic-design.camera.lensType": "iPhone 16 Pro 24mm"})
    ultra = estimate_from_exif(1920, 1080, wide, 0.85, "exif_lens_label")[0]
    normal = estimate_from_exif(1920, 1080, main, 0.85, "exif_lens_label")[0]

    assert ultra.fx == pytest.approx(693.3, abs=0.5)
    assert normal.fx == pytest.approx(1280.0, abs=0.5)
    # The guess put both at 1632 px. That is 2.35x wrong for the ultra-wide.
    assert ultra.horizontal_fov_deg == pytest.approx(108.4, abs=0.5)
    assert normal.horizontal_fov_deg == pytest.approx(73.7, abs=0.5)


def test_source_label_passes_through():
    _, source = estimate_from_exif(1920, 1080, 13.0, 0.85, "exif_lens_label")
    assert source == "exif_lens_label"


def test_missing_focal_still_reports_the_guess():
    intr, source = estimate_from_exif(1920, 1080, None, 0.85, "exif_lens_label")
    assert source == "fallback_guess"
    assert intr.fx == pytest.approx(1632.0)


def test_prior_length_matches_every_model():
    for model, names in CAMERA_MODEL_PARAMS.items():
        prior = camera_params_prior(model, 693.3, 693.3, 960.0, 540.0)
        assert len(prior.split(",")) == len(names), model


def test_fisheye_prior_carries_four_distortion_zeros():
    prior = camera_params_prior("OPENCV_FISHEYE", 693.3, 693.3, 960.0, 540.0)
    assert prior.split(",") == ["693.3", "693.3", "960.0", "540.0", "0.0", "0.0", "0.0", "0.0"]


def test_simple_radial_prior_is_unchanged():
    """The existing SIMPLE_RADIAL path must produce exactly what it did before."""
    assert camera_params_prior("SIMPLE_RADIAL", 1632.0, 1632.0, 960.0, 540.0) == "1632.0,960.0,540.0,0.0"


def test_unknown_model_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="unknown COLMAP camera model"):
        camera_params_prior("SUPER_FISHEYE_9000", 1.0, 1.0, 1.0, 1.0)
