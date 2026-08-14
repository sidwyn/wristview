"""Camera intrinsics and the deployment camera model."""

import numpy as np
import pytest

from wristview.camera import Intrinsics, apply_fisheye, estimate_from_exif, fisheye_maps


@pytest.fixture
def intrinsics() -> Intrinsics:
    return Intrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)


def test_from_fov_gives_back_that_fov():
    for fov in (45.0, 60.0, 90.0, 120.0):
        intrinsics = Intrinsics.from_fov(800, 600, fov)
        assert intrinsics.horizontal_fov_deg == pytest.approx(fov, abs=1e-9)


def test_project_unproject_roundtrip(intrinsics):
    rng = np.random.default_rng(0)
    points = np.c_[
        rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50), rng.uniform(0.5, 5.0, 50)
    ]
    pixels, depth = intrinsics.project(points)
    assert np.allclose(intrinsics.unproject(pixels, depth), points, atol=1e-9)


def test_principal_point_projects_to_the_centre(intrinsics):
    pixels, _ = intrinsics.project(np.array([[0.0, 0.0, 2.0]]))
    assert np.allclose(pixels[0], [intrinsics.cx, intrinsics.cy])


def test_project_handles_zero_depth_without_dividing_by_zero(intrinsics):
    pixels, _ = intrinsics.project(np.array([[1.0, 1.0, 0.0]]))
    assert np.isfinite(pixels).all()


def test_scaled_preserves_field_of_view(intrinsics):
    half = intrinsics.scaled(320, 240)
    assert half.horizontal_fov_deg == pytest.approx(intrinsics.horizontal_fov_deg, abs=1e-9)
    assert half.fx == pytest.approx(intrinsics.fx / 2)
    assert half.cx == pytest.approx(intrinsics.cx / 2)


def test_dict_roundtrip(intrinsics):
    assert Intrinsics.from_dict(intrinsics.to_dict()) == intrinsics


def test_matrix_layout(intrinsics):
    matrix = intrinsics.matrix
    assert matrix[0, 0] == intrinsics.fx
    assert matrix[1, 1] == intrinsics.fy
    assert matrix[0, 2] == intrinsics.cx
    assert matrix[2, 2] == 1.0


class TestEstimateFromExif:
    def test_uses_the_35mm_focal_when_present(self):
        intrinsics, source = estimate_from_exif(2160, 1214, focal_35mm=26.0, fallback_ratio=0.85)
        # 35mm film is 36mm wide, so focal in pixels is f/36 * width.
        assert intrinsics.fx == pytest.approx(26.0 / 36.0 * 2160)
        assert source == "exif_focal35"

    def test_falls_back_when_exif_is_missing(self):
        intrinsics, source = estimate_from_exif(2160, 1214, focal_35mm=None, fallback_ratio=0.85)
        assert intrinsics.fx == pytest.approx(0.85 * 2160)
        assert source == "fallback_guess"

    def test_treats_a_nonsense_focal_as_missing(self):
        _, source = estimate_from_exif(1920, 1080, focal_35mm=0.4, fallback_ratio=0.85)
        assert source == "fallback_guess"


class TestFisheye:
    def test_maps_have_image_shape(self, intrinsics):
        map_x, map_y = fisheye_maps(intrinsics, [-0.08, 0.01, 0.0, 0.0])
        assert map_x.shape == (intrinsics.height, intrinsics.width)
        assert map_y.shape == (intrinsics.height, intrinsics.width)
        assert map_x.dtype == np.float32

    def test_zero_coefficients_are_close_to_identity(self, intrinsics):
        map_x, map_y = fisheye_maps(intrinsics, [0.0, 0.0, 0.0, 0.0])
        # Even with no distortion the fisheye projection is not the pinhole
        # one, but the principal point must stay put.
        centre_y, centre_x = intrinsics.height // 2, intrinsics.width // 2
        assert map_x[centre_y, centre_x] == pytest.approx(intrinsics.cx, abs=1.5)
        assert map_y[centre_y, centre_x] == pytest.approx(intrinsics.cy, abs=1.5)

    def test_apply_preserves_shape_and_dtype(self, intrinsics):
        image = np.random.default_rng(0).integers(
            0, 255, (intrinsics.height, intrinsics.width, 3), dtype=np.uint8
        )
        warped = apply_fisheye(image, intrinsics, [-0.08, 0.01, 0.0, 0.0])
        assert warped.shape == image.shape
        assert warped.dtype == image.dtype

    def test_barrel_distortion_pulls_the_edges_inward(self, intrinsics):
        map_x, _ = fisheye_maps(intrinsics, [-0.25, 0.0, 0.0, 0.0])
        centre_row = intrinsics.height // 2
        # Sampling the far right output column reaches beyond the right edge
        # of the source, which is what barrel distortion does.
        assert map_x[centre_row, -1] > intrinsics.width * 0.9
