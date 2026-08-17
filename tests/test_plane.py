"""Plane fitting and plane-constrained object pose.

Session 4's object floated 85 mm above the desk it was resting on, with an ICP
residual of 1.1 mm and 100 per cent mask coverage. These cover the arithmetic
that replaces that path, on geometry where the right answer is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristview.camera import Intrinsics
from wristview.geometry import make_pose
from wristview.plane import fit_plane, object_pose_on_plane, ray_plane_intersection

UP = np.array([0.0, 0.0, 1.0])


def desk_points(n=800, height=-0.35, noise=0.001, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.5, 0.5, size=(n, 2))
    z = np.full(n, height) + rng.normal(0, noise, n)
    return np.column_stack([xy, z])


class TestFitPlane:
    def test_recovers_a_known_plane(self):
        normal, offset, report = fit_plane(desk_points(), UP)
        assert abs(offset - (-0.35)) < 0.002
        assert np.degrees(np.arccos(abs(normal @ UP))) < 1.0
        assert report["inlier_fraction"] > 0.9

    def test_ignores_a_wall_that_is_bigger_than_the_desk(self):
        """The largest plane is not always the one being sat on."""
        rng = np.random.default_rng(1)
        wall = np.column_stack([
            np.full(2000, 0.6), rng.uniform(-1, 1, 2000), rng.uniform(-1, 1, 2000)
        ])
        points = np.vstack([desk_points(400), wall])
        normal, offset, _ = fit_plane(points, UP)
        assert abs(offset - (-0.35)) < 0.01
        assert abs(normal @ UP) > 0.98

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            fit_plane(np.zeros((4, 3)), UP)


class TestRayPlane:
    def test_hits_where_expected(self):
        hit = ray_plane_intersection(np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), UP, 0.0)
        assert np.allclose(hit, [0, 0, 0])

    def test_parallel_ray_misses(self):
        assert ray_plane_intersection(
            np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), UP, 0.0) is None

    def test_ray_pointing_away_misses(self):
        assert ray_plane_intersection(
            np.array([0, 0, 1.0]), np.array([0, 0, 1.0]), UP, 0.0) is None


class TestObjectPoseOnPlane:
    def setup_method(self):
        self.intr = Intrinsics(fx=800, fy=800, cx=320, cy=240, width=640, height=480)
        # Camera a metre above the desk, looking straight down.
        rot = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        self.cam = make_pose(rot, np.array([0.0, 0.0, 0.65]))

    def test_places_a_cube_at_half_its_height(self):
        """The whole point: the centre sits height/2 above the plane, exactly."""
        mask = np.zeros((480, 640), bool)
        mask[220:260, 300:340] = True
        out = object_pose_on_plane(mask, self.cam, self.intr, UP, -0.35, 0.0762)
        assert out is not None
        pose, report = out
        assert abs(report["height_above_plane_m"] - 0.0381) < 1e-6

    def test_a_mask_off_to_one_side_moves_the_object_sideways(self):
        centred = np.zeros((480, 640), bool)
        centred[220:260, 300:340] = True
        offset = np.zeros((480, 640), bool)
        offset[220:260, 400:440] = True
        a, _ = object_pose_on_plane(centred, self.cam, self.intr, UP, -0.35, 0.0762)
        b, _ = object_pose_on_plane(offset, self.cam, self.intr, UP, -0.35, 0.0762)
        assert b[0, 3] > a[0, 3] + 0.05
        # but both stay on the same horizontal plane
        assert abs(a[2, 3] - b[2, 3]) < 1e-9

    def test_the_pose_is_a_proper_rigid_transform(self):
        mask = np.zeros((480, 640), bool)
        mask[210:270, 290:350] = True
        pose, _ = object_pose_on_plane(mask, self.cam, self.intr, UP, -0.35, 0.0762)
        r = pose[:3, :3]
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-8)
        assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-6)

    def test_a_tiny_mask_is_refused_rather_than_guessed(self):
        mask = np.zeros((480, 640), bool)
        mask[100:103, 100:103] = True
        assert object_pose_on_plane(mask, self.cam, self.intr, UP, -0.35, 0.0762) is None

    def test_object_height_is_the_only_scale_input(self):
        """A wrong ruler measurement moves the object along the viewing ray."""
        mask = np.zeros((480, 640), bool)
        mask[220:260, 300:340] = True
        a, _ = object_pose_on_plane(mask, self.cam, self.intr, UP, -0.35, 0.0762)
        b, _ = object_pose_on_plane(mask, self.cam, self.intr, UP, -0.35, 0.10)
        assert abs(b[2, 3] - a[2, 3]) == pytest.approx((0.10 - 0.0762) / 2, abs=1e-6)
