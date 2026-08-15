"""The z-buffered rasterizer that composites the gripper and object.

Stage 5 layers the splat, the object, and the gripper by depth. Getting the
ordering wrong produces a gripper that floats behind the scene, which looks
plausible in a thumbnail and is useless as training data.
"""

import numpy as np
import pytest

from wristview.backends.mesh_render import (
    box_mesh,
    combine,
    composite,
    rasterize_mesh,
    rasterize_points,
)

FX = FY = 300.0
WIDTH, HEIGHT = 128, 128
CX, CY = WIDTH / 2, HEIGHT / 2


def identity_view() -> np.ndarray:
    return np.eye(4)


class TestBoxMesh:
    def test_vertex_and_face_counts(self):
        vertices, faces = box_mesh(np.zeros(3), np.array([1.0, 1.0, 1.0]))
        assert vertices.shape == (8, 3)
        assert faces.shape == (12, 3)

    def test_half_extents_are_respected(self):
        half = np.array([0.5, 1.0, 2.0])
        vertices, _ = box_mesh(np.array([1.0, 2.0, 3.0]), half)
        assert np.allclose(vertices.max(axis=0), np.array([1.0, 2.0, 3.0]) + half)
        assert np.allclose(vertices.min(axis=0), np.array([1.0, 2.0, 3.0]) - half)


class TestCombine:
    def test_offsets_face_indices(self):
        a = box_mesh(np.zeros(3), np.ones(3))
        b = box_mesh(np.array([3.0, 0, 0]), np.ones(3))
        vertices, faces = combine([a, b])
        assert vertices.shape == (16, 3)
        assert faces.shape == (24, 3)
        assert faces.max() == 15

    def test_empty_input(self):
        vertices, faces = combine([])
        assert len(vertices) == 0
        assert len(faces) == 0


class TestRasterizeMesh:
    def test_draws_a_box_in_front_of_the_camera(self):
        vertices, faces = box_mesh(np.array([0.0, 0.0, 2.0]), np.array([0.3, 0.3, 0.3]))
        color, depth = rasterize_mesh(
            vertices, faces, identity_view(), FX, FY, CX, CY, WIDTH, HEIGHT, (1.0, 0.0, 0.0)
        )
        assert np.isfinite(depth).any(), "nothing was drawn"
        assert color[int(CY), int(CX)].sum() > 0

    def test_depth_is_the_near_face(self):
        vertices, faces = box_mesh(np.array([0.0, 0.0, 2.0]), np.array([0.3, 0.3, 0.3]))
        _, depth = rasterize_mesh(
            vertices, faces, identity_view(), FX, FY, CX, CY, WIDTH, HEIGHT, (1.0, 0.0, 0.0)
        )
        assert depth[int(CY), int(CX)] == pytest.approx(1.7, abs=1e-6)

    def test_nearer_box_wins(self):
        near = box_mesh(np.array([0.0, 0.0, 1.0]), np.array([0.2, 0.2, 0.2]))
        far = box_mesh(np.array([0.0, 0.0, 4.0]), np.array([0.2, 0.2, 0.2]))
        vertices, faces = combine([far, near])
        _, depth = rasterize_mesh(
            vertices, faces, identity_view(), FX, FY, CX, CY, WIDTH, HEIGHT, (1.0, 1.0, 1.0)
        )
        assert depth[int(CY), int(CX)] == pytest.approx(0.8, abs=1e-6)

    def test_geometry_behind_the_camera_is_clipped(self):
        vertices, faces = box_mesh(np.array([0.0, 0.0, -2.0]), np.array([0.3, 0.3, 0.3]))
        _, depth = rasterize_mesh(
            vertices, faces, identity_view(), FX, FY, CX, CY, WIDTH, HEIGHT, (1.0, 0.0, 0.0)
        )
        assert not np.isfinite(depth).any()

    def test_empty_mesh_draws_nothing(self):
        color, depth = rasterize_mesh(
            np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
            identity_view(), FX, FY, CX, CY, WIDTH, HEIGHT, (1.0, 0.0, 0.0),
        )
        assert not np.isfinite(depth).any()
        assert color.sum() == 0

    def test_shading_varies_across_faces(self):
        # A single flat colour everywhere means the normals are not being used.
        vertices, faces = box_mesh(np.array([0.4, 0.3, 2.0]), np.array([0.4, 0.4, 0.4]))
        color, depth = rasterize_mesh(
            vertices, faces, identity_view(), FX, FY, CX, CY, WIDTH, HEIGHT, (0.8, 0.8, 0.8)
        )
        drawn = color[np.isfinite(depth)]
        assert drawn.std() > 0.0


class TestRasterizePoints:
    def test_draws_points(self):
        rng = np.random.default_rng(0)
        points = rng.normal(0, 0.05, (300, 3))
        points[:, 2] += 1.0
        color, depth = rasterize_points(
            points, None, identity_view(), FX, FY, CX, CY, WIDTH, HEIGHT
        )
        assert np.isfinite(depth).any()
        assert color.sum() > 0

    def test_nearer_points_win(self):
        points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        color, depth = rasterize_points(
            points, colors, identity_view(), FX, FY, CX, CY, WIDTH, HEIGHT,
            point_radius_m=0.05,
        )
        centre = color[int(CY), int(CX)]
        assert centre[0] > centre[2]
        assert depth[int(CY), int(CX)] == pytest.approx(1.0, abs=1e-6)

    def test_empty_cloud(self):
        color, depth = rasterize_points(
            np.zeros((0, 3)), None, identity_view(), FX, FY, CX, CY, WIDTH, HEIGHT
        )
        assert not np.isfinite(depth).any()
        assert color.sum() == 0

    def test_points_behind_the_camera_are_skipped(self):
        points = np.array([[0.0, 0.0, -1.0]])
        _, depth = rasterize_points(
            points, None, identity_view(), FX, FY, CX, CY, WIDTH, HEIGHT
        )
        assert not np.isfinite(depth).any()


class TestComposite:
    def test_nearest_layer_wins_per_pixel(self):
        background = np.array([0.0, 0.0, 0.0])
        far_color = np.zeros((4, 4, 3))
        far_color[:] = [0.0, 0.0, 1.0]
        far_depth = np.full((4, 4), 5.0)

        near_color = np.zeros((4, 4, 3))
        near_color[:] = [1.0, 0.0, 0.0]
        near_depth = np.full((4, 4), 2.0)
        near_depth[0, 0] = np.inf  # a hole in the near layer

        color, depth = composite([(far_color, far_depth), (near_color, near_depth)], background)
        assert np.allclose(color[1, 1], [1.0, 0.0, 0.0])
        assert np.allclose(color[0, 0], [0.0, 0.0, 1.0])
        assert depth[1, 1] == pytest.approx(2.0)

    def test_layer_order_does_not_matter(self):
        background = np.zeros(3)
        a_color = np.tile([1.0, 0.0, 0.0], (4, 4, 1))
        a_depth = np.full((4, 4), 2.0)
        b_color = np.tile([0.0, 1.0, 0.0], (4, 4, 1))
        b_depth = np.full((4, 4), 5.0)

        forward, _ = composite([(a_color, a_depth), (b_color, b_depth)], background)
        backward, _ = composite([(b_color, b_depth), (a_color, a_depth)], background)
        assert np.allclose(forward, backward)

    def test_background_shows_where_nothing_is_drawn(self):
        background = np.array([0.2, 0.3, 0.4])
        color, _ = composite(
            [(np.zeros((3, 3, 3)), np.full((3, 3), np.inf))], background
        )
        assert np.allclose(color[0, 0], background)
