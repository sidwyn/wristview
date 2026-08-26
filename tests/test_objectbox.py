"""The object had no body. These tests are why that cannot recur silently."""

from __future__ import annotations

import numpy as np
import pytest

from wristview.objectbox import box_mesh, resolve_dimensions, sample_colour


def test_the_box_is_centred_on_the_origin():
    """The pose puts the centre above the desk, so the model must be centred."""
    vertices, faces = box_mesh([0.0762, 0.0762, 0.0762])
    assert np.allclose(vertices.mean(axis=0), 0.0)
    assert np.allclose(vertices.max(axis=0), 0.0381)
    assert np.allclose(vertices.min(axis=0), -0.0381)
    assert len(faces) == 12


def test_the_box_is_closed():
    """Every edge is shared by exactly two triangles, or the depth test leaks."""
    _, faces = box_mesh([0.05, 0.06, 0.07])
    edges: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((int(u), int(v))))] = edges.get(tuple(sorted((int(u), int(v)))), 0) + 1
    assert set(edges.values()) == {2}


def test_non_cube_dimensions_are_kept():
    vertices, _ = box_mesh([0.02, 0.04, 0.08])
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    assert np.allclose(extent, [0.02, 0.04, 0.08])


def test_a_degenerate_box_raises_rather_than_drawing_nothing():
    with pytest.raises(ValueError):
        box_mesh([0.0, 0.05, 0.05])


def test_dimensions_fall_back_to_a_cube_of_the_solver_height():
    """The renderer must use the size the pose solver used, not another one."""
    dims, source = resolve_dimensions({"object_height_m": 0.0762})
    assert np.allclose(dims, 0.0762)
    assert "object_height_m" in source


def test_explicit_dimensions_win():
    dims, source = resolve_dimensions(
        {"object_height_m": 0.0762, "object_dimensions_m": [0.07, 0.07, 0.07]}
    )
    assert np.allclose(dims, 0.07)
    assert source == "object_dimensions_m"


def test_no_size_at_all_raises():
    """A model invented from nothing is what the single point already was."""
    with pytest.raises(ValueError, match="object_height_m"):
        resolve_dimensions({"object_height_m": 0.0})


def test_colour_comes_from_the_object_pixels():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :] = (10, 10, 10)
    image[1:3, 1:3] = (200, 180, 40)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    (r, g, b), n = sample_colour([image], [mask])
    assert n == 4
    assert r == pytest.approx(200 / 255, abs=1e-6)
    assert g == pytest.approx(180 / 255, abs=1e-6)


def test_a_leaky_mask_does_not_move_the_median():
    """A mean would drag toward the desk. A median does not."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :] = (200, 180, 40)
    image[0, 0] = (5, 5, 5)
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    (r, _, _), n = sample_colour([image], [mask])
    assert n == 100
    assert r == pytest.approx(200 / 255, abs=1e-6)


def test_no_object_pixels_returns_a_zero_count_so_it_cannot_be_called_a_measurement():
    colour, n = sample_colour([], [])
    assert len(colour) == 3
    assert n == 0


def test_a_mask_smaller_than_its_image_is_still_sampled():
    """Stage 3 segments at the work resolution; the frames are full size.

    Skipping the pair on a shape mismatch discarded every sample on real26 and
    the box was then labelled as measured anyway.
    """
    import numpy as np
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :] = (200, 180, 40)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    (r, g, _), n = sample_colour([image], [mask])
    assert n == 400
    assert r == pytest.approx(200 / 255, abs=1e-6)
