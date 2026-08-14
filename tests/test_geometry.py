"""Geometry helpers. Every later stage depends on these being exactly right."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wristview.geometry import (
    frame_from_axes,
    invert_pose,
    look_at,
    make_pose,
    orthonormalize,
    quat_to_rotmat,
    rotmat_to_quat,
    sim3_matrix,
    slerp_fill,
    smooth_poses,
    transform_points,
    umeyama_sim3,
    world_from_colmap_image,
)


def random_rotation(seed: int) -> np.ndarray:
    return Rotation.random(random_state=seed).as_matrix()


def test_invert_pose_roundtrip():
    pose = make_pose(random_rotation(0), np.array([1.5, -2.0, 0.75]))
    assert np.allclose(pose @ invert_pose(pose), np.eye(4), atol=1e-12)


def test_invert_pose_matches_numpy_inverse():
    pose = make_pose(random_rotation(3), np.array([0.1, 0.2, 0.3]))
    assert np.allclose(invert_pose(pose), np.linalg.inv(pose), atol=1e-12)


def test_quaternion_roundtrip():
    rotation = random_rotation(1)
    assert np.allclose(quat_to_rotmat(rotmat_to_quat(rotation)), rotation, atol=1e-12)


def test_world_from_colmap_image_puts_camera_centre_in_translation():
    # COLMAP stores world-to-camera. The camera centre is -R^T t.
    rotation = random_rotation(2)
    tvec = np.array([0.3, -1.2, 4.0])
    quat = rotmat_to_quat(rotation)
    pose = world_from_colmap_image(quat, tvec)
    assert np.allclose(pose[:3, 3], -rotation.T @ tvec, atol=1e-12)


class TestUmeyama:
    def test_recovers_a_known_similarity(self):
        rng = np.random.default_rng(0)
        source = rng.normal(size=(40, 3))
        scale_true = 2.75
        rotation_true = random_rotation(5)
        translation_true = np.array([1.0, -3.0, 0.5])
        target = scale_true * source @ rotation_true.T + translation_true

        scale, rotation, translation = umeyama_sim3(source, target)
        assert scale == pytest.approx(scale_true, rel=1e-9)
        assert np.allclose(rotation, rotation_true, atol=1e-9)
        assert np.allclose(translation, translation_true, atol=1e-9)

    def test_scale_is_the_metric_ratio(self):
        # This is the property Stage 1 depends on: ARKit metres against
        # scale-free COLMAP units gives the metres-per-unit factor.
        rng = np.random.default_rng(1)
        colmap = rng.normal(size=(30, 3))
        metres = 0.137 * colmap
        scale, _, _ = umeyama_sim3(colmap, metres)
        assert scale == pytest.approx(0.137, rel=1e-9)

    def test_without_scale_returns_unit(self):
        rng = np.random.default_rng(2)
        source = rng.normal(size=(20, 3))
        target = source @ random_rotation(7).T + np.array([1.0, 2.0, 3.0])
        scale, _, _ = umeyama_sim3(source, target, with_scale=False)
        assert scale == pytest.approx(1.0)

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="3 or more"):
            umeyama_sim3(np.zeros((2, 3)), np.zeros((2, 3)))

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            umeyama_sim3(np.zeros((5, 3)), np.zeros((4, 3)))

    def test_no_reflection_on_planar_input(self):
        # Coplanar points are the degenerate case that makes a naive SVD fit
        # return a reflection instead of a rotation.
        rng = np.random.default_rng(3)
        source = np.c_[rng.normal(size=(25, 2)), np.zeros(25)]
        target = source @ random_rotation(9).T + np.array([0.5, 0.5, 0.5])
        _, rotation, _ = umeyama_sim3(source, target)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)


def test_sim3_matrix_scales_points():
    transform = sim3_matrix(3.0, np.eye(3), np.array([1.0, 0.0, 0.0]))
    moved = transform_points(transform, np.array([[1.0, 2.0, 3.0]]))
    assert np.allclose(moved, [[4.0, 6.0, 9.0]])


class TestFrameFromAxes:
    def test_axes_are_orthonormal_and_right_handed(self):
        pose = frame_from_axes(
            np.array([1.0, 2.0, 3.0]),
            approach=np.array([0.0, 0.0, 1.0]),
            closing=np.array([1.0, 0.0, 0.0]),
        )
        rotation = pose[:3, :3]
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)

    def test_z_is_the_approach_direction(self):
        approach = np.array([0.3, -0.5, 0.8])
        approach = approach / np.linalg.norm(approach)
        pose = frame_from_axes(np.zeros(3), approach, np.array([1.0, 0.0, 0.0]))
        assert np.allclose(pose[:3, 2], approach, atol=1e-12)

    def test_closing_axis_is_orthogonalized_not_ignored(self):
        # Gram-Schmidt: the closing input need not be perpendicular already.
        approach = np.array([0.0, 0.0, 1.0])
        closing = np.array([1.0, 0.0, 0.7])
        pose = frame_from_axes(np.zeros(3), approach, closing)
        assert np.allclose(pose[:3, 0], [1.0, 0.0, 0.0], atol=1e-12)

    def test_survives_closing_parallel_to_approach(self):
        pose = frame_from_axes(
            np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])
        )
        rotation = pose[:3, :3]
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
        assert np.isfinite(rotation).all()

    def test_origin_is_preserved(self):
        origin = np.array([0.11, -0.22, 0.33])
        pose = frame_from_axes(origin, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        assert np.allclose(pose[:3, 3], origin)


class TestSlerpFill:
    def test_fills_an_interior_gap(self):
        poses = np.repeat(np.eye(4)[None], 5, axis=0)
        poses[0] = make_pose(np.eye(3), np.array([0.0, 0.0, 0.0]))
        poses[4] = make_pose(np.eye(3), np.array([4.0, 0.0, 0.0]))
        valid = np.array([True, False, False, False, True])
        filled = slerp_fill(poses, valid)
        assert filled[2, 0, 3] == pytest.approx(2.0)

    def test_holds_the_nearest_pose_at_the_edges(self):
        poses = np.repeat(np.eye(4)[None], 4, axis=0)
        poses[1, :3, 3] = [1.0, 0.0, 0.0]
        poses[2, :3, 3] = [2.0, 0.0, 0.0]
        valid = np.array([False, True, True, False])
        filled = slerp_fill(poses, valid)
        assert filled[0, 0, 3] == pytest.approx(1.0)
        assert filled[3, 0, 3] == pytest.approx(2.0)

    def test_interpolated_rotations_stay_valid(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[0, :3, :3] = np.eye(3)
        poses[2, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        filled = slerp_fill(poses, np.array([True, False, True]))
        middle = filled[1, :3, :3]
        assert np.allclose(middle @ middle.T, np.eye(3), atol=1e-10)
        angle = Rotation.from_matrix(middle).as_euler("xyz", degrees=True)[2]
        assert angle == pytest.approx(45.0, abs=1e-6)

    def test_raises_when_nothing_is_valid(self):
        with pytest.raises(ValueError, match="no valid poses"):
            slerp_fill(np.repeat(np.eye(4)[None], 3, axis=0), np.zeros(3, dtype=bool))


class TestSmoothPoses:
    def test_reduces_translation_jitter(self):
        rng = np.random.default_rng(0)
        clean = np.repeat(np.eye(4)[None], 60, axis=0)
        clean[:, 0, 3] = np.linspace(0, 1, 60)
        noisy = clean.copy()
        noisy[:, 0, 3] += rng.normal(0, 0.02, 60)

        smoothed = smooth_poses(noisy, window=7)
        before = np.abs(noisy[:, 0, 3] - clean[:, 0, 3]).mean()
        after = np.abs(smoothed[:, 0, 3] - clean[:, 0, 3]).mean()
        assert after < before

    def test_rotations_stay_orthonormal(self):
        poses = np.repeat(np.eye(4)[None], 20, axis=0)
        for i in range(20):
            poses[i, :3, :3] = Rotation.from_euler("y", i * 3, degrees=True).as_matrix()
        smoothed = smooth_poses(poses, window=5)
        for pose in smoothed:
            rotation = pose[:3, :3]
            assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)

    def test_window_of_one_is_a_no_op(self):
        poses = np.repeat(np.eye(4)[None], 5, axis=0)
        poses[:, 0, 3] = [0, 5, -3, 2, 9]
        assert np.allclose(smooth_poses(poses, window=1), poses)

    def test_handles_quaternion_sign_flips(self):
        # Neighbouring rotations can come back with opposite quaternion signs.
        # Averaging without hemisphere alignment cancels them to near zero.
        poses = np.repeat(np.eye(4)[None], 10, axis=0)
        for i in range(10):
            poses[i, :3, :3] = Rotation.from_euler("z", 179 + i * 0.2, degrees=True).as_matrix()
        smoothed = smooth_poses(poses, window=5)
        for pose in smoothed:
            assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0, abs=1e-9)


def test_look_at_points_z_at_the_target():
    pose = look_at(np.array([0.0, -2.0, 1.0]), np.array([0.0, 0.0, 1.0]))
    direction = pose[:3, 2]
    assert np.allclose(direction, [0.0, 1.0, 0.0], atol=1e-12)


def test_look_at_raises_when_eye_meets_target():
    with pytest.raises(ValueError, match="coincide"):
        look_at(np.zeros(3), np.zeros(3))


def test_orthonormalize_repairs_a_drifted_rotation():
    rotation = random_rotation(11) + np.random.default_rng(0).normal(0, 0.01, (3, 3))
    fixed = orthonormalize(rotation)
    assert np.allclose(fixed @ fixed.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(fixed) == pytest.approx(1.0)
