"""Rigid and similarity transform helpers.

Conventions, fixed for the whole pipeline:

- A pose is a 4x4 matrix.
- `T_world_cam` maps a point in camera coordinates to world coordinates.
  Its translation column is the camera centre in the world.
- Camera axes follow COLMAP and OpenCV: x right, y down, z forward.
- Quaternions are `(w, x, y, z)`, matching COLMAP.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """COLMAP `(w, x, y, z)` quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return Rotation.from_quat(np.array([q[1], q[2], q[3], q[0]])).as_matrix()


def rotmat_to_quat(rotmat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to a COLMAP `(w, x, y, z)` quaternion."""
    x, y, z, w = Rotation.from_matrix(np.asarray(rotmat, dtype=np.float64)).as_quat()
    return np.array([w, x, y, z])


def make_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 pose from a 3x3 rotation and a 3-vector."""
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(translation).reshape(3)
    return pose


def invert_pose(pose: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform without a general matrix inverse."""
    out = np.eye(4)
    rot = pose[:3, :3]
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ pose[:3, 3]
    return out


def world_from_colmap_image(quat_wxyz: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """COLMAP stores world-to-camera. Return the camera-to-world pose."""
    rot = quat_to_rotmat(quat_wxyz)
    return invert_pose(make_pose(rot, tvec))


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N, 3) point array."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ pose[:3, :3].T + pose[:3, 3]


def umeyama_sim3(
    source: np.ndarray, target: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform mapping `source` onto `target`.

    Umeyama 1991. Returns `(scale, rotation, translation)` such that
    `scale * rotation @ source_i + translation` approximates `target_i`.

    This is how metric scale enters the pipeline: fit the ARKit trajectory,
    which is in real metres, against the scale-free COLMAP trajectory.
    """
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape:
        raise ValueError(f"shape mismatch: {source.shape} against {target.shape}")
    if len(source) < 3:
        raise ValueError(f"need 3 or more correspondences, got {len(source)}")

    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src_centered = source - src_mean
    dst_centered = target - dst_mean

    covariance = dst_centered.T @ src_centered / len(source)
    u_mat, singular, vt_mat = np.linalg.svd(covariance)

    # Guard against a reflection when the point set is near-degenerate.
    correction = np.eye(3)
    if np.linalg.det(u_mat) * np.linalg.det(vt_mat) < 0:
        correction[2, 2] = -1.0

    rotation = u_mat @ correction @ vt_mat

    if with_scale:
        src_variance = (src_centered**2).sum() / len(source)
        scale = float((singular * np.diag(correction)).sum() / max(src_variance, 1e-12))
    else:
        scale = 1.0

    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def sim3_matrix(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Pack a similarity transform into a 4x4 matrix."""
    out = np.eye(4)
    out[:3, :3] = scale * rotation
    out[:3, 3] = translation
    return out


def apply_sim3_to_pose(sim3: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Transform a camera-to-world pose by a similarity transform.

    The rotation block must stay orthonormal, so the scale is divided back out
    of the rotation while the translation keeps it.
    """
    scale = float(np.cbrt(max(np.linalg.det(sim3[:3, :3]), 1e-30)))
    out = sim3 @ pose
    out[:3, :3] = out[:3, :3] / max(scale, 1e-12)
    return out


def slerp_fill(poses: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill gaps in a pose sequence: slerp the rotation, lerp the translation.

    `poses` is (N, 4, 4) and `valid` is a boolean mask of length N. Leading and
    trailing gaps hold the nearest valid pose.
    """
    poses = np.asarray(poses, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool)
    indices = np.arange(len(poses))
    known = indices[valid]
    if len(known) == 0:
        raise ValueError("no valid poses to interpolate from")
    if len(known) == 1:
        poses[~valid] = poses[known[0]]
        return poses

    rotations = Rotation.from_matrix(poses[known, :3, :3])
    slerp = Slerp(known.astype(float), rotations)
    missing = indices[~valid]
    clamped = np.clip(missing.astype(float), known[0], known[-1])

    poses[missing, :3, :3] = slerp(clamped).as_matrix()
    for axis in range(3):
        poses[missing, axis, 3] = np.interp(clamped, known.astype(float), poses[known, axis, 3])
    poses[missing, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return poses


def smooth_poses(poses: np.ndarray, window: int) -> np.ndarray:
    """Smooth a pose sequence with a centred moving average.

    Translations average directly. Rotations average as quaternions with sign
    alignment, then renormalize. Good enough for handheld camera jitter and it
    avoids a dependency on a full pose-graph smoother.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if window <= 1 or len(poses) < 3:
        return poses.copy()
    window = min(window, len(poses))
    if window % 2 == 0:
        window += 1
    half = window // 2

    quats = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    # Align hemispheres so averaging does not cancel neighbouring rotations.
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    out = np.repeat(np.eye(4)[None], len(poses), axis=0)
    for i in range(len(poses)):
        lo = max(0, i - half)
        hi = min(len(poses), i + half + 1)
        mean_quat = quats[lo:hi].mean(axis=0)
        norm = np.linalg.norm(mean_quat)
        out[i, :3, :3] = (
            Rotation.from_quat(mean_quat / norm).as_matrix() if norm > 1e-9 else poses[i, :3, :3]
        )
        out[i, :3, 3] = poses[lo:hi, :3, 3].mean(axis=0)
    return out


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Camera-to-world pose looking from `eye` at `target`.

    Uses the OpenCV convention: x right, y down, z forward.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64).reshape(3)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("eye and target coincide")
    forward = forward / norm

    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-12)
    down = np.cross(forward, right)

    return make_pose(np.stack([right, down, forward], axis=1), eye)


def orthonormalize(rotation: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3)."""
    u_mat, _, vt_mat = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    out = u_mat @ vt_mat
    if np.linalg.det(out) < 0:
        u_mat[:, -1] *= -1
        out = u_mat @ vt_mat
    return out


def frame_from_axes(
    origin: np.ndarray, approach: np.ndarray, closing: np.ndarray
) -> np.ndarray:
    """Build a gripper pose from an approach direction and a closing axis.

    Gripper convention, fixed for Stage 4 and Stage 5:
      z is the approach direction, pointing out of the jaws
      x is the closing axis, thumb to index
      y completes the right-handed frame

    `closing` is orthogonalized against `approach` with Gram-Schmidt, so the
    two inputs need not be perpendicular.
    """
    approach = np.asarray(approach, dtype=np.float64).reshape(3)
    closing = np.asarray(closing, dtype=np.float64).reshape(3)

    z_axis = approach / max(np.linalg.norm(approach), 1e-12)
    x_axis = closing - np.dot(closing, z_axis) * z_axis
    norm = np.linalg.norm(x_axis)
    if norm < 1e-6:
        # Degenerate: closing is parallel to approach. Pick any perpendicular.
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, z_axis)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        x_axis = fallback - np.dot(fallback, z_axis) * z_axis
        norm = np.linalg.norm(x_axis)
    x_axis = x_axis / norm
    y_axis = np.cross(z_axis, x_axis)

    return make_pose(np.stack([x_axis, y_axis, z_axis], axis=1), origin)
