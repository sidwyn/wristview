"""The work surface, and object poses solved against it.

Session 4 tracked its object perfectly in 2D and placed it wrongly in 3D: the
cube floated 85 mm above a desk it was resting on, and sat 113 to 145 mm to one
side of the hand holding it. Every reported number was healthy. The ICP
residual was 1.1 mm, mask coverage was 100 per cent, and none of that
constrains where the object is in the world, because the residual only measures
how well a model fits its own back-projected points.

The error came in through monocular depth. Depth is recovered up to an unknown
affine transform per frame and anchored against sparse observations, and a
small object contributes almost nothing to that fit, so it lands wherever the
background puts it.

This removes the dependency rather than repairing it. The desk is the one
surface in the scene that is large, flat, textured and reconstructed well. An
object resting on it has only three degrees of freedom, two across the surface
and one of yaw, and all three follow from the silhouette without any depth
network at all.
"""

from __future__ import annotations

import numpy as np

from .camera import Intrinsics
from .logging_setup import get

log = get(__name__)


def fit_plane(
    points: np.ndarray,
    normal_hint: np.ndarray,
    iterations: int = 400,
    tolerance_m: float = 0.006,
    seed: int = 0,
) -> tuple[np.ndarray, float, dict]:
    """RANSAC the dominant plane whose normal agrees with `normal_hint`.

    Returns the unit normal, the plane offset `d` in `n . x = d`, and a report.
    The hint matters: a desk scene also contains a large wall and a monitor,
    and the biggest plane is not always the one being sat on.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    hint = np.asarray(normal_hint, dtype=np.float64).reshape(3)
    hint = hint / max(np.linalg.norm(hint), 1e-12)
    if len(points) < 16:
        raise ValueError(f"need 16 or more points to fit a plane, got {len(points)}")

    rng = np.random.default_rng(seed)
    best_normal = hint
    best_offset = float(np.median(points @ hint))
    best_count = -1

    for _ in range(iterations):
        trio = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(trio[1] - trio[0], trio[2] - trio[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal = normal / length
        # Only planes that face the way the surface faces.
        if abs(float(normal @ hint)) < 0.86:      # within about 30 degrees
            continue
        if float(normal @ hint) < 0:
            normal = -normal
        offset = float(normal @ trio[0])
        inliers = int((np.abs(points @ normal - offset) < tolerance_m).sum())
        if inliers > best_count:
            best_normal, best_offset, best_count = normal, offset, inliers

    if best_count < 0:
        raise ValueError("no plane agreed with the normal hint")

    # Refit on the inliers, which is what makes the estimate precise rather
    # than merely correct.
    for _ in range(3):
        keep = np.abs(points @ best_normal - best_offset) < tolerance_m
        if keep.sum() < 16:
            break
        block = points[keep]
        centre = block.mean(axis=0)
        _, _, vt = np.linalg.svd(block - centre)
        normal = vt[2]
        if float(normal @ hint) < 0:
            normal = -normal
        best_normal, best_offset = normal, float(normal @ centre)

    keep = np.abs(points @ best_normal - best_offset) < tolerance_m
    residual = np.abs(points[keep] @ best_normal - best_offset)
    report = {
        "inliers": int(keep.sum()),
        "points": int(len(points)),
        "inlier_fraction": round(float(keep.mean()), 4),
        "residual_median_mm": round(float(np.median(residual) * 1000), 3),
        "residual_p90_mm": round(float(np.percentile(residual, 90) * 1000), 3),
        "normal": [round(float(v), 6) for v in best_normal],
        "offset_m": round(float(best_offset), 6),
        "angle_to_hint_deg": round(
            float(np.degrees(np.arccos(np.clip(best_normal @ hint, -1, 1)))), 3
        ),
    }
    return best_normal, float(best_offset), report


def ray_plane_intersection(
    origin: np.ndarray, direction: np.ndarray, normal: np.ndarray, offset: float
) -> np.ndarray | None:
    """Where a ray meets a plane, or None if it runs parallel or behind."""
    denominator = float(direction @ normal)
    if abs(denominator) < 1e-9:
        return None
    t = (offset - float(origin @ normal)) / denominator
    if t <= 0:
        return None
    return origin + t * direction


def object_pose_on_plane(
    mask: np.ndarray,
    camera_pose: np.ndarray,
    intrinsics: Intrinsics,
    normal: np.ndarray,
    offset: float,
    height_m: float,
    reference_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, dict] | None:
    """Pose of an object resting on the plane, from its silhouette alone.

    An object standing on a known plane has its centre on a second plane,
    parallel to the first and `height_m / 2` above it. So the ray through the
    silhouette's centroid meets that offset plane at the object's centre, and
    no depth estimate is involved anywhere.

    `height_m` is the object's real height, measured with a ruler. It is the
    only scale input, and it is the one number here that cannot be got wrong
    quietly: an error in it moves the object along the viewing ray.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 30:
        return None

    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    centre_offset = offset + height_m / 2.0

    centroid_px = np.array([xs.mean(), ys.mean()], dtype=np.float64)
    ray_cam = intrinsics.unproject(centroid_px[None, :], np.ones(1))[0]
    ray_cam = ray_cam / max(np.linalg.norm(ray_cam), 1e-12)

    origin = camera_pose[:3, 3]
    direction = camera_pose[:3, :3] @ ray_cam
    centre = ray_plane_intersection(origin, direction, normal, centre_offset)
    if centre is None:
        return None

    # Yaw about the surface normal, from the silhouette's principal axis. The
    # other two angles are fixed by the object sitting flat.
    rotation = np.eye(3) if reference_rotation is None else np.asarray(reference_rotation)
    pixels = np.stack([xs, ys], axis=1).astype(np.float64)
    spread = pixels - pixels.mean(axis=0)
    if len(spread) > 60:
        _, _, vt = np.linalg.svd(spread, full_matrices=False)
        major_px = vt[0]
        # Carry the image-space axis onto the plane, then orthogonalise.
        tip = intrinsics.unproject((centroid_px + major_px * 40)[None, :], np.ones(1))[0]
        tip = camera_pose[:3, :3] @ (tip / max(np.linalg.norm(tip), 1e-12))
        hit = ray_plane_intersection(origin, tip, normal, centre_offset)
        if hit is not None:
            axis = hit - centre
            axis = axis - normal * float(axis @ normal)
            length = np.linalg.norm(axis)
            if length > 1e-6:
                axis = axis / length
                rotation = np.stack([axis, np.cross(normal, axis), normal], axis=1)

    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = centre
    return pose, {
        "mask_pixels": int(len(ys)),
        "centroid_px": [round(float(v), 2) for v in centroid_px],
        "height_above_plane_m": round(float(centre @ normal - offset), 5),
        # The world-frame ray through the silhouette centroid. It points at the
        # object whether the object is resting or carried, so it is the part of
        # this solve that survives the object leaving the plane. `carry` uses it
        # to place the object along the ray once the hand picks it up.
        "ray_world": direction,
        "camera_position": origin,
    }
