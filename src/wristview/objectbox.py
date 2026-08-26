"""A box model for the manipulated object, at its measured size and colour.

The plane solve returns a pose and no shape. Stage 3 recorded that as
`np.zeros((1, 3))`, one point at the origin, and Stage 5 drew it with
`rasterize_points` at a 5 mm radius. So every wrist view real26 produced
contained a dot where the object was, five to thirteen pixels across.

The cube visible in those renders is the splat's own copy of the cube, frozen
at the position it held during the scan. It does not move when the operator
picks the object up, because a splat is a static reconstruction. A policy
trained on those frames sees a static scene and a dot: it cannot learn that the
object moves, or where it is relative to the gripper.

This module gives the object a body. A box at the measured dimensions, at the
tracked pose, rasterised as a mesh so the existing depth test occludes it
against the splat and the gripper correctly.

A box is not the object's true shape. It is the shape we have measured, and it
is the difference between an object and a marker. A Sam3D reconstruction
replaces it later without changing anything downstream: Stage 5 asks for
vertices and faces, and does not care where they came from.
"""

from __future__ import annotations

import numpy as np

from .logging_setup import get

log = get(__name__)


def box_mesh(dimensions_m: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return the vertices and triangles of a box centred on the origin.

    The object pose puts the centre at `offset + height / 2` above the desk, so
    the model must be centred too. A model built with one corner at the origin
    would sit half a body off, and would still look plausible.
    """
    half = np.asarray(dimensions_m, dtype=np.float64) / 2.0
    if half.shape != (3,) or not np.all(half > 0):
        raise ValueError(f"dimensions_m must be three positive lengths, got {dimensions_m}")
    signs = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float64)
    vertices = signs * half
    faces = np.array([
        [0, 2, 1], [0, 3, 2],   # -z
        [4, 5, 6], [4, 6, 7],   # +z
        [0, 1, 5], [0, 5, 4],   # -y
        [3, 7, 6], [3, 6, 2],   # +y
        [0, 4, 7], [0, 7, 3],   # -x
        [1, 2, 6], [1, 6, 5],   # +x
    ], dtype=np.int64)
    return vertices, faces


def resolve_dimensions(pose_cfg: dict) -> tuple[np.ndarray, str]:
    """Return the object's size in metres, and where the number came from.

    `object_dimensions_m` wins when it is set. Otherwise fall back to a cube of
    `object_height_m`, which is the number the plane solve already uses to place
    the centre. Using a different size here than the solver used would put a box
    of one size at a position computed for another.
    """
    explicit = pose_cfg.get("object_dimensions_m")
    if explicit:
        dims = np.asarray(explicit, dtype=np.float64)
        if dims.shape != (3,):
            raise ValueError(
                f"object_dimensions_m must hold three lengths, got {explicit}"
            )
        return dims, "object_dimensions_m"

    height = float(pose_cfg.get("object_height_m") or 0.0)
    if height <= 0:
        raise ValueError(
            "cannot build an object model: object_height_m is not set and "
            "object_dimensions_m is not set. One of them must give the size "
            "the object really is."
        )
    return np.array([height, height, height], dtype=np.float64), "cube of object_height_m"


def sample_colour(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    default: tuple[float, float, float] = (0.75, 0.65, 0.15),
) -> tuple[tuple[float, float, float], int]:
    """Return the object's median RGB in 0 to 1, and how many pixels made it.

    The count is returned, not logged, so the caller cannot label a default as
    a measurement. The first version of this function returned only a colour
    and the caller wrote "median of the object mask pixels" beside a default it
    had never sampled. That is the defect this whole module exists to remove.

    Take the median, not the mean. A mask that leaks a few pixels of desk pulls
    a mean; it does not move a median.

    `images` are RGB. Masks may be a different size from their image: Stage 3
    segments at the work resolution and the frames are full size. Resize the
    image to the mask rather than skipping the pair, which is what silently
    discarded every sample on real26.
    """
    import cv2

    samples: list[np.ndarray] = []
    for image, mask in zip(images, masks, strict=False):
        if image is None or mask is None:
            continue
        flag = mask > (127 if mask.dtype == np.uint8 else 0.5)
        if not flag.any():
            continue
        if flag.shape != image.shape[:2]:
            image = cv2.resize(
                image, (flag.shape[1], flag.shape[0]), interpolation=cv2.INTER_AREA
            )
        samples.append(image[flag].reshape(-1, 3))
    if not samples:
        return default, 0
    stacked = np.concatenate(samples, axis=0).astype(np.float64)
    if stacked.max() > 1.5:
        stacked = stacked / 255.0
    median = np.median(stacked, axis=0)
    return (float(median[0]), float(median[1]), float(median[2])), len(stacked)
