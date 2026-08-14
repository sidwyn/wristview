"""Camera intrinsics and the deployment camera model.

The pipeline reconstructs and renders with a pinhole model. Stage 5 applies
the deployment camera's distortion as a final image warp, so the renderer
itself stays simple and the output still matches the real robot camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Intrinsics:
    """Pinhole intrinsics in pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float) -> Intrinsics:
        """Build intrinsics from a horizontal field of view."""
        focal = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
        return cls(width, height, focal, focal, width / 2.0, height / 2.0)

    @classmethod
    def from_dict(cls, payload: dict) -> Intrinsics:
        return cls(
            width=int(payload["width"]),
            height=int(payload["height"]),
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
        )

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "model": "PINHOLE",
            "horizontal_fov_deg": self.horizontal_fov_deg,
        }

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def horizontal_fov_deg(self) -> float:
        return float(np.rad2deg(2.0 * np.arctan((self.width / 2.0) / self.fx)))

    def scaled(self, width: int, height: int) -> Intrinsics:
        """Rescale intrinsics to a different image size."""
        sx = width / self.width
        sy = height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy)

    def project(self, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N, 3) camera-frame points. Returns pixels and depths."""
        points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        depth = points_cam[:, 2]
        safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        pixels = np.stack(
            [self.fx * points_cam[:, 0] / safe + self.cx, self.fy * points_cam[:, 1] / safe + self.cy],
            axis=1,
        )
        return pixels, depth

    def unproject(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Lift (N, 2) pixels at (N,) depths into camera-frame points."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        depth = np.asarray(depth, dtype=np.float64).reshape(-1)
        x = (pixels[:, 0] - self.cx) / self.fx * depth
        y = (pixels[:, 1] - self.cy) / self.fy * depth
        return np.stack([x, y, depth], axis=1)


def estimate_from_exif(
    width: int, height: int, focal_35mm: float | None, fallback_ratio: float
) -> tuple[Intrinsics, str]:
    """Derive intrinsics from a 35mm-equivalent focal length when EXIF has one.

    Returns the intrinsics and the source label recorded in `intrinsics.json`.
    """
    if focal_35mm and focal_35mm > 1.0:
        # 35mm film is 36mm wide. Focal in pixels scales with image width.
        fx = focal_35mm / 36.0 * width
        return Intrinsics(width, height, fx, fx, width / 2.0, height / 2.0), "exif_focal35"

    fx = fallback_ratio * width
    return Intrinsics(width, height, fx, fx, width / 2.0, height / 2.0), "fallback_guess"


def apply_fisheye(
    image: np.ndarray, intrinsics: Intrinsics, coeffs: list[float]
) -> np.ndarray:
    """Warp a pinhole render into the OpenCV fisheye model.

    The maps depend only on the intrinsics and coefficients, so they are built
    once per episode by `fisheye_maps` and reused for every frame.
    """
    map_x, map_y = fisheye_maps(intrinsics, coeffs)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def fisheye_maps(intrinsics: Intrinsics, coeffs: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Build the remap tables that take a pinhole image to a fisheye image.

    For each fisheye output pixel, find the pinhole input pixel that lands
    there. That is the inverse direction, which is what `cv2.remap` needs.
    """
    k_mat = intrinsics.matrix
    dist = np.array(coeffs, dtype=np.float64).reshape(4, 1)

    ys, xs = np.meshgrid(
        np.arange(intrinsics.height, dtype=np.float64),
        np.arange(intrinsics.width, dtype=np.float64),
        indexing="ij",
    )
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)

    # Fisheye output pixel to a normalized ray.
    rays = cv2.fisheye.undistortPoints(pixels, k_mat, dist).reshape(-1, 2)

    # That ray through the pinhole model gives the source pixel.
    map_x = (rays[:, 0] * intrinsics.fx + intrinsics.cx).reshape(intrinsics.height, intrinsics.width)
    map_y = (rays[:, 1] * intrinsics.fy + intrinsics.cy).reshape(intrinsics.height, intrinsics.width)
    return map_x.astype(np.float32), map_y.astype(np.float32)
