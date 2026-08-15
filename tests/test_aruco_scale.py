"""ArUco metric scale in Stage 1.

A printed marker of known size is the cheapest way to give a reconstruction
real units, and unlike the known-object route it needs no detector, no
segmentation and no guess about what is in the shot.

The test builds a scene where the answer is known exactly: a marker of a
chosen side length, in a world whose units are deliberately not metres, seen
from cameras whose poses are given. The recovered scale must be the ratio
between the two.
"""

import cv2
import numpy as np
import pytest

from wristview.camera import Intrinsics
from wristview.stages.s01_scene import _fit_aruco_scale

DICT = "DICT_4X4_50"
MARKER_ID = 7
# The marker is 3 units across in the reconstruction's arbitrary units and
# 0.15 m across in reality, so the scale must come back as 0.05 m per unit.
SIDE_UNITS = 3.0
SIDE_METRES = 0.15
EXPECTED_SCALE = SIDE_METRES / SIDE_UNITS


@pytest.fixture
def intrinsics() -> Intrinsics:
    return Intrinsics(width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0)


def marker_corners_world(side: float = SIDE_UNITS) -> np.ndarray:
    """Marker lying flat on z=0, centred on the origin.

    Corner order matches OpenCV's: top-left, top-right, bottom-right,
    bottom-left, seen from +z looking down.
    """
    half = side / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ]
    )


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world pose, OpenCV axes."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(forward @ up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)
    pose[:3, 3] = eye
    return pose


def render_marker(pose: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """Warp a generated marker image onto its projected quad."""
    from wristview.geometry import invert_pose, transform_points

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT))
    tile = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, 400)
    # A white border is part of the marker definition; without it detection
    # is unreliable.
    bordered = np.full((480, 480), 255, np.uint8)
    bordered[40:440, 40:440] = tile

    corners_cam = transform_points(invert_pose(pose), marker_corners_world())
    if corners_cam[:, 2].min() <= 1e-6:
        raise ValueError("marker is behind the camera")
    pixels, _ = intrinsics.project(corners_cam)

    # Map the marker's own corners onto the projected quad, not the corners of
    # the bordered canvas. The border is quiet zone, outside the marker, so
    # warping the whole canvas would render the marker 400/480 too small and
    # the recovered scale would be off by exactly that ratio.
    source = np.array(
        [[40, 40], [439, 40], [439, 439], [40, 439]], dtype=np.float32
    )
    homography = cv2.getPerspectiveTransform(source, pixels.astype(np.float32))
    canvas = np.full((intrinsics.height, intrinsics.width), 255, np.uint8)
    warped = cv2.warpPerspective(
        bordered, homography, (intrinsics.width, intrinsics.height),
        borderMode=cv2.BORDER_TRANSPARENT, dst=canvas,
    )
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def scene(tmp_path, intrinsics):
    """Frames of the marker from several viewpoints, plus their true poses."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    poses = {}
    names = []
    rng = np.random.default_rng(0)

    for index in range(10):
        angle = index / 10 * 2 * np.pi
        eye = np.array(
            [4.0 * np.cos(angle), 4.0 * np.sin(angle), 5.0 + rng.uniform(-0.6, 0.6)]
        )
        pose = look_at(eye, np.zeros(3))
        image = render_marker(pose, intrinsics)
        name = f"frame_{index:03d}.png"
        cv2.imwrite(str(frames_dir / name), image)
        names.append(name)
        poses[name] = pose

    return frames_dir, names, poses


class TestArucoScale:
    def test_recovers_the_known_scale(self, scene, intrinsics):
        frames_dir, names, poses = scene
        result = _fit_aruco_scale(
            frames_dir, names, poses, intrinsics, DICT, SIDE_METRES
        )
        assert result is not None, "the marker was not found or not triangulated"
        scale, transform, diagnostics = result

        assert scale == pytest.approx(EXPECTED_SCALE, rel=0.02), (
            f"expected {EXPECTED_SCALE:.5f} m per unit, got {scale:.5f}"
        )
        assert diagnostics["method"] == "aruco"
        assert diagnostics["measured_side_colmap_units"] == pytest.approx(
            SIDE_UNITS, rel=0.02
        )

    def test_transform_scales_points_correctly(self, scene, intrinsics):
        from wristview.geometry import transform_points

        frames_dir, names, poses = scene
        scale, transform, _ = _fit_aruco_scale(
            frames_dir, names, poses, intrinsics, DICT, SIDE_METRES
        )
        # The marker's two opposite corners are SIDE_UNITS apart, and must
        # become SIDE_METRES apart after the transform.
        corners = transform_points(transform, marker_corners_world())
        side = np.linalg.norm(corners[0] - corners[1])
        assert side == pytest.approx(SIDE_METRES, rel=0.02)

    def test_a_different_marker_size_gives_a_different_scale(self, scene, intrinsics):
        frames_dir, names, poses = scene
        half, _, _ = _fit_aruco_scale(
            frames_dir, names, poses, intrinsics, DICT, SIDE_METRES / 2
        )
        full, _, _ = _fit_aruco_scale(
            frames_dir, names, poses, intrinsics, DICT, SIDE_METRES
        )
        assert half == pytest.approx(full / 2, rel=0.02)

    def test_returns_none_when_no_marker_is_present(self, tmp_path, intrinsics):
        frames_dir = tmp_path / "blank"
        frames_dir.mkdir()
        names, poses = [], {}
        for index in range(4):
            name = f"blank_{index}.png"
            cv2.imwrite(
                str(frames_dir / name),
                np.full((intrinsics.height, intrinsics.width, 3), 200, np.uint8),
            )
            names.append(name)
            poses[name] = look_at(np.array([0.0, 0.0, 5.0]), np.zeros(3))

        assert _fit_aruco_scale(frames_dir, names, poses, intrinsics, DICT, SIDE_METRES) is None

    def test_ignores_frames_with_no_pose(self, scene, intrinsics):
        """Unregistered frames must be skipped, not crash the fit."""
        frames_dir, names, poses = scene
        partial = {k: v for k, v in list(poses.items())[:4]}
        result = _fit_aruco_scale(
            frames_dir, names, partial, intrinsics, DICT, SIDE_METRES
        )
        assert result is not None
        assert result[0] == pytest.approx(EXPECTED_SCALE, rel=0.05)

    def test_needs_at_least_two_views(self, scene, intrinsics):
        frames_dir, names, poses = scene
        one = {names[0]: poses[names[0]]}
        assert _fit_aruco_scale(frames_dir, names, one, intrinsics, DICT, SIDE_METRES) is None
