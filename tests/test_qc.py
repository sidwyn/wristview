"""The two QC gates that guard a capture.

Both exist because a real capture got past everything else. Stage 2 reported
100 percent of frames registered on footage whose demos shared almost no
viewpoint with the scan, and produced an 85 m camera trajectory across a
0.69 m desk.
"""

import cv2
import numpy as np
import pytest

from wristview.camera import Intrinsics
from wristview.geometry import invert_pose, make_pose
from wristview.qc import (
    MARKER_POSITION_MAX_CM,
    PREFLIGHT_PASS_RATIO,
    detect_marker,
    marker_agreement,
    marker_corners_local,
    marker_gates,
    marker_pose_in_camera,
    preflight_gate,
)

DICT = "DICT_4X4_50"
MARKER_ID = 0
SIDE_M = 0.10


@pytest.fixture
def intrinsics() -> Intrinsics:
    return Intrinsics(width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0)


ORIGIN = np.zeros(3)
IDENTITY = np.eye(4)


def look_at(eye, target=None) -> np.ndarray:
    target = ORIGIN if target is None else target
    forward = np.asarray(target, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(forward @ up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return make_pose(np.stack([right, down, forward], axis=1), np.asarray(eye, float))


def render_marker(pose, intrinsics, marker_world=None) -> np.ndarray:
    """A marker at `marker_world`, seen from `pose`."""
    marker_world = IDENTITY if marker_world is None else marker_world
    from wristview.geometry import transform_points

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT))
    tile = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, 400)
    bordered = np.full((480, 480), 255, np.uint8)
    bordered[40:440, 40:440] = tile

    world = transform_points(marker_world, marker_corners_local(SIDE_M))
    cam = transform_points(invert_pose(pose), world)
    if cam[:, 2].min() <= 1e-6:
        raise ValueError("marker behind camera")
    pixels, _ = intrinsics.project(cam)

    source = np.array([[40, 40], [439, 40], [439, 439], [40, 439]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, pixels.astype(np.float32))
    canvas = np.full((intrinsics.height, intrinsics.width), 255, np.uint8)
    warped = cv2.warpPerspective(
        bordered, homography, (intrinsics.width, intrinsics.height),
        borderMode=cv2.BORDER_TRANSPARENT, dst=canvas,
    )
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def clip(tmp_path, intrinsics):
    """A short clip of a fixed marker with exactly known camera poses."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    names, poses = [], []
    for index in range(12):
        angle = index / 12 * 2 * np.pi
        pose = look_at([0.5 * np.cos(angle), 0.5 * np.sin(angle), 0.6])
        cv2.imwrite(str(frames_dir / f"f{index:03d}.png"), render_marker(pose, intrinsics))
        names.append(f"f{index:03d}.png")
        poses.append(pose)
    return frames_dir, names, np.stack(poses), np.ones(12, bool)


class TestPreflightGate:
    def test_passes_a_good_overlap(self):
        g = preflight_gate(436, 572)          # session 3, measured
        assert g.passed and g.value == pytest.approx(0.76, abs=0.01)

    def test_fails_the_capture_that_could_not_localize(self):
        g = preflight_gate(126, 700)          # session 1, measured
        assert not g.passed

    def test_marginal_capture_does_not_pass(self):
        g = preflight_gate(212, 549)          # session 2, measured, ratio 0.39
        assert not g.passed
        assert g.value < PREFLIGHT_PASS_RATIO

    def test_zero_ceiling_does_not_divide_by_zero(self):
        assert preflight_gate(100, 0).value == 0.0

    def test_records_its_evidence(self):
        assert "126" in preflight_gate(436, 572).detail


class TestMarkerPose:
    def test_recovers_a_known_camera_pose(self, intrinsics):
        pose = look_at([0.3, -0.2, 0.7])
        corners = detect_marker(render_marker(pose, intrinsics), DICT, MARKER_ID)
        assert corners is not None
        marker_cam = marker_pose_in_camera(corners, SIDE_M, intrinsics)
        assert marker_cam is not None
        implied = invert_pose(marker_cam)   # marker at world origin
        # 2 cm: the synthetic marker spans about 86 px at this distance, so
        # sub-pixel corner error plus warp quantisation is a centimetre or two.
        assert np.linalg.norm(implied[:3, 3] - pose[:3, 3]) < 0.02

    def test_returns_none_without_a_marker(self, intrinsics):
        blank = np.full((intrinsics.height, intrinsics.width, 3), 200, np.uint8)
        assert detect_marker(blank, DICT, MARKER_ID) is None


class TestMarkerAgreement:
    def test_correct_poses_agree_closely(self, clip, intrinsics):
        frames_dir, names, poses, valid = clip
        r = marker_agreement(frames_dir, names, poses, valid, intrinsics, SIDE_M, DICT,
                             MARKER_ID, marker_world=IDENTITY)
        assert r["frames"] >= 10
        assert r["position_median_cm"] < 3.0, r
        assert r["rotation_median_deg"] < 2.0, r
        assert all(g.passed for g in marker_gates(r))

    def test_catches_a_displaced_camera(self, clip, intrinsics):
        """The check has to fail when the pose is wrong. That is its job."""
        frames_dir, names, poses, valid = clip
        wrong = poses.copy()
        wrong[:, 0, 3] += 0.25          # 25 cm sideways
        # The reference must be independent, or the error cancels. That was a
        # real defect in this check before a test caught it.
        r = marker_agreement(frames_dir, names, wrong, valid, intrinsics, SIDE_M, DICT,
                             MARKER_ID, marker_world=IDENTITY)
        assert r["position_median_cm"] > MARKER_POSITION_MAX_CM
        assert not marker_gates(r)[0].passed

    def test_catches_a_rotated_camera(self, clip, intrinsics):
        from scipy.spatial.transform import Rotation

        frames_dir, names, poses, valid = clip
        wrong = poses.copy()
        spin = Rotation.from_euler("z", 20, degrees=True).as_matrix()
        wrong[:, :3, :3] = spin @ wrong[:, :3, :3]
        r = marker_agreement(frames_dir, names, wrong, valid, intrinsics, SIDE_M, DICT,
                             MARKER_ID, marker_world=IDENTITY)
        assert r["rotation_median_deg"] > 5.0

    def test_a_self_referential_reference_is_flagged(self, clip, intrinsics):
        """Without an independent reference the check cannot see a rigid error."""
        frames_dir, names, poses, valid = clip
        wrong = poses.copy()
        wrong[:, 0, 3] += 0.25
        r = marker_agreement(frames_dir, names, wrong, valid, intrinsics,
                             SIDE_M, DICT, MARKER_ID)
        assert r["reference"] == "self"
        # The rigid shift cancels, which is exactly why Stage 2 must not do this.
        assert r["position_median_cm"] < 3.0

    def test_reports_when_no_marker_is_visible(self, tmp_path, intrinsics):
        frames_dir = tmp_path / "blank"
        frames_dir.mkdir()
        names = []
        for i in range(4):
            cv2.imwrite(str(frames_dir / f"b{i}.png"),
                        np.full((intrinsics.height, intrinsics.width, 3), 200, np.uint8))
            names.append(f"b{i}.png")
        poses = np.repeat(np.eye(4)[None], 4, axis=0)
        r = marker_agreement(frames_dir, names, poses, np.ones(4, bool),
                             intrinsics, SIDE_M, DICT, MARKER_ID, marker_world=IDENTITY)
        assert r["frames"] == 0
        assert not marker_gates(r)[0].passed
