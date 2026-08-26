"""The gate that would have caught the 25x hand error on day one.

Written against the real defect. WiLoR reports the hand for a camera with a
focal length near 37500 px; the pipeline rescaled the translation onto our
intrinsics by scaling all three components, which leaves X/Z alone while the
focal changes and so shrinks the hand toward the optical axis by the focal
ratio. The first test reproduces exactly that and requires the gate to fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristview.camera import Intrinsics
from wristview.reprojection import (
    MAX_HAND_REPROJECTION_PX,
    effector_check,
    errors,
    gate,
    hand_check,
    object_check,
    project,
    standoff_check,
    summarise,
    to_camera,
)

NUM_LANDMARKS = 21


@pytest.fixture
def intrinsics() -> Intrinsics:
    return Intrinsics(width=1920, height=1080, fx=1468.0, fy=1468.0, cx=960.0, cy=540.0)


def _hand_in_camera(offset_x: float, depth: float = 0.45) -> np.ndarray:
    """A hand at a given lateral offset, as real joints in the camera frame."""
    rng = np.random.default_rng(0)
    joints = rng.normal(0, 0.03, (NUM_LANDMARKS, 3))
    joints[:, 2] = np.abs(joints[:, 2])
    return joints + np.array([offset_x, 0.0, depth])


# --------------------------------------------------------------------------
# the primitives
# --------------------------------------------------------------------------


def test_projection_round_trips(intrinsics):
    points = _hand_in_camera(0.2)
    pixels = project(points, intrinsics)
    assert np.isfinite(pixels).all()
    assert errors(points, pixels, intrinsics).max() < 1e-9


def test_points_behind_the_camera_are_not_silently_projected(intrinsics):
    behind = np.array([[0.1, 0.0, -0.5]])
    assert np.isnan(project(behind, intrinsics)).all()


def test_world_to_camera_inverts_a_camera_pose(intrinsics):
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    angle = 0.4
    pose[:3, :3] = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    world = np.array([[1.2, 2.3, 3.4]])
    cam = to_camera(world, pose)
    back = pose[:3, :3] @ cam[0] + pose[:3, 3]
    assert np.allclose(back, world[0])


# --------------------------------------------------------------------------
# the defect
# --------------------------------------------------------------------------


def test_the_hand_check_fails_on_the_uniform_scaling_bug(intrinsics):
    """Reproduce the real defect and require a failure.

    The hand is genuinely 30 cm off the optical axis. Scaling all three
    translation components by fx/wilor_focal keeps X/Z and so keeps the hand
    at the same *ratio*, but the projected offset is f * X/Z, so against our
    smaller focal it lands far too close to the centre.
    """
    wilor_focal = 37500.0
    scale = intrinsics.fx / wilor_focal

    truth = _hand_in_camera(0.30, depth=0.45)
    observed_px = project(truth, intrinsics)

    # What the broken code produced: the same hand expressed at WiLoR's scale,
    # then every component multiplied down.
    at_wilor_scale = truth / scale
    broken = at_wilor_scale * scale       # uniform scaling of the whole thing
    # ...which is the identity on the joints, so the real bug is that the
    # translation alone was scaled while the projection used our focal. Model
    # it directly: lateral position divided by the focal ratio.
    broken = truth.copy()
    broken[:, 0] *= scale
    broken[:, 1] *= scale

    report = hand_check(
        broken[None, :, :], observed_px[None, :, :],
        np.ones(1, dtype=bool), intrinsics,
    )
    assert report["passed"] is False
    assert report["median_px"] > 100, report
    assert gate([report]), "the gate must stop a stage on this"


def test_the_hand_check_reports_the_all_joint_figure_too(intrinsics):
    """Gating on the root must not hide what the other joints are doing."""
    truth = _hand_in_camera(0.30)
    observed = project(truth, intrinsics)
    report = hand_check(
        truth[None, :, :], observed[None, :, :], np.ones(1, dtype=bool), intrinsics
    )
    assert "all_joints_median_px" in report
    assert "all_joints_weak_perspective_median_px" in report
    assert "weak perspective" in report["why_the_root"]


def test_the_hand_check_passes_a_correct_lift(intrinsics):
    truth = _hand_in_camera(0.30)
    observed_px = project(truth, intrinsics)
    report = hand_check(
        truth[None, :, :], observed_px[None, :, :], np.ones(1, dtype=bool), intrinsics
    )
    assert report["passed"] is True
    assert report["median_px"] < 1e-6
    assert report["evidence"] == "independent"
    assert gate([report]) == []


def test_a_small_jitter_still_passes(intrinsics):
    """The bound has to tolerate the honest residual, near 13 to 16 px."""
    truth = _hand_in_camera(0.30)
    observed = project(truth, intrinsics) + np.array([8.0, 6.0])
    report = hand_check(
        truth[None, :, :], observed[None, :, :], np.ones(1, dtype=bool), intrinsics
    )
    assert report["median_px"] < MAX_HAND_REPROJECTION_PX
    assert report["passed"] is True


# --------------------------------------------------------------------------
# evidence versus arithmetic
# --------------------------------------------------------------------------


def test_a_by_construction_check_is_labelled_as_such(intrinsics):
    """A zero here must never read as proof the pose is right."""
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, :3, 3] = [0.1, 0.05, 0.6]
    cams = np.repeat(np.eye(4)[None], 3, axis=0)
    centroids = project(poses[:, :3, 3], intrinsics)
    report = object_check(
        poses, np.ones(3, dtype=bool), centroids, np.ones(3, dtype=bool),
        cams, intrinsics,
    )
    assert report["median_px"] < 1e-6
    assert report["evidence"] == "by construction"
    assert "only a large value is informative" in report["note"]


def test_a_by_construction_check_still_fails_when_it_is_badly_wrong(intrinsics):
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, :3, 3] = [0.1, 0.05, 0.6]
    cams = np.repeat(np.eye(4)[None], 3, axis=0)
    centroids = project(poses[:, :3, 3], intrinsics) + np.array([400.0, 0.0])
    report = object_check(
        poses, np.ones(3, dtype=bool), centroids, np.ones(3, dtype=bool),
        cams, intrinsics,
    )
    assert report["passed"] is False
    assert gate([report])


def test_carried_frames_are_reported_separately_as_independent(intrinsics):
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, :3, 3] = [0.1, 0.05, 0.6]
    cams = np.repeat(np.eye(4)[None], 4, axis=0)
    centroids = project(poses[:, :3, 3], intrinsics)
    sources = np.array(["plane", "plane", "carried", "carried"], dtype=object)
    report = object_check(
        poses, np.ones(4, dtype=bool), centroids, np.ones(4, dtype=bool),
        cams, intrinsics, sources=sources,
    )
    assert report["carried_frames"]["evidence"] == "independent"
    assert report["carried_frames"]["frames"] == 2


# --------------------------------------------------------------------------
# the remaining quantities
# --------------------------------------------------------------------------


def test_effector_check_is_metric_and_expects_a_deliberate_pullback():
    """The effector is meant to sit behind the fingertips, not on them.

    A pixel version of this check reported 45 to 95 px on four session 6 clips
    that were doing exactly what Stage 4 tells them to, because the pullback is
    the design. What is checkable is its size.
    """
    hand = np.zeros((3, NUM_LANDMARKS, 3))
    hand[:, 4] = [0.10, 0.0, 0.45]
    hand[:, 8] = [0.10, 0.0, 0.45]
    tip = hand[:, 4]
    correct = tip - np.array([0.0, 0.0, 0.02])
    report = effector_check(
        correct, np.ones(3, dtype=bool), hand, np.ones(3, dtype=bool), 0.02
    )
    assert report["passed"] is True
    assert report["unit"] == "m"
    assert report["median_m"] == pytest.approx(0.02, abs=1e-6)


def test_effector_check_catches_a_pullback_that_never_arrived():
    hand = np.zeros((3, NUM_LANDMARKS, 3))
    hand[:, 4] = [0.10, 0.0, 0.45]
    hand[:, 8] = [0.10, 0.0, 0.45]
    on_the_tips = hand[:, 4].copy()
    report = effector_check(
        on_the_tips, np.ones(3, dtype=bool), hand, np.ones(3, dtype=bool), 0.20
    )
    assert report["passed"] is False
    assert gate([report])


def test_standoff_check_catches_a_standoff_that_never_reached_the_renderer():
    """real06b rendered with `standoff_m` unset and nothing reported it."""
    grasp = np.tile(np.array([0.1, 0.0, 0.45]), (4, 1))
    default_mount = grasp + np.array([0.0, -0.07, -0.12])   # the fallback
    report = standoff_check(default_mount, grasp, np.ones(4, dtype=bool), 0.25)
    assert report["passed"] is False
    assert gate([report])


def test_standoff_check_is_labelled_by_construction():
    """It can only report the config, and says so rather than implying more."""
    grasp = np.tile(np.array([0.1, 0.0, 0.45]), (4, 1))
    eye = grasp + np.array([0.0, -0.15, -0.2])              # exactly 0.25 m
    report = standoff_check(eye, grasp, np.ones(4, dtype=bool), 0.25)
    assert report["passed"] is True
    assert report["evidence"] == "by construction"
    assert "not a camera in the wrong place" in report["note"]


def test_an_empty_check_reports_nothing_rather_than_passing(intrinsics):
    report = summarise(np.array([]), "nothing", True, 25.0)
    assert report["passed"] is None
    assert report["frames"] == 0
    assert gate([report]) == [], "an empty check must not fail a stage either"
