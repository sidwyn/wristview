"""Where the wrist camera sits on the gripper.

Lives here rather than in Stage 5 because Stage 4 has to check it and stages
are not allowed to import each other, a rule `test_no_stage_imports_another_stage`
enforces. Two copies of this geometry is the alternative, and a second copy is
how a check ends up measuring its own arithmetic instead of the pipeline's: an
independently written version of this reported a 0.15 m standoff against a
correct 0.25 m, because it measured from the wrist while the mount is specified
from the finger tips.
"""

from __future__ import annotations

import numpy as np

from .geometry import make_pose


def wrist_camera_offset(config: dict) -> np.ndarray:
    """The wrist camera's pose in the gripper frame.

    A real wrist camera is bolted to the wrist: it sits back from the jaws and
    above them, and it is aimed down the approach axis at the point the
    fingers close on. So this is a fixed mount, built by aiming, not a pose
    derived from hand anatomy.

    The previous version took a translation and roll-pitch-yaw in the gripper
    frame and pointed the camera along the gripper's own +z. That fails on a
    wrapped grasp: the approach axis of a hand curled around a mug handle is
    roughly horizontal, so the camera looked across the desk at the monitor
    instead of down at the object. Aiming at the finger-tip midpoint makes the
    framing independent of how the hand happens to be oriented.

    Gripper frame, as everywhere else in this pipeline:
        z  approach, out of the jaws
        x  closing axis, thumb to index
        y  completes the right-handed frame, which is "up" out of the back

    Config:
        mount_back_m   how far behind the finger tips the camera sits, along -z
        mount_up_m     how far above the jaw line, along -y
        aim_ahead_m    how far beyond the finger tips the camera looks. Larger
                       values push the finger tips lower in frame.
        grasp_offset_m distance from the gripper origin to the finger tips
                       along +z. Matches retarget.origin_offset_m.
        standoff_m     when set, the mount is scaled so the camera sits exactly
                       this far from the finger tips, keeping the direction and
                       the framing. This is the one knob worth turning, because
                       it has to be matched to the scan, not chosen by eye: the
                       splat can only be sharp where training views actually
                       were. Session 4 measured a median camera-to-surface
                       distance of 0.264 m in its closest pass, with 2 frames
                       under 0.15 m, so 0.25 m is where the data supports a
                       render and 0.15 m is not.
    """
    back = float(config.get("mount_back_m", 0.12))
    up = float(config.get("mount_up_m", 0.07))
    aim_ahead = float(config.get("aim_ahead_m", 0.06))

    standoff = config.get("standoff_m")
    if standoff is not None:
        # Scale the whole mount, so framing is preserved and only the distance
        # changes. Without scaling aim_ahead too, moving the camera back would
        # also swing the finger tips up the frame.
        current = float(np.hypot(back, up))
        if current > 1e-9:
            factor = float(standoff) / current
            back, up, aim_ahead = back * factor, up * factor, aim_ahead * factor
    grasp_offset = float(config.get("grasp_offset_m", 0.02))

    # Everything is expressed in the gripper frame, so the axes are the
    # identity basis and the geometry reads directly.
    fingertips = np.array([0.0, 0.0, grasp_offset])
    eye = fingertips + np.array([0.0, -up, -back])
    target = fingertips + np.array([0.0, 0.0, aim_ahead])

    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    # The gripper's -y is up, so the camera's down axis starts from +y.
    reference_up = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, reference_up)
    norm = np.linalg.norm(right)
    if norm < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / norm
    down = np.cross(forward, right)

    return make_pose(np.stack([right, down, forward], axis=1), eye)
