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

from .camera import Intrinsics
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
        pitch_down_deg rotate the camera down about its own right axis, after
                       aiming. Position is unchanged, so this cannot move the
                       viewpoint or affect the coverage gate. It exists because
                       the mount's aim is set in metres and the lens is set in
                       degrees, and narrowing the lens to match the real camera
                       drops the finger tips out of frame without it.
        grasp_offset_m distance from the gripper origin to the finger tips
                       along +z. Matches retarget.origin_offset_m.
        standoff_m     REQUIRED. How far the camera sits from the finger tips.
                       The mount is scaled to this, keeping the direction and
                       the framing.

    `standoff_m` has no default and no fallback.

    It was optional, and null. Stage 4 warned that the wrist camera therefore
    could not be checked, and the warning fired on every take of real06b and
    real26 without stopping anything. An unset physical quantity is the trap
    this project keeps falling into: a distance the rig decides is not a
    distance the code may invent. Rule 6 in CLAUDE.md.

    0.25 m is the placement the UMI rig uses and it is what this project
    renders at. It was raised to 0.35 m once, as a workaround for an 18 cm
    scan floor, which traded framing for coverage. That is no longer needed:
    real26/bm's floor is 8.47 cm. Do not raise it again to hide a scan.
    """
    if config.get("standoff_m") is None:
        raise ValueError(
            "render.wrist_camera.standoff_m is not set. It is the distance "
            "from the finger tips to the camera and the rig decides it, so "
            "there is no default. Set it to 0.25 for the UMI placement. It "
            "was optional until real26/bm, and every run since real06b "
            "carried a warning that the wrist camera could not be checked."
        )
    standoff = float(config["standoff_m"])
    if not 0.05 <= standoff <= 1.0:
        raise ValueError(
            f"render.wrist_camera.standoff_m is {standoff} m, outside 0.05 to "
            f"1.0 m. That is not a wrist camera mount."
        )

    back = float(config.get("mount_back_m", 0.12))
    up = float(config.get("mount_up_m", 0.07))
    aim_ahead = float(config.get("aim_ahead_m", 0.06))

    # Scale the whole mount, so framing is preserved and only the distance
    # changes. Without scaling aim_ahead too, moving the camera back would
    # also swing the finger tips up the frame.
    current = float(np.hypot(back, up))
    if current > 1e-9:
        factor = standoff / current
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

    pose = make_pose(np.stack([right, down, forward], axis=1), eye)

    pitch = float(config.get("pitch_down_deg", 0.0))
    if pitch:
        angle = np.radians(-pitch)
        rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ])
        pose = pose @ make_pose(rotation, np.zeros(3))
    return pose


def fingertip_row_fraction(config: dict, width: int, height: int) -> float:
    """Where the finger tips land, as a fraction of frame height.

    The config carries the intended value in `framing_row_fraction` and this
    computes the delivered one. They are compared in
    `tests/test_mount_framing.py`.

    The check exists because the comment on these keys read "about 77 percent
    of the way down the frame" while the code put them at 86.6, and the two
    were never compared. A number in a comment is a claim with no reader, which
    is this project's most repeated defect.

    Values above 1.0 mean the finger tips are below the bottom of the frame and
    the gripper is not in the picture at all.
    """
    pose = wrist_camera_offset(config)
    tips = np.array([0.0, 0.0, float(config.get("grasp_offset_m", 0.02))])
    direction = np.linalg.inv(pose)[:3, :3] @ (tips - pose[:3, 3])
    if direction[2] <= 1e-9:
        raise ValueError(
            "the finger tips are behind the wrist camera, so the mount is not "
            "aimed at them. Check mount_back_m, mount_up_m and aim_ahead_m."
        )
    intrinsics = Intrinsics.from_fov(width, height, float(config["fov_deg"]))
    row = intrinsics.fy * direction[1] / direction[2] + intrinsics.cy
    return float(row / height)


def assert_above_plane(
    camera_positions: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    clip_id: str = "",
    valid: np.ndarray | None = None,
) -> dict:
    """Raise if any wrist camera is placed below the work surface.

    `wrist_camera_offset` checks `standoff_m`, a parameter. Nothing checked the
    result. real27 exported 309 cameras of 4,673 below the desk plane, the
    lowest at -20.3 cm, and they rendered as black or as a flat blue wash.

    A camera under the desk is not a viewpoint. There is no threshold to tune
    here and no band where it is acceptable, so this raises rather than warns.

    The report is returned as well as raised on, because the caller wants the
    per-clip count even when it passes.
    """
    positions = np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3)
    normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    keep = (np.ones(len(positions), dtype=bool) if valid is None
            else np.asarray(valid, dtype=bool))

    heights = positions @ normal - float(plane_offset)
    below = keep & (heights < 0.0)
    report = {
        "check": "wrist_camera_above_desk",
        "clip_id": clip_id,
        "frames": int(keep.sum()),
        "frames_below": int(below.sum()),
        "lowest_m": round(float(heights[keep].min()), 4) if keep.any() else None,
        "below_frames": [int(i) for i in np.nonzero(below)[0]],
        "passed": not bool(below.any()),
    }
    if below.any():
        worst = float(heights[below].min())
        raise ValueError(
            f"{clip_id or 'clip'}: {int(below.sum())} of {int(keep.sum())} wrist "
            f"cameras are below the desk plane, the lowest at "
            f"{worst * 100:.1f} cm. A camera under the work surface is not a "
            f"viewpoint and renders as background. Frames "
            f"{report['below_frames'][:12]}"
            f"{' ...' if len(report['below_frames']) > 12 else ''}. "
            f"Do not raise standoff_m to lift them: the mount is scaled along "
            f"the GRIPPER's up axis, so when the gripper is rolled a larger "
            f"standoff drives the camera further under, not clear of it."
        )
    return report
