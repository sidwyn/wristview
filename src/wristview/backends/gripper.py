"""Parallel-jaw gripper model.

Generic, configured by URDF path. When no URDF is given, a parametric gripper
is built from the config: two jaws on a palm, which is all Stage 4 needs for
width limits and all Stage 5 needs to draw.

Gripper frame, fixed across Stage 4 and Stage 5:
    z  approach direction, pointing out of the jaws
    x  closing axis, from one jaw to the other
    y  completes the right-handed frame
The origin sits between the jaw tips at the grasp point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..logging_setup import get

log = get(__name__)


@dataclass
class GripperSpec:
    """Everything the pipeline needs to know about the end effector."""

    max_width_m: float
    min_width_m: float
    finger_length_m: float
    palm_depth_m: float
    source: str
    urdf_path: str | None = None

    def clamp_width(self, width: float | np.ndarray):
        return np.clip(width, self.min_width_m, self.max_width_m)

    def to_dict(self) -> dict:
        return {
            "max_width_m": self.max_width_m,
            "min_width_m": self.min_width_m,
            "finger_length_m": self.finger_length_m,
            "palm_depth_m": self.palm_depth_m,
            "source": self.source,
            "urdf_path": self.urdf_path,
        }


def load(config: dict) -> GripperSpec:
    """Build a gripper spec, from a URDF when one is configured."""
    urdf_path = (config.get("urdf") or "").strip()
    defaults = config.get("gripper", {})

    if urdf_path:
        spec = _from_urdf(Path(urdf_path), defaults)
        if spec is not None:
            return spec
        log.warning("could not read gripper limits from %s; using config values", urdf_path)

    return GripperSpec(
        max_width_m=float(defaults.get("max_width_m", 0.085)),
        min_width_m=float(defaults.get("min_width_m", 0.0)),
        finger_length_m=float(defaults.get("finger_length_m", 0.055)),
        palm_depth_m=float(defaults.get("palm_depth_m", 0.06)),
        source="config",
        urdf_path=None,
    )


def _from_urdf(path: Path, defaults: dict) -> GripperSpec | None:
    """Read the jaw travel from a URDF's prismatic finger joints.

    A parallel-jaw gripper's opening is twice one finger's travel, because
    both jaws move. A single-finger design is handled by taking the span.
    """
    if not path.exists():
        log.warning("gripper URDF not found: %s", path)
        return None
    try:
        import yourdfpy

        model = yourdfpy.URDF.load(str(path), load_meshes=False, build_scene_graph=False)
    except Exception as exc:  # noqa: BLE001 - a bad URDF must not kill the run
        log.warning("failed to parse URDF %s: %s", path, exc)
        return None

    travels = []
    for joint in model.robot.joints:
        if joint.type != "prismatic" or joint.limit is None:
            continue
        lower = float(joint.limit.lower or 0.0)
        upper = float(joint.limit.upper or 0.0)
        travels.append(abs(upper - lower))

    if not travels:
        log.warning("URDF %s has no prismatic joints; cannot read jaw travel", path)
        return None

    travels.sort()
    max_width = travels[-1] * (2.0 if len(travels) >= 2 else 1.0)
    log.info(
        "gripper from URDF %s: %d prismatic joints, max opening %.4f m",
        path.name, len(travels), max_width,
    )
    return GripperSpec(
        max_width_m=max_width,
        min_width_m=0.0,
        finger_length_m=float(defaults.get("finger_length_m", 0.055)),
        palm_depth_m=float(defaults.get("palm_depth_m", 0.06)),
        source="urdf",
        urdf_path=str(path),
    )


def jaw_boxes(spec: GripperSpec, width: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Boxes making up the gripper at a given opening, in the gripper frame.

    Each entry is a centre and a half-extent. Stage 5 rasterizes these.
    """
    half_width = float(np.clip(width, spec.min_width_m, spec.max_width_m)) / 2.0
    jaw_thickness = 0.008
    jaw_height = 0.016
    finger = spec.finger_length_m
    palm = spec.palm_depth_m

    boxes = [
        # Palm, sitting behind the jaws along -z.
        (
            np.array([0.0, 0.0, -finger - palm / 2.0]),
            np.array([max(half_width, 0.02) + 0.012, jaw_height, palm / 2.0]),
        )
    ]
    for side in (-1.0, 1.0):
        boxes.append(
            (
                np.array([side * (half_width + jaw_thickness / 2.0), 0.0, -finger / 2.0]),
                np.array([jaw_thickness / 2.0, jaw_height / 2.0, finger / 2.0]),
            )
        )
    return boxes


def write_default_urdf(path: Path, spec: GripperSpec) -> Path:
    """Write a URDF for the parametric gripper, so runs are reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    travel = spec.max_width_m / 2.0
    path.write_text(
        f"""<?xml version="1.0"?>
<robot name="wristview_parallel_jaw">
  <link name="palm">
    <visual>
      <geometry><box size="{spec.max_width_m + 0.024} 0.032 {spec.palm_depth_m}"/></geometry>
    </visual>
  </link>
  <link name="left_finger">
    <visual>
      <geometry><box size="0.008 0.016 {spec.finger_length_m}"/></geometry>
    </visual>
  </link>
  <link name="right_finger">
    <visual>
      <geometry><box size="0.008 0.016 {spec.finger_length_m}"/></geometry>
    </visual>
  </link>
  <joint name="left_finger_joint" type="prismatic">
    <parent link="palm"/>
    <child link="left_finger"/>
    <axis xyz="-1 0 0"/>
    <limit lower="0.0" upper="{travel}" effort="80" velocity="0.2"/>
  </joint>
  <joint name="right_finger_joint" type="prismatic">
    <parent link="palm"/>
    <child link="right_finger"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="{travel}" effort="80" velocity="0.2"/>
  </joint>
</robot>
"""
    )
    return path
