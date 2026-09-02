"""The config says where the finger tips land. The code must agree.

`configs/default.yaml` carried the comment "These three put the finger tips
about 77 percent of the way down the frame" from the first commit. The code put
them at 86.6 per cent. Nobody compared the two, because a number in a comment
has no reader.

The claim now lives in `render.wrist_camera.framing_row_fraction` and these
tests are its reader. They also pin the two facts that made the framing matter:
the finger tips sit at a fixed angle below the optical axis, and narrowing the
lens to the real camera's field of view drops them out of frame unless the
mount is pitched.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from wristview.mount import fingertip_row_fraction, wrist_camera_offset

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
WIDTH, HEIGHT = 640, 360
# The real wrist camera, from wrist_K.npy at 1280x720: fx 1062.5, fy 943.6.
REAL_HORIZONTAL_FOV_DEG = 62.1


@pytest.fixture
def config() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text())["render"]["wrist_camera"]
    # standoff_m is deliberately required and unset in the shipped config.
    cfg["standoff_m"] = 0.25
    return cfg


def test_the_delivered_framing_matches_the_declared_framing(config):
    declared = float(config["framing_row_fraction"])
    delivered = fingertip_row_fraction(config, WIDTH, HEIGHT)
    assert delivered == pytest.approx(declared, abs=0.005), (
        f"the config declares the finger tips at {declared:.1%} of frame "
        f"height and the mount delivers {delivered:.1%}. Change the mount or "
        f"change the declaration, but do not leave them disagreeing."
    )


def test_the_finger_tips_are_inside_the_frame(config):
    assert 0.0 < fingertip_row_fraction(config, WIDTH, HEIGHT) < 1.0


def test_the_lens_matches_the_camera_we_recorded(config):
    assert float(config["fov_deg"]) == pytest.approx(REAL_HORIZONTAL_FOV_DEG, abs=0.5), (
        "the rendered lens must match the wrist camera the experiment records, "
        "or arm A and arm C are not comparable"
    )


def test_the_old_settings_would_fail_this_check(config):
    """90 degrees with no pitch delivered 86.6 per cent, not the claimed 77."""
    old = copy.deepcopy(config)
    old["fov_deg"], old["pitch_down_deg"] = 90.0, 0.0
    assert fingertip_row_fraction(old, WIDTH, HEIGHT) == pytest.approx(0.866, abs=0.005)


def test_the_real_lens_without_pitch_puts_the_gripper_out_of_frame(config):
    """Why the pitch key exists. Narrowing alone loses the gripper."""
    unpitched = copy.deepcopy(config)
    unpitched["pitch_down_deg"] = 0.0
    assert fingertip_row_fraction(unpitched, WIDTH, HEIGHT) > 1.0


@pytest.mark.parametrize("standoff", [0.20, 0.25, 0.30, 0.35])
def test_framing_does_not_depend_on_standoff(config, standoff):
    """The mount scales, so distance changes and framing does not."""
    cfg = copy.deepcopy(config)
    cfg["standoff_m"] = standoff
    assert fingertip_row_fraction(cfg, WIDTH, HEIGHT) == pytest.approx(
        float(config["framing_row_fraction"]), abs=0.005
    )


def test_pitch_rotates_and_does_not_translate(config):
    """The coverage gate reads camera position. Pitch must not touch it."""
    unpitched = copy.deepcopy(config)
    unpitched["pitch_down_deg"] = 0.0
    a = wrist_camera_offset(config)
    b = wrist_camera_offset(unpitched)
    assert np.allclose(a[:3, 3], b[:3, 3])
    assert not np.allclose(a[:3, :3], b[:3, :3])


def test_pitching_down_raises_the_finger_tips_in_frame(config):
    """Sign check. Positive pitch_down_deg must move them up the image."""
    more = copy.deepcopy(config)
    more["pitch_down_deg"] = float(config["pitch_down_deg"]) + 5.0
    assert fingertip_row_fraction(more, WIDTH, HEIGHT) < fingertip_row_fraction(
        config, WIDTH, HEIGHT
    )
