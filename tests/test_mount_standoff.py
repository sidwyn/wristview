"""`standoff_m` is a physical distance, so the code may not invent one.

It was optional and null. Stage 4 warned on every take of real06b and real26
that the wrist camera could not be checked, and nothing stopped. CLAUDE.md
rule 6: defaults for physical quantities must not exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristview.mount import wrist_camera_offset

BASE = {"mount_back_m": 0.10, "mount_up_m": 0.10, "aim_ahead_m": 0.14,
        "grasp_offset_m": 0.02}


def test_an_unset_standoff_raises_rather_than_falling_back():
    with pytest.raises(ValueError, match="standoff_m is not set"):
        wrist_camera_offset(dict(BASE))


def test_an_explicit_none_also_raises():
    """null in YAML arrives as None, which is how it went unnoticed."""
    with pytest.raises(ValueError, match="standoff_m is not set"):
        wrist_camera_offset({**BASE, "standoff_m": None})


def test_the_camera_sits_exactly_the_requested_distance_from_the_fingertips():
    for standoff in (0.15, 0.25, 0.35):
        pose = wrist_camera_offset({**BASE, "standoff_m": standoff})
        fingertips = np.array([0.0, 0.0, BASE["grasp_offset_m"]])
        assert np.linalg.norm(pose[:3, 3] - fingertips) == pytest.approx(standoff, abs=1e-9)


def test_framing_is_preserved_across_standoffs():
    """Scaling the mount must move the camera, not re-aim it."""
    a = wrist_camera_offset({**BASE, "standoff_m": 0.25})
    b = wrist_camera_offset({**BASE, "standoff_m": 0.35})
    assert np.allclose(a[:3, :3], b[:3, :3], atol=1e-9)


def test_an_absurd_standoff_raises():
    for bad in (0.0, 0.001, 3.0):
        with pytest.raises(ValueError):
            wrist_camera_offset({**BASE, "standoff_m": bad})


def test_the_shipped_default_is_the_umi_placement():
    """0.25 m. 0.35 m was a workaround for an 18 cm scan floor, now retired."""
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    assert cfg["render"]["wrist_camera"]["standoff_m"] == 0.25
