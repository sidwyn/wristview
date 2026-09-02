"""A wrist camera below the work surface is not a viewpoint.

`wrist_camera_offset` validates `standoff_m`, a parameter it is handed, and
nothing validated the position that comes out. real27 exported 309 of 4,673
cameras below the desk plane, the lowest at -20.3 cm, and they rendered as
black or as a flat blue wash.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from wristview.mount import assert_above_plane

UP = np.array([0.0, 0.0, 1.0])
TOOL = Path(__file__).resolve().parents[1] / "tools" / "export_wrist_cameras.py"


def test_a_camera_above_the_desk_passes():
    report = assert_above_plane(np.array([[0, 0, 0.25], [0, 0, 0.10]]), UP, 0.0)
    assert report["passed"] is True
    assert report["frames_below"] == 0
    assert report["lowest_m"] == pytest.approx(0.10)


def test_one_camera_below_raises_and_names_the_frame():
    with pytest.raises(ValueError, match=r"1 of 3 wrist cameras are below"):
        assert_above_plane(np.array([[0, 0, 0.25], [0, 0, -0.02], [0, 0, 0.20]]),
                           UP, 0.0, clip_id="demo_9")


def test_the_message_names_the_clip_and_the_depth():
    with pytest.raises(ValueError) as excinfo:
        assert_above_plane(np.array([[0, 0, -0.203], [0, 0, 0.2]]), UP, 0.0,
                           clip_id="demo_28")
    message = str(excinfo.value)
    assert "demo_28" in message
    assert "-20.3 cm" in message
    # It must also stop the obvious wrong fix, which the mount geometry makes
    # worse rather than better.
    assert "standoff_m" in message


def test_an_offset_plane_is_respected():
    """The desk is rarely at the origin."""
    positions = np.array([[0, 0, 0.30], [0, 0, 0.34]])
    with pytest.raises(ValueError):
        assert_above_plane(positions, UP, 0.32)
    assert assert_above_plane(positions, UP, 0.29)["passed"] is True


def test_invalid_frames_are_not_judged():
    """A frame nothing will render cannot fail a render check."""
    positions = np.array([[0, 0, -0.05], [0, 0, 0.20]])
    report = assert_above_plane(positions, UP, 0.0, valid=np.array([False, True]))
    assert report["passed"] is True
    assert report["frames"] == 1


def test_a_tilted_plane_uses_the_normal_not_z():
    """A desk is never exactly level, and z is not the same as height.

    This camera has a POSITIVE z and still sits under a tilted desk, so a
    check written against the z axis would pass it.
    """
    normal = np.array([0.0, 0.5, 0.866])
    camera = np.array([[0.0, -0.20, 0.05]])
    assert camera[0, 2] > 0
    assert float(camera[0] @ (normal / np.linalg.norm(normal))) < 0
    with pytest.raises(ValueError):
        assert_above_plane(camera, normal, 0.0)


def test_the_export_calls_it_before_writing():
    """The point is to stop the upload, not to describe it afterwards."""
    source = TOOL.read_text()
    tree = ast.parse(source)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert_line = next(
        n.lineno for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "assert_above_plane"
    )
    write_line = next(
        n.lineno for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "savez_compressed"
    )
    assert assert_line < write_line
