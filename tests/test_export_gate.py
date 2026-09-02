"""The viewpoint gate must stop the export, not the download.

`coverage.render_viewpoint_coverage` was correct and ran only in Stage 5, which
happens after a pod has trained a splat and drawn every frame. real27full
exported 4,673 cameras, trained for 40 minutes, rendered 14,019 frames and
downloaded 1.9 GB before reaching it. 22 of its 30 clips fail the gate.

So these tests do not check the arithmetic, which `test_coverage.py` covers.
They check the two things that make it a gate: the export refuses to write
cameras when coverage fails, and it reads the scan poses from the file that
holds them in metres.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS.parent))

from tools import export_wrist_cameras as exporter  # noqa: E402


def _run_dir(tmp_path: Path, wrist_height_m: float, scan_heights_m) -> Path:
    """A minimal run: a flat desk at z=0, scan views above it, one clip.

    The plane normal is +z and the offset is 0, so a camera's height above the
    desk is just its z. That keeps the fixture readable: the only quantity
    under test is where the wrist camera sits relative to the scan.
    """
    run = tmp_path / "run"
    scene = run / "01_scene"
    scene.mkdir(parents=True)
    (scene / "scale.json").write_text(json.dumps({
        "scale_factor": 1.0,
        "diagnostics": {"desk_plane": {"normal": [0.0, 0.0, 1.0], "offset_m": 0.0}},
    }))
    # Put the scan directly over the wrist camera in plan view, so the only
    # quantity separating them is height. A scan placed to one side would fail
    # the gate on lateral distance and prove nothing about the floor.
    offset = exporter.wrist_camera_offset(
        yaml.safe_load((TOOLS.parent / "configs" / "default.yaml").read_text())
        ["render"]["wrist_camera"]
    )
    ee = np.eye(4)
    ee[:3, 3] = [0.2, 0.0, 0.0]
    camera_xy = (ee @ offset)[:2, 3]

    frames = []
    for index, height in enumerate(scan_heights_m):
        pose = np.eye(4)
        pose[:3, 3] = [camera_xy[0] + 0.02 * index, camera_xy[1], height]
        frames.append({"name": f"scan_{index:05d}.jpg",
                       "pose_world_from_cam": pose.tolist()})
    (scene / "cameras.json").write_text(json.dumps({
        "intrinsics": {"width": 1920, "height": 1080, "fx": 1400.0, "fy": 1400.0,
                       "cx": 960.0, "cy": 540.0, "model": "PINHOLE"},
        "frames": frames,
    }))

    clip = run / "04_retarget" / "demo_0"
    clip.mkdir(parents=True)
    # The mount lifts the camera off the end effector, so place the end
    # effector low enough that the camera lands at `wrist_height_m`.
    poses = np.tile(ee, (12, 1, 1))
    poses[:, 2, 3] += wrist_height_m - float((ee @ offset)[2, 3])
    np.savez(clip / "ee_trajectory.npz", poses=poses,
             poses_video_rate=poses, hand_valid=np.ones(len(poses), bool),
             hand_valid_control=np.ones(len(poses), bool))
    return run


def _export(run: Path, out: Path) -> int:
    argv = sys.argv
    sys.argv = ["export_wrist_cameras", "--run", str(run), "--clip", "demo_0",
                "--out", str(out)]
    try:
        return exporter.main()
    finally:
        sys.argv = argv


def test_export_refuses_when_the_wrist_camera_is_below_the_scan(tmp_path):
    """Every wrist frame under the scan floor. Nothing may be written."""
    run = _run_dir(tmp_path, wrist_height_m=0.05, scan_heights_m=[0.4] * 12)
    out = tmp_path / "out"

    assert _export(run, out) == 1

    assert not (out / "wrist_cameras.npz").exists()
    assert not (out / "cameras.json").exists()
    # The evidence survives the refusal, or the next person cannot see why.
    report = json.loads((out / "viewpoint_coverage.json").read_text())
    assert report["passed"] is False
    assert report["fraction_below_scan_floor"] == 1.0
    assert report["failures"]


def test_export_refuses_a_run_with_no_control_rate_validity(tmp_path):
    """An older run would otherwise export every camera as valid."""
    run = _run_dir(tmp_path, wrist_height_m=0.25, scan_heights_m=[0.2, 0.3])
    traj = run / "04_retarget" / "demo_0" / "ee_trajectory.npz"
    data = dict(np.load(traj))
    data.pop("hand_valid_control")
    np.savez(traj, **data)
    with pytest.raises(SystemExit, match="hand_valid_control"):
        _export(run, tmp_path / "out")


def test_export_drops_unmeasured_cameras(tmp_path):
    """Marking is not enough; an unmeasured camera must not be shipped."""
    run = _run_dir(tmp_path, wrist_height_m=0.25,
                   scan_heights_m=[0.18, 0.22, 0.25, 0.28, 0.32, 0.36])
    traj = run / "04_retarget" / "demo_0" / "ee_trajectory.npz"
    data = dict(np.load(traj))
    keep = np.ones(len(data["poses"]), bool)
    keep[:3] = False
    data["hand_valid_control"] = keep
    np.savez(traj, **data)
    out = tmp_path / "out"

    assert _export(run, out) == 0

    stored = np.load(out / "wrist_cameras.npz")
    assert len(stored["view_matrices"]) == int(keep.sum())
    assert stored["source_index"].tolist() == list(range(3, len(keep)))
    meta = json.loads((out / "cameras.json").read_text())
    assert meta["frames_dropped_unmeasured"] == 3
    assert meta["frames"] == int(keep.sum())


def test_export_writes_cameras_when_the_scan_covers_the_wrist(tmp_path):
    run = _run_dir(tmp_path, wrist_height_m=0.25,
                   scan_heights_m=[0.18, 0.22, 0.25, 0.28, 0.32, 0.36])
    out = tmp_path / "out"

    assert _export(run, out) == 0

    assert (out / "wrist_cameras.npz").exists()
    assert (out / "cameras.json").exists()
    assert json.loads((out / "viewpoint_coverage.json").read_text())["passed"] is True


def test_scan_centres_are_read_from_the_metric_poses(tmp_path):
    """Not from the COLMAP model, which Stage 1 has already transformed.

    Reading `01_scene/colmap/sparse` back and multiplying by `scale_factor`
    applies the scale twice. On real27full that shrank every scan camera by
    12.34x and moved the scan floor from 15.2 cm to 32.6 cm, which reads as a
    capture that never went low enough.
    """
    run = _run_dir(tmp_path, wrist_height_m=0.25, scan_heights_m=[0.2, 0.5])
    centres = exporter.scan_centres(run)

    assert centres.shape == (2, 3)
    assert centres[:, 2].tolist() == [0.2, 0.5]

    source = ast.parse((TOOLS / "export_wrist_cameras.py").read_text())
    names = {node.value for node in ast.walk(source)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "01_scene/cameras.json" in " ".join(names) or "cameras.json" in names
    assert not any("sparse" in name for name in names), (
        "the export must not read the COLMAP model; its units differ from "
        "Stage 1's poses depending on whether the transform was applied"
    )


def test_the_gate_is_called_before_anything_is_written():
    """A gate after the write is a report. Order is the whole point."""
    source = (TOOLS / "export_wrist_cameras.py").read_text()
    tree = ast.parse(source)
    main = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "main")

    gate_line = next(
        node.lineno for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "render_viewpoint_coverage"
    )
    write_line = next(
        node.lineno for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "savez_compressed"
    )
    assert gate_line < write_line


@pytest.mark.parametrize("missing", ["scale.json", "cameras.json"])
def test_export_stops_when_the_gate_cannot_run(tmp_path, missing):
    """No plane, no scan poses, no export. Silence is not a pass."""
    run = _run_dir(tmp_path, wrist_height_m=0.25, scan_heights_m=[0.2, 0.3])
    (run / "01_scene" / missing).unlink()
    with pytest.raises((SystemExit, FileNotFoundError)):
        _export(run, tmp_path / "out")
    assert not (tmp_path / "out" / "wrist_cameras.npz").exists()
