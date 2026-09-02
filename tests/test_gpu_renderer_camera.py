"""The CUDA renderer must draw the camera the config declares.

`tools/cuda_job/train_gsplat.py` had the wrist camera as literals:

    width, height, fov = 640, 480, 90.0
    back, up, aim_ahead, grasp_offset = 0.10, 0.10, 0.14, 0.02

That is the mount as it stood BEFORE the real27 correction. The correction
narrowed the lens from 90 to 62.1 degrees to match the camera the experiment
actually records, and pitched the mount 12.02 degrees down so the finger tips
stayed in frame, and it moved measured alpha from 0.635 to 0.939. It changed
`configs/default.yaml` and `src/wristview/mount.py`. Nothing linked the CUDA
renderer, so it silently kept drawing the old camera for every GPU render
after that date, at the wrong size, with no pitch.

`--wrist-fov-deg` and `--wrist-size` look like they control it and do not:
they are read only by the holdout path.

These tests pin the two halves. The exporter must ship the camera, and the
renderer must read it from there rather than from literals.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "tools" / "cuda_job" / "train_gsplat.py"
EXPORTER = ROOT / "tools" / "export_for_gsplat.py"
CONFIG = ROOT / "configs" / "default.yaml"


def test_the_exporter_ships_the_wrist_camera():
    """Without this the renderer has nothing to read."""
    source = EXPORTER.read_text()
    assert '"wrist_camera": wrist_camera' in source, (
        "export.json must carry render.wrist_camera, or the CUDA renderer "
        "cannot know what lens to draw"
    )


def test_the_renderer_reads_the_camera_rather_than_hardcoding_it():
    source = RENDERER.read_text()
    assert 'export.get("wrist_camera")' in source, (
        "the trajectory render must take the camera from export.json"
    )
    assert "width, height, fov = 640, 480, 90.0" not in source, (
        "the pre-correction camera literals are back"
    )


def test_the_renderer_applies_the_pitch():
    """Pitch is what keeps the finger tips in frame at 62.1 degrees.

    Without it `fingertip_row_fraction` puts them past the bottom edge, which
    `test_mount_framing.py` already pins for the local renderer.
    """
    source = RENDERER.read_text()
    assert 'wc.get("pitch_down_deg"' in source, (
        "the CUDA renderer must apply pitch_down_deg like mount.py does"
    )


def test_the_renderer_uses_control_rate_poses():
    """A wrist frame is only useful paired with the action at that instant.

    It rendered `poses_video_rate` while s05_render reads `poses`, so the
    layer's frame count could never match the episode it composites with:
    268 against 209 on real31's demo_1.
    """
    source = RENDERER.read_text()
    assert 'traj["poses"] if "poses" in traj' in source, (
        "the CUDA renderer must prefer control-rate poses"
    )


def test_no_literal_ninety_degree_lens_survives_in_the_render_path():
    """90 degrees is the number the correction removed. It must not reappear."""
    source = RENDERER.read_text()
    start = source.index("# --- render the wrist trajectories")
    body = source[start:]
    assert not re.search(r"\bfov\s*=\s*90", body), "a 90 degree lens is hardcoded again"


def test_the_config_still_declares_the_corrected_mount():
    """If these move, the renderer follows them; that is the point."""
    wrist = yaml.safe_load(CONFIG.read_text())["render"]["wrist_camera"]
    assert wrist["fov_deg"] == 62.1
    assert wrist["pitch_down_deg"] == 12.02


def test_both_files_still_parse():
    for path in (RENDERER, EXPORTER):
        ast.parse(path.read_text())
