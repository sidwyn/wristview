"""A run must be reconstructable from its own directory.

real26's Stage 4 was run with `--set retarget.frame_range=[57,231]`. That trim
cut 148 tracked frames and produced 181 frozen render frames. Nothing in the
run directory recorded it: `--set` does not reach `config.yaml`, the `stage`
subcommand never rewrites that file, and no `meta.json` held the overrides.
Explaining the frozen tail needed a re-run of the stage.
"""

from __future__ import annotations

import json

from wristview.config import Config
from wristview.runctx import RunContext, StageRecorder


def test_config_remembers_its_overrides():
    config = Config.load(None, {"render": {"width": 800}})
    assert config.overrides == {"render": {"width": 800}}
    assert config.get("render.width") == 800


def test_config_without_overrides_records_none():
    assert Config.load(None).overrides == {}


def test_the_invocation_lands_in_the_run_record(tmp_path):
    config = Config.load(None, {"retarget": {"frame_range": [57, 231]}})
    ctx = RunContext(root=tmp_path / "r", config=config, run_id="r")
    ctx.record_invocation([4])

    entry = json.loads((tmp_path / "r" / "run_record.jsonl").read_text().strip())
    assert entry["overrides"] == {"retarget": {"frame_range": [57, 231]}}
    assert entry["stages"] == [4]
    assert entry["run_id"] == "r"


def test_the_record_appends_rather_than_replacing(tmp_path):
    """Every run of every stage, in order. The last does not hide the rest."""
    ctx = RunContext(root=tmp_path / "r", config=Config.load(None), run_id="r")
    ctx.record_invocation([3])
    ctx.record_invocation([4])
    ctx.record_invocation([5])
    lines = (tmp_path / "r" / "run_record.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["stages"] for line in lines] == [[3], [4], [5]]


def test_stage_meta_carries_the_overrides(tmp_path):
    config = Config.load(None, {"render": {"height": 360}})
    ctx = RunContext(root=tmp_path / "r", config=config, run_id="r")
    StageRecorder(ctx, 5, "render").write("ok")
    meta = json.loads((tmp_path / "r" / "05_render" / "meta.json").read_text())
    assert meta["config_overrides"] == {"render": {"height": 360}}
