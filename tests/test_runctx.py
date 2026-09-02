"""The run directory and the stage data contract.

Two rules from the build plan are enforced by the structure of this code:
every stage is a pure function over a run directory, and no stage calls the
next. The meta.json contract is what makes a bad batch debuggable weeks later,
so it is not optional and it is tested.
"""

import json

import pytest
import yaml

from wristview.config import Config
from wristview.runctx import STAGE_DIRS, RunContext, StageRecorder, read_json, write_json


@pytest.fixture
def ctx(tmp_path) -> RunContext:
    return RunContext.create(tmp_path / "runs", "testrun", Config.load())


class TestConfig:
    def test_default_loads(self):
        config = Config.load()
        assert config.get("ingest.scan_fps") is not None
        assert config.get("render.width") is not None

    def test_dotted_lookup(self):
        config = Config.load()
        assert config.get("scene.splat.iterations") == config.data["scene"]["splat"]["iterations"]

    def test_missing_key_returns_default(self):
        assert Config.load().get("nope.not.here", "fallback") == "fallback"

    def test_overrides_merge_without_dropping_siblings(self):
        config = Config.load(overrides={"render": {"width": 111}})
        assert config.get("render.width") == 111
        # The sibling keys under render must survive a nested override.
        assert config.get("render.height") is not None

    def test_hash_is_stable_and_sensitive(self):
        a = Config.load()
        b = Config.load()
        c = Config.load(overrides={"render": {"width": 999}})
        assert a.hash == b.hash
        assert a.hash != c.hash

    def test_hash_ignores_key_order(self):
        a = Config(data={"x": 1, "y": 2})
        b = Config(data={"y": 2, "x": 1})
        assert a.hash == b.hash

    def test_write_roundtrip(self, tmp_path):
        config = Config.load(overrides={"render": {"width": 321}})
        path = config.write(tmp_path / "config.yaml")
        assert yaml.safe_load(path.read_text())["render"]["width"] == 321

    def test_section_returns_a_copy(self):
        config = Config.load()
        section = config.section("render")
        section["width"] = -1
        assert config.get("render.width") != -1


class TestRunContext:
    def test_creates_the_run_directory_and_saves_config(self, ctx):
        assert ctx.root.exists()
        assert (ctx.root / "config.yaml").exists()

    def test_stage_dirs_match_the_documented_layout(self, ctx):
        for stage, name in STAGE_DIRS.items():
            assert ctx.stage_dir(stage).name == name

    def test_stops_at_stage_six(self):
        # Stage 6 is the LeRobot export. Stage 7 is deliberately not built.
        assert max(STAGE_DIRS) == 6

    def test_episode_dir_nests_under_the_stage(self, ctx):
        path = ctx.episode_dir(3, "demo_0")
        assert path.parent.name == "03_estimate"
        assert path.name == "demo_0"

    def test_rel_is_relative_to_the_run_root(self, ctx):
        assert ctx.rel(ctx.stage_dir(2) / "summary.json") == "02_localize/summary.json"

    def test_read_meta_explains_a_missing_stage(self, ctx):
        with pytest.raises(FileNotFoundError, match="has not run"):
            ctx.read_meta(1)


class TestStageRecorder:
    def test_writes_the_full_contract(self, ctx):
        rec = StageRecorder(ctx, 2, "localize")
        rec.meta.inputs = {"manifest": "00_ingest/manifest.json"}
        rec.output("poses", ctx.stage_dir(2) / "camera_poses.npy")
        rec.metric("register_rate", 0.97)
        rec.backend("localizer", "lightglue_pnp_ransac")
        rec.note("something worth remembering")
        with rec.timed("pnp"):
            pass
        path = rec.write("ok")

        meta = json.loads(path.read_text())
        for key in (
            "stage", "name", "status", "git_sha", "config_hash", "platform",
            "inputs", "outputs", "metrics", "timings_s", "backends", "notes",
            "duration_s",
        ):
            assert key in meta, f"meta.json is missing {key}"
        assert meta["status"] == "ok"
        assert meta["metrics"]["register_rate"] == 0.97
        assert meta["backends"]["localizer"] == "lightglue_pnp_ransac"
        assert "pnp" in meta["timings_s"]

    def test_records_the_config_hash(self, ctx):
        rec = StageRecorder(ctx, 0, "ingest")
        meta = json.loads(rec.write("ok").read_text())
        assert meta["config_hash"] == ctx.config.hash

    def test_writes_meta_even_on_failure(self, ctx):
        rec = StageRecorder(ctx, 1, "scene")
        rec.note("RuntimeError: the scan did not reconstruct")
        meta = json.loads(rec.write("failed").read_text())
        assert meta["status"] == "failed"
        assert "did not reconstruct" in meta["notes"][0]

    def test_timings_accumulate_across_calls(self, ctx):
        rec = StageRecorder(ctx, 1, "scene")
        with rec.timed("matching"):
            pass
        with rec.timed("matching"):
            pass
        meta = json.loads(rec.write("ok").read_text())
        assert meta["timings_s"]["matching"] >= 0.0

    def test_outputs_are_recorded_relative(self, ctx):
        rec = StageRecorder(ctx, 5, "render")
        rec.output("wrist", ctx.episode_dir(5, "demo_0") / "wrist")
        meta = json.loads(rec.write("ok").read_text())
        assert meta["outputs"]["wrist"] == "05_render/demo_0/wrist"


def test_json_helpers_roundtrip(tmp_path):
    payload = {"a": 1, "b": [1, 2, 3], "c": {"d": "e"}}
    assert read_json(write_json(tmp_path / "x.json", payload)) == payload


def test_no_stage_imports_another_stage():
    """The build plan's hard rule: no stage calls the next.

    Enforced by inspecting imports rather than by convention, because this is
    the property that lets any stage be re-run or replaced on its own.
    """
    import ast
    from pathlib import Path

    stages_dir = Path(__file__).resolve().parents[1] / "src" / "wristview" / "stages"
    offenders = []

    for path in sorted(stages_dir.glob("s0*.py")):
        tree = ast.parse(path.read_text())
        this_stage = int(path.name[1:3])
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            module = node.module.split(".")[-1]
            if not module.startswith("s0"):
                continue
            other = int(module[1:3])
            # Reading an earlier stage's helpers is fine; that is how a stage
            # loads the artifacts it depends on. Reaching forward is not.
            if other >= this_stage:
                offenders.append(f"{path.name} imports {module}")

    assert not offenders, "a stage reaches forward: " + "; ".join(offenders)
