"""Run directory and the stage data contract.

Two rules from the build plan, enforced here:

1. Every stage is a pure function over a run directory. It reads files and
   writes files.
2. No stage calls the next stage. The driver in `cli.py` sequences them.

Every stage writes a `meta.json` recording its inputs, the git SHA, the config
hash, and its timings. That file is how a bad batch gets debugged weeks later.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config

STAGE_DIRS = {
    0: "00_ingest",
    1: "01_scene",
    2: "02_localize",
    3: "03_estimate",
    4: "04_retarget",
    5: "05_render",
}


def git_sha() -> str:
    """Return the current git SHA, or `unknown` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return sha + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


@dataclass
class StageMeta:
    """The meta.json payload for one stage."""

    stage: int
    name: str
    status: str = "running"
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    backends: dict[str, str] = field(default_factory=dict)
    git_sha: str = ""
    config_hash: str = ""
    config_overrides: dict[str, Any] = field(default_factory=dict)
    platform: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "stage": self.stage,
            "name": self.name,
            "status": self.status,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            # What `--set` changed for this stage. Without it a run directory
            # cannot say why its numbers differ from the config beside them.
            "config_overrides": self.config_overrides,
            "platform": self.platform,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(max(0.0, self.finished_at - self.started_at), 3),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "timings_s": {k: round(v, 3) for k, v in self.timings_s.items()},
            "backends": self.backends,
            "notes": self.notes,
        }
        return payload


class RunContext:
    """Owns one run directory and hands out per-stage directories."""

    def __init__(self, root: str | Path, config: Config, run_id: str):
        self.root = Path(root).resolve()
        self.config = config
        self.run_id = run_id
        self.git_sha = git_sha()
        self.platform = f"{platform.system()}-{platform.machine()}-py{platform.python_version()}"
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, runs_dir: str | Path, run_id: str, config: Config) -> RunContext:
        root = Path(runs_dir) / run_id
        ctx = cls(root=root, config=config, run_id=run_id)
        config.write(ctx.root / "config.yaml")
        return ctx

    def record_invocation(self, stages: list[int] | None = None) -> Path:
        """Append what was run, and with which overrides, to the run record.

        `--set` used to leave no trace. It does not reach `config.yaml`, which
        the `stage` subcommand never rewrites, and it did not reach any
        `meta.json`. So a run directory could not answer what produced it.

        This log is append-only on purpose. Rewriting `config.yaml` with the
        overrides would make a one-off setting sticky for every later stage,
        which is a different way to lose track of what happened.
        """
        record = {
            "at": time.time(),
            "run_id": self.run_id,
            "stages": stages,
            "git_sha": self.git_sha,
            "config_hash": self.config.hash,
            "config_source": str(self.config.source_path) if self.config.source_path else None,
            "overrides": dict(getattr(self.config, "overrides", {}) or {}),
            "argv": sys.argv[1:],
            "platform": self.platform,
        }
        path = self.root / "run_record.jsonl"
        with open(path, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return path

    def stage_dir(self, stage: int, create: bool = True) -> Path:
        path = self.root / STAGE_DIRS[stage]
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def episode_dir(self, stage: int, episode: str, create: bool = True) -> Path:
        path = self.stage_dir(stage, create=create) / episode
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def read_meta(self, stage: int) -> dict[str, Any]:
        """Read a previous stage's meta.json. Raises if that stage has not run."""
        path = self.stage_dir(stage, create=False) / "meta.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Stage {stage} has not run: {path} is missing. "
                f"Run stage {stage} before this one."
            )
        with open(path) as handle:
            return json.load(handle)

    def rel(self, path: str | Path) -> str:
        """Path relative to the run root, for readable meta.json entries."""
        try:
            return str(Path(path).resolve().relative_to(self.root))
        except ValueError:
            return str(path)


class StageRecorder:
    """Collects meta.json for one stage and writes it, even on failure."""

    def __init__(self, ctx: RunContext, stage: int, name: str):
        self.ctx = ctx
        self.meta = StageMeta(
            stage=stage,
            name=name,
            git_sha=ctx.git_sha,
            config_hash=ctx.config.hash,
            config_overrides=dict(getattr(ctx.config, "overrides", {}) or {}),
            platform=ctx.platform,
            started_at=time.time(),
        )
        self.path = ctx.stage_dir(stage) / "meta.json"

    @contextmanager
    def timed(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.meta.timings_s[label] = (
                self.meta.timings_s.get(label, 0.0) + time.perf_counter() - start
            )

    def note(self, message: str) -> None:
        self.meta.notes.append(message)

    def backend(self, role: str, name: str) -> None:
        """Record which implementation actually ran, for example hand=mediapipe."""
        self.meta.backends[role] = name

    def output(self, key: str, path: str | Path) -> None:
        self.meta.outputs[key] = self.ctx.rel(path)

    def metric(self, key: str, value: Any) -> None:
        self.meta.metrics[key] = value

    def write(self, status: str) -> Path:
        self.meta.status = status
        self.meta.finished_at = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as handle:
            json.dump(self.meta.to_dict(), handle, indent=2, sort_keys=False)
        return self.path


def verify_frames_present(frames_dir: Path, names: list[str], clip_id: str) -> None:
    """Fail now if the manifest and the frames on disk disagree.

    Stages 1, 2 and 3 all read frames a previous stage recorded. When those
    files are missing the failure surfaces late and unrecognisably: a missing
    scan frame showed up as `KeyError: 'scan_00000.jpg'` from inside hloc's
    match importer, sixteen minutes into matching, long after the cause.
    """
    missing = [name for name in names if not (frames_dir / name).exists()]
    if not missing:
        return

    raise FileNotFoundError(
        f"{clip_id}: {len(missing)} of {len(names)} frames named in the manifest "
        f"are missing from {frames_dir}. First missing: {missing[:3]}. "
        f"The frames directory has {len(list(frames_dir.glob('*.jpg')))} jpgs. "
        f"Re-run stage 0 for this run."
    )


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
    return target


def read_json(path: str | Path) -> Any:
    with open(path) as handle:
        return json.load(handle)
