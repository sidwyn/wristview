"""Do the stages actually call the helpers the way the helpers are written?

Twice now a change passed its own unit tests and broke the pipeline anyway,
because the tests exercised the function and nothing exercised the call. Stage 1
called `rec.metrics`, which does not exist, after a 33 minute reconstruction.
Stage 3 called `resting_pose(..., clearance_m=...)` for 155 seconds before
raising TypeError, and `--keep-going` then let Stage 4 run on stale inputs and
report four trajectories that looked exactly like real ones.

A signature check is cheap and catches the whole class. It does not prove the
arguments are semantically right, only that the call is one the callee can
accept, which is precisely the failure that keeps happening.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from wristview import carry, grasp, plane, qc

STAGES = Path(__file__).resolve().parents[1] / "src" / "wristview" / "stages"

# Module alias as written in the stage, mapped to the real module.
WATCHED = {
    "carry": carry,
    "plane_module": plane,
    "plane": plane,
    "qc": qc,
    "grasp": grasp,
}


def _calls_in(path: Path):
    """Every `alias.function(...)` call in a stage, with its keywords."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            continue
        alias = func.value.id
        if alias not in WATCHED:
            continue
        yield alias, func.attr, node, path


def _stage_files():
    return sorted(STAGES.glob("s0*.py"))


@pytest.mark.parametrize("path", _stage_files(), ids=lambda p: p.stem)
def test_every_helper_call_matches_its_signature(path: Path):
    checked = 0
    for alias, name, node, _ in _calls_in(path):
        module = WATCHED[alias]
        target = getattr(module, name, None)
        if target is None or not callable(target):
            pytest.fail(
                f"{path.name} calls {alias}.{name}, which {module.__name__} "
                f"does not define"
            )
        signature = inspect.signature(target)

        positional = len(node.args)
        keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
        has_star = any(kw.arg is None for kw in node.keywords) or any(
            isinstance(a, ast.Starred) for a in node.args
        )
        if has_star:
            continue      # forwarding a dict or list; nothing to check statically

        try:
            signature.bind_partial(*range(positional), **dict.fromkeys(keywords))
        except TypeError as exc:
            pytest.fail(
                f"{path.name} line {node.lineno} calls {alias}.{name}"
                f"{signature} with {positional} positional and {sorted(keywords)}: "
                f"{exc}"
            )
        checked += 1

    # A stage that watches nothing is fine; this just records the coverage.
    assert checked >= 0


def test_the_check_would_have_caught_the_resting_pose_break():
    """The specific regression, stated so the guard cannot rot into a no-op."""
    signature = inspect.signature(carry.resting_pose)
    with pytest.raises(TypeError):
        signature.bind_partial(1, 2, 3, 4, clearance_m=0.15)
    # And the call the stage makes now is accepted.
    signature.bind_partial(1, 2, stillness_m=0.01, min_rest_frames=5)


def test_stage3_reads_the_object_prompt_from_the_config():
    """`estimate.object.prompt` must reach Grounding DINO.

    It was declared, documented, and read by nothing. real27 therefore passed
    the whole task sentence to the detector, "pick up the tea box and place it
    on the mat", which returned a box covering 93 per cent of the frame. A
    declared key that nothing reads survives the unknown-key check, so only a
    test like this catches it.
    """
    source = Path("src/wristview/stages/s03_estimate.py").read_text()
    assert '(cfg.get("object") or {}).get("prompt")' in source, (
        "Stage 3 does not read estimate.object.prompt"
    )
    assert "prompt_source" in source, "Stage 3 does not report which prompt it used"
