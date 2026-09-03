"""The gate-failure branch must survive being taken.

`s01_scene` logged `gate.evidence` on failure. GateResult has no such field:
it is `detail`. The line only runs when a gate FAILS, so it was the untested
branch, and the first genuine failure raised AttributeError instead of
naming the gate. A 711-frame reconstruction that registered 100 per cent died
after 3571 seconds reporting a typo rather than the number that failed.
"""

from __future__ import annotations

import dataclasses

from wristview.qc import GateResult


def test_gate_result_has_detail_not_evidence():
    fields = {f.name for f in dataclasses.fields(GateResult)}
    assert "detail" in fields
    assert "evidence" not in fields


def test_every_attribute_the_failure_log_reads_exists():
    """Pin the exact attributes s01_scene touches when a gate fails."""
    gate = GateResult("reprojection", False, 1.504, 1.5, "px", "over the limit")
    for attr in ("name", "passed", "value", "unit", "threshold", "detail"):
        getattr(gate, attr)
    assert gate.detail == "over the limit"


def test_the_failure_log_line_formats():
    gate = GateResult("reprojection", False, 1.504, 1.5, "px", "mean over 711 frames")
    rendered = (f"GATE FAILED {gate.name}: {gate.value} {gate.unit} "
                f"against a limit of {gate.threshold}. {gate.detail}")
    assert "reprojection" in rendered and "1.504" in rendered
