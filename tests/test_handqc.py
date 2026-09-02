"""A hand detection with no neighbour is a phantom.

WiLoR returns confidence 1.000 on every frame it accepts, so the score cannot
be low and cannot be read. Two frames of real27's demo_28 hold no hand at all,
on desk clutter in the top right corner, and they arrived at full confidence.
They made the clip look like a take shot with the hand already in frame, and it
was reported that way.

Landmark count cannot separate the two cases: real frames where a hand is
entering the shot hold as few as 2 of 21 landmarks inside the image. Adjacency
can.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from wristview.handqc import drop_isolated

STAGE = Path(__file__).resolve().parents[1] / "src" / "wristview" / "stages" / "s03_estimate.py"


def mask(pattern: str) -> np.ndarray:
    return np.array([c == "1" for c in pattern], dtype=bool)


def test_the_demo_28_pattern():
    """Its real shape: a phantom at 0, another at 17, the hand from 21."""
    valid = mask("1" + "." * 16 + "1" + "..." + "1" * 20)
    kept, report = drop_isolated(valid)

    assert report["frames_dropped"] == 2
    assert report["dropped_frames"] == [0, 17]
    assert int(kept.sum()) == 20
    assert not kept[0] and not kept[17]
    assert kept[21]


def test_a_pair_survives():
    """Two adjacent frames support each other. Only singletons go."""
    kept, report = drop_isolated(mask("..11....1..."))
    assert report["dropped_frames"] == [8]
    assert kept[2] and kept[3]


def test_a_run_is_untouched():
    valid = mask("...111111...")
    kept, report = drop_isolated(valid)
    assert report["frames_dropped"] == 0
    assert np.array_equal(kept, valid)


def test_the_first_and_last_frame_can_be_isolated():
    """Frame 0 has no frame before it, which must not count as support."""
    kept, report = drop_isolated(mask("1...1"))
    assert report["dropped_frames"] == [0, 4]
    assert not kept.any()


def test_a_run_at_the_very_start_survives():
    kept, _ = drop_isolated(mask("11..."))
    assert kept[0] and kept[1]


def test_nothing_valid_is_not_an_error():
    kept, report = drop_isolated(mask("....."))
    assert report["frames_dropped"] == 0
    assert not kept.any()


def test_a_short_clip_is_not_judged():
    _, report = drop_isolated(mask("1"))
    assert report["frames_dropped"] == 0
    assert "too short" in report["note"]


def test_the_report_names_the_frames():
    """A gate that says how many without saying which is not evidence."""
    _, report = drop_isolated(mask("1..11..1"))
    assert report["dropped_frames"] == [0, 7]
    assert report["valid_before"] == 4
    assert report["valid_after"] == 2


def test_stage_3_calls_it_and_records_it():
    """The house defect is a number nothing reads. Pin both ends."""
    source = STAGE.read_text()
    tree = ast.parse(source)

    called = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "drop_isolated"
        for node in ast.walk(tree)
    )
    assert called, "Stage 3 must call drop_isolated"
    assert '"isolated_hand_frames": isolated_report' in source, (
        "Stage 3 must write the report into its status, or the drop is invisible"
    )
