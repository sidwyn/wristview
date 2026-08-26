"""Find the demonstration inside a take that also contains sync segments.

Session real25 recorded a sync clock at each end of the demo. The head camera
pointed at a monitor for the first 2.3 s and the last 3.0 s. Stage 2 registered
224 of 305 frames and then rejected the clip. A few clock frames registered
against room geometry and returned wrong poses. Those poses turned a 60 cm
camera path into 25.78 m.

These tests use that clip's shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristview.workspace import detect_workspace_segment


def _clip(pattern: list[tuple[int, bool]], fps: float = 20.0):
    """Build a visibility array and its times from runs of (count, visible)."""
    visible = np.concatenate([np.full(n, v, dtype=bool) for n, v in pattern])
    times = np.arange(len(visible)) / fps
    return visible, times


def test_finds_the_demonstration_between_two_sync_segments():
    """The real25 shape: 2.3 s of clock, 12.6 s of work, 3.0 s of clock."""
    visible, times = _clip([(46, False), (252, True), (60, False)])
    lo, hi, report = detect_workspace_segment(visible, times)
    assert lo == 46
    assert hi == 298
    assert report["duration_s"] == pytest.approx(12.55, abs=0.05)
    assert report["fraction_of_clip"] > 0.6


def test_bridges_a_short_gap_where_a_hand_covers_the_marker():
    """A reach hides the marker. That is not the end of the segment."""
    visible, times = _clip([(20, False), (100, True), (10, False), (100, True), (20, False)])
    lo, hi, report = detect_workspace_segment(visible, times)
    # 100 visible, then 10 bridged, then 100 visible: 210 frames from index 20.
    assert lo == 20
    assert hi == 230
    assert report["frames"] == 210


def test_does_not_bridge_a_long_gap():
    """Two separate takes in one file stay separate."""
    visible, times = _clip([(100, True), (60, False), (100, True)])
    lo, hi, _ = detect_workspace_segment(visible, times)
    assert (hi - lo) == 100


def test_raises_when_the_marker_never_appears():
    """real24 take 1: the wrist camera saw the marker in 1 of 411 frames."""
    visible, times = _clip([(300, False)])
    with pytest.raises(ValueError, match="never appears"):
        detect_workspace_segment(visible, times)


def test_raises_when_the_segment_is_too_short():
    """A brief glimpse is a detection failure, not a demonstration."""
    visible, times = _clip([(100, False), (20, True), (100, False)])
    with pytest.raises(ValueError, match="under the"):
        detect_workspace_segment(visible, times)


def test_raises_when_the_segment_is_a_small_share_of_the_clip():
    """The operator recorded a demonstration, not a marker test."""
    visible, times = _clip([(400, False), (60, True), (400, False)])
    with pytest.raises(ValueError, match="per cent"):
        detect_workspace_segment(visible, times)


def test_never_falls_back_to_the_whole_clip():
    """A silent fall-through returns the failure this module prevents."""
    visible, times = _clip([(300, False)])
    with pytest.raises(ValueError):
        detect_workspace_segment(visible, times)


def test_reports_the_span_for_the_run_summary():
    visible, times = _clip([(46, False), (252, True), (60, False)])
    _, _, report = detect_workspace_segment(visible, times)
    for key in ("start_frame", "end_frame", "start_s", "end_s", "duration_s",
                "frames", "fraction_of_clip", "marker_visible_frames"):
        assert key in report


def test_rejects_mismatched_input_lengths():
    with pytest.raises(ValueError, match="entries"):
        detect_workspace_segment(np.ones(10, bool), np.arange(5))
