"""Find the part of a demo clip that shows the scanned workspace.

A take holds more than the demonstration. Session real25 opened and closed with
a sync clock on a monitor. The head camera pointed at a screen for the first
2.3 s and the last 3.0 s. Those frames show no part of the scan.

Stage 2 registered 224 of 305 frames on that clip and then rejected it. The
missing frames were not the fault. A few clock frames matched room geometry and
returned wrong poses. Those poses added 247 cm to a 307 cm camera path. The
true path is 60 cm.

So trim the clip before Stage 2 runs.

**Marker visibility was tried first and rejected. Keep this reasoning.**

The first design used the table marker as the workspace test. The marker lies
on the work surface, so the camera sees it when it looks at the workspace. The
data refutes that:

    t= 0.00- 2.25 s   marker  0%   registration   7%    sync clock
    t= 3.05- 6.10 s   marker 100%  registration 100%    work
    t= 6.90- 9.00 s   marker  0%   registration 100%    work, hand covers it
    t=11.30-13.55 s   marker 100%  registration 100%    work
    t=14.90-17.90 s   marker  0%   registration 20-62%  sync clock

The hand covers the marker for 4.25 s in the middle of the manipulation. That
gap is longer than either sync segment, at 2.3 s and 3.0 s. No bridge length
separates the two cases. Bridge far enough to cross the occlusion and the sync
segments join as well. Bridge less and the demonstration splits in two.

Marker visibility is a proxy. It cannot see the failure it must catch.

This module uses the scan-match count for each frame instead. That measures the
question directly: can this frame localize against the scan? The registration
column above is that signal, and it separates the two cases cleanly.

This module never falls back to the full clip. A silent fall-through returns
the failure that it exists to prevent. It raises an error instead.
"""

from __future__ import annotations

import numpy as np

from .logging_setup import get

log = get(__name__)

# The marker can leave the frame for short periods during a reach. Bridge a gap
# of this length. Do not bridge a longer one.
MAX_BRIDGE_S = 1.0

# Refuse a segment shorter than this. A shorter span is a detection failure,
# not a demonstration.
MIN_SEGMENT_S = 2.0

# Refuse a segment that holds less than this share of the clip. The operator
# recorded a demonstration, not a marker test.
MIN_SEGMENT_FRACTION = 0.25


def matched_frames(match_counts: np.ndarray, warn_matches: float) -> np.ndarray:
    """Mark the frames that match the scan well enough to localize.

    `match_counts` holds the best scan-match count for each frame. Stage 2 needs
    a frame to reach the warn bar before its pose is worth solving.
    """
    return np.asarray(match_counts, dtype=float) >= float(warn_matches)


def detect_workspace_segment(
    visible: np.ndarray,
    times_s: np.ndarray,
    max_bridge_s: float = MAX_BRIDGE_S,
    min_segment_s: float = MIN_SEGMENT_S,
    min_fraction: float = MIN_SEGMENT_FRACTION,
) -> tuple[int, int, dict]:
    """Return the frame span that shows the workspace.

    `visible` marks the frames where the detector found the table marker.
    `times_s` gives the source time of each frame.

    The function bridges short gaps. It then takes the longest span. It raises
    a ValueError when no span qualifies.
    """
    visible = np.asarray(visible, dtype=bool)
    times = np.asarray(times_s, dtype=float)
    if len(visible) != len(times):
        raise ValueError(
            f"visible has {len(visible)} entries and times_s has {len(times)}"
        )
    if not visible.any():
        raise ValueError(
            "the table marker never appears in this clip, so the workspace "
            "segment cannot be found. Check the marker id and that the camera "
            "sees the work surface."
        )

    # Bridge the short gaps. A hand passing over the marker hides it briefly.
    filled = visible.copy()
    starts = np.nonzero(visible)[0]
    for left, right in zip(starts, starts[1:], strict=False):
        if right - left <= 1:
            continue
        if times[right] - times[left] <= max_bridge_s:
            filled[left : right + 1] = True

    spans: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(filled):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(filled)))

    best = max(spans, key=lambda s: times[s[1] - 1] - times[s[0]])
    lo, hi = best
    duration = float(times[hi - 1] - times[lo])
    fraction = (hi - lo) / len(visible)

    report = {
        "start_frame": int(lo),
        "end_frame": int(hi),
        "start_s": round(float(times[lo]), 3),
        "end_s": round(float(times[hi - 1]), 3),
        "duration_s": round(duration, 3),
        "frames": int(hi - lo),
        "fraction_of_clip": round(float(fraction), 4),
        "marker_visible_frames": int(visible.sum()),
        "spans_found": len(spans),
        "max_bridge_s": max_bridge_s,
    }

    if duration < min_segment_s:
        raise ValueError(
            f"the longest workspace segment is {duration:.2f} s, under the "
            f"{min_segment_s:.1f} s minimum. The camera rarely saw the work "
            f"surface. Detected {visible.sum()} marker frames of {len(visible)}."
        )
    if fraction < min_fraction:
        raise ValueError(
            f"the workspace segment holds {fraction * 100:.0f} per cent of the "
            f"clip, under the {min_fraction * 100:.0f} per cent minimum. Check "
            f"that the camera points at the work surface for the demonstration."
        )
    return int(lo), int(hi), report
