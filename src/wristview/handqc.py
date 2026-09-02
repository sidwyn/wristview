"""Reject hand detections that no neighbouring frame supports.

WiLoR returned confidence 1.000 on every valid frame of all 30 real27 clips, so
the score cannot be low and cannot be read. Defect 37. Two frames of demo_28
hold no hand at all, in the top right corner on desk clutter, and they arrived
at full confidence and made the clip look like a take shot with the hand
already in frame.

Landmark count does not separate a phantom from a real hand. Legitimate frames
where a hand is entering the shot hold as few as 2 of 21 landmarks inside the
image, demo_19 among them. Adjacency does: a hand takes time to arrive, so a
detection with no detection on either side of it is not a hand.

Across all 30 real27 clips exactly two valid frames have no valid neighbour,
and both are demo_28's phantoms.
"""

from __future__ import annotations

import numpy as np

from .logging_setup import get

log = get(__name__)


def drop_isolated_labels(labels) -> tuple[np.ndarray, dict]:
    """Drop hand-side labels that neither neighbour supports.

    Sibling of `drop_isolated`, and it exists because that one is not enough.
    `drop_isolated` judges hand_valid, which is detected against not detected.
    It says nothing about hand_side, so a single "Left" inside a run of
    "Right" survives it untouched.

    That single label then costs the whole clip. The identity check walks
    consecutive pairs and counts a crossing whenever the label changes, so one
    isolated label produces TWO changes, Right->Left and Left->Right. Two
    changes clear the `max(na, nb) >= 2` test and the clip is declared INVALID
    with "the selected hand changed side 2 times". About 11 of 30 clips were
    lost to this in one session.

    A hand does not swap identity for one frame and swap back. When both
    neighbours agree with each other and disagree with the frame between them,
    the frame is a detector mislabel, not a crossing.

    Returns a mask of the labels to trust, and a report naming what it removed.
    """
    labels = np.asarray([str(x) for x in labels], dtype=object)
    trusted = np.ones(len(labels), dtype=bool)
    if len(labels) < 3:
        return trusted, {
            "check": "isolated_hand_labels",
            "labels_dropped": 0,
            "dropped_indices": [],
            "note": "too few labelled frames to judge",
        }

    # Work in runs, not in neighbours. A neighbour test alone deletes every
    # interior frame of an ALTERNATING sequence, because there each frame
    # differs from both its neighbours and they agree with each other. That
    # sequence is a real fault and must stay visible; laundering it into a
    # clean clip is the opposite of what this function is for.
    #
    # So a label is only isolated when it is a run of exactly ONE sitting
    # between two runs of at least two that carry the SAME label. That is a
    # single frame mislabelled inside stable footage, and nothing else.
    runs: list[list] = []
    for index, label in enumerate(labels):
        if runs and runs[-1][0] == label:
            runs[-1][2] = index
        else:
            runs.append([label, index, index])

    for position in range(1, len(runs) - 1):
        label, start, stop = runs[position]
        before, after = runs[position - 1], runs[position + 1]
        singleton = start == stop
        flanked = before[0] == after[0] != label
        stable = (before[2] - before[1] + 1) >= 2 and (after[2] - after[1] + 1) >= 2
        if singleton and flanked and stable:
            trusted[start] = False

    return trusted, {
        "check": "isolated_hand_labels",
        "labels_dropped": int((~trusted).sum()),
        "dropped_indices": [int(i) for i in np.nonzero(~trusted)[0]],
        "labels_before": int(len(labels)),
        "labels_after": int(trusted.sum()),
        "note": (
            "a hand does not change identity for one frame and change back. "
            "An isolated label is the detector mislabelling one hand, and left "
            "in place it scores as TWO crossings and invalidates the clip."
        ),
    }


def drop_isolated(valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop valid frames that have no valid frame immediately before or after.

    Returns the filtered mask and a report. The report names the frames, because
    a gate that says how many it removed without saying which is not evidence.
    """
    valid = np.asarray(valid, dtype=bool)
    if len(valid) < 3:
        return valid.copy(), {"check": "isolated_hand_frames", "frames_dropped": 0,
                              "dropped_frames": [], "note": "clip too short to judge"}

    padded = np.concatenate([[False], valid, [False]])
    supported = padded[:-2] | padded[2:]
    isolated = valid & ~supported

    kept = valid & ~isolated
    return kept, {
        "check": "isolated_hand_frames",
        "frames_dropped": int(isolated.sum()),
        "dropped_frames": [int(i) for i in np.nonzero(isolated)[0]],
        "valid_before": int(valid.sum()),
        "valid_after": int(kept.sum()),
        "note": (
            "a hand takes time to arrive, so a detection no neighbour supports "
            "is a false positive. The detector's own confidence cannot say so: "
            "it is 1.000 on every frame it accepts."
        ),
    }
