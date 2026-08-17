"""Grasp detection from contact, not from finger separation.

The width test asks whether the fingers are close together. That is a proxy,
and on session 3 it was the wrong one: the operator held a mug by its handle,
so thumb-to-index distance barely changed through the whole episode. Grasp
state agreed with ground truth on 57.9 per cent of frames, which is close to
what guessing the majority class would score, and the object was reported
STATIC right through transport.

This asks the two questions that actually define a grasp:

  contact   is a fingertip touching the object
  carried   is the object holding still *relative to the hand*

Either alone is not enough. A finger resting on an object satisfies contact
while nothing is held. An object sitting on the desk while the hand moves in
parallel above it can briefly satisfy stability. Together they are specific:
the object is being touched, and it is going where the hand goes.

The second test is what the width signal could never express, and it is the
reason a handle grasp is now detectable.
"""

from __future__ import annotations

import numpy as np

from .backends import hands as hand_backend
from .logging_setup import get
from .qc import palm_rotations

log = get(__name__)

# MANO fingertips that oppose in a pinch or wrap.
FINGERTIPS = (4, 8, 12)


def _hysteresis(signal: np.ndarray, min_frames: int) -> np.ndarray:
    """Absorb runs shorter than `min_frames`, never the first or last run."""
    out = signal.copy()
    if min_frames <= 1 or len(signal) == 0:
        return out
    boundaries = [0]
    for index in range(1, len(out)):
        if out[index] != out[index - 1]:
            boundaries.append(index)
    boundaries.append(len(out))
    for run in range(1, len(boundaries) - 2):
        start, stop = boundaries[run], boundaries[run + 1]
        if stop - start < min_frames:
            out[start:stop] = out[start - 1]
    return out


def fingertip_distances(
    landmarks: np.ndarray, valid: np.ndarray, object_positions: np.ndarray,
    object_valid: np.ndarray, object_radius_m: float,
) -> np.ndarray:
    """Closest fingertip to the object surface, per frame. NaN where unknown.

    The object is treated as a sphere of `object_radius_m` about its centre,
    which is crude but honest: the alternative needs a tracked object mesh, and
    a sphere is enough to separate touching from not touching at these scales.
    """
    count = len(landmarks)
    out = np.full(count, np.nan)
    for index in range(count):
        if not (valid[index] and object_valid[index]):
            continue
        tips = landmarks[index][list(FINGERTIPS)]
        centre = object_positions[index]
        out[index] = float(np.linalg.norm(tips - centre, axis=1).min()) - object_radius_m
    return out


def relative_pose_stability(
    landmarks: np.ndarray, valid: np.ndarray, object_positions: np.ndarray,
    object_valid: np.ndarray, window: int,
) -> np.ndarray:
    """How still the object sits in the hand's own frame, per frame, in metres.

    A carried object has a near-constant position in the hand frame, whatever
    the hand does in the world. An object on the desk does not.

    Returns NaN where the window has too few usable frames.
    """
    count = len(landmarks)
    out = np.full(count, np.nan)
    usable = valid & object_valid
    if usable.sum() < 3:
        return out

    # The hand frame comes from the palm, which is rigid. Fingertips move
    # relative to the wrist during a grasp, so a fingertip-based frame would
    # register the grasp itself as instability.
    rotations = np.full((count, 3, 3), np.nan)
    rotations[usable] = palm_rotations(landmarks[usable])
    wrist = landmarks[:, 0]

    relative = np.full((count, 3), np.nan)
    for index in np.nonzero(usable)[0]:
        relative[index] = rotations[index].T @ (object_positions[index] - wrist[index])

    half = max(window // 2, 1)
    for index in range(count):
        lo, hi = max(0, index - half), min(count, index + half + 1)
        block = relative[lo:hi]
        block = block[np.isfinite(block).all(axis=1)]
        if len(block) < 3:
            continue
        out[index] = float(np.linalg.norm(block.std(axis=0)))
    return out


def detect_grasp_by_contact(
    landmarks: np.ndarray,
    valid: np.ndarray,
    object_positions: np.ndarray,
    object_valid: np.ndarray,
    fps: float,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """Per-frame closed flag from contact and carry, plus diagnostics."""
    count = len(landmarks)
    radius = float(cfg.get("object_radius_m", 0.035))
    contact_margin = float(cfg.get("contact_margin_m", 0.03))
    release_margin = float(cfg.get("release_margin_m", 0.05))
    stability_tol = float(cfg.get("stability_tol_m", 0.02))
    window = int(cfg.get("stability_window", max(3, int(round(fps * 0.3)))))
    min_state = int(cfg.get("min_state_frames", max(3, int(round(fps * 0.15)))))

    gap = fingertip_distances(landmarks, valid, object_positions, object_valid, radius)
    wobble = relative_pose_stability(
        landmarks, valid, object_positions, object_valid, window
    )

    usable = np.isfinite(gap)
    if not usable.any():
        widths = np.array([
            hand_backend.grasp_width(landmarks[i]) if valid[i] else np.nan
            for i in range(count)
        ])
        return np.zeros(count, dtype=bool), {
            "signal": "unavailable",
            "reason": (
                "no frame had both a hand and an object position, so contact "
                "could not be evaluated. Grasp state is reported as never closed "
                "rather than guessed from finger width."
            ),
            "closed_frames": 0,
            "closed_fraction": 0.0,
            "transitions": 0,
            "grasp_onset_frame": None,
            "release_frame": None,
            "width_median_m": (
                round(float(np.nanmedian(widths)), 4) if np.isfinite(widths).any() else None
            ),
        }

    # Schmitt trigger on the contact gap, so a fingertip hovering at the
    # threshold does not chatter the grasp open and closed.
    touching = np.zeros(count, dtype=bool)
    state = False
    for index in range(count):
        value = gap[index]
        if np.isnan(value):
            touching[index] = state
            continue
        if state and value > release_margin:
            state = False
        elif not state and value < contact_margin:
            state = True
        touching[index] = state

    carried = np.zeros(count, dtype=bool)
    carried[np.isfinite(wobble)] = wobble[np.isfinite(wobble)] < stability_tol

    closed = _hysteresis(touching & carried, min_state)

    transitions = np.nonzero(np.diff(closed.astype(int)))[0]
    onsets = np.nonzero(closed)[0]
    diagnostics = {
        "signal": "contact_and_carry",
        "closed_frames": int(closed.sum()),
        "closed_fraction": round(float(closed.mean()), 4),
        "transitions": int(len(transitions)),
        "grasp_onset_frame": int(onsets[0]) if len(onsets) else None,
        "release_frame": int(onsets[-1] + 1) if len(onsets) and onsets[-1] + 1 < count else None,
        "frames_with_contact": int(touching.sum()),
        "frames_carried": int(carried.sum()),
        "frames_evaluable": int(usable.sum()),
        "contact_gap_median_m": round(float(np.nanmedian(gap)), 4),
        "contact_gap_min_m": round(float(np.nanmin(gap)), 4),
        "wobble_median_m": (
            round(float(np.nanmedian(wobble)), 4) if np.isfinite(wobble).any() else None
        ),
        "thresholds": {
            "object_radius_m": radius,
            "contact_margin_m": contact_margin,
            "release_margin_m": release_margin,
            "stability_tol_m": stability_tol,
            "stability_window": window,
        },
    }
    return closed, diagnostics
