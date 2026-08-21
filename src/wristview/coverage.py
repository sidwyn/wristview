"""Was the splat ever shown the viewpoints it is being asked to render from?

A Gaussian splat is an interpolator. It is sharp where training views were and
soft to wrong where they were not, and no amount of training, densification or
Gaussian count changes that. CAPTURE-SOP.md has said so since session 3:

    The virtual wrist camera renders from 0.10 m. Every session so far was
    scanned from 0.5 m and further, so every wrist frame is an extrapolation
    rather than an interpolation ... This is the one defect in the current
    renders that only a capture change can fix.

Session 6 shipped anyway, because nothing measured it. Its scan:

    max baseline between any two views    2.96 cm
    total camera path                     18.5 cm over 327 frames
    height above the desk                 41.1 to 43.9 cm, a 2.7 cm band
    baseline over subject distance        0.069

while the wrist camera renders from 19.8 to 38.5 cm and travels 1.4 m. Every
rendered frame sat below the lowest scan view. The reconstruction still
reported 327 of 327 registered at 1.3565 px, and the splat still reached
30.21 dB, because both of those measure the scan against itself.

That last point is worth stating plainly. PSNR computed on the scan's own views
is nearly meaningless as a novel-view figure when every view sits inside a 3 cm
ball: it is one viewpoint measured 327 times. A high number there is consistent
with a render that falls apart 20 cm lower.

So this module asks two questions the existing gates do not:

  is the scan itself capable of constraining geometry
      a capture that only rotates has no parallax and cannot fix depth at all,
      whatever its reprojection error says

  do the render viewpoints lie inside it
      per rendered frame, how far away is the nearest scan view, and is the
      frame below the scan's floor
"""

from __future__ import annotations

import numpy as np

from .logging_setup import get

log = get(__name__)

# Thresholds, written down rather than passed in, so that changing one is a
# visible edit to this file.
#
# A workspace is roughly 0.6 m across and the SOP asks for a wide orbit of it,
# so an obedient scan has a baseline of that order. 0.30 m is half of it, and
# session 6 managed 0.0296 m, so this fails the real failure by a factor of ten
# rather than sitting just above it.
MIN_SCAN_BASELINE_M = 0.30

# The SOP asks for a wide orbit with varied height, a close pass at the demo
# camera's height, and a contact pass at 0.10 to 0.15 m. Obeying it spans well
# over 0.30 m. Session 6 spanned 0.027 m.
MIN_SCAN_HEIGHT_SPAN_M = 0.15

# How far a rendered viewpoint may sit from the nearest view that actually saw
# the scene. Beyond this the render is extrapolating. Half the standoff is the
# scale at which the parallax of nearby surfaces changes visibly.
MAX_VIEWPOINT_GAP_M = 0.15

# A few frames may stray below the scan floor at the extremes of a reach.
# Session 6 had 100 per cent of frames there.
MAX_FRACTION_BELOW_SCAN_FLOOR = 0.10


def _heights(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 3) @ np.asarray(normal) - offset


def scan_geometry(
    camera_centres: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    view_directions: np.ndarray | None = None,
) -> dict:
    """Whether the scan can constrain geometry at all, before any render.

    Reported in Stage 1, where it is still cheap to reshoot. The numbers that
    matter are translation, not count: 327 views taken from one spot are one
    view, and a splat trained on them has no way to tell a near surface from a
    far one.
    """
    centres = np.asarray(camera_centres, dtype=np.float64).reshape(-1, 3)
    if len(centres) < 2:
        return {"check": "scan_geometry", "views": int(len(centres)), "passed": None,
                "note": "too few views to say anything"}

    gaps = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    baseline = float(gaps.max())
    path = float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum())
    heights = _heights(centres, plane_normal, plane_offset)
    span = float(heights.max() - heights.min())

    report = {
        "check": "scan_geometry",
        "views": int(len(centres)),
        "max_baseline_m": round(baseline, 4),
        "path_length_m": round(path, 3),
        "height_min_m": round(float(heights.min()), 4),
        "height_max_m": round(float(heights.max()), 4),
        "height_span_m": round(span, 4),
        "baseline_limit_m": MIN_SCAN_BASELINE_M,
        "height_span_limit_m": MIN_SCAN_HEIGHT_SPAN_M,
    }

    if view_directions is not None:
        dirs = np.asarray(view_directions, dtype=np.float64).reshape(-1, 3)
        dirs = dirs / np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12, None)
        pick = dirs[np.linspace(0, len(dirs) - 1, min(len(dirs), 80)).astype(int)]
        angles = np.degrees(np.arccos(np.clip(pick @ pick.T, -1, 1)))
        report["view_direction_spread_deg"] = round(float(angles.max()), 1)

    # Distance to the subject, so the baseline can be read as a ratio. A normal
    # object scan orbits at a ratio near 1; below about 0.2 depth is weakly
    # constrained and the splat fills in with floaters.
    subject = centres.mean(axis=0) - np.asarray(plane_normal) * float(np.median(heights))
    distance = float(np.median(np.linalg.norm(centres - subject, axis=1)))
    if distance > 1e-9:
        report["subject_distance_m"] = round(distance, 3)
        report["baseline_over_distance"] = round(baseline / distance, 4)

    failures = []
    if baseline < MIN_SCAN_BASELINE_M:
        failures.append(
            f"the scan translated {baseline * 100:.1f} cm, under the "
            f"{MIN_SCAN_BASELINE_M * 100:.0f} cm needed for parallax. A capture that "
            f"mostly rotates cannot constrain depth however well it reprojects"
        )
    if span < MIN_SCAN_HEIGHT_SPAN_M:
        failures.append(
            f"the scan stayed within a {span * 100:.1f} cm height band, under the "
            f"{MIN_SCAN_HEIGHT_SPAN_M * 100:.0f} cm the SOP's three passes produce"
        )
    report["passed"] = not failures
    report["failures"] = failures
    return report


def render_viewpoint_coverage(
    scan_centres: np.ndarray,
    render_centres: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    render_valid: np.ndarray | None = None,
) -> dict:
    """Per rendered frame, how far it sits from anything that saw the scene.

    Position only. Direction matters too, but position is where session 6 went
    wrong by 20 cm and it is the part a capture change fixes.
    """
    scan = np.asarray(scan_centres, dtype=np.float64).reshape(-1, 3)
    render = np.asarray(render_centres, dtype=np.float64).reshape(-1, 3)
    if render_valid is not None:
        render = render[np.asarray(render_valid, dtype=bool)]
    if not len(scan) or not len(render):
        return {"check": "render_viewpoint_coverage", "frames": int(len(render)),
                "passed": None, "note": "nothing to compare"}

    gaps = np.linalg.norm(render[:, None, :] - scan[None, :, :], axis=2).min(axis=1)
    scan_floor = float(_heights(scan, plane_normal, plane_offset).min())
    render_heights = _heights(render, plane_normal, plane_offset)
    below = float((render_heights < scan_floor).mean())

    failures = []
    median_gap = float(np.median(gaps))
    if median_gap > MAX_VIEWPOINT_GAP_M:
        failures.append(
            f"the median rendered frame sits {median_gap * 100:.1f} cm from the "
            f"nearest view that ever saw this scene, over the "
            f"{MAX_VIEWPOINT_GAP_M * 100:.0f} cm limit"
        )
    if below > MAX_FRACTION_BELOW_SCAN_FLOOR:
        failures.append(
            f"{below * 100:.0f} per cent of rendered frames are below the scan's "
            f"lowest view at {scan_floor * 100:.1f} cm, so they extrapolate "
            f"downward from every training view at once"
        )

    return {
        "check": "render_viewpoint_coverage",
        "frames": int(len(render)),
        "nearest_scan_view_median_m": round(median_gap, 4),
        "nearest_scan_view_p90_m": round(float(np.percentile(gaps, 90)), 4),
        "nearest_scan_view_max_m": round(float(gaps.max()), 4),
        "scan_floor_m": round(scan_floor, 4),
        "render_height_min_m": round(float(render_heights.min()), 4),
        "render_height_median_m": round(float(np.median(render_heights)), 4),
        "fraction_below_scan_floor": round(below, 4),
        "gap_limit_m": MAX_VIEWPOINT_GAP_M,
        "passed": not failures,
        "failures": failures,
        "note": (
            "a splat is an interpolator; this is the one render defect that "
            "only a capture change can fix"
        ),
    }
