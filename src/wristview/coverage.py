"""Check that the scan covers the viewpoints the render needs.

A Gaussian splat interpolates. It is sharp where training views exist. It is
soft or wrong where they do not. More training does not change this. A higher
Gaussian count does not change this.

CAPTURE-SOP.md states the rule since session 3. The wrist camera renders from
0.10 m. Earlier sessions scanned from 0.5 m and further. Every wrist frame was
therefore an extrapolation. Only a capture change corrects this defect.

Session 6 shipped with the defect. No check measured it. Its scan gave these
numbers:

    max baseline between two views        2.96 cm
    total camera path                     18.5 cm over 327 frames
    height above the desk                 41.1 to 43.9 cm
    baseline over subject distance        0.069

The wrist camera rendered from 19.8 to 38.5 cm. It travelled 1.4 m. Every
rendered frame was below the lowest scan view. The reconstruction reported 327
of 327 frames registered at 1.3565 px. The splat reached 30.21 dB. Both numbers
measure the scan against itself.

Read that last point carefully. PSNR on the scan's own views means little. Every
view sits inside a 3 cm ball. The number describes one viewpoint measured 327
times. A high number there permits a render that fails 20 cm lower.

This module asks two questions that the other gates do not ask.

First: can the scan constrain geometry? A capture that only rotates has no
parallax. It cannot fix depth. Its reprojection error stays low.

Second: do the render viewpoints lie inside the scan? For each rendered frame,
measure the distance to the nearest scan view. Also count the frames below the
scan floor.
"""

from __future__ import annotations

import numpy as np

from .logging_setup import get

log = get(__name__)

# Keep the thresholds in this file. A change to a threshold is then a visible
# edit.
#
# A workspace is about 0.6 m across. The SOP asks for a wide orbit of it. A
# correct scan therefore gives a baseline of that size. This limit is half of
# it. Session 6 gave 0.0296 m. The limit rejects that failure by ten times.
MIN_SCAN_BASELINE_M = 0.30

# The SOP asks for three passes. The first is a wide orbit with varied height.
# The second is a close pass at the demo camera's height. The third is a contact
# pass at 0.10 to 0.15 m. The three passes span more than 0.30 m. Session 6
# spanned 0.027 m.
MIN_SCAN_HEIGHT_SPAN_M = 0.15

# This is the largest permitted distance from a rendered viewpoint to the
# nearest scan view. Above this distance the render extrapolates. The value is
# half the standoff. At that distance the parallax of near surfaces changes
# visibly.
MAX_VIEWPOINT_GAP_M = 0.15

# A few frames can fall below the scan floor at the end of a reach. Session 6
# had 100 per cent of its frames below the floor.
MAX_FRACTION_BELOW_SCAN_FLOOR = 0.10


def _heights(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 3) @ np.asarray(normal) - offset


def scan_geometry(
    camera_centres: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    view_directions: np.ndarray | None = None,
) -> dict:
    """Measure whether the scan can constrain geometry. Run this before a render.

    Stage 1 reports this. A reshoot is still cheap at that point.

    Translation matters here, not the view count. 327 views from one spot give
    the information of one view. A splat trained on them cannot separate a near
    surface from a far one.
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

    # Measure the distance to the subject. The baseline then reads as a ratio.
    # A normal object scan gives a ratio near 1. Below about 0.2 the depth is
    # weakly constrained. The splat then adds floaters.
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
    """Measure the distance from each rendered frame to the nearest scan view.

    This function uses position only. Direction also matters. Session 6 failed
    on position by 20 cm. A capture change corrects position.
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


def splat_pixel_coverage(
    means: np.ndarray,
    view_matrices: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near_plane: float = 0.02,
    far_plane: float = 1e10,
    valid: np.ndarray | None = None,
    stride: int = 1,
) -> dict:
    """Fraction of pixels holding at least one projected Gaussian centre.

    `render_viewpoint_coverage` asks whether a camera stands somewhere the scan
    stood. This asks a different question: whether the splat has anything to
    draw where this camera looks. The two come apart. real27full's demo_0
    frames 60 to 120 sit 1.8 to 5.8 cm from a scan view, which is excellent by
    the first measure, and 47 per cent of their pixels hold no Gaussian centre
    at all, because a 90 degree field of view from 23 cm looks past the edge of
    a reconstruction that reaches 1.6 m.

    Rendered alpha measures the same thing and needs a GPU. This needs the
    centres and a projection, so it runs on the laptop before anything is
    uploaded, which is the whole point.

    Centres, not footprints: a Gaussian covers more than the pixel its centre
    lands in, so this reads lower than rendered alpha. It is a lower bound on
    coverage and the two track closely, measured on real27full at 0.996 mean
    alpha where 10 or more centres land per pixel and 0.534 where none do.
    """
    means = np.ascontiguousarray(np.asarray(means, dtype=np.float32).reshape(-1, 3))
    views = np.asarray(view_matrices, dtype=np.float64).reshape(-1, 4, 4)
    if valid is not None:
        views = views[np.asarray(valid, dtype=bool)]
    views = views[::max(int(stride), 1)]
    if not len(views) or not len(means):
        return {"check": "splat_pixel_coverage", "frames": int(len(views)),
                "passed": None, "note": "nothing to measure"}

    pixels = int(width) * int(height)
    fractions = np.empty(len(views), dtype=np.float64)
    for index, world_to_cam in enumerate(views):
        rotation = np.ascontiguousarray(world_to_cam[:3, :3].T, dtype=np.float32)
        camera = means @ rotation + world_to_cam[:3, 3].astype(np.float32)
        depth = camera[:, 2]
        live = (depth > near_plane) & (depth < far_plane)
        if not live.any():
            fractions[index] = 0.0
            continue
        front = camera[live]
        inv = 1.0 / front[:, 2]
        u = (fx * front[:, 0] * inv + cx).astype(np.int32)
        v = (fy * front[:, 1] * inv + cy).astype(np.int32)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not inside.any():
            fractions[index] = 0.0
            continue
        hit = np.unique(v[inside].astype(np.int64) * width + u[inside])
        fractions[index] = len(hit) / pixels

    return {
        "check": "splat_pixel_coverage",
        "frames": int(len(views)),
        "gaussians": int(len(means)),
        "stride": int(stride),
        "coverage_median": round(float(np.median(fractions)), 4),
        "coverage_mean": round(float(fractions.mean()), 4),
        "coverage_min": round(float(fractions.min()), 4),
        "coverage_p10": round(float(np.percentile(fractions, 10)), 4),
        "per_frame": [round(float(f), 4) for f in fractions],
    }
