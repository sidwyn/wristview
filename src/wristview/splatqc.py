"""Check that a splat can reproduce the views it was trained on.

A splat that cannot redraw its own training frames cannot draw anything else.
This is the cheapest possible test and it needs no new data.

Session real26 shipped a splat that failed it. The splat held 10455 Gaussians
and reported a train PSNR of 17.51 dB. Rendered from registered scan camera
poses, with no retarget and no wrist camera in the path, it produced a blur:
15.71 to 17.29 dB against the photographs, with the mat pattern and the marker
both absent.

Nothing downstream reported a problem. The render stage measured viewpoint
coverage and splat alpha, and both looked healthy. Alpha was 100 per cent
because Gaussians were drawn at every pixel. They were the wrong Gaussians.

Run this on any splat before it is used, wherever it was trained.
"""

from __future__ import annotations

import numpy as np

from .logging_setup import get

log = get(__name__)

# A splat that reproduces its training views scores well above this. real03 and
# real06 reached about 35 dB. The real26 failure sat at 15.7 to 17.3 dB.
# 25 dB separates the two without sitting close to either.
MIN_TRAIN_VIEW_PSNR_DB = 25.0

# Gaussians per registered training view. real26 held 10455 over 298 views, or
# 35 each, and could not represent the scene. real06 held 2.73 M over 327, or
# 8360 each.
MIN_GAUSSIANS_PER_VIEW = 200.0


def evaluate(
    render_psnr_db: list[float],
    gaussian_count: int,
    view_count: int,
    min_psnr_db: float = MIN_TRAIN_VIEW_PSNR_DB,
    min_per_view: float = MIN_GAUSSIANS_PER_VIEW,
) -> dict:
    """Judge a splat on its own training views.

    `render_psnr_db` holds one measurement per sampled training view, taken by
    rendering from that view's registered camera pose and comparing against the
    photograph.
    """
    values = np.asarray([v for v in render_psnr_db if np.isfinite(v)], dtype=float)
    per_view = gaussian_count / max(view_count, 1)

    failures: list[str] = []
    if not len(values):
        return {
            "check": "splat_reproduces_training_views",
            "views_tested": 0,
            "passed": None,
            "note": "no view was rendered, so this says nothing either way",
        }

    median = float(np.median(values))
    if median < min_psnr_db:
        failures.append(
            f"the splat renders its own training views at {median:.2f} dB, "
            f"under {min_psnr_db:.0f} dB. It cannot reproduce the scene it was "
            f"trained on, so it cannot render a new viewpoint either."
        )
    if per_view < min_per_view:
        failures.append(
            f"{gaussian_count} Gaussians over {view_count} views is "
            f"{per_view:.0f} each, under {min_per_view:.0f}. The splat is too "
            f"thin to hold the scene. Check the initialisation count and the "
            f"densification window."
        )

    return {
        "check": "splat_reproduces_training_views",
        "views_tested": int(len(values)),
        "psnr_median_db": round(median, 2),
        "psnr_min_db": round(float(values.min()), 2),
        "psnr_p10_db": round(float(np.percentile(values, 10)), 2),
        "gaussians": int(gaussian_count),
        "views": int(view_count),
        "gaussians_per_view": round(float(per_view), 1),
        "min_psnr_db": min_psnr_db,
        "min_gaussians_per_view": min_per_view,
        "passed": not failures,
        "failures": failures,
    }
