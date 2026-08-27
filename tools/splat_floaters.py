"""Count the Gaussians in a splat that are dead or degenerate.

PSNR on training views cannot see these. A Gaussian parked a few centimetres
in front of a training camera reproduces that camera's pixels perfectly and
blankets any later view from nearby, which is exactly what a wrist camera is.
A Gaussian wider than the room is invisible in a wide shot and covers the
frame in a close one.

The categories and the first measurements come from the real03 GPU run, in
`tools/cuda_job/README.md`: 9.8 per cent of its Gaussians finished below the
0.005 opacity that pruning removes, 5.0 per cent were wider than a tenth of
the scene, and 20 were wider than the whole 0.59 m scene, the largest 31.3 m
across. Together 14.6 per cent of that splat was dead or degenerate.

    python -m tools.splat_floaters --splat runs/real26d/01_scene/splat.pt \
        --run runs/real26d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Below this opacity a Gaussian contributes nothing a viewer can see, and it is
# the value `DefaultStrategy` prunes at. Anything left below it survived only
# because training stopped before the next prune.
DEAD_OPACITY = 0.005

# A Gaussian wider than this fraction of the scene is not modelling a surface.
WIDE_FRACTION = 0.10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splat", required=True)
    parser.add_argument("--run", default=None,
                        help="run root, to take the scene extent from the reconstruction")
    parser.add_argument("--scene-extent-m", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    state = torch.load(args.splat, map_location="cpu", weights_only=True)
    # Accept either the gsplat checkpoint or the converted MPS form.
    scales = state["log_scales"] if "log_scales" in state else state["scales"]
    opacity = state["opacity_logit"] if "opacity_logit" in state else state["opacities"]
    means = state["means"]

    widths = torch.exp(scales).max(dim=-1).values.numpy()
    alpha = torch.sigmoid(opacity).squeeze().numpy()
    count = len(alpha)

    # Take the extent Stage 1 already measured and recorded. Reading the
    # reconstruction here instead would import pycolmap alongside torch, which
    # links a second OpenMP runtime and aborts the process.
    extent = args.scene_extent_m
    if extent is None and args.run:
        meta = json.loads((Path(args.run) / "01_scene" / "meta.json").read_text())
        recorded = meta.get("metrics", {}).get("scene_extent")
        if recorded:
            extent = float(np.linalg.norm(np.asarray(recorded, dtype=float)))
    if extent is None:
        raise SystemExit(
            "no scene extent. Give --scene-extent-m, or --run pointing at a run "
            "whose 01_scene/meta.json records scene_extent."
        )

    dead = alpha < DEAD_OPACITY
    wide = widths > WIDE_FRACTION * extent
    huge = widths > extent
    degenerate = dead | wide

    report = {
        "splat": str(Path(args.splat).resolve()),
        "gaussians": int(count),
        "scene_extent_m": round(extent, 4),
        "dead_opacity_below": DEAD_OPACITY,
        "dead": int(dead.sum()),
        "dead_pct": round(100.0 * float(dead.mean()), 3),
        "wider_than_tenth_of_scene": int(wide.sum()),
        "wide_pct": round(100.0 * float(wide.mean()), 3),
        "wider_than_whole_scene": int(huge.sum()),
        "widest_m": round(float(widths.max()), 3),
        "degenerate_total": int(degenerate.sum()),
        "degenerate_pct": round(100.0 * float(degenerate.mean()), 3),
        "opacity_median": round(float(np.median(alpha)), 4),
        "width_median_m": round(float(np.median(widths)), 5),
        "width_p99_m": round(float(np.percentile(widths, 99)), 4),
        "reference_real03": {
            "dead_pct": 9.8, "wide_pct": 5.0,
            "wider_than_whole_scene": 20, "widest_m": 31.3,
            "degenerate_pct": 14.6,
        },
    }
    # How close do the surviving Gaussians sit to the desk? A floater shell in
    # front of the training cameras is the failure this exists to catch, and it
    # shows up as a population far from any surface.
    report["means_span_m"] = [round(float(v), 3) for v in
                              (means.numpy().max(axis=0) - means.numpy().min(axis=0))]

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
