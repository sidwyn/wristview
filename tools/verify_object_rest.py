"""Before the hand touches it, the tracked object must be where the scan left it.

Same object, same place, nothing moved. So the tracked pose in the pre-contact
frames and the object's position in the scan reconstruction must agree, and any
difference is error with no other explanation available.

This is the check that costs nothing and would have caught session 4, where the
cube floated 85 mm above the desk and sat 113 to 145 mm to one side of the hand
holding it, while every reported diagnostic looked healthy.

    python -m tools.verify_object_rest --run <run> --object-height 0.0762
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.logging_setup import get, setup  # noqa: E402
from wristview.runctx import read_json  # noqa: E402

log = get(__name__)


def scan_object_centre(run: Path, saturation: float, value: float) -> np.ndarray | None:
    """Where the object sits in the scan, from the sparse cloud's colour.

    The task object is deliberately a saturated print on a desk of wood, grey
    keyboard and white paper, so hue separates it without a network.
    """
    import pycolmap

    rec = pycolmap.Reconstruction(str(run / "01_scene" / "colmap" / "sparse"))
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    rgb = np.array([p.color for p in rec.points3D.values()], dtype=float) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    keep = (hsv[:, 1] > saturation) & (hsv[:, 2] > value)
    if keep.sum() < 30:
        return None
    points = xyz[keep]
    centre = np.median(points, axis=0)
    for _ in range(6):
        points = points[np.linalg.norm(points - centre, axis=1) < 0.10]
        if len(points) < 20:
            return None
        centre = np.median(points, axis=0)
    return centre


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--object-height", type=float, default=0.0762)
    parser.add_argument("--saturation", type=float, default=0.55)
    parser.add_argument("--value", type=float, default=0.35)
    parser.add_argument("--move-threshold-m", type=float, default=0.02,
                        help="the object counts as moved once it travels this far")
    args = parser.parse_args()

    setup(None, verbose=False)
    root = Path(args.run).resolve()

    scan_centre = scan_object_centre(root, args.saturation, args.value)
    if scan_centre is None:
        log.error("could not find the object in the scan cloud by colour")
        return 1
    log.info("object in the scan reconstruction: %s", np.round(scan_centre, 4).tolist())

    plane = (read_json(root / "01_scene" / "scale.json").get("diagnostics") or {}).get("desk_plane")
    if plane and plane.get("offset_m") is not None:
        normal = np.asarray(plane["normal"], dtype=np.float64)
        height = float(scan_centre @ normal - plane["offset_m"])
        log.info(
            "  its centre sits %.1f mm above the fitted desk plane (a %.1f mm object "
            "resting flat should read %.1f mm)",
            height * 1000, args.object_height * 1000, args.object_height * 500,
        )

    manifest = read_json(root / "00_ingest" / "manifest.json")["clips"]
    rows = {}
    log.info("")
    log.info("%-9s %8s %10s %10s %10s", "clip", "frames", "median cm", "p90 cm", "max cm")
    for clip in sorted(manifest):
        pose_path = root / "03_estimate" / clip / "object_pose.npy"
        if not pose_path.exists():
            continue
        poses = np.load(pose_path)
        valid = np.load(root / "03_estimate" / clip / "object_valid.npy").astype(bool)
        centres = poses[:, :3, 3]

        # Pre-contact is simply "before the object has moved anywhere".
        travelled = np.linalg.norm(centres - np.median(centres[valid][:15], axis=0), axis=1)
        resting = valid & (travelled < args.move_threshold_m)
        # and only the leading run of it, so a replaced object does not count
        first_move = np.argmax(~resting) if (~resting).any() else len(resting)
        resting[first_move:] = False
        if resting.sum() < 5:
            log.info("%-9s %8s  no settled pre-contact stretch", clip, int(resting.sum()))
            continue

        error = np.linalg.norm(centres[resting] - scan_centre, axis=1) * 100
        rows[clip] = {
            "frames": int(resting.sum()),
            "median_cm": round(float(np.median(error)), 2),
            "p90_cm": round(float(np.percentile(error, 90)), 2),
            "max_cm": round(float(error.max()), 2),
        }
        log.info("%-9s %8d %10.2f %10.2f %10.2f", clip, resting.sum(),
                 rows[clip]["median_cm"], rows[clip]["p90_cm"], rows[clip]["max_cm"])

    out = root / "object_rest_check.json"
    out.write_text(json.dumps(
        {"scan_centre_m": scan_centre.round(5).tolist(), "per_clip": rows}, indent=2))
    log.info("")
    if rows:
        worst = max(r["median_cm"] for r in rows.values())
        log.info("worst median error across clips: %.2f cm", worst)
        log.info("session 4 measured 11 to 16 cm laterally, with the object 8.5 cm high")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
