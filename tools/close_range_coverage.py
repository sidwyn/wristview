"""How much of the scan was shot at wrist-camera range, and from where?

The wrist camera renders from about 0.15 m. A scan shot from 0.5 m renders
soft at contact, and no amount of training fixes it, because no training view
was ever there. This measures the close-range coverage directly instead of
judging it from the render.

Coverage is reported three ways, because a thin pass can be thin in three
different ways: too few frames, frames clustered on one side, or frames all at
one height.

    python -m tools.close_range_coverage --run <run> --radius 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.logging_setup import get, setup  # noqa: E402
from wristview.runctx import read_json  # noqa: E402

log = get(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--radius", type=float, default=0.25,
                        help="what counts as close, in metres")
    parser.add_argument("--target-frames", type=int, default=120,
                        help="frames wanted inside the radius, see the note in the output")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup(None, verbose=False)
    root = Path(args.run).resolve()

    scene = read_json(root / "01_scene" / "cameras.json")
    frames = scene["frames"]
    poses = np.array([f["pose_world_from_cam"] for f in frames], dtype=np.float64)
    centres = poses[:, :3, 3]

    scale = read_json(root / "01_scene" / "scale.json")
    diagnostics = scale.get("diagnostics", {})

    # Anchor the working area on the marker. The capture procedure puts it
    # within 5 to 10 cm of the task object, so it stands in for the object
    # without needing segmentation to have run.
    anchor = diagnostics.get("marker_world_position")
    if anchor is None and diagnostics.get("method") == "aruco":
        # Older runs did not record it. Solve it from the marker detections,
        # which is the same estimate Stage 1 would have written.
        from wristview.camera import Intrinsics
        from wristview.qc import estimate_marker_world_pose

        manifest = read_json(root / "00_ingest" / "manifest.json")["clips"]["scan"]
        pose = estimate_marker_world_pose(
            root / manifest["frames_dir"],
            [f["name"] for f in frames],
            poses,
            np.ones(len(frames), dtype=bool),
            Intrinsics.from_dict(scene["intrinsics"]),
            float(diagnostics["known_side_m"]),
            "DICT_4X4_50",
            int(diagnostics["marker_id"]),
        )
        if pose is not None:
            anchor = pose[:3, 3]
            log.info("marker world position solved from %d detections: %s",
                     diagnostics.get("detections", 0), np.round(anchor, 4).tolist())
    if anchor is None:
        from wristview.backends import dense_cloud as dc
        cloud_path = root / "01_scene" / "dense.ply"
        if not cloud_path.exists():
            log.error("no marker position in scale.json and no dense.ply to fall back on")
            return 1
        points, _ = dc.read_ply(cloud_path)
        anchor = np.median(points, axis=0)
        log.warning("no marker position recorded; using the cloud median as the "
                    "working-area centre, which is cruder")
    anchor = np.asarray(anchor, dtype=np.float64).reshape(3)

    distance = np.linalg.norm(centres - anchor, axis=1)
    close = distance <= args.radius

    fps = float(read_json(root / "00_ingest" / "manifest.json")["clips"]["scan"]
                .get("effective_fps") or 6.0)

    log.info("working-area centre %s", np.round(anchor, 4).tolist())
    log.info("%d scan frames, camera distance to it: median %.2f m, min %.2f m, max %.2f m",
             len(centres), np.median(distance), distance.min(), distance.max())
    log.info("")
    for edge in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        n = int((distance <= edge).sum())
        log.info("  within %.2f m: %4d frames  (%5.1f s at %.0f fps)  %5.1f%% of the scan",
                 edge, n, n / fps, fps, n / len(centres) * 100)

    report = {
        "anchor_m": anchor.round(5).tolist(),
        "frames_total": len(centres),
        "scan_fps": fps,
        "radius_m": args.radius,
        "frames_within_radius": int(close.sum()),
        "seconds_within_radius": round(float(close.sum() / fps), 2),
        "distance_median_m": round(float(np.median(distance)), 4),
        "distance_min_m": round(float(distance.min()), 4),
        "by_radius": {f"{e:.2f}": int((distance <= e).sum())
                      for e in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50)},
    }

    if close.sum() == 0:
        log.error("no scan frame came within %.2f m of the working area", args.radius)
        report["verdict"] = "no close-range coverage at all"
    else:
        # Where are the close frames, around the object and in elevation? A
        # pass that circles one side only leaves the other side unmodelled,
        # and that is invisible in a frame count.
        offset = centres[close] - anchor
        up = np.asarray(diagnostics.get("world_up", [0, 0, 1]), dtype=np.float64)
        up = up / max(np.linalg.norm(up), 1e-12)
        height = offset @ up
        flat = offset - np.outer(height, up)
        basis_a = np.cross(up, [1.0, 0.0, 0.0])
        if np.linalg.norm(basis_a) < 1e-6:
            basis_a = np.cross(up, [0.0, 1.0, 0.0])
        basis_a /= np.linalg.norm(basis_a)
        basis_b = np.cross(up, basis_a)
        azimuth = np.degrees(np.arctan2(flat @ basis_b, flat @ basis_a)) % 360.0

        sectors = np.zeros(8, dtype=int)
        for a in azimuth:
            sectors[int(a // 45)] += 1
        empty = int((sectors == 0).sum())
        log.info("")
        log.info("azimuth around the working area, %d sectors of 45 degrees:", 8)
        log.info("  %s", "  ".join(f"{s:3d}" for s in sectors))
        log.info("  %d of 8 sectors empty, busiest %d frames, thinnest non-empty %d",
                 empty, sectors.max(), sectors[sectors > 0].min() if (sectors > 0).any() else 0)
        log.info("elevation above the working plane: median %.2f m, range %.2f to %.2f m",
                 float(np.median(height)), float(height.min()), float(height.max()))

        report.update({
            "azimuth_sectors": sectors.tolist(),
            "empty_sectors": empty,
            "height_median_m": round(float(np.median(height)), 4),
            "height_min_m": round(float(height.min()), 4),
            "height_max_m": round(float(height.max()), 4),
        })

        # What to shoot next, in seconds, and where.
        deficit = max(args.target_frames - int(close.sum()), 0)
        report["extra_seconds_recommended"] = round(deficit / fps, 1)
        report["target_frames"] = args.target_frames
        log.info("")
        if deficit == 0 and empty == 0:
            log.info("VERDICT: close-range coverage is sufficient and evenly distributed.")
            report["verdict"] = "sufficient"
        else:
            parts = []
            if deficit:
                parts.append(f"{deficit / fps:.0f} more seconds at this range")
            if empty:
                gaps = [i for i, s in enumerate(sectors) if s == 0]
                parts.append("covering the "
                             + ", ".join(f"{g * 45}-{(g + 1) * 45} degree" for g in gaps)
                             + " sector" + ("s" if len(gaps) > 1 else ""))
            log.info("VERDICT: shoot %s.", " ".join(parts))
            report["verdict"] = "thin: " + "; ".join(parts)

    out = Path(args.out) if args.out else root / "close_range_coverage.json"
    out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
