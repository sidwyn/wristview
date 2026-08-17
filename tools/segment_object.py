"""Cut the task object out of the scan point cloud.

The wrist render needs the object the gripper is holding. Stage 3 would give a
tracked object pose, but that needs splat depth. This is the cheaper route: the
object does not move during the scan, so segment it once in the scan images,
back-project those pixels, and carry the resulting points rigidly with the
gripper after the grasp.

    python -m tools.segment_object --run runs/real03 --prompt "a coffee mug"

Writes `01_scene/object.ply` plus `object_meta.json`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.backends import dense_cloud as dc  # noqa: E402
from wristview.backends import objects as ob  # noqa: E402
from wristview.camera import Intrinsics  # noqa: E402
from wristview.geometry import transform_points  # noqa: E402
from wristview.logging_setup import get, setup  # noqa: E402
from wristview.runctx import read_json  # noqa: E402

log = get(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--prompt", default="a coffee mug")
    parser.add_argument("--anchors", default="/tmp/anchors.pkl",
                        help="poses and per-frame depth anchors, from the pycolmap phase")
    parser.add_argument("--max-frames", type=int, default=25)
    parser.add_argument("--work-resolution", type=int, default=640)
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--voxel-m", type=float, default=0.002)
    parser.add_argument("--cluster-radius-m", type=float, default=0.12,
                        help="drop points further than this from the object's median centre")
    args = parser.parse_args()

    setup(None, verbose=False)
    root = Path(args.run).resolve()

    with open(args.anchors, "rb") as handle:
        anchors = pickle.load(handle)
    poses, observations = anchors["poses"], anchors["obs"]

    cameras = read_json(root / "01_scene" / "cameras.json")
    intrinsics = Intrinsics.from_dict(cameras["intrinsics"])
    scan = read_json(root / "00_ingest" / "manifest.json")["clips"]["scan"]
    frames_dir = root / scan["frames_dir"]

    usable = [n for n in scan["frame_names"] if n in poses and n in observations]
    picked = [usable[i] for i in np.linspace(0, len(usable) - 1, args.max_frames).astype(int)]

    from wristview.backends.depth import DepthAnythingV2

    depth_model = DepthAnythingV2("mps")
    detector = ob.GroundingDinoDetector("mps")
    segmenter = ob.Sam2Segmenter("mps")

    scale = min(1.0, args.work_resolution / max(intrinsics.width, intrinsics.height))
    width = max(32, int(round(intrinsics.width * scale)))
    height = max(32, int(round(intrinsics.height * scale)))
    work = intrinsics.scaled(width, height)

    all_points, all_colors = [], []
    found = 0
    for name in picked:
        image = cv2.imread(str(frames_dir / name))
        if image is None:
            continue
        small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        box = detector.detect(small, args.prompt, args.box_threshold, 0.25)
        if box is None:
            continue
        mask = ob.clean_mask(segmenter.segment(small, box=box), 300, 0.35)
        if mask is None:
            continue

        relative = depth_model.predict(small)
        pixels, depths = observations[name]
        fit = dc._fit_affine_depth(relative, pixels * scale, depths)
        if fit is None:
            continue
        metric, stats = fit
        if stats["correlation"] < 0.5:
            continue

        valid = mask & (metric > 0.05) & (metric < 5.0)
        ys, xs = np.nonzero(valid)
        if len(ys) < 100:
            continue

        cam = work.unproject(
            np.stack([xs, ys], axis=1).astype(np.float64), metric[ys, xs].astype(np.float64)
        )
        all_points.append(transform_points(poses[name], cam))
        all_colors.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)[ys, xs] / 255.0)
        found += 1

    if not all_points:
        log.error("the object was never segmented; try a different --prompt")
        return 1

    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    log.info("segmented %s in %d of %d frames, %d raw points",
             args.prompt, found, len(picked), len(points))

    # Masks leak background at the silhouette, and a stray patch of desk would
    # be carried along with the object. Keep the dominant cluster.
    centre = np.median(points, axis=0)
    keep = np.linalg.norm(points - centre, axis=1) < args.cluster_radius_m
    points, colors = points[keep], colors[keep]
    points, colors = dc.voxel_downsample(points, colors, args.voxel_m)

    extent = points.max(axis=0) - points.min(axis=0)
    log.info("after clustering and downsample: %d points, extent %s m",
             len(points), extent.round(3).tolist())

    dc.write_ply(root / "01_scene" / "object.ply", points, colors)
    (root / "01_scene" / "object_meta.json").write_text(json.dumps({
        "prompt": args.prompt,
        "frames_segmented": found,
        "frames_tried": len(picked),
        "points": int(len(points)),
        "centre_m": centre.round(5).tolist(),
        "extent_m": extent.round(5).tolist(),
        "cluster_radius_m": args.cluster_radius_m,
        "note": (
            "Segmented from the scan, so these points are the object where it "
            "sat during the scan. The renderer carries them rigidly with the "
            "gripper after the grasp event and leaves them here before it."
        ),
    }, indent=2))
    log.info("wrote %s", root / "01_scene" / "object.ply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
