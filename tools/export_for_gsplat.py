"""Package a run's reconstruction for gsplat training on a rented CUDA box.

Writes a self-contained folder: the COLMAP model, the scan images, the metric
scale, and the wrist trajectories. Nothing in it depends on this repository, so
the remote machine only needs gsplat and its dependencies.

    python -m tools.export_for_gsplat --run runs/real03 --out exports/real03
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.logging_setup import get, setup  # noqa: E402
from wristview.runctx import read_json  # noqa: E402

log = get(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--images", action="store_true", default=True)
    parser.add_argument("--splat-only", action="store_true",
                        help="ship only what splat training reads: the COLMAP "
                             "model and the scan images. Skip the trajectories.")
    args = parser.parse_args()

    setup(None, verbose=False)
    run_root = Path(args.run).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # COLMAP model, already in metric world coordinates.
    sparse_out = out / "sparse" / "0"
    sparse_out.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        source = run_root / "01_scene" / "colmap" / "sparse" / name
        if not source.exists():
            raise FileNotFoundError(f"missing {source}; run Stage 1 first")
        shutil.copy2(source, sparse_out / name)
    log.info("copied COLMAP model to %s", sparse_out)

    manifest = read_json(run_root / "00_ingest" / "manifest.json")
    scan = manifest["clips"]["scan"]

    if args.images:
        images_out = out / "images"
        images_out.mkdir(parents=True, exist_ok=True)
        frames_dir = run_root / scan["frames_dir"]
        for name in scan["frame_names"]:
            shutil.copy2(frames_dir / name, images_out / name)
        log.info("copied %d scan images", len(scan["frame_names"]))

    # Wrist trajectories, so the remote box can render without this repo.
    # `--splat-only` drops them. Training never reads them, and a GPU pod
    # that only trains has no use for a retarget result.
    exported = []
    if not args.splat_only:
        traj_out = out / "trajectories"
        traj_out.mkdir(parents=True, exist_ok=True)
        retarget = read_json(run_root / "04_retarget" / "summary.json")
        for clip_id, entry in retarget.items():
            if entry.get("rejected"):
                continue
            source = run_root / "04_retarget" / clip_id / "ee_trajectory.npz"
            if source.exists():
                shutil.copy2(source, traj_out / f"{clip_id}.npz")
                exported.append(clip_id)
        log.info("copied %d trajectories", len(exported))

    scale = read_json(run_root / "01_scene" / "scale.json")
    config = (run_root / "config.yaml").read_text()
    (out / "config.yaml").write_text(config)

    meta = {
        "run": run_root.name,
        "scan_frames": scan["frame_count"],
        "image_size": [scan["video_info"]["width"], scan["video_info"]["height"]],
        "metric_scale_m_per_unit": scale["scale_factor"],
        "scale_method": scale["diagnostics"]["method"],
        "clips": exported,
        "note": (
            "The COLMAP model is already in metric world coordinates: the "
            "Sim(3) in scale.json has been applied to it. Do not apply it "
            "again."
        ),
    }
    (out / "export.json").write_text(json.dumps(meta, indent=2))

    total_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    log.info("wrote %s (%.0f MB)", out, total_mb)
    log.info("  sparse model, %d images, %d trajectories", scan["frame_count"], len(exported))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
