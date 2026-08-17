"""Do hand-tracking dropouts happen where two hands are in shot?

Stage 3 keeps the largest detected hand and discards the rest, so a second
hand cannot directly cause a dropout. But an idle hand in frame changes what
the detector sees: it can win the size comparison, or split confidence, or sit
across the manipulating hand.

This measures the correlation rather than assuming it, because the answer
decides whether the fix belongs in the capture procedure or in the code.

    python -m tools.analyse_two_hands --run runs/real03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.backends import mano_compat  # noqa: E402
from wristview.logging_setup import get, setup  # noqa: E402
from wristview.runctx import read_json  # noqa: E402

log = get(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup(None, verbose=False)
    mano_compat.apply()

    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    pipeline = WiLorHandPose3dEstimationPipeline(
        device=torch.device("mps"), dtype=torch.float32, verbose=False
    )

    root = Path(args.run).resolve()
    manifest = read_json(root / "00_ingest" / "manifest.json")["clips"]
    results = {}

    for clip_id, meta in manifest.items():
        if meta["kind"] != "demo":
            continue
        hand_path = root / "03_estimate" / clip_id / "hand.npz"
        if not hand_path.exists():
            continue
        tracked = np.load(hand_path)["valid"]

        frames_dir = root / meta["frames_dir"]
        counts = np.zeros(len(meta["frame_names"]), dtype=int)
        for index, name in enumerate(meta["frame_names"]):
            image = cv2.imread(str(frames_dir / name))
            if image is None:
                continue
            try:
                detections = pipeline.predict(image[:, :, ::-1])
            except Exception:  # noqa: BLE001 - one bad frame must not stop the sweep
                detections = []
            counts[index] = len(detections)
            if (index + 1) % 60 == 0:
                log.info("  %s: %d/%d", clip_id, index + 1, len(counts))

        n = min(len(counts), len(tracked))
        counts, tracked = counts[:n], tracked[:n].astype(bool)

        two_plus = counts >= 2
        one = counts == 1
        zero = counts == 0

        results[clip_id] = {
            "frames": int(n),
            "tracked_fraction": round(float(tracked.mean()), 4),
            "frames_zero_hands": int(zero.sum()),
            "frames_one_hand": int(one.sum()),
            "frames_two_plus_hands": int(two_plus.sum()),
            "tracked_given_one_hand": round(float(tracked[one].mean()), 4) if one.any() else None,
            "tracked_given_two_plus": round(float(tracked[two_plus].mean()), 4) if two_plus.any() else None,
            "dropouts_total": int((~tracked).sum()),
            "dropouts_with_zero_hands": int((~tracked & zero).sum()),
            "dropouts_with_two_plus": int((~tracked & two_plus).sum()),
        }
        r = results[clip_id]
        log.info(
            "%s: %d frames, tracked %.0f%%. one hand in %d frames, two or more in %d. "
            "tracked given one hand %.0f%%, given two or more %s",
            clip_id, n, r["tracked_fraction"] * 100, r["frames_one_hand"],
            r["frames_two_plus_hands"],
            (r["tracked_given_one_hand"] or 0) * 100,
            f"{r['tracked_given_two_plus'] * 100:.0f}%" if r["tracked_given_two_plus"] is not None else "n/a",
        )

    # Pool across clips, which is where a weak per-clip signal becomes visible.
    pooled = {
        "frames": sum(r["frames"] for r in results.values()),
        "frames_one_hand": sum(r["frames_one_hand"] for r in results.values()),
        "frames_two_plus_hands": sum(r["frames_two_plus_hands"] for r in results.values()),
        "frames_zero_hands": sum(r["frames_zero_hands"] for r in results.values()),
        "dropouts_total": sum(r["dropouts_total"] for r in results.values()),
        "dropouts_with_zero_hands": sum(r["dropouts_with_zero_hands"] for r in results.values()),
        "dropouts_with_two_plus": sum(r["dropouts_with_two_plus"] for r in results.values()),
    }
    if pooled["dropouts_total"]:
        pooled["share_of_dropouts_with_zero_hands"] = round(
            pooled["dropouts_with_zero_hands"] / pooled["dropouts_total"], 4)
        pooled["share_of_dropouts_with_two_plus"] = round(
            pooled["dropouts_with_two_plus"] / pooled["dropouts_total"], 4)

    out = Path(args.out) if args.out else root / "two_hand_analysis.json"
    out.write_text(json.dumps({"per_clip": results, "pooled": pooled}, indent=2))

    log.info("")
    log.info("POOLED over %d frames:", pooled["frames"])
    log.info("  frames with 0 hands detected : %d", pooled["frames_zero_hands"])
    log.info("  frames with 1 hand           : %d", pooled["frames_one_hand"])
    log.info("  frames with 2 or more hands  : %d", pooled["frames_two_plus_hands"])
    log.info("  dropouts                     : %d", pooled["dropouts_total"])
    log.info("    of which 0 hands detected  : %d (%.0f%%)",
             pooled["dropouts_with_zero_hands"],
             pooled.get("share_of_dropouts_with_zero_hands", 0) * 100)
    log.info("    of which 2 or more hands   : %d (%.0f%%)",
             pooled["dropouts_with_two_plus"],
             pooled.get("share_of_dropouts_with_two_plus", 0) * 100)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
