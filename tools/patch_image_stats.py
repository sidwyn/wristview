"""Recompute the image statistics of an exported LeRobot dataset, as float.

lerobot's stats accumulator does `np.mean(batch ** 2)` on the raw uint8 frame.
In uint8, 255 ** 2 wraps to 1 mod 256, so the mean of squares comes out far
below the square of the mean, the variance goes negative, and the sqrt guard
clamps it to near zero. sept02_final recorded an ego std of
[0.0171, 0.0127, 0.0111] where the true value is about [0.242, 0.215, 0.196].

`MEAN_STD` normalisation divides by that number, so every image reaching a
policy was scaled about 14x too large. See DEFECT-TABLE row 48.

Only `std` is rebuilt, and `mean` is checked. `min`, `max` and the quantiles
never square anything and are already correct. The quantiles are in fact the
cross-check nobody read: a q10 to q90 spread of 0.62 cannot coexist with a
std of 0.017.

    python -m tools.patch_image_stats --dataset runs/.../lerobot [--apply]

Without --apply it reports and changes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

# A std estimate over this many frames is stable to well under a percent:
# each frame contributes about 230,000 pixels per channel.
SAMPLE_FRAMES = 400

# Below this, a std cannot have come from real image pixels and is the
# overflow signature rather than an unusually flat dataset.
IMPOSSIBLE_STD = 0.05


def working_video_backend() -> str:
    """A decode backend that actually decodes here.

    Inlined rather than imported from `wristview.lerobot_export`. The editable
    install in this repo stops resolving on its own, four times in one session,
    and a data-repair tool must not fail for that reason. See CLAUDE.md.
    """
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception:  # noqa: BLE001 - any failure means it cannot decode
        return "pyav"
    return "torchcodec"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frames", type=int, default=SAMPLE_FRAMES)
    ap.add_argument("--apply", action="store_true",
                    help="write the file. Without it, this only reports.")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.dataset).resolve()
    path = root / "meta" / "stats.json"
    stats = json.loads(path.read_text())
    image_keys = sorted(k for k in stats if k.startswith("observation.images"))
    if not image_keys:
        print("no image features in this dataset")
        return 1

    ds = LeRobotDataset(repo_id=f"wristview/{root.parents[1].name}", root=root,
                        video_backend=working_video_backend())
    picks = np.linspace(0, len(ds) - 1, min(args.frames, len(ds))).astype(int)
    print(f"{root}\n  {len(ds)} frames, sampling {len(picks)} of them\n")

    changed = {}
    for key in image_keys:
        px = np.stack([ds[int(i)][key].numpy() for i in picks])   # (N,3,H,W) float
        mean = px.mean(axis=(0, 2, 3)).astype(np.float64)
        std = px.std(axis=(0, 2, 3)).astype(np.float64)
        old_std = np.asarray(stats[key]["std"], dtype=np.float64).ravel()
        old_mean = np.asarray(stats[key]["mean"], dtype=np.float64).ravel()

        flag = "  OVERFLOW SIGNATURE" if old_std.max() < IMPOSSIBLE_STD else ""
        print(f"  {key}")
        print(f"    std  recorded {np.round(old_std, 4).tolist()}"
              f"  ->  measured {np.round(std, 4).tolist()}"
              f"  ({std.max() / max(old_std.max(), 1e-12):.1f}x){flag}")
        print(f"    mean recorded {np.round(old_mean, 4).tolist()}"
              f"  ->  measured {np.round(mean, 4).tolist()}")
        shape = np.asarray(stats[key]["std"]).shape
        changed[key] = {"std": std.reshape(shape).tolist(),
                        "mean": mean.reshape(shape).tolist()}

    if not args.apply:
        print("\nreported only. Pass --apply to write.")
        return 0

    backup = path.with_suffix(".json.corrupt.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"\nkept the original at {backup.name}")
    for key, vals in changed.items():
        stats[key].update(vals)
    path.write_text(json.dumps(stats, indent=4))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
