"""Sample each real wrist clip at the instants the exported rows already use.

Pixels only. The actions, the state, the 15 Hz grid and the row identities all
come from the ego path; this adds a third view of instants the dataset already
defines. The wrist clips are NOT put through the pipeline: localising a wrist
camera during a grasp fails at 28 per cent inliers and puts the camera inside
a wall.

Each row's ego instant is `timestamps_s[source_frame_index]`, in the ego clip's
own timebase. The wrist clip runs on its own clock:

    t_wrist = rate * t_ego + offset_s

The RATE TERM is not optional. This read `t_ego + offset_s`, which asserts the
two clocks tick at the same speed. They do not: measured drift across a ~16 s
take was +23 ms in one block and +305 ms in another, so a pure offset fitted
at the start of a take is a third of a second wrong by the end of it, and
wrong by a different amount in every block. `tools/fit_sync.py` fits both
terms over every whistle in a block.

A sync file with no `rate` is read as rate 1.0, which is exactly what the old
tap-cross-correlation entries meant.

Frames come out at 1280x720 and are written as-is; `load_image` in
`lerobot_export` does the 640x360 downscale with INTER_AREA, so every camera
goes through one resize path rather than two.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--demos", required=True, help="folder holding wrist_*.mov")
    ap.add_argument("--sync", required=True, help="tap-sync JSON")
    ap.add_argument("--pairs", required=True, help="clip -> (ego, wrist) JSON")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sync = json.loads(Path(args.sync).read_text())
    pairs = json.loads(Path(args.pairs).read_text())

    done, skipped = [], []
    for clip in sorted(sync, key=lambda s: int(s.split("_")[1])):
        entry = sync[clip]
        if entry is None:
            print(f"  {clip}: NO RELIABLE SYNC, skipped")
            skipped.append(clip)
            continue
        offset = float(entry["offset_s"])
        # Absent in the old tap-sync files, which had no rate term to record.
        rate = float(entry.get("rate", 1.0))
        status = json.loads((run / "05_render" / clip / "status.json").read_text())
        idx = np.asarray(status["source_frame_index"])
        traj = np.load(run / "04_retarget" / clip / "ee_trajectory.npz")
        times = rate * np.asarray(traj["timestamps_s"])[idx] + offset

        wrist_mov = Path(args.demos) / pairs[clip][1]
        target = out / clip
        target.mkdir(parents=True, exist_ok=True)
        # One ffmpeg per frame is slow but exact. A select filter on a list of
        # timestamps rounds to the nearest decoded frame and silently drifts on
        # a variable frame rate, which is the fault this whole channel exists
        # to avoid.
        listing = "\n".join(f"{t:.6f}" for t in times)
        (target / "instants.txt").write_text(listing + "\n")
        for i, t in enumerate(times):
            dst = target / f"{i:05d}.png"
            if dst.exists():
                continue
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(t, 0):.6f}",
                 "-i", str(wrist_mov), "-frames:v", "1", str(dst)],
                check=True,
            )
        n = len(list(target.glob("*.png")))
        if n != len(times):
            print(f"  {clip}: wanted {len(times)} frames, got {n}")
            skipped.append(clip)
            continue
        drift_ms = abs(rate - 1.0) * (times[-1] - times[0]) * 1000 if len(times) > 1 else 0.0
        print(f"  {clip}: {n} frames, rate {rate:.9f} offset {offset:+.4f} s "
              f"({drift_ms:.0f} ms of drift across the take)")
        done.append(clip)

    json.dump({"clips": done, "skipped": skipped}, open(out / "index.json", "w"), indent=1)
    print(f"\n{len(done)} clips extracted, {len(skipped)} skipped: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
