"""Cut a continuously recorded block into per-take ego and wrist clips.

A block is one unbroken ego recording and one unbroken wrist recording of
about ten takes, with a whistle at every take boundary. The whistles are the
cut points, and they are also the only clock the two cameras share, so every
emitted segment KEEPS both of its bounding whistles. Cutting on a whistle
would destroy the thing `fit_sync.py` needs.

    python -m tools.split_takes --ego block1_ego.mov --wrist block1_wrist.mov \
        --out runs/real-sept02/takes --block block1 --expect-whistles 11

WHAT COUNTS AS A TAKE

Whistles bound intervals; not every interval is a take. Resetting the cube
between takes happens on camera, and that footage is not wanted. Rather than
assume a fixed whistle grammar, this classifies by duration: an interval
shorter than `--min-take-s` is a reset and is dropped. The report prints every
interval with its duration so the structure is visible before anything is
encoded, and `--dry-run` stops there.

WHY IT RE-ENCODES

`-c copy` cuts on keyframes, which moves a cut by up to a group-of-pictures
and silently shifts the segment's timebase. Every segment here is the input to
a metric reconstruction, so the cut is exact and the video is re-encoded.
`hevc_videotoolbox` makes that cheap on Apple silicon.

Each segment records `block_start_s`, its own start within the original block.
Sync is fitted in BLOCK time, so anything applying that fit to a segment adds
this back and does not care whether the cut was frame-exact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.audio import find_whistles, load_audio  # noqa: E402
from wristview.logging_setup import get, setup  # noqa: E402
from wristview.videoio import ffmpeg_binary, probe  # noqa: E402

log = get(__name__)

# Keep this much footage outside each bounding whistle, so the whistle is
# whole and has a little room around it for the envelope to settle.
PAD_S = 0.35


def cut(
    source: Path, start_s: float, end_s: float, target: Path, encoder: str
) -> None:
    """Cut [start_s, end_s) exactly, re-encoding video and copying audio."""
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_binary(), "-v", "error", "-y",
        # Seek AFTER -i so the seek is frame-exact rather than keyframe-fast.
        "-i", str(source),
        "-ss", f"{max(start_s, 0.0):.6f}", "-to", f"{end_s:.6f}",
        "-c:v", encoder,
    ]
    if encoder == "hevc_videotoolbox":
        command += ["-q:v", "55", "-tag:v", "hvc1"]
    else:
        command += ["-crf", "18", "-preset", "veryfast"]
    command += ["-c:a", "copy", str(target)]
    subprocess.run(command, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ego", required=True)
    ap.add_argument("--wrist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", required=True, help="block name, prefixes every clip id")
    ap.add_argument("--expect-whistles", type=int, default=None,
                    help="take the N strongest candidates instead of trusting --z-min")
    ap.add_argument("--z-min", type=float, default=20.0,
                    help="robust z above the noise floor. Room dependent; calibrate "
                         "from the printed candidate table")
    ap.add_argument("--min-ms", type=float, default=300.0)
    ap.add_argument("--max-ms", type=float, default=1500.0)
    ap.add_argument("--min-take-s", type=float, default=5.0,
                    help="intervals shorter than this are resets, not takes")
    ap.add_argument("--pad-s", type=float, default=PAD_S)
    ap.add_argument("--encoder", default="hevc_videotoolbox")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the whistles and intervals, encode nothing")
    args = ap.parse_args()

    setup(None, verbose=True)
    ego, wrist = Path(args.ego).resolve(), Path(args.wrist).resolve()
    for path in (ego, wrist):
        if not path.exists():
            log.error("not found: %s", path)
            return 2

    ego_info, wrist_info = probe(ego), probe(wrist)
    log.info("ego   %s  %.2f s", ego.name, ego_info.duration_s)
    log.info("wrist %s  %.2f s", wrist.name, wrist_info.duration_s)

    whistles = find_whistles(
        load_audio(ego), z_min=args.z_min, min_ms=args.min_ms,
        max_ms=args.max_ms, expect=args.expect_whistles,
    )
    log.info("")
    log.info("%d whistles in the ego audio:", len(whistles))
    for index, whistle in enumerate(whistles):
        log.info("  %2d  %8.3f s  %5.0f ms  peak z %8.1f",
                 index, whistle.centre_s, whistle.duration_s * 1000, whistle.peak_z)

    if len(whistles) < 2:
        log.error("need at least two whistles to bound one take. Lower --z-min, or "
                  "pass --expect-whistles to take the N strongest.")
        return 1

    # Intervals between consecutive whistles. Each keeps BOTH bounding
    # whistles, so consecutive segments deliberately overlap by one whistle.
    intervals = []
    for index in range(len(whistles) - 1):
        opening, closing = whistles[index], whistles[index + 1]
        start = max(opening.start_s - args.pad_s, 0.0)
        end = closing.end_s + args.pad_s
        span = closing.start_s - opening.end_s
        intervals.append({
            "index": index,
            "start_s": start,
            "end_s": end,
            "content_s": span,
            "is_take": span >= args.min_take_s,
            "open_whistle_s": opening.centre_s,
            "close_whistle_s": closing.centre_s,
        })

    log.info("")
    log.info("%d intervals, content is the gap between the bounding whistles:", len(intervals))
    for item in intervals:
        log.info("  %2d  %7.3f to %7.3f s   content %6.2f s   %s",
                 item["index"], item["start_s"], item["end_s"], item["content_s"],
                 "TAKE" if item["is_take"] else "reset, dropped")

    takes = [i for i in intervals if i["is_take"]]
    log.info("")
    log.info("%d takes, %d intervals dropped as resets",
             len(takes), len(intervals) - len(takes))
    if not takes:
        log.error("no interval reached --min-take-s %.1f s. Check the table above.",
                  args.min_take_s)
        return 1

    out = Path(args.out).resolve()
    manifest = {
        "block": args.block,
        "ego_source": str(ego),
        "wrist_source": str(wrist),
        "ego_duration_s": round(ego_info.duration_s, 4),
        "wrist_duration_s": round(wrist_info.duration_s, 4),
        "pad_s": args.pad_s,
        "whistles_ego": [w.to_dict() for w in whistles],
        "clips": {},
    }

    if args.dry_run:
        log.info("dry run: nothing encoded")
        print(json.dumps(manifest, indent=1))
        return 0

    pairs = {}
    for order, item in enumerate(takes):
        clip_id = f"{args.block}_take_{order:02d}"
        ego_name, wrist_name = f"{clip_id}_ego.mov", f"{clip_id}_wrist.mov"
        log.info("cutting %s  %.3f to %.3f s", clip_id, item["start_s"], item["end_s"])
        cut(ego, item["start_s"], item["end_s"], out / ego_name, args.encoder)
        # The wrist is cut on the SAME block-time window. It is not yet
        # clock-corrected: the pad exists to absorb the drift, and fit_sync
        # resolves it properly against the whistles both files kept.
        cut(wrist, item["start_s"], item["end_s"], out / wrist_name, args.encoder)
        manifest["clips"][clip_id] = {
            "ego": ego_name,
            "wrist": wrist_name,
            "block_start_s": round(item["start_s"], 6),
            "block_end_s": round(item["end_s"], 6),
            "content_s": round(item["content_s"], 4),
            "open_whistle_block_s": round(item["open_whistle_s"], 6),
            "close_whistle_block_s": round(item["close_whistle_s"], 6),
        }
        pairs[clip_id] = [ego_name, wrist_name]

    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.block}_manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / f"{args.block}_pairs.json").write_text(json.dumps(pairs, indent=1))

    log.info("")
    log.info("wrote %d take pairs to %s", len(pairs), out)
    log.info("")
    log.info("feed the ego clips to the pipeline with:")
    log.info("  --demos %s", " ".join(f"{out.name}/{v[0]}" for v in pairs.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
