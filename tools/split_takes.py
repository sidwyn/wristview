"""Cut a continuously recorded block into per-take ego and wrist clips.

A block is one unbroken ego recording and one unbroken wrist recording of
about ten takes, with a whistle at every take boundary. The whistles are the
cut points, and they are also the only clock the two cameras share, so every
emitted segment KEEPS both of its bounding whistles. Cutting on a whistle
would destroy the thing `fit_sync.py` needs.

    python -m tools.split_takes --ego block1_ego.mov --wrist block1_wrist.mov \
        --out runs/real-sept02/takes --block block1 --expect-whistles 11

WHAT COUNTS AS A TAKE

Every interval between two whistles is a take. Measured on block 1: 11
whistles, 10 intervals, all 13.5 to 15.5 s. The cube is reset INSIDE the take,
not in a gap between whistles, so there are no short intervals to discard.

`--min-take-s` therefore exists to catch a FAULT, not to classify footage. An
interval under it means a spurious whistle split one take in two, and silently
dropping it would delete real footage, so it is an error unless `--allow-drop`.

THE OVERLAP, AND WHY THE VIDEO IS TRIMMED

Cutting outside both bounding whistles makes segment N and segment N+1 share
the whistle between them: about 1.8 s, 36 ego frames at 20 fps, ~27 rows at
15 Hz. The headline result is a paired comparison measured on held-out
episodes, so an eval episode next to a training episode would have had some
of its rows seen during training. On a 367-row eval set one shared boundary
is about 7 per cent, which is the same order as the effect being measured.

So the VIDEO is cut to the content window, strictly between the whistles.
Consecutive takes then share no frame at all. Nothing real is lost: the
trimmed footage is the whistle itself, where the hand is out of frame.

The whistles still have to survive, because they are the only clock the two
cameras share. They survive as AUDIO: each take gets a .wav spanning both
bounding whistles, which is what any re-verification actually reads. Sync
never needed those pixels.

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


def cut_audio(source: Path, start_s: float, end_s: float, target: Path) -> None:
    """Extract [start_s, end_s) of audio as 16 kHz mono WAV.

    This is the whistle-bearing span, kept so a take can be re-synced on its
    own. It is audio because sync reads audio; carrying the pixels too would
    put the same frames in two episodes.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg_binary(), "-v", "error", "-y", "-i", str(source),
         "-ss", f"{max(start_s, 0.0):.6f}", "-to", f"{end_s:.6f}",
         "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(target)],
        check=True,
    )


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
        # CARRY THE LENS TAG. The focal length lives in
        # com.blackmagic-design.camera.lensType, which is a custom QuickTime
        # tag. `-map_metadata 0` alone does NOT carry it: ffmpeg writes only
        # tags it recognises into mov, and silently drops the rest. Without
        # `use_metadata_tags` every cut take lost its lens and Stage 0 fell
        # back to a 1632 px guess where the label gives 1280, an 11 per cent
        # error in the camera every demo pose is solved against.
        "-map_metadata", "0", "-movflags", "use_metadata_tags",
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
    ap.add_argument("--z-min", type=float, default=None,
                    help="robust z above the noise floor. Room dependent; calibrate "
                         "from the printed candidate table")
    ap.add_argument("--min-ms", type=float, default=300.0)
    ap.add_argument("--max-ms", type=float, default=2000.0)
    ap.add_argument("--min-take-s", type=float, default=5.0,
                    help="an interval shorter than this is not a take. It is an ERROR "
                         "unless --allow-drop: on real blocks every interval is a take")
    ap.add_argument("--allow-drop", action="store_true",
                    help="permit intervals under --min-take-s to be discarded")
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

    audio = load_audio(ego)

    # Every run above the threshold at ANY duration, so the separation between
    # real whistles and spurious bursts is visible rather than assumed.
    #
    # Do not conclude from one block that duration is enough. On block 1 every
    # spurious burst was 15 to 115 ms and the 300 ms floor removed all of
    # them. Block 2 carried a 355 ms burst in the ego and a 645 ms one in the
    # wrist, which cleared the floor and would each have split a take in two.
    # z is what separates them: real 5,212 to 19,599, spurious at most 145.
    candidates = find_whistles(audio, z_min=args.z_min if args.z_min is not None else 100.0,
                               min_ms=0.0, max_ms=1e9)
    log.info("")
    log.info("%d runs at any duration (z floor %s):", len(candidates),
             args.z_min if args.z_min is not None else "self-calibrated")
    for candidate in sorted(candidates, key=lambda w: -w.peak_z):
        verdict = "kept" if args.min_ms <= candidate.duration_s * 1000 <= args.max_ms else "dropped on duration"
        log.info("    z %9.0f   %6.0f ms   t %8.2f s   %s",
                 candidate.peak_z, candidate.duration_s * 1000, candidate.centre_s, verdict)

    whistles = find_whistles(
        audio, z_min=args.z_min, min_ms=args.min_ms,
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
            # The non-overlapping window: strictly between the two whistles.
            # Consecutive content windows do not touch, so trimming to these
            # removes the shared frames entirely.
            "content_start_s": opening.end_s,
            "content_end_s": closing.start_s,
        })

    log.info("")
    log.info("%d intervals, content is the gap between the bounding whistles:", len(intervals))
    for item in intervals:
        log.info("  %2d  %7.3f to %7.3f s   content %6.2f s   %s",
                 item["index"], item["start_s"], item["end_s"], item["content_s"],
                 "TAKE" if item["is_take"] else "reset, dropped")

    takes = [i for i in intervals if i["is_take"]]
    dropped = [i for i in intervals if not i["is_take"]]
    log.info("")
    log.info("%d intervals, %d at or above --min-take-s %.1f s",
             len(intervals), len(takes), args.min_take_s)
    if not takes:
        log.error("no interval reached --min-take-s %.1f s. Check the table above.",
                  args.min_take_s)
        return 1

    # Dropping is never silent. On block 1 every interval is a take: the reset
    # happens INSIDE the ~15 s, not between whistles, so a short interval here
    # means a MIS-DETECTED whistle split one take in two, and dropping it
    # would quietly delete real footage. Refuse, and make the operator say so.
    if dropped and not args.allow_drop:
        log.error("")
        log.error("%d interval(s) fall under --min-take-s %.1f s:", len(dropped),
                  args.min_take_s)
        for item in dropped:
            log.error("    interval %d, %.2f s of content between whistles at %.2f and %.2f s",
                      item["index"], item["content_s"], item["open_whistle_s"],
                      item["close_whistle_s"])
        log.error("")
        log.error("A short interval usually means a spurious whistle split one take, "
                  "not that a reset was recorded. Fix the detection, or pass "
                  "--allow-drop if these really are not takes.")
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
        # VIDEO: content only, so consecutive takes share no frame.
        start, end = item["content_start_s"], item["content_end_s"]
        log.info("cutting %s  content %.3f to %.3f s  (%.2f s)",
                 clip_id, start, end, end - start)
        cut(ego, start, end, out / ego_name, args.encoder)
        # The wrist is cut on the SAME block-time window, deliberately. It is
        # not clock-corrected here: fit_sync resolves that, and correcting it
        # twice would double-count the offset.
        cut(wrist, start, end, out / wrist_name, args.encoder)

        # AUDIO: the whistle-bearing span, for re-syncing this take alone.
        sync_start = max(item["start_s"], 0.0)
        sync_end = item["end_s"]
        cut_audio(ego, sync_start, sync_end, out / f"{clip_id}_ego_sync.wav")
        cut_audio(wrist, sync_start, sync_end, out / f"{clip_id}_wrist_sync.wav")

        manifest["clips"][clip_id] = {
            "ego": ego_name,
            "wrist": wrist_name,
            "ego_sync_audio": f"{clip_id}_ego_sync.wav",
            "wrist_sync_audio": f"{clip_id}_wrist_sync.wav",
            # Block time of the VIDEO, which is what a sync fit must be
            # rebased onto.
            "block_start_s": round(start, 6),
            "block_end_s": round(end, 6),
            "content_s": round(end - start, 4),
            # Block time of the whistle-bearing audio span.
            "sync_audio_block_start_s": round(sync_start, 6),
            "sync_audio_block_end_s": round(sync_end, 6),
            "open_whistle_block_s": round(item["open_whistle_s"], 6),
            "close_whistle_block_s": round(item["close_whistle_s"], 6),
        }
        pairs[clip_id] = [ego_name, wrist_name]

    # Prove the lens survived the re-encode rather than trusting the flag.
    # This is the defect this codebase repeats: compute the number, then never
    # read it. The first version of this tool dropped the lens tag on all
    # twenty takes and nothing noticed until Stage 0 printed fallback_guess.
    source_focal = probe(ego).focal_35mm
    if source_focal is not None:
        for clip_id in pairs:
            cut_focal = probe(out / pairs[clip_id][0]).focal_35mm
            if cut_focal != source_focal:
                log.error("%s lost the lens metadata: source %s mm, cut %s mm. "
                          "Stage 0 would fall back to a guessed focal length.",
                          clip_id, source_focal, cut_focal)
                return 1
        log.info("lens metadata survived every cut: %.0f mm", source_focal)
    else:
        log.warning("the source block carries no lens tag, so the cuts cannot "
                    "either. Stage 0 will self-calibrate from a guess.")

    # Prove the leak is gone rather than asserting it.
    ordered = [manifest["clips"][k] for k in pairs]
    overlaps = [
        round(ordered[i]["block_end_s"] - ordered[i + 1]["block_start_s"], 6)
        for i in range(len(ordered) - 1)
    ]
    worst = max(overlaps) if overlaps else 0.0
    log.info("")
    log.info("worst overlap between consecutive takes: %.3f s (must be <= 0)", worst)
    if worst > 0:
        log.error("takes still overlap by %.3f s; episodes would share frames", worst)
        return 1
    manifest["worst_overlap_s"] = worst

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
