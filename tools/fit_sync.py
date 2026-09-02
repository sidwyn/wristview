"""Fit the ego-to-wrist clock for one block, from every whistle in it.

Two phones free-running for ten minutes do not agree. Measured drift over a
~16 s take was +23 ms on one sample pair and +305 ms on another, so the rate
difference is real, it is not shared between blocks, and a single additive
offset cannot express it. This fits

    t_wrist = a * t_ego + b

over ALL the whistles in the block, and reports the residual at each one so a
bad block is visible rather than averaged away.

    python -m tools.fit_sync --ego block1_ego.mov --wrist block1_wrist.mov \
        --manifest runs/.../block1_manifest.json --out runs/.../block1_sync.json

READ THE RESIDUALS, NOT THE FIT. A two-parameter fit through eleven whistles
is heavily over-determined, so a large residual at one whistle means that
whistle was mis-detected, and a residual that grows across the block means the
clocks are not related by a constant rate and this model is the wrong one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.audio import (  # noqa: E402
    coarse_offset,
    find_whistles,
    fit_clock,
    load_audio,
)
from wristview.logging_setup import get, setup  # noqa: E402

log = get(__name__)

# A whistle in one recording is the same whistle in the other if their times
# agree to this once the coarse offset is removed. Whistles are seconds apart,
# so this is loose enough for any plausible drift and far too tight to pair
# the wrong two.
MATCH_TOLERANCE_S = 1.0

# Above this the fit is not trustworthy enough to sample wrist frames with.
MAX_RESIDUAL_MS = 50.0


def match(ego_s: np.ndarray, wrist_s: np.ndarray, shift: float,
          tolerance: float = MATCH_TOLERANCE_S) -> tuple[np.ndarray, np.ndarray]:
    """Pair whistles across the two recordings, nearest neighbour after `shift`.

    Each wrist whistle is claimed at most once, so a spurious detection in one
    recording drops a pair rather than corrupting every later one.
    """
    taken: set[int] = set()
    pairs = []
    for time in ego_s:
        expected = time + shift
        best, best_gap = None, tolerance
        for index, candidate in enumerate(wrist_s):
            if index in taken:
                continue
            gap = abs(candidate - expected)
            if gap < best_gap:
                best, best_gap = index, gap
        if best is not None:
            taken.add(best)
            pairs.append((time, float(wrist_s[best])))
    if not pairs:
        return np.zeros(0), np.zeros(0)
    matched = np.asarray(pairs, dtype=np.float64)
    return matched[:, 0], matched[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ego", required=True)
    ap.add_argument("--wrist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None,
                    help="split_takes manifest, to write a per-take entry as well")
    ap.add_argument("--expect-whistles", type=int, default=None)
    ap.add_argument("--z-min", type=float, default=1000.0)
    ap.add_argument("--min-ms", type=float, default=300.0)
    ap.add_argument("--max-ms", type=float, default=2000.0)
    args = ap.parse_args()

    setup(None, verbose=True)
    ego, wrist = Path(args.ego).resolve(), Path(args.wrist).resolve()

    ego_audio, wrist_audio = load_audio(ego), load_audio(wrist)
    kwargs = dict(z_min=args.z_min, min_ms=args.min_ms, max_ms=args.max_ms,
                  expect=args.expect_whistles)
    ego_whistles = find_whistles(ego_audio, **kwargs)
    wrist_whistles = find_whistles(wrist_audio, **kwargs)
    log.info("whistles: %d in ego, %d in wrist", len(ego_whistles), len(wrist_whistles))

    if len(ego_whistles) < 2 or len(wrist_whistles) < 2:
        log.error("need at least two whistles in each recording to fit a rate")
        return 1

    shift = coarse_offset(ego_audio, wrist_audio)
    log.info("coarse envelope offset %+.3f s", shift)

    ego_times = np.array([w.centre_s for w in ego_whistles])
    wrist_times = np.array([w.centre_s for w in wrist_whistles])
    matched_ego, matched_wrist = match(ego_times, wrist_times, shift)
    log.info("matched %d of %d ego whistles", len(matched_ego), len(ego_times))

    if len(matched_ego) < 2:
        log.error("only %d whistle pairs matched; cannot fit a rate. Check that both "
                  "recordings cover the same block.", len(matched_ego))
        return 1

    rate, offset, residual = fit_clock(matched_ego, matched_wrist)
    residual_ms = residual * 1000.0
    drift_ppm = (rate - 1.0) * 1e6

    log.info("")
    log.info("  t_wrist = %.9f * t_ego + %+.6f", rate, offset)
    log.info("  rate departs unity by %+.0f ppm (%+.1f ms per minute)",
             drift_ppm, (rate - 1.0) * 60_000)
    log.info("")
    log.info("  per-whistle residual:")
    for index, (t_ego, value) in enumerate(zip(matched_ego, residual_ms, strict=True)):
        flag = "  <-- worst" if abs(value) == np.abs(residual_ms).max() else ""
        log.info("    %2d  t_ego %8.3f s   residual %+7.1f ms%s", index, t_ego, value, flag)

    worst = float(np.abs(residual_ms).max())
    rms = float(np.sqrt(np.mean(residual_ms ** 2)))
    log.info("")
    log.info("  residual rms %.1f ms, worst %.1f ms", rms, worst)

    # How much a pure offset would have cost, which is the case for the rate term.
    span = float(matched_ego.max() - matched_ego.min())
    log.info("  a pure offset would drift %.0f ms across this block's %.0f s",
             abs(rate - 1.0) * span * 1000, span)

    verified = worst <= MAX_RESIDUAL_MS
    if not verified:
        log.warning("worst residual %.1f ms exceeds %.1f ms. Treat this block as "
                    "unsynced rather than sampling wrist frames from it.",
                    worst, MAX_RESIDUAL_MS)

    payload = {
        "ego_source": str(ego),
        "wrist_source": str(wrist),
        "rate": rate,
        "offset_s": offset,
        "drift_ppm": round(drift_ppm, 1),
        "matched_whistles": int(len(matched_ego)),
        "ego_whistles": len(ego_whistles),
        "wrist_whistles": len(wrist_whistles),
        "residual_ms": [round(float(v), 3) for v in residual_ms],
        "residual_rms_ms": round(rms, 3),
        "residual_worst_ms": round(worst, 3),
        "verified": verified,
        "max_residual_ms": MAX_RESIDUAL_MS,
        "note": "t_wrist = rate * t_ego + offset_s, both in the BLOCK timebase.",
    }

    # Per-take entries, in each take's own timebase, so a consumer holding one
    # cut clip does not have to know where in the block it came from.
    #
    # THE VERDICT IS PER TAKE, NOT PER BLOCK. Sync quality is a property of
    # the two whistles bounding a take. Condemning a whole block for one bad
    # whistle throws away nine good takes to punish one: on block 2, whistle 9
    # residual -63.3 ms while the first nine were all inside 26 ms. The
    # threshold is unchanged at 50 ms; only the scope it is applied to is
    # corrected.
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        clips = {}
        for clip_id, clip in manifest.get("clips", {}).items():
            start = float(clip["block_start_s"])

            # The residuals at this take's own two bounding whistles.
            bounds = []
            for key in ("open_whistle_block_s", "close_whistle_block_s"):
                if key not in clip:
                    continue
                nearest = int(np.argmin(np.abs(matched_ego - float(clip[key]))))
                bounds.append(abs(float(residual_ms[nearest])))
            local_worst = max(bounds) if bounds else worst

            # t_ego_clip = t_ego_block - start, and the same cut window was
            # used for the wrist, so the clip-local offset absorbs both.
            clips[clip_id] = {
                "rate": rate,
                "offset_s": rate * start + offset - start,
                "block_start_s": start,
                "bounding_residual_worst_ms": round(local_worst, 3),
                "verified": local_worst <= MAX_RESIDUAL_MS,
            }
        payload["clips"] = clips
        good = sum(1 for c in clips.values() if c["verified"])
        log.info("  per-take sync: %d of %d takes verified at %.0f ms",
                 good, len(clips), MAX_RESIDUAL_MS)
        for clip_id, clip in clips.items():
            if not clip["verified"]:
                log.warning("    %s UNSYNCED: bounding whistle residual %.1f ms",
                            clip_id, clip["bounding_residual_worst_ms"])
        payload["takes_verified"] = good
        payload["takes_total"] = len(clips)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    log.info("wrote %s", out)

    # Non-zero only when NOTHING in the block is usable. A block with one bad
    # whistle still delivers its other takes, and the caller should carry on.
    if "takes_verified" in payload:
        return 0 if payload["takes_verified"] else 1
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
