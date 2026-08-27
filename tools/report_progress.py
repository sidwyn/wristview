"""Keep a markdown report current while a long run is still going.

A run of 30 episodes takes hours. Watching a log is not the same as knowing
where it is, and re-reading a log to answer "how is it going" wastes the time
it is meant to save. This parses the run directory itself, writes a table, and
replaces one marked section of a report file, leaving everything else alone.

It reads artifacts, not stdout: `status.json` per clip and the per-stage
`meta.json`. Those are written as each stage finishes, so a partial run
produces a partial table rather than nothing.

    python -m tools.report_progress --run runs/real27full \
        --report runs/real27/REPORT-real27.md --watch 120
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

BEGIN = "<!-- PROGRESS:BEGIN -->"
END = "<!-- PROGRESS:END -->"

# A lift should raise the object about 10 cm. Below this the take did not
# lift, or the object track did not follow it.
LIFT_MIN_SPREAD_CM = 5.0


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def clip_row(run: Path, clip: str) -> dict:
    """Everything known about one clip so far. Missing stages stay None."""
    row: dict = {"clip": clip}

    loc = read_json(run / "02_localize" / "summary.json") or {}
    entry = loc.get(clip) or {}
    row["registered"] = entry.get("frames_registered")
    row["frames"] = entry.get("frames")
    row["inliers"] = entry.get("median_inliers")
    row["inlier_frac"] = entry.get("median_inlier_ratio")

    est = read_json(run / "03_estimate" / f"{clip}/status.json") or {}
    row["hand_rate"] = est.get("hand_detection_rate")
    row["switches"] = est.get("hand_side_switches")
    qc = read_json(run / "03_estimate" / f"{clip}/qc.json") or {}
    for key in ("hand_position_vs_detected_wrist",):
        block = qc.get(key) or {}
        row["reproj_med"] = block.get("median_px")
        row["reproj_p90"] = block.get("p90_px")
    carry = est.get("carry") or {}
    row["contact_frames"] = carry.get("frames_carried")
    row["carry_rejected"] = bool(carry.get("rejected"))

    ret = read_json(run / "04_retarget" / "summary.json") or {}
    rentry = ret.get(clip) or {}
    vel = rentry.get("hand_velocity") or {}
    row["velocity_flags"] = vel.get("frames_flagged")
    dropped = rentry.get("hand_velocity_dropped") or {}
    row["dropped"] = dropped.get("frames_dropped")

    # Hand detection over the manipulation window only, and the miss pattern.
    # The whole-clip rate penalises a take that was shot correctly, with the
    # hand out of frame at both ends.
    hand_path = run / "03_estimate" / clip / "hand.npz"
    if hand_path.exists():
        try:
            valid = np.load(hand_path)["valid"].astype(bool)
            if valid.any():
                first = int(np.argmax(valid))
                last = len(valid) - 1 - int(np.argmax(valid[::-1]))
                inside = valid[first:last + 1]
                row["hand_manip"] = float(inside.mean())
                row["lead_in"] = first
                row["lead_out"] = len(valid) - 1 - last
                row["gaps_inside"] = int((~inside).sum())
        except Exception:
            pass

    # Object lift, the thing a Lift take is for.
    pose = run / "03_estimate" / clip / "object_pose.npy"
    valid_p = run / "03_estimate" / clip / "object_valid.npy"
    scale = read_json(run / "01_scene" / "scale.json")
    if pose.exists() and valid_p.exists() and scale:
        try:
            plane = scale["diagnostics"]["desk_plane"]
            normal = np.asarray(plane["normal"], dtype=float)
            offset = float(plane["offset_m"])
            poses = np.load(pose)
            ok = np.load(valid_p).astype(bool)
            if ok.any():
                height = (poses[ok][:, :3, 3] @ normal) - offset
                row["height_spread_cm"] = float(height.max() - height.min()) * 100
                row["height_max_cm"] = float(height.max()) * 100
        except Exception:
            pass
    return row


def render(run: Path, clips: list[str]) -> str:
    rows = [clip_row(run, c) for c in clips]
    done = [r for r in rows if r.get("registered") is not None]
    with_hand = [r for r in rows if r.get("hand_manip") is not None]

    out = [BEGIN, "",
           f"_Live. Updated {datetime.now().strftime('%H:%M:%S')}. "
           f"{len(done)} of {len(clips)} takes through Stage 2, "
           f"{len(with_hand)} through Stage 3._", ""]

    if not done:
        out += ["Stage 1 is still running. No take has reached Stage 2 yet.", "", END]
        return "\n".join(out)

    out += ["| take | kept | reg | inliers | hand in manip | lead in/out | gaps | reproj med/p90 | switch | vel | contact | lift cm |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def fmt(row):
        def g(k, spec="{}", dash="-"):
            v = row.get(k)
            return dash if v is None else spec.format(v)
        reg = "-"
        if row.get("registered") is not None and row.get("frames"):
            reg = f"{row['registered']}/{row['frames']}"
        hand = "-" if row.get("hand_manip") is None else f"{row['hand_manip'] * 100:.0f}%"
        lead = "-" if row.get("lead_in") is None else f"{row['lead_in']}/{row['lead_out']}"
        rep = "-"
        if row.get("reproj_med") is not None:
            rep = f"{row['reproj_med']:.1f}/{row.get('reproj_p90') or 0:.1f}"
        lift = "-" if row.get("height_spread_cm") is None else f"{row['height_spread_cm']:.1f}"
        flag = ""
        if row.get("height_spread_cm") is not None and row["height_spread_cm"] < LIFT_MIN_SPREAD_CM:
            flag = " **LOW**"
        if row.get("carry_rejected"):
            flag += " **NO CARRY**"
        if row.get("gaps_inside"):
            flag += " **GAPS**"
        return (f"| {row['clip']} | {g('frames')} | {reg} | {g('inliers')} | {hand} | {lead} | "
                f"{g('gaps_inside', dash='-')} | {rep} | {g('switches')} | "
                f"{g('velocity_flags')} | {g('contact_frames')} | {lift}{flag} |")

    # Worst first: no carry, then gaps inside the manipulation, then least lift.
    def badness(r):
        return (
            0 if r.get("carry_rejected") else 1,
            -(r.get("gaps_inside") or 0),
            r.get("height_spread_cm") if r.get("height_spread_cm") is not None else 1e9,
            -(r.get("velocity_flags") or 0),
        )

    for row in sorted(done, key=badness):
        out.append(fmt(row))

    spreads = [r["height_spread_cm"] for r in done if r.get("height_spread_cm") is not None]
    regs = [r["registered"] / r["frames"] for r in done
            if r.get("registered") and r.get("frames")]
    if spreads:
        out += ["", f"Lift spread across {len(spreads)} takes: median "
                    f"{np.median(spreads):.1f} cm, min {min(spreads):.1f}, max {max(spreads):.1f}. "
                    f"{sum(1 for s in spreads if s < LIFT_MIN_SPREAD_CM)} under "
                    f"{LIFT_MIN_SPREAD_CM:.0f} cm."]
    if regs:
        out += [f"Registration across {len(regs)} takes: median {100 * np.median(regs):.1f}%, "
                f"min {100 * min(regs):.1f}%."]
    out += ["", END]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--watch", type=float, default=0,
                        help="seconds between updates; 0 writes once and exits")
    parser.add_argument("--until-clips", type=int, default=0,
                        help="stop once this many clips have reached Stage 3")
    args = parser.parse_args()

    run, report = Path(args.run), Path(args.report)
    while True:
        manifest = read_json(run / "00_ingest" / "manifest.json") or {}
        clips = [c for c in (manifest.get("clips") or {}) if c != "scan"]
        clips.sort()
        section = render(run, clips) if clips else (
            f"{BEGIN}\n\n_Live. Stage 0 has not written a manifest yet._\n\n{END}")

        text = report.read_text() if report.exists() else "# Report\n"
        if BEGIN in text and END in text:
            head, rest = text.split(BEGIN, 1)
            _, tail = rest.split(END, 1)
            text = head + section + tail
        else:
            text = text.rstrip("\n") + "\n\n---\n\n## Live progress\n\n" + section + "\n"
        report.write_text(text)

        finished = sum(
            1 for c in clips
            if (run / "03_estimate" / c / "status.json").exists()
        )
        if not args.watch:
            return 0
        if args.until_clips and finished >= args.until_clips:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
