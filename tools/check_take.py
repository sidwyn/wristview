"""Gate a take from the raw video, before anything is processed.

Reads video files only. No reconstruction, no COLMAP, no Stage 0. It answers
one question in under a minute, while the operator is still at the mat:

    is it safe to shoot the other 30?

The two failures it exists to catch both happened on real27, and both cost a
session:

- The demo began with the hand already on the object. There was no resting
  window before the grasp, so the carry could not be solved and the object
  stayed welded to the desk for all 300 frames. Nothing before Stage 3 saw it,
  and Stage 3 is 40 minutes downstream.
- The scan spent 88 per cent of itself above 38 cm and 5.5 per cent in the
  20 to 30 cm band the wrist camera actually renders from. real26/bm failed
  the viewpoint gate for the same reason at 15.05 cm against a 15.00 limit.

    python -m tools.check_take --scan room_scan.mov --demo demo_1.mov
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

# --- check 1 -----------------------------------------------------------------
# EXPERIMENT-PLAN 4.4 asks for about 10 s. Outside this band the take is either
# clipped or carrying dead air, and dead air is the cheapest thing to fix.
DEMO_SECONDS_MIN, DEMO_SECONDS_MAX = 8.0, 13.0

# A tapped take carries a tap at each end, so it is longer.
TAPPED_SECONDS_MIN, TAPPED_SECONDS_MAX = 12.0, 18.0

# --- check 2 -----------------------------------------------------------------
# The hand must be out of frame at the start, so a resting pose exists before
# the grasp. real27 had the hand within 1.8 cm of the object at frame 0.
HAND_FREE_FRAMES = 30

# A tapped take does not begin hand-free: the hand enters, taps the table,
# leaves, and only then reaches for the object. So the window to find is the
# GAP between the tap and the reach, not the head of the clip. Measuring from
# frame 0 would find the tap and fail every correctly shot group-C take.
TAPPED_GAP_MIN_FRAMES = 30
TAPPED_GAP_MIN_SECONDS = 1.5
# How far in to look before giving up. A tap and a reach inside 15 s covers
# the 12 to 18 s a tapped take should run.
TAPPED_SEARCH_SECONDS = 15.0

# --- check 4 -----------------------------------------------------------------
MARKER_SIDE_M = 0.100
MARKER_ID = 0
BANDS_M = [(0.00, 0.15), (0.15, 0.20), (0.20, 0.30),
           (0.30, 0.38), (0.38, 0.60), (0.60, 99.0)]
# EXPERIMENT-PLAN 4.2 allocates 45 s of an 85 s scan below 30 cm.
SECONDS_BELOW_30CM_MIN = 45.0

# --- check 5 -----------------------------------------------------------------
AZIMUTH_SECTORS = 8
RENDER_BAND_M = (0.20, 0.30)

# --- measurement coverage ----------------------------------------------------
# Below this the CHECK failed, not the scan.
MARKER_VISIBLE_MIN_FRACTION = 0.30

# CHECKS 4 AND 5 CAN ONLY EVER PASS, NEVER FAIL. Here is why.
#
# The marker lies flat on the desk. A camera low down and looking ACROSS the
# mat sees it obliquely or not at all, and that is exactly where the render
# band is. Measured on real26/d, against heights from its reconstruction:
#
#   height      frames   marker seen    rate
#   0-15 cm        105             0      0%
#   15-20 cm       127             6      5%
#   20-30 cm       102            26     25%
#   30-38 cm       143            66     46%
#   38-60 cm       254           183     72%
#
# A frame below 30 cm is 0.15x as likely to show the marker as one above. On
# that scan the true share below 30 cm is 45.7 per cent; the marker-visible
# subset says 11.4 per cent, a four-fold undercount. Below 15 cm the marker was
# never seen at all, in 105 frames.
#
# So a scan shot CORRECTLY, spending its time low, is the scan whose low frames
# the marker cannot see. Failing it on this measurement would reject exactly
# what it is meant to accept: real26/d passed the real viewpoint gate at
# 7.47 cm and this check scores it at 5.8 s below 30 cm.
#
# The measurement is therefore a LOWER BOUND. If the marker-visible frames
# alone already clear the bar, the scan clears it, because the unseen frames
# are lower still. If they do not, nothing has been shown either way.
#
# CLAUDE.md warns that marker PnP has produced selection bias three times.
# This is the fourth.
#
# To make it a real gate, the capture has to change, not the tool: a second
# marker standing VERTICAL beside the mat would be visible from the low
# across-the-mat views, and its pose gives the same height and azimuth. Until
# that exists, the honest pre-flight is checks 1 to 3, and the render band is
# confirmed after Stage 1 by close_range_coverage.

# Focal length in pixels at 1920 wide, when the video carries no usable
# metadata. Calibrated from real27, whose Stage 1 refined the scan camera to
# f = 1451.2 px at 1920x1080 across 327 registered views.
FOCAL_PX_AT_1920 = 1451.2


def rotation_tag(path: Path) -> int:
    """Return the display rotation ffmpeg would apply, in degrees."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream_side_data=rotation:stream_tags=rotate",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out or "{}")
    for stream in data.get("streams", []):
        for side in stream.get("side_data_list", []) or []:
            if "rotation" in side:
                return int(float(side["rotation"])) % 360
        tag = (stream.get("tags") or {}).get("rotate")
        if tag is not None:
            return int(float(tag)) % 360
    return 0


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = (stream["r_frame_rate"].split("/") + ["1"])[:2]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(num) / max(float(den), 1.0),
        "duration_s": float(data["format"]["duration"]),
        "rotation": rotation_tag(path),
    }


def hand_presence(path: Path, frames: int) -> list[bool]:
    """Per-frame hand presence for the first `frames` frames."""
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    )
    capture = cv2.VideoCapture(str(path))
    present: list[bool] = []
    try:
        for _ in range(frames):
            ok, frame = capture.read()
            if not ok:
                break
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            present.append(bool(result.multi_hand_landmarks))
    finally:
        capture.release()
        hands.close()
    return present


def tapped_gap(present: list[bool], fps: float) -> dict:
    """Find the hand-free window between the tap and the reach.

    A tapped take runs: hand in (tap), hand out (the gap), hand in (the reach).
    The gap is what the resting pose is measured from, so it has to exist and
    be long enough. Looking only at the head of the clip, as the untapped check
    does, would find the tap and fail every correctly shot take.
    """
    runs = []
    index = 0
    while index < len(present):
        start = index
        value = present[index]
        while index < len(present) and present[index] == value:
            index += 1
        runs.append({"value": value, "start": start, "stop": index, "length": index - start})

    # Detection flickers. A single frame of hand, or a two-frame dropout in the
    # middle of a reach, are not a tap and not a gap. Requiring both sides to
    # be sustained stops a flicker being read as the structure we are looking
    # for. Measured on an untapped take, the naive version found a "tap" at
    # frame 178 and a 2-frame "gap" at 179-180, in the middle of a grasp.
    MIN_RUN = 5

    candidates = []
    saw_hand = False
    for position, run in enumerate(runs):
        if run["value"]:
            if run["length"] >= MIN_RUN:
                saw_hand = True
            continue
        if not saw_hand:
            # A hand-free head, before any tap. Not the gap we want.
            continue
        following = runs[position + 1] if position + 1 < len(runs) else None
        if following is None or not following["value"] or following["length"] < MIN_RUN:
            continue
        previous = next((r for r in reversed(runs[:position])
                         if r["value"] and r["length"] >= MIN_RUN), None)
        candidates.append({
            "found": True,
            "start": run["start"], "stop": run["stop"],
            "frames": run["length"], "seconds": run["length"] / fps,
            "tap_run": [previous["start"], previous["stop"]] if previous else None,
            "reach_starts": following["start"],
        })

    # Take the first gap that is long enough. Returning the first gap of any
    # length would let a flicker mask the real one further in.
    for candidate in candidates:
        if candidate["frames"] >= TAPPED_GAP_MIN_FRAMES and \
                candidate["seconds"] >= TAPPED_GAP_MIN_SECONDS:
            candidate["candidates_seen"] = len(candidates)
            return candidate
    if candidates:
        longest = max(candidates, key=lambda c: c["frames"])
        longest["candidates_seen"] = len(candidates)
        return longest
    return {"found": False, "runs": len(runs),
            "hand_frames": int(sum(present)), "frames_examined": len(present)}


def first_hand_frame(path: Path, frames: int) -> tuple[int | None, int]:
    """Return the first frame index holding a hand, and how many were read."""
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    )
    capture = cv2.VideoCapture(str(path))
    found, read = None, 0
    try:
        for index in range(frames):
            ok, frame = capture.read()
            if not ok:
                break
            read = index + 1
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                found = index
                break
    finally:
        capture.release()
        hands.close()
    return found, read


def scan_distances(path: Path, info: dict, sample_fps: float = 5.0) -> dict:
    """Camera HEIGHT above the desk and azimuth, from marker id 0.

    Height, not range. The marker lies flat on the desk, so the marker's own
    Z axis is the desk normal and the camera centre's Z in the marker frame is
    its height above the desk. That is the quantity `close_range_coverage`
    reports from the reconstruction, and the quantity the render band is
    defined in.

    Range to the marker is NOT the same thing and must not be substituted. A
    camera 25 cm above the desk but 40 cm to one side is 47 cm from the
    marker. Using range would have put real27's whole scan two bands too high
    and would reject a scan that was shot correctly but wide.

    Both come from one IPPE_SQUARE solve. The apparent side is kept only as a
    cross-check on that solve.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    # Detect at 1280 wide. This is not an arbitrary speed compromise: on
    # real27's scan the marker was found in 19 per cent of frames at 960,
    # 47 per cent at 1280, and 31 per cent at full 1920. ArUco's adaptive
    # threshold does worse on full-resolution sensor noise than on a modestly
    # downscaled frame, so the relationship is not monotonic and the middle
    # size wins. A detector that misses the marker turns a measurable scan
    # into "could not measure".
    scale = 1280.0 / info["width"]
    focal = FOCAL_PX_AT_1920 * (info["width"] / 1920.0) * scale
    width, height = int(info["width"] * scale), int(info["height"] * scale)
    K = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1.0]])
    corners_local = np.array([
        [-MARKER_SIDE_M / 2, MARKER_SIDE_M / 2, 0], [MARKER_SIDE_M / 2, MARKER_SIDE_M / 2, 0],
        [MARKER_SIDE_M / 2, -MARKER_SIDE_M / 2, 0], [-MARKER_SIDE_M / 2, -MARKER_SIDE_M / 2, 0],
    ], dtype=np.float64)

    capture = cv2.VideoCapture(str(path))
    step = max(1, int(round(info["fps"] / sample_fps)))
    distances: list[float] = []      # height above the desk
    azimuths: list[float] = []
    ranges: list[float] = []         # straight-line range, cross-check only
    sampled = 0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step:
                index += 1
                continue
            index += 1
            sampled += 1
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            found, ids, _ = detector.detectMarkers(small)
            if ids is None or MARKER_ID not in ids.flatten():
                continue
            slot = int(np.nonzero(ids.flatten() == MARKER_ID)[0][0])
            pixels = found[slot].reshape(4, 2)
            side = float(np.mean([
                np.linalg.norm(pixels[i] - pixels[(i + 1) % 4]) for i in range(4)
            ]))
            if side < 4:
                continue

            ok_pnp, rvec, tvec = cv2.solvePnP(
                corners_local, pixels.astype(np.float64), K, None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok_pnp:
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            # Camera centre expressed in the marker's frame. The marker lies on
            # the desk, so Z is height above the desk and the XY bearing is the
            # azimuth around the mat.
            centre = (-rotation.T @ tvec).ravel()
            distances.append(abs(float(centre[2])))
            azimuths.append(float(np.degrees(np.arctan2(centre[1], centre[0]))))
            ranges.append(focal * MARKER_SIDE_M / side)
    finally:
        capture.release()

    return {
        "sampled": sampled,
        "seconds_per_sample": info["duration_s"] / max(sampled, 1),
        "height_m": np.array(distances),
        "azimuth_deg": np.array(azimuths),
        "range_m": np.array(ranges),
        "focal_px_used": focal / scale,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--demo", required=True)
    parser.add_argument("--tapped", action="store_true",
                        help="a group-C take, with a tap at each end. Widens the "
                             "duration limit and looks for the hand-free window "
                             "BETWEEN the tap and the reach.")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    scan, demo = Path(args.scan), Path(args.demo)
    scan_info, demo_info = probe(scan), probe(demo)
    report: dict = {"scan": str(scan), "demo": str(demo),
                    "scan_info": scan_info, "demo_info": demo_info}
    hard: list[str] = []
    unmeasured: list[str] = []

    print(f"scan  {scan.name}: {scan_info['duration_s']:.1f}s "
          f"{scan_info['width']}x{scan_info['height']} @{scan_info['fps']:.0f} "
          f"rotation {scan_info['rotation']}")
    print(f"demo  {demo.name}: {demo_info['duration_s']:.1f}s "
          f"{demo_info['width']}x{demo_info['height']} @{demo_info['fps']:.0f} "
          f"rotation {demo_info['rotation']}")
    print()

    # --- 1. demo duration ---
    low, high = ((TAPPED_SECONDS_MIN, TAPPED_SECONDS_MAX) if args.tapped
                 else (DEMO_SECONDS_MIN, DEMO_SECONDS_MAX))
    seconds = demo_info["duration_s"]
    ok1 = low <= seconds <= high
    report["mode"] = "tapped" if args.tapped else "plain"
    report["check1_duration_s"] = round(seconds, 2)
    print(f"1 demo duration      {seconds:6.1f} s   "
          f"[{low:.0f} to {high:.0f}{', tapped' if args.tapped else ''}]   "
          f"{'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        hard.append(
            f"the demo is {seconds:.1f} s, outside {low:.0f} to {high:.0f} s"
            + (" for a tapped take" if args.tapped else "") + ". "
            + ("Trim the dead air; it costs compute on every take."
               if seconds > high else "The take is clipped.")
        )

    # --- 2. a hand-free window before the grasp ---
    if args.tapped:
        budget = int(TAPPED_SEARCH_SECONDS * demo_info["fps"])
        present = hand_presence(demo, budget)
        gap = tapped_gap(present, demo_info["fps"])
        report["check2_tapped_gap"] = gap
        ok2 = bool(gap.get("found")) and gap["frames"] >= TAPPED_GAP_MIN_FRAMES \
            and gap["seconds"] >= TAPPED_GAP_MIN_SECONDS
        if gap.get("found"):
            # Say where it looked. A gate that passes without showing its
            # working is not evidence.
            print(f"2 hand-free gap      {gap['seconds']:5.2f} s   "
                  f"[{TAPPED_GAP_MIN_SECONDS:.1f} s and {TAPPED_GAP_MIN_FRAMES} frames]  "
                  f"{'PASS' if ok2 else 'FAIL'}")
            print(f"    tap        frames {gap['tap_run'][0]:4d}-{gap['tap_run'][1] - 1:<4d}"
                  f" ({gap['tap_run'][0] / demo_info['fps']:5.2f}-"
                  f"{(gap['tap_run'][1] - 1) / demo_info['fps']:5.2f} s)")
            print(f"    GAP        frames {gap['start']:4d}-{gap['stop'] - 1:<4d}"
                  f" ({gap['start'] / demo_info['fps']:5.2f}-"
                  f"{(gap['stop'] - 1) / demo_info['fps']:5.2f} s)  "
                  f"{gap['frames']} frames")
            print(f"    reach from frame {gap['reach_starts']:4d}"
                  f"      ({gap['reach_starts'] / demo_info['fps']:5.2f} s)")
        else:
            print(f"2 hand-free gap      {'none':>6}     "
                  f"[searched {len(present)} frames]   FAIL")
            print(f"    hand present in {gap['hand_frames']} of "
                  f"{gap['frames_examined']} frames examined, in {gap['runs']} run(s)")
        if not ok2:
            if not gap.get("found"):
                hard.append(
                    f"no hand-free window was found between a tap and a reach in "
                    f"the first {TAPPED_SEARCH_SECONDS:.0f} s. A tapped take must "
                    f"go: tap, hand fully out of frame, then reach. Without the "
                    f"gap the object is never seen at rest and the carry cannot "
                    f"be solved."
                )
            else:
                hard.append(
                    f"the hand-free gap between the tap and the reach is "
                    f"{gap['seconds']:.2f} s over {gap['frames']} frames, under "
                    f"{TAPPED_GAP_MIN_SECONDS:.1f} s. Pause longer after the tap, "
                    f"with the hand fully out of frame."
                )
    else:
        first, _read = first_hand_frame(demo, HAND_FREE_FRAMES)
        ok2 = first is None
        report["check2_first_hand_frame"] = first
        report["check2_frames_checked"] = HAND_FREE_FRAMES
        print(f"2 hand-free start    {'none' if ok2 else f'frame {first}':>6}     "
              f"[first {HAND_FREE_FRAMES} frames]   {'PASS' if ok2 else 'FAIL'}")
        if not ok2:
            hard.append(
                f"a hand appears at frame {first} of the first {HAND_FREE_FRAMES}. "
                f"The take must start with the hand OUT of frame, so the object "
                f"is seen at rest before the grasp. Without that the carry cannot "
                f"be solved and the object stays pinned to the desk for the whole "
                f"clip."
            )

    # --- 3. rotation tags agree ---
    ok3 = scan_info["rotation"] == demo_info["rotation"]
    print(f"3 rotation match     {scan_info['rotation']:>3} vs {demo_info['rotation']:<3}"
          f"          {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        hard.append(
            f"the scan is tagged {scan_info['rotation']} degrees and the demo "
            f"{demo_info['rotation']}. Hold the camera the same way up for both."
        )

    # --- 4 and 5. scan geometry from the marker ---
    geometry = scan_distances(scan, scan_info, args.sample_fps)
    visible = len(geometry["height_m"])
    fraction = visible / max(geometry["sampled"], 1)
    report["check4_marker_visible_fraction"] = round(fraction, 4)
    report["focal_px_used"] = round(geometry["focal_px_used"], 1)
    print()
    print(f"  marker id {MARKER_ID} visible in {visible} of {geometry['sampled']} "
          f"sampled frames ({fraction * 100:.0f}%), f={geometry['focal_px_used']:.0f} px")

    measurable = fraction >= MARKER_VISIBLE_MIN_FRACTION
    if not measurable:
        unmeasured.append(
            f"marker id {MARKER_ID} was visible in only {fraction * 100:.0f} per cent "
            f"of sampled scan frames, under {MARKER_VISIBLE_MIN_FRACTION * 100:.0f}. "
            f"Checks 4 and 5 COULD NOT BE MEASURED. This is a failure of the "
            f"check, not a pass for the scan: place the marker where the camera "
            f"sees it from every pass, and run this again."
        )

    distances = geometry["height_m"]
    per_sample = geometry["seconds_per_sample"]
    if visible:
        print(f"  height above the desk from the marker pose; straight-line range "
              f"median {np.median(geometry['range_m']) * 100:.0f} cm for comparison")
    print(f"\n  {'band':>12} {'frames':>7} {'seconds':>8}")
    bands = []
    for low, high in BANDS_M:
        count = int(((distances >= low) & (distances < high)).sum())
        label = f"{low * 100:.0f}-{high * 100:.0f} cm" if high < 99 else f">{low * 100:.0f} cm"
        bands.append({"band_cm": [low * 100, min(high, 1.0) * 100],
                      "frames": count, "seconds": round(count * per_sample, 1)})
        print(f"  {label:>12} {count:7d} {count * per_sample:8.1f}")
    report["check4_bands"] = bands

    below = float(((distances < 0.30).sum()) * per_sample)
    report["check4_seconds_below_30cm"] = round(below, 1)
    # A lower bound: the marker cannot see the lowest frames, so the true
    # figure is higher than this one. Clearing the bar here clears it outright.
    ok4 = measurable and below >= SECONDS_BELOW_30CM_MIN
    verdict4 = "PASS" if ok4 else "COULD NOT MEASURE"
    print(f"\n4 below 30 cm        {below:6.1f} s   "
          f"[{SECONDS_BELOW_30CM_MIN:.0f} s minimum, LOWER BOUND]   {verdict4}")
    if not ok4:
        unmeasured.append(
            f"the marker-visible frames show {below:.1f} s below 30 cm, under "
            f"{SECONDS_BELOW_30CM_MIN:.0f} s. That is NOT a failure: the marker "
            f"lies flat and is invisible from the low across-the-mat views this "
            f"check is about, so the true figure is higher and unknown. On "
            f"real26/d, which passed the real viewpoint gate at 7.47 cm, this "
            f"same measurement read 5.8 s. Confirm the render band after Stage 1 "
            f"with close_range_coverage."
        )

    band = (distances >= RENDER_BAND_M[0]) & (distances < RENDER_BAND_M[1])
    azimuth = geometry["azimuth_deg"][band]
    azimuth = azimuth[np.isfinite(azimuth)]
    occupied = 0
    if len(azimuth):
        counts, _ = np.histogram(azimuth, bins=AZIMUTH_SECTORS, range=(-180, 180))
        occupied = int((counts > 0).sum())
        report["check5_sector_counts"] = counts.tolist()
    report["check5_sectors_occupied"] = occupied
    ok5 = measurable and occupied >= AZIMUTH_SECTORS
    verdict5 = "PASS" if ok5 else "COULD NOT MEASURE"
    print(f"5 azimuth sectors    {occupied:>3}/{AZIMUTH_SECTORS}       "
          f"[all {AZIMUTH_SECTORS}, LOWER BOUND]      {verdict5}")
    if not ok5:
        unmeasured.append(
            f"the 20 to 30 cm band shows {occupied} of {AZIMUTH_SECTORS} azimuth "
            f"sectors. Same limitation: only 25 per cent of frames in that band "
            f"see the marker at all, so absent sectors may simply be sectors the "
            f"marker faced away from. Walking right around the mat is still the "
            f"right thing to do."
        )

    report["hard_failures"] = hard
    report["could_not_measure"] = unmeasured
    report["passed"] = not hard and not unmeasured
    print()
    if unmeasured:
        for line in unmeasured:
            print(f"COULD NOT MEASURE: {line}")
    if hard:
        for line in hard:
            print(f"FAIL: {line}")
    print()
    print("GO" if report["passed"] else "NO-GO")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=float))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
