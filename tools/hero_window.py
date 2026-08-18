"""Find the one window in a session that survives being shown as the website asset.

A wrist render is convincing for a short stretch and no longer. The object is
composited where the scan left it, so it is right up to the moment the operator
picks it up, and wrong for the whole of the transport after that. On real04 the
cube leaves its resting position by 12.4 cm on demo_0, and every frame past that
shows a cube sitting on a desk nobody is touching. So the window ends about one
second past the grasp, and not later.

The grasp comes from the object's own displacement, not from the contact
detector in `wristview.grasp`. On real04 the contact detector reported 46 closed
frames on demo_0, 4 on demo_1 and 0 on demo_3, while all three clips visibly
pick the cube up, so its onset is not usable for cutting. Displacement is
unambiguous over the same three clips: the cube stays within 0.92 cm of its
resting position for the whole approach, then leaves by 7.7 to 12.4 cm.

Displacement is measured across the work surface rather than in full 3D. The
operator slides the cube along the desk, so the lateral component carries 7.7 to
12.4 cm of the motion while the vertical component never exceeds 4 cm. Removing
the vertical component also removes the axis where the pose estimate is
weakest.

    python -m tools.hero_window --run <run> --videos <dir of clean mp4s> --out hero.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.logging_setup import get, setup  # noqa: E402
from wristview.runctx import read_json  # noqa: E402
from wristview.videoio import ffmpeg_binary  # noqa: E402

log = get(__name__)

# render_wrist_videos lays out each frame as <panel><divider><panel>, and fills
# the divider with grey 40. The divider width is the only layout constant the
# renderer does not write anywhere, so it is repeated here.
PANEL_DIVIDER_PX = 4

# The wrist panel is cleared to RGB (0.04, 0.04, 0.05) before anything is
# composited into it. That is BGR (12, 10, 10) once quantised, measured on a
# lossless frame of real04 demo_3 where 181,195 of 307,200 panel pixels sit at
# exactly that value. Pixels still at it are scene the scan never modelled.
BACKGROUND_BGR = (12, 10, 10)

# H.264 moves the background off its exact value by a few counts. At tolerance 6
# the measured background fraction of the same frame reads 0.5835 from the mp4
# against 0.5899 from the source png, a 0.6 point disagreement. Tolerance 2
# costs four times that, and tolerance 14 starts eating real dark geometry.
BACKGROUND_TOL = 6

# A frame counts as near black when its wrist panel averages below one eighth of
# full scale. On real04 that separates cleanly: demo_0 bottoms out at mean gray
# 65 and demo_1 at 50, while demo_3 drops to 24 with a tenth percentile of 28.
NEAR_BLACK_GRAY = 32.0

# Coverage leads because a hole in the reconstruction is the defect a viewer
# sees first. Sharpness comes second because the whole render is soft anyway at
# the 0.25 m standoff the scan supports. Near-black frames are counted last
# because they are already reflected in coverage, and double counting them would
# let one dark clip lose on a defect it is only charged for once.
COVERAGE_WEIGHT = 0.5
SHARPNESS_WEIGHT = 0.3
DARKNESS_WEIGHT = 0.2

# verify_object_rest.py calls the object moved at 2 cm, and this agrees with it
# on purpose so the two tools cannot disagree about when a clip stops resting.
MOVE_THRESHOLD_M = 0.02

# The resting position is the median of this many leading valid frames. Copied
# from verify_object_rest.py, which uses 15 for the same reason: it is long
# enough to average out tracker jitter and short enough to end before any reach.
SETTLE_FRAMES = 15


@dataclass(frozen=True)
class GraspEvent:
    """When the object leaves its resting place, and how safe that call is."""

    frame: int
    threshold_m: float
    rest_noise_p95_m: float
    peak_displacement_m: float

    @property
    def margin(self) -> float:
        """How many times the resting noise the threshold sits above."""
        return self.threshold_m / max(self.rest_noise_p95_m, 1e-9)


@dataclass(frozen=True)
class Window:
    """A half-open frame range `[start, end)` of one rendered clip."""

    clip: str
    start: int
    end: int
    fps: float
    grasp_frame: int
    reach_frame: int | None
    at_target_length: bool

    @property
    def frames(self) -> int:
        return self.end - self.start

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps

    @property
    def start_s(self) -> float:
        return self.start / self.fps


@dataclass(frozen=True)
class ClipScore:
    """What the wrist panel looks like over one window, measured on the render."""

    coverage_mean: float
    coverage_min: float
    sharpness_mean: float
    near_black_frames: int
    frames_measured: int

    @property
    def near_black_fraction(self) -> float:
        return self.near_black_frames / max(self.frames_measured, 1)


def surface_normal(run_root: Path) -> np.ndarray | None:
    """The work surface normal, so displacement can be split across it and up it.

    Two runs write it under different keys. Newer runs fit the desk explicitly
    and store `desk_plane`; real04 predates that and only has the ArUco world up
    vector. Either serves, because both name the axis the desk is normal to.
    """
    scale_path = run_root / "01_scene" / "scale.json"
    if not scale_path.exists():
        return None
    diagnostics = read_json(scale_path).get("diagnostics") or {}

    plane = diagnostics.get("desk_plane") or {}
    raw = plane.get("normal") or diagnostics.get("world_up")
    if raw is None:
        return None

    normal = np.asarray(raw, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(normal))
    if length < 1e-9:
        return None
    return normal / length


def lateral_displacement(
    poses: np.ndarray,
    valid: np.ndarray,
    normal: np.ndarray | None,
    settle_frames: int = SETTLE_FRAMES,
) -> np.ndarray:
    """Distance of the object from its resting place, measured across the surface.

    Returns one distance in metres per frame. Frames without a valid pose return
    NaN, so a caller cannot mistake a dropped track for a stationary object.

    Passing `normal` as None falls back to full 3D distance. That is the honest
    result when no surface has been fitted, and it costs little here because the
    motion is almost entirely lateral anyway.
    """
    poses = np.asarray(poses, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected an (N, 4, 4) pose array, got {poses.shape}")
    if len(poses) != len(valid):
        raise ValueError(f"{len(poses)} poses against {len(valid)} validity flags")

    centres = poses[:, :3, 3]
    settled = centres[valid][:settle_frames]
    if len(settled) == 0:
        return np.full(len(poses), np.nan)
    rest = np.median(settled, axis=0)

    offset = centres - rest
    if normal is not None:
        # Project out the component along the surface normal. What remains is
        # travel across the desk, which is the motion a pick actually makes.
        offset = offset - np.outer(offset @ normal, normal)

    distance = np.linalg.norm(offset, axis=1)
    distance[~valid] = np.nan
    return distance


def find_grasp_frame(
    displacement: np.ndarray,
    threshold_m: float = MOVE_THRESHOLD_M,
    min_hold_frames: int = 3,
) -> int | None:
    """First frame where the object leaves rest and stays gone.

    The hold requirement is what stops a single bad pose from cutting the clip
    early. One outlier frame is common in this tracker, and a run of three is
    not, so the crossing has to persist before it counts.

    Returns None when the object never travels far enough, which happens on a
    clip where the operator never picks anything up.
    """
    displacement = np.asarray(displacement, dtype=np.float64)
    if min_hold_frames < 1:
        raise ValueError(f"min_hold_frames must be 1 or more, got {min_hold_frames}")

    # NaN compares false, so a dropped track breaks the run rather than
    # extending it. That is the conservative reading: an unobserved frame is not
    # evidence the object moved.
    moved = np.isfinite(displacement) & (displacement > threshold_m)

    run = 0
    for index, flag in enumerate(moved):
        run = run + 1 if flag else 0
        if run >= min_hold_frames:
            return index - run + 1
    return None


def describe_grasp(
    displacement: np.ndarray,
    grasp_frame: int,
    threshold_m: float = MOVE_THRESHOLD_M,
    quiet_margin_frames: int = 20,
) -> GraspEvent:
    """Report the grasp with the noise floor it was called against.

    The margin matters more than the frame number. A threshold that sits just
    above the tracker's resting jitter is a coin toss, and the only way to see
    that is to print both.
    """
    finite = np.isfinite(displacement)
    quiet_end = max(grasp_frame - quiet_margin_frames, 1)
    quiet = displacement[:quiet_end][finite[:quiet_end]]
    noise = float(np.percentile(quiet, 95)) if len(quiet) else float("nan")
    peak = float(np.nanmax(displacement)) if finite.any() else float("nan")
    return GraspEvent(
        frame=grasp_frame,
        threshold_m=threshold_m,
        rest_noise_p95_m=noise,
        peak_displacement_m=peak,
    )


def find_reach_start(
    positions: np.ndarray,
    measured: np.ndarray,
    fps: float,
    search_lo: int,
    search_hi: int,
    quiet_speed_mps: float = 0.03,
    quiet_frames: int = 6,
    min_measured_fraction: float = 0.6,
) -> int | None:
    """Last frame before the grasp where the hand was still holding still.

    Walks backwards from `search_hi`. The reach starts where the hand stops
    being quiet, so the first sustained quiet stretch found going back is the
    frame to cut on.

    Returns None when the hand track is too sparse to trust. On real04 that is
    the answer for all three clips. Hand detection runs 71.6 to 80.8 per cent
    overall, but it is front loaded against the reach: demo_0's first measured
    hand frame is 77, which leaves 25 of the 81 frames in its search band
    measured, against the 60 per cent this asks for. A held pose reports zero
    speed, so without the check the cut would land on missing data and call it
    a still hand.
    """
    positions = np.asarray(positions, dtype=np.float64)
    measured = np.asarray(measured, dtype=bool)
    search_lo = max(0, search_lo)
    search_hi = min(len(positions) - 1, search_hi)
    if search_hi - search_lo < quiet_frames:
        return None

    band = measured[search_lo : search_hi + 1]
    if band.mean() < min_measured_fraction:
        return None

    step = np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps
    speed = np.concatenate([[0.0], step])
    # Only measured frames carry a speed. Filled frames repeat the last pose, so
    # their zero speed says nothing about the hand.
    speed[~measured] = np.nan

    run = 0
    for index in range(search_hi, search_lo - 1, -1):
        value = speed[index]
        run = run + 1 if np.isfinite(value) and value < quiet_speed_mps else 0
        if run >= quiet_frames:
            return index + run - 1
    return None


def choose_window(
    clip: str,
    grasp_frame: int,
    frame_count: int,
    fps: float,
    tail_s: float = 1.0,
    target_s: float = 6.0,
    min_s: float = 6.0,
    max_s: float = 10.0,
    reach_frame: int | None = None,
) -> Window:
    """Frame range to cut, ending about `tail_s` after the grasp.

    The end is the hard constraint and the length is the soft one. Past the
    tail the composited object is provably in the wrong place, so the window
    never runs long to reach a target duration. It shortens instead, and reports
    that it did.

    When no reach onset is supplied the window falls back to `target_s`, which
    defaults to the low end of the band. Extra lead-in before the hand moves is
    dead air on a website asset, so the shorter default is the better guess.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if min_s > max_s:
        raise ValueError(f"min_s {min_s} exceeds max_s {max_s}")

    tail = int(round(tail_s * fps))
    min_frames = int(round(min_s * fps))
    max_frames = int(round(max_s * fps))

    # Inclusive of the grasp frame itself, then `tail` frames of carry.
    end = min(frame_count, grasp_frame + tail + 1)
    preferred = reach_frame if reach_frame is not None else end - int(round(target_s * fps))

    start = min(max(preferred, end - max_frames), end - min_frames)
    start = max(0, start)

    frames = end - start
    return Window(
        clip=clip,
        start=start,
        end=end,
        fps=fps,
        grasp_frame=grasp_frame,
        reach_frame=reach_frame,
        at_target_length=min_frames <= frames <= max_frames,
    )


def reach_search_bounds(
    grasp_frame: int, frame_count: int, fps: float,
    tail_s: float = 1.0, min_s: float = 6.0, max_s: float = 10.0,
) -> tuple[int, int]:
    """Frames a reach onset may fall in, given the end and the length band.

    Searching outside the band is wasted work: `choose_window` clamps the start
    back into it, so an onset found beyond it cannot change the cut.
    """
    end = min(frame_count, grasp_frame + int(round(tail_s * fps)) + 1)
    lo = max(0, end - int(round(max_s * fps)))
    hi = max(lo, end - int(round(min_s * fps)))
    return lo, hi


def right_panel(frame: np.ndarray, divider_px: int = PANEL_DIVIDER_PX) -> np.ndarray:
    """The synthesised wrist view, which is the half being judged.

    The source half is real footage and always scores well, so including it
    would flatter every clip equally and separate none of them.
    """
    width = (frame.shape[1] - divider_px) // 2
    return frame[:, frame.shape[1] - width :]


def score_window(
    video: Path,
    window: Window,
    near_black_gray: float = NEAR_BLACK_GRAY,
    background_tol: int = BACKGROUND_TOL,
) -> ClipScore:
    """Measure the wrist panel over one window, on the render a viewer will see.

    Reads the encoded mp4 rather than the render report. The report's coverage
    counts depth samples in the scene layer only, so it misses the gripper proxy
    and it ignores what the encoder did. This counts pixels in the delivered
    file instead.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video}")

    background = np.array(BACKGROUND_BGR, dtype=np.int16)
    coverages: list[float] = []
    sharpness: list[float] = []
    near_black = 0
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, window.start)
        for _ in range(window.frames):
            ok, frame = capture.read()
            if not ok:
                break
            panel = right_panel(frame)
            gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
            empty = np.abs(panel.astype(np.int16) - background).max(axis=2) <= background_tol
            coverages.append(1.0 - float(empty.mean()))
            sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            if float(gray.mean()) < near_black_gray:
                near_black += 1
    finally:
        capture.release()

    if not coverages:
        raise RuntimeError(f"read no frames from {video} at frame {window.start}")

    return ClipScore(
        coverage_mean=float(np.mean(coverages)),
        coverage_min=float(np.min(coverages)),
        sharpness_mean=float(np.mean(sharpness)),
        near_black_frames=near_black,
        frames_measured=len(coverages),
    )


def rank_clips(scores: Mapping[str, ClipScore]) -> list[tuple[str, float]]:
    """Rank clips best first, on coverage, sharpness and freedom from dark frames.

    Sharpness is normalised against the best clip in the group because it has no
    absolute scale. Laplacian variance depends on resolution and on how much
    texture the scan happened to capture, so only the comparison means anything.
    """
    if not scores:
        return []
    peak = max(score.sharpness_mean for score in scores.values())
    peak = peak if peak > 0 else 1.0

    ranked = [
        (
            clip,
            COVERAGE_WEIGHT * score.coverage_mean
            + SHARPNESS_WEIGHT * (score.sharpness_mean / peak)
            + DARKNESS_WEIGHT * (1.0 - score.near_black_fraction),
        )
        for clip, score in scores.items()
    ]
    # Ties break on clip name, so the same run always yields the same asset.
    ranked.sort(key=lambda row: (-row[1], row[0]))
    return ranked


def export_window(source: Path, window: Window, out_path: Path, crf: int = 18) -> Path:
    """Cut the window out with ffmpeg, as H.264 with no audio and a movable index.

    Cuts on the decoded frame rather than by timestamp. The renders are constant
    rate, but seeking by time still lands on the nearest keyframe, and a hero
    clip that starts a keyframe early shows the hand already moving.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_binary(),
        "-loglevel", "error",
        "-y",
        "-i", str(source),
        "-vf", f"trim=start_frame={window.start}:end_frame={window.end},setpts=PTS-STARTPTS",
        "-an",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(command, check=True)
    return out_path


def video_fps(video: Path, fallback: float) -> float:
    """Playback rate of the render, which is what sets the window's real length.

    Frames were extracted at an uneven rate. real04 demo_0 asked for 20 fps and
    got an effective 18.93 after 24 blurred frames were dropped, and its frame
    times contain gaps up to 0.4 s. None of that survives into the render, which
    writes every frame at a constant rate, so window length follows the render's
    rate and not the capture's.
    """
    capture = cv2.VideoCapture(str(video))
    try:
        rate = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return rate if rate > 0 else fallback


def frame_count(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def clip_reach_start(
    run_root: Path, clip: str, fps: float, bounds: tuple[int, int]
) -> int | None:
    """Reach onset from the retargeted end effector, or None when unavailable.

    The positions are the video-rate track, so they are indexed the same way as
    the rendered frames and the object poses. `fps` is the render's rate for the
    same reason.
    """
    traj_path = run_root / "04_retarget" / clip / "ee_trajectory.npz"
    if not traj_path.exists():
        return None
    traj = np.load(traj_path)
    if "poses_video_rate" not in traj or "hand_measured" not in traj:
        return None
    return find_reach_start(
        positions=traj["poses_video_rate"][:, :3, 3],
        measured=traj["hand_measured"],
        fps=fps,
        search_lo=bounds[0],
        search_hi=bounds[1],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--videos", required=True, help="directory of rendered clean mp4s")
    parser.add_argument("--out", required=True, help="output mp4")
    parser.add_argument("--clips", nargs="*", default=None)
    parser.add_argument("--move-threshold-m", type=float, default=MOVE_THRESHOLD_M)
    parser.add_argument("--tail-seconds", type=float, default=1.0)
    parser.add_argument("--target-seconds", type=float, default=6.0)
    parser.add_argument("--min-seconds", type=float, default=6.0)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=20.0, help="used only if the mp4 omits it")
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()

    setup(None, verbose=False)
    run_root = Path(args.run).resolve()
    videos_dir = Path(args.videos).resolve()
    out_path = Path(args.out).resolve()

    normal = surface_normal(run_root)
    log.info("surface normal: %s", "none, falling back to 3D distance"
             if normal is None else np.round(normal, 4).tolist())

    candidates = args.clips or sorted(p.stem for p in videos_dir.glob("*.mp4"))
    if not candidates:
        log.error("no mp4 files in %s", videos_dir)
        return 1

    windows: dict[str, Window] = {}
    grasps: dict[str, GraspEvent] = {}
    scores: dict[str, ClipScore] = {}

    for clip in candidates:
        video = videos_dir / f"{clip}.mp4"
        pose_path = run_root / "03_estimate" / clip / "object_pose.npy"
        if not video.exists() or not pose_path.exists():
            log.warning("%s: skipped, missing %s", clip,
                        video if not video.exists() else pose_path)
            continue

        poses = np.load(pose_path)
        valid = np.load(run_root / "03_estimate" / clip / "object_valid.npy").astype(bool)
        displacement = lateral_displacement(poses, valid, normal)

        grasp_frame = find_grasp_frame(displacement, args.move_threshold_m)
        if grasp_frame is None:
            log.warning("%s: skipped, the object never travelled %.0f cm from rest",
                        clip, args.move_threshold_m * 100)
            continue

        rendered = frame_count(video)
        if rendered != len(poses):
            # The renderer stops at min(poses, frames), so a shorter video is
            # expected. A longer one means the two came from different runs.
            log.warning("%s: %d rendered frames against %d object poses",
                        clip, rendered, len(poses))

        fps = video_fps(video, args.fps)
        bounds = reach_search_bounds(
            grasp_frame, rendered, fps,
            args.tail_seconds, args.min_seconds, args.max_seconds,
        )
        reach_frame = clip_reach_start(run_root, clip, fps, bounds)
        if reach_frame is None:
            log.info("%s: no reach onset in frames %d to %d, using the %.0fs default",
                     clip, bounds[0], bounds[1], args.target_seconds)

        window = choose_window(
            clip=clip,
            grasp_frame=grasp_frame,
            frame_count=rendered,
            fps=fps,
            tail_s=args.tail_seconds,
            target_s=args.target_seconds,
            min_s=args.min_seconds,
            max_s=args.max_seconds,
            reach_frame=reach_frame,
        )
        windows[clip] = window
        grasps[clip] = describe_grasp(displacement, grasp_frame, args.move_threshold_m)
        scores[clip] = score_window(video, window)

    if not scores:
        log.error("no clip yielded a window")
        return 1

    ranked = rank_clips(scores)

    log.info("")
    log.info("%-9s %6s %8s %9s %9s %10s %11s %7s", "clip", "grasp", "window",
             "length", "coverage", "sharpness", "near-black", "score")
    for clip, total in ranked:
        window, score = windows[clip], scores[clip]
        log.info("%-9s %6d %8s %8.2fs %8.1f%% %10.0f %11d %7.4f",
                 clip, window.grasp_frame, f"{window.start}-{window.end}",
                 window.duration_s, score.coverage_mean * 100,
                 score.sharpness_mean, score.near_black_frames, total)

    log.info("")
    for clip, _ in ranked:
        grasp = grasps[clip]
        log.info("%-9s grasp at %d: object left rest by %.0f cm against a %.1f cm "
                 "resting noise floor (%.1fx margin), travelling %.1f cm in all",
                 clip, grasp.frame, grasp.threshold_m * 100,
                 grasp.rest_noise_p95_m * 100, grasp.margin,
                 grasp.peak_displacement_m * 100)
        if not windows[clip].at_target_length:
            log.warning("%-9s window is %.2fs, outside the %.0f to %.0fs target",
                        clip, windows[clip].duration_s, args.min_seconds, args.max_seconds)

    best, best_score = ranked[0]
    window = windows[best]
    log.info("")
    log.info("hero: %s frames %d to %d, %.2f s, score %.4f",
             best, window.start, window.end, window.duration_s, best_score)
    export_window(videos_dir / f"{best}.mp4", window, out_path, crf=args.crf)
    log.info("wrote %s", out_path)

    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "run": str(run_root),
        "hero": best,
        "window": {
            "start_frame": window.start,
            "end_frame": window.end,
            "fps": window.fps,
            "duration_s": round(window.duration_s, 3),
            "grasp_frame": window.grasp_frame,
            "reach_frame": window.reach_frame,
            "at_target_length": window.at_target_length,
        },
        "ranking": [
            {
                "clip": clip,
                "score": round(total, 4),
                "grasp_frame": grasps[clip].frame,
                "window": [windows[clip].start, windows[clip].end],
                "duration_s": round(windows[clip].duration_s, 3),
                "coverage_mean": round(scores[clip].coverage_mean, 4),
                "coverage_min": round(scores[clip].coverage_min, 4),
                "sharpness_mean": round(scores[clip].sharpness_mean, 1),
                "near_black_frames": scores[clip].near_black_frames,
                "rest_noise_p95_m": round(grasps[clip].rest_noise_p95_m, 5),
                "peak_displacement_m": round(grasps[clip].peak_displacement_m, 4),
            }
            for clip, total in ranked
        ],
    }, indent=2))
    log.info("wrote %s", sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
