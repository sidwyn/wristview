"""Stage 4 · Retargeting.

In:  hand pose and object pose from Stage 3.
Out: end-effector trajectory, 6DoF plus gripper width, at a fixed control rate.

The pose mapping is a design choice, not a formula, and the build plan says
so. The rule used here, straight from the plan:

  do not put the gripper where the wrist was.

A wrist pose is not a grasp. The gripper is derived from the grasp itself:

  origin     thumb-index fingertip midpoint, pulled back along the approach
             direction, because a real jaw closes behind the fingertips
  approach   from the palm centroid out through the grasp point
  roll       the thumb-to-index axis, which is what the jaws close along
  width      thumb-index distance, clamped to the URDF's limits

Grasp onset and release come from two signals that must agree: the fingers
are close enough to be holding something, and the object is moving with the
hand. Finger distance alone fires on any pinch in mid-air. Motion coupling
alone fires when the object is pushed. Together they are reliable.
"""

from __future__ import annotations

import numpy as np

from ..backends import gripper as gripper_backend
from ..backends import hands as hand_backend
from ..geometry import (
    canonicalise_roll,
    frame_from_axes,
    orthonormalize,
    slerp_fill,
    smooth_poses,
)
from .. import reprojection
from ..mount import wrist_camera_offset
from ..grasp import detect_grasp_by_contact
from ..logging_setup import get
from ..qc import hand_velocity_gates, hand_velocity_outliers
from ..runctx import RunContext, StageRecorder, read_json, write_json

log = get(__name__)

STAGE = 4
NAME = "retarget"


def _hysteresis(signal: np.ndarray, min_frames: int) -> np.ndarray:
    """Absorb runs shorter than `min_frames` into the preceding run.

    Without this the grasp flag chatters open and closed for a few frames
    around the transition, and a chattering gripper is not trainable.

    The first and last runs are never absorbed. A short run at the end is the
    episode's terminal state, not chatter: absorbing a two-frame release at
    the end of a clip would report the gripper as still closed, which is
    exactly backwards.
    """
    out = signal.copy()
    if min_frames <= 1 or len(signal) == 0:
        return out

    # Collect run boundaries first, so the final run can be identified.
    boundaries = [0]
    for index in range(1, len(out)):
        if out[index] != out[index - 1]:
            boundaries.append(index)
    boundaries.append(len(out))

    # Skip the first run, which has nothing before it, and the last run,
    # which the clip merely ran out of time to continue.
    for run in range(1, len(boundaries) - 2):
        start, stop = boundaries[run], boundaries[run + 1]
        if stop - start < min_frames:
            out[start:stop] = out[start - 1]
    return out


def detect_grasp(
    landmarks: np.ndarray,
    valid: np.ndarray,
    object_positions: np.ndarray,
    object_valid: np.ndarray,
    fps: float,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """Return a per-frame closed or open flag, plus diagnostics."""
    count = len(landmarks)
    widths = np.full(count, np.nan)
    contact = np.zeros(count, dtype=bool)

    close_distance = float(cfg.get("close_distance_m", 0.070))
    open_distance = float(cfg.get("open_distance_m", 0.095))
    contact_distance = float(cfg.get("contact_distance_m", 0.05))
    coupling_threshold = float(cfg.get("motion_coupling_threshold", 0.6))
    min_speed = float(cfg.get("min_object_speed_m_s", 0.02))

    for index in range(count):
        if not valid[index]:
            continue
        widths[index] = hand_backend.grasp_width(landmarks[index])
        # Fingertips close to the object surface. Computed here, per frame,
        # and not inside the threshold block below: an earlier edit folded it
        # in there, where it ran once against a stale loop index.
        if object_valid[index]:
            distance = np.linalg.norm(
                hand_backend.grasp_center(landmarks[index]) - object_positions[index]
            )
            contact[index] = distance < contact_distance

    # Fixed thresholds in metres only suit one object size. A wrapped grasp on
    # a drinking glass holds the thumb and index about 5.8 cm apart, while a
    # small block closes to under 4 cm, so a single pair of numbers cannot
    # serve both and the failure is silent: the trigger simply never latches.
    #
    # So derive them from this episode's own width distribution, and fall back
    # to the configured metres when the hand barely changes shape, which is
    # what a clip with no grasp in it looks like.
    finite = np.isfinite(widths)
    threshold_source = "config"
    if bool(cfg.get("auto_threshold", True)) and int(finite.sum()) >= 10:
        low, high = np.percentile(widths[finite], [15, 85])
        if high - low > 0.015:
            close_distance = float(low + 0.35 * (high - low))
            open_distance = float(low + 0.65 * (high - low))
            threshold_source = "auto_percentile"

    # Motion coupling: is the object travelling with the hand.
    coupled = np.zeros(count, dtype=bool)
    dt = 1.0 / max(fps, 1e-6)
    for index in range(1, count):
        if not (valid[index] and valid[index - 1]):
            continue
        if not (object_valid[index] and object_valid[index - 1]):
            continue
        hand_velocity = (
            hand_backend.grasp_center(landmarks[index])
            - hand_backend.grasp_center(landmarks[index - 1])
        ) / dt
        object_velocity = (object_positions[index] - object_positions[index - 1]) / dt
        object_speed = float(np.linalg.norm(object_velocity))
        if object_speed < min_speed:
            # A stationary object says nothing either way.
            continue
        hand_speed = float(np.linalg.norm(hand_velocity))
        if hand_speed < 1e-6:
            continue
        cosine = float(np.dot(hand_velocity, object_velocity) / (hand_speed * object_speed))
        coupled[index] = cosine > coupling_threshold

    # Schmitt trigger on finger distance, so one threshold does not chatter.
    fingers_closed = np.zeros(count, dtype=bool)
    state = False
    for index in range(count):
        width = widths[index]
        if np.isnan(width):
            fingers_closed[index] = state
            continue
        if state and width > open_distance:
            state = False
        elif not state and width < close_distance:
            state = True
        fingers_closed[index] = state

    # Contact and motion coupling both need an object position. When object
    # tracking is unavailable the two signals are vacuously false, and
    # requiring them reports no grasp at all on a clip that plainly contains
    # one. Fall back to finger distance, and record that it happened: finger
    # distance alone fires on any pinch in mid-air, so a fallback episode is
    # weaker evidence than a confirmed one.
    object_available = bool(object_valid.any())
    if object_available:
        holding = fingers_closed & (contact | coupled)
        # Carrying the object counts as holding even if the fingers read wide,
        # which happens when a fingertip is occluded and its landmark drifts.
        holding = holding | (coupled & contact)
    else:
        holding = fingers_closed
    holding = _hysteresis(holding, int(cfg.get("min_state_frames", 3)))

    transitions = np.nonzero(np.diff(holding.astype(int)))[0]
    diagnostics = {
        "closed_frames": int(holding.sum()),
        "closed_fraction": round(float(holding.mean()), 4),
        "transitions": len(transitions),
        "grasp_onset_frame": int(transitions[0] + 1) if len(transitions) else None,
        "release_frame": int(transitions[-1] + 1) if len(transitions) > 1 else None,
        "contact_frames": int(contact.sum()),
        "coupled_frames": int(coupled.sum()),
        "median_width_m": (
            round(float(np.nanmedian(widths)), 4) if np.isfinite(widths).any() else None
        ),
        # Recorded so a wrong threshold is visible rather than silent.
        "threshold_source": threshold_source,
        "signal": "fingers+object" if object_available else "fingers_only",
        "object_available": object_available,
        "close_distance_m": round(close_distance, 4),
        "open_distance_m": round(open_distance, 4),
        "width_percentiles_m": (
            [round(float(v), 4) for v in np.percentile(widths[finite], [5, 25, 50, 75, 95])]
            if bool(finite.any())
            else None
        ),
    }
    return holding, diagnostics


def gripper_poses_from_hand(
    landmarks: np.ndarray, valid: np.ndarray, origin_offset_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map hand landmarks to end-effector poses. The core of Stage 4."""
    count = len(landmarks)
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    widths = np.zeros(count)

    for index in range(count):
        if not valid[index]:
            continue
        hand = landmarks[index]
        approach = hand_backend.approach_direction(hand)
        closing = hand_backend.grasp_axis(hand)
        center = hand_backend.grasp_center(hand)

        # Pull the origin back along the approach: the jaws grip behind the
        # fingertips, so putting the origin at the fingertips would place the
        # gripper a jaw-length too far forward every frame.
        origin = center - approach * origin_offset_m

        poses[index] = frame_from_axes(origin, approach, closing)
        widths[index] = hand_backend.grasp_width(hand)

    return poses, widths


def resample(
    poses: np.ndarray,
    widths: np.ndarray,
    closed: np.ndarray,
    source_fps: float,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample the trajectory to a fixed control rate.

    Rotations slerp, translations and widths interpolate linearly, and the
    binary grasp flag takes the nearest sample so it never lands between
    states.
    """
    from scipy.spatial.transform import Rotation, Slerp

    count = len(poses)
    if count < 2:
        return poses, widths, closed, np.zeros(count)

    duration = (count - 1) / source_fps
    target_count = max(2, int(np.floor(duration * target_hz)) + 1)
    source_times = np.arange(count) / source_fps
    target_times = np.arange(target_count) / target_hz
    target_times = np.clip(target_times, source_times[0], source_times[-1])

    rotations = Rotation.from_matrix(poses[:, :3, :3])
    slerp = Slerp(source_times, rotations)

    out = np.repeat(np.eye(4)[None], target_count, axis=0)
    out[:, :3, :3] = slerp(target_times).as_matrix()
    for axis in range(3):
        out[:, axis, 3] = np.interp(target_times, source_times, poses[:, axis, 3])

    out_widths = np.interp(target_times, source_times, widths)
    nearest = np.clip(np.round(target_times * source_fps).astype(int), 0, count - 1)
    out_closed = closed[nearest]
    return out, out_widths, out_closed, target_times


def _retarget_episode(
    ctx: RunContext,
    rec: StageRecorder,
    clip_id: str,
    clip: dict,
    spec: gripper_backend.GripperSpec,
    cfg: dict,
    world_up: np.ndarray | None,
) -> dict:
    """Retarget one episode."""
    out_dir = ctx.episode_dir(STAGE, clip_id)
    estimate_dir = ctx.episode_dir(3, clip_id, create=False)

    hand = np.load(estimate_dir / "hand.npz")
    landmarks = hand["landmarks_world"]
    landmarks_px = hand["landmarks_px"]
    hand_valid = hand["valid"]

    # Stage 1's refined camera, not Stage 0's prior. The prior has been wrong
    # by a factor of 1.5 on real footage, and a wrong focal moves depth, which
    # is exactly what the checks below are trying to detect.
    from ..camera import Intrinsics

    cameras_path = ctx.stage_dir(1, create=False) / "cameras.json"
    intrinsics = (
        Intrinsics.from_dict(read_json(cameras_path)["intrinsics"])
        if cameras_path.exists()
        else None
    )

    # A clip whose selected hand changed side is not a weak episode, it is a
    # different hand from one moment to the next. Nothing downstream can
    # recover that, so it is rejected rather than smoothed over.
    switches = int(hand["hand_side_switches"]) if "hand_side_switches" in hand else 0
    if switches:
        sides = [str(x) for x in hand["hand_side"]] if "hand_side" in hand else []
        seen = {x for x in sides if x}
        status = {
            "clip_id": clip_id,
            "rejected": True,
            "reject_reason": (
                f"hand identity changed {switches} times between {sorted(seen)}. "
                "The largest-box selector crossed between two hands mid-episode, "
                "so this trajectory describes neither hand."
            ),
            "hand_side_switches": switches,
        }
        write_json(out_dir / "status.json", status)
        log.error("%s REJECTED: %s", clip_id, status["reject_reason"])
        return status

    object_poses = np.load(estimate_dir / "object_pose.npy")
    object_valid = np.load(estimate_dir / "object_valid.npy")
    object_positions = object_poses[:, :3, 3]

    fps = float(clip.get("effective_fps") or clip["video_info"]["fps"] or 30.0)

    if not hand_valid.any():
        status = {
            "clip_id": clip_id,
            "rejected": True,
            "reject_reason": "no hand was detected on any frame, so nothing can be retargeted",
        }
        write_json(out_dir / "status.json", status)
        log.error("%s REJECTED: no hand detected", clip_id)
        return status

    # ---- hand plausibility ------------------------------------------------
    # A wrist cannot turn faster than a wrist can turn. Frames that claim it did
    # are tracking failures, and they have to be found before anything is built
    # on them.
    velocity = hand_velocity_outliers(landmarks, hand_valid, fps)
    velocity_gates = hand_velocity_gates(velocity)
    outlier = velocity.pop("outlier")
    log.info(
        "%s: %d of %d tracked frames above %.0f deg/s (%.1f%%), longest run %d, peak %s deg/s",
        clip_id, velocity["frames_flagged"], velocity["frames_checked"],
        900.0, velocity["flagged_fraction"] * 100, velocity["longest_run"],
        velocity["max_rate_deg_s"],
    )
    failed = [gate for gate in velocity_gates if not gate.passed]
    if failed:
        reason = "; ".join(f"{g.name} {g.value} against {g.threshold} {g.unit}" for g in failed)
        status = {
            "clip_id": clip_id,
            "rejected": True,
            "reject_reason": f"hand tracking is not physically plausible: {reason}",
            "hand_velocity": velocity,
        }
        write_json(out_dir / "status.json", status)
        log.error("%s REJECTED: %s", clip_id, reason)
        return status

    # Isolated bad frames are dropped from the measured set and filled by the
    # same interpolation that covers a missed detection. Which frames were
    # filled is recorded, because the export has to distinguish measurement
    # from fill.
    hand_measured = hand_valid & ~outlier
    hand_filled = hand_valid & outlier
    if outlier.any():
        log.info(
            "%s: filling %d flagged frame(s) from neighbours, at %s",
            clip_id, int(hand_filled.sum()), velocity["flagged_frames"],
        )
    hand_valid = hand_measured

    # ---- grasp ----------------------------------------------------------
    # Contact and carry, not finger separation. The width test scored 57.9 per
    # cent against ground truth on a handle grasp, because thumb-to-index
    # distance barely changes when an object is held by its handle.
    grasp_cfg = cfg.get("grasp", {})
    method = str(grasp_cfg.get("method", "contact"))

    # Stage 3 decides contact against the object's resting pose, which is the
    # only pose that is trustworthy at the moment contact happens. The detector
    # below instead measures fingertip distance to the *tracked* object, and on
    # session 6 that was pinned to the desk for the whole clip, so it returned 0
    # closed frames on all four clips while the operator visibly picked things
    # up. Prefer the stage that measured the right thing, and record both so the
    # disagreement stays visible.
    contact_path = estimate_dir / "contact_runs.json"
    stage3_runs = []
    if contact_path.exists():
        stage3_runs = read_json(contact_path).get("runs") or []

    if stage3_runs:
        closed = np.zeros(len(landmarks), dtype=bool)
        for start, stop in stage3_runs:
            closed[int(start):int(stop)] = True
        onsets = np.nonzero(closed)[0]
        _, detector_diagnostics = detect_grasp_by_contact(
            landmarks, hand_valid, object_positions, object_valid, fps, grasp_cfg
        )
        grasp_diagnostics = {
            "signal": "stage3_contact_runs",
            "why": (
                "contact decided in Stage 3 against the object's resting pose. "
                "The fingertip-distance detector is run alongside for comparison "
                "but not used, because it measures distance to the tracked "
                "object, which is wrong for every carried frame."
            ),
            "closed_frames": int(closed.sum()),
            "closed_fraction": round(float(closed.mean()), 4),
            "transitions": int(len(np.nonzero(np.diff(closed.astype(int)))[0])),
            "runs": [[int(a), int(b)] for a, b in stage3_runs],
            "grasp_onset_frame": int(onsets[0]) if len(onsets) else None,
            "release_frame": (
                int(onsets[-1] + 1) if len(onsets) and onsets[-1] + 1 < len(closed) else None
            ),
            "detector_for_comparison": {
                "closed_frames": detector_diagnostics.get("closed_frames"),
                "contact_gap_median_m": detector_diagnostics.get("contact_gap_median_m"),
            },
        }
        log.info(
            "%s: grasp from Stage 3 contact runs, closed on %d/%d frames (%.0f%%), "
            "runs %s. The fingertip detector would have said %s.",
            clip_id, int(closed.sum()), len(closed), closed.mean() * 100,
            grasp_diagnostics["runs"], detector_diagnostics.get("closed_frames"),
        )
    elif method == "contact":
        closed, grasp_diagnostics = detect_grasp_by_contact(
            landmarks, hand_valid, object_positions, object_valid, fps, grasp_cfg
        )
        if grasp_diagnostics["signal"] == "unavailable":
            # Falling back is recorded, never silent: a width-derived grasp is
            # weaker evidence and the export has to say so.
            log.warning(
                "%s: no object position on any frame, so contact could not be "
                "evaluated. Falling back to finger width, which is the weaker "
                "signal this replaced.", clip_id,
            )
            closed, grasp_diagnostics = detect_grasp(
                landmarks, hand_valid, object_positions, object_valid, fps, grasp_cfg
            )
            grasp_diagnostics["signal"] = "width_fallback"
            grasp_diagnostics["fallback_reason"] = "no object position on any frame"
    else:
        closed, grasp_diagnostics = detect_grasp(
            landmarks, hand_valid, object_positions, object_valid, fps, grasp_cfg
        )
    log.info(
        "%s: grasp by %s, closed on %d/%d frames (%.0f%%), %d transitions, "
        "contact frame %s, release frame %s",
        clip_id, grasp_diagnostics["signal"], grasp_diagnostics["closed_frames"],
        len(closed), grasp_diagnostics["closed_fraction"] * 100,
        grasp_diagnostics["transitions"], grasp_diagnostics["grasp_onset_frame"],
        grasp_diagnostics.get("release_frame"),
    )

    # ---- pose mapping -----------------------------------------------------
    poses, widths = gripper_poses_from_hand(
        landmarks, hand_valid, float(cfg.get("origin_offset_m", 0.02))
    )

    # Resolve the gripper's roll ambiguity before anything else touches the
    # poses. It has to happen here, on the raw per-frame poses: smoothing
    # averages a flip away into a slow tumble instead of a sharp one, so a
    # sequence that is half upside down still reads as zero flips afterwards.
    roll_report = None
    if world_up is not None:
        poses[hand_valid], roll_report = canonicalise_roll(poses[hand_valid], world_up)
        log.info(
            "%s: roll flips %d -> %d, up axis %.0f deg from world up (%d frames inverted)",
            clip_id, roll_report["flips_before"], roll_report["flips_after"],
            roll_report["median_up_angle_deg"], roll_report["frames_up_inverted"],
        )
        if roll_report["flips_after"]:
            log.warning(
                "%s: %d roll flip(s) remain inside the episode",
                clip_id, roll_report["flips_after"],
            )

    # Fill the gaps where the hand was not detected, then smooth.
    poses = slerp_fill(poses, hand_valid)
    widths = np.interp(
        np.arange(len(widths)),
        np.nonzero(hand_valid)[0],
        widths[hand_valid],
    )
    poses = smooth_poses(poses, int(cfg.get("smoothing_window", 7)))
    for index in range(len(poses)):
        poses[index, :3, :3] = orthonormalize(poses[index, :3, :3])

    widths = spec.clamp_width(widths)
    # A closed gripper reports its true opening on the object, not the noisy
    # fingertip distance, so clamp the closed frames to the minimum reached.
    if closed.any():
        held_width = float(np.median(widths[closed]))
        widths = np.where(closed, held_width, widths)

    # ---- resample ---------------------------------------------------------
    target_hz = float(cfg.get("control_rate_hz", 15.0))
    resampled_poses, resampled_widths, resampled_closed, times = resample(
        poses, widths, closed, fps, target_hz
    )

    # ---- reprojection gate, continued from Stage 3 ------------------------
    #
    # Stage 3 checks the hand and the object. The two quantities this stage
    # invents are the end effector and the wrist camera, and both are checked
    # against what the source frame shows, for the same reason: a 3D quantity
    # nobody projects back is a quantity nobody has checked.
    poses_path = ctx.stage_dir(2, create=False) / clip_id / "camera_poses.npy"
    camera_poses = np.load(poses_path) if poses_path.exists() else None
    wrist_cfg = (ctx.config.get("render", {}) or {}).get("wrist_camera", {}) or {}
    standoff_m = wrist_cfg.get("standoff_m")
    grasp_offset_m = float(wrist_cfg.get("grasp_offset_m", 0.02))

    qc_reports = [] if (intrinsics is None or camera_poses is None) else [
        reprojection.effector_check(
            poses[:, :3, 3], hand_valid, landmarks, hand_valid, grasp_offset_m,
        )
    ]
    if intrinsics is None or camera_poses is None:
        log.warning(
            "%s: no refined camera or no localized poses, so this clip goes out "
            "unchecked", clip_id,
        )
    if standoff_m:
        # Use Stage 5's own mount geometry rather than a second copy of it.
        # Computing the eye independently is how the first version of this
        # check reported 0.15 m against a correct 0.25 m: it measured from the
        # wrist, while the mount is specified from the fingertips.
        offset = wrist_camera_offset(wrist_cfg)
        eye_local = np.asarray(offset[:3, 3], dtype=np.float64)
        origins = np.array([
            pose[:3, :3] @ eye_local + pose[:3, 3] for pose in poses
        ])
        fingertip_local = np.array([0.0, 0.0, grasp_offset_m])
        grasp_points = np.array([
            pose[:3, :3] @ fingertip_local + pose[:3, 3] for pose in poses
        ])
        qc_reports.append(
            reprojection.standoff_check(
                origins, grasp_points, hand_valid, float(standoff_m)
            )
        )
    else:
        log.warning(
            "%s: render.wrist_camera.standoff_m is unset, so the wrist camera "
            "cannot be checked. real06b shipped this way and nothing said so.",
            clip_id,
        )

    for report in qc_reports:
        log.info("%s: %s -> %s", clip_id, report["check"],
                 {k: v for k, v in report.items()
                  if k in ("median_px", "p90_px", "median_m", "frames", "passed")})
    qc_failures = reprojection.gate(qc_reports)
    for failure in qc_failures:
        log.error("%s: REPROJECTION GATE FAILED: %s", clip_id, failure)
    write_json(out_dir / "qc.json", {
        "clip_id": clip_id,
        "reprojection": qc_reports,
        "passed": not qc_failures,
        "failures": qc_failures,
    })

    trajectory_path = out_dir / "ee_trajectory.npy"
    np.save(trajectory_path, resampled_poses)
    np.savez_compressed(
        out_dir / "ee_trajectory.npz",
        poses=resampled_poses,
        width_m=resampled_widths,
        closed=resampled_closed,
        timestamps_s=times,
        source_fps=fps,
        control_rate_hz=target_hz,
        # Kept at video rate too, because Stage 5 renders per video frame.
        poses_video_rate=poses,
        width_video_rate=widths,
        closed_video_rate=closed,
        hand_valid=hand_valid,
        # Per frame, what the trajectory rests on. `hand_measured` is observed,
        # `hand_filled` is interpolated across a frame the velocity gate
        # rejected, and neither means the hand was never detected.
        hand_measured=hand_measured,
        hand_filled=hand_filled,
    )

    _plot_trajectory(out_dir / "trajectory.png", poses, closed, clip_id)

    speeds = np.linalg.norm(np.diff(resampled_poses[:, :3, 3], axis=0), axis=1) * target_hz
    status = {
        "clip_id": clip_id,
        "rejected": False,
        "roll": roll_report,
        "hand_velocity": velocity,
        "frames_filled_velocity": int(hand_filled.sum()),
        "source_fps": round(fps, 3),
        "control_rate_hz": target_hz,
        "frames_video_rate": len(poses),
        "frames_control_rate": len(resampled_poses),
        "duration_s": round(float(times[-1]) if len(times) else 0.0, 3),
        "grasp": grasp_diagnostics,
        "gripper": spec.to_dict(),
        "width_range_m": [round(float(resampled_widths.min()), 4),
                          round(float(resampled_widths.max()), 4)],
        "ee_speed_median_m_s": round(float(np.median(speeds)), 4) if len(speeds) else 0.0,
        "ee_speed_max_m_s": round(float(speeds.max()), 4) if len(speeds) else 0.0,
        "path_length_m": round(float(np.sum(np.linalg.norm(
            np.diff(resampled_poses[:, :3, 3], axis=0), axis=1))), 4),
        "trajectory": ctx.rel(trajectory_path),
    }

    # A success-labeled episode where the gripper never closed is not a grasp.
    # The build plan lists this as a Stage 7 QC check; catching it here costs
    # nothing and stops a bad episode reaching the renderer.
    if grasp_diagnostics["closed_frames"] == 0:
        status["warning"] = (
            "the gripper never closed. Either the grasp was not detected or the "
            "episode contains no grasp."
        )
        log.warning("%s: %s", clip_id, status["warning"])

    write_json(out_dir / "status.json", status)
    log.info(
        "%s: %d control samples over %.1fs, path %.3f m, width %.3f to %.3f m",
        clip_id, status["frames_control_rate"], status["duration_s"],
        status["path_length_m"], status["width_range_m"][0], status["width_range_m"][1],
    )
    return status


def _plot_trajectory(path, poses: np.ndarray, closed: np.ndarray, title: str) -> None:
    """Plot the gripper path in 3D. The M4 check from the milestone table."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    positions = poses[:, :3, 3]
    figure = plt.figure(figsize=(11, 5))

    axis3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="0.6", linewidth=1)
    axis3d.scatter(
        positions[closed, 0], positions[closed, 1], positions[closed, 2],
        c="crimson", s=9, label="closed",
    )
    axis3d.scatter(
        positions[~closed, 0], positions[~closed, 1], positions[~closed, 2],
        c="steelblue", s=5, label="open",
    )
    # Draw the approach axis every so often, so orientation is visible.
    for index in range(0, len(poses), max(1, len(poses) // 18)):
        origin = positions[index]
        approach = poses[index, :3, 2] * 0.035
        axis3d.plot(
            [origin[0], origin[0] + approach[0]],
            [origin[1], origin[1] + approach[1]],
            [origin[2], origin[2] + approach[2]],
            color="darkorange", linewidth=1.1,
        )
    axis3d.set_xlabel("x (m)")
    axis3d.set_ylabel("y (m)")
    axis3d.set_zlabel("z (m)")
    axis3d.set_title(f"{title}: end-effector path")
    axis3d.legend(loc="upper left", fontsize=8)

    axis2d = figure.add_subplot(1, 2, 2)
    axis2d.plot(positions[:, 2], color="seagreen", label="height z (m)")
    axis2d.fill_between(
        np.arange(len(closed)), 0, closed.astype(float) * positions[:, 2].max(),
        color="crimson", alpha=0.18, label="gripper closed",
    )
    axis2d.set_xlabel("frame")
    axis2d.set_ylabel("height (m)")
    axis2d.set_title("height against grasp state")
    axis2d.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def run(ctx: RunContext) -> dict:
    """Run Stage 4 over the run directory."""
    rec = StageRecorder(ctx, STAGE, NAME)
    cfg = ctx.config.section("retarget")

    try:
        ingest_dir = ctx.stage_dir(0, create=False)
        manifest = read_json(ingest_dir / "manifest.json")
        estimate_summary = read_json(ctx.stage_dir(3, create=False) / "summary.json")
        rec.meta.inputs = {
            "estimate_summary": ctx.rel(ctx.stage_dir(3, create=False) / "summary.json")
        }

        # World up comes from the scene, not from a convention: COLMAP's world
        # orientation is arbitrary. Stage 1 reads it off the marker plane.
        # A run with no marker has no scale.json at all, so this is a missing
        # capability rather than an error.
        scale_path = ctx.stage_dir(1, create=False) / "scale.json"
        world_up = None
        if scale_path.exists():
            world_up = read_json(scale_path).get("diagnostics", {}).get("world_up")
        if world_up is None:
            log.warning(
                "no world_up in Stage 1 scale.json, so the gripper roll cannot be "
                "canonicalised and the wrist camera may be inverted"
            )
        else:
            world_up = np.asarray(world_up, dtype=np.float64)
            log.info("world up %s, from the marker plane", np.round(world_up, 4).tolist())

        spec = gripper_backend.load(cfg)
        rec.backend("gripper", spec.source)
        log.info(
            "gripper: max opening %.3f m, finger %.3f m, from %s",
            spec.max_width_m, spec.finger_length_m, spec.source,
        )
        if spec.source == "config":
            urdf_path = gripper_backend.write_default_urdf(
                ctx.stage_dir(STAGE) / "gripper.urdf", spec
            )
            rec.output("gripper_urdf", urdf_path)

        statuses = {}
        for clip_id in estimate_summary:
            log.info("--- Stage 4: %s ---", clip_id)
            with rec.timed(f"retarget.{clip_id}"):
                statuses[clip_id] = _retarget_episode(
                    ctx, rec, clip_id, manifest["clips"][clip_id], spec, cfg, world_up
                )
            if not statuses[clip_id].get("rejected"):
                rec.output(
                    f"{clip_id}_trajectory",
                    ctx.episode_dir(STAGE, clip_id) / "ee_trajectory.npy",
                )
                rec.output(
                    f"{clip_id}_plot", ctx.episode_dir(STAGE, clip_id) / "trajectory.png"
                )

        summary_path = write_json(ctx.stage_dir(STAGE) / "summary.json", statuses)
        rec.output("summary", summary_path)
        rec.metric("episodes", len(statuses))
        rec.metric("accepted", sum(1 for s in statuses.values() if not s.get("rejected")))
        rec.metric(
            "roll_flips_inside_episode",
            {
                k: (v.get("roll") or {}).get("flips_after")
                for k, v in statuses.items()
                if not v.get("rejected")
            },
        )
        rec.metric(
            "median_up_angle_deg",
            {
                k: (v.get("roll") or {}).get("median_up_angle_deg")
                for k, v in statuses.items()
                if not v.get("rejected")
            },
        )
        rec.metric(
            "grasp_closed_fraction",
            {
                k: v.get("grasp", {}).get("closed_fraction")
                for k, v in statuses.items()
                if not v.get("rejected")
            },
        )

        accepted = [k for k, v in statuses.items() if not v.get("rejected")]
        rec.write("ok" if accepted else "failed")
        return statuses

    except Exception as exc:
        rec.note(f"{type(exc).__name__}: {exc}")
        rec.write("failed")
        raise


def load_summary(ctx: RunContext) -> dict:
    return read_json(ctx.stage_dir(STAGE, create=False) / "summary.json")
