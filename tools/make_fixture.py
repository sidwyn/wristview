"""Render a synthetic scan clip and demo clips, with ground truth.

Usage:
    python -m tools.make_fixture --out fixtures/room01

Writes:
    scan.mp4              slow orbit of the room, no hands, no people
    demo_0.mp4 ...        pick-and-place clips, hand in frame
    arkit_scan.json       camera trajectory in metres, ARKit's format
    groundtruth.json      camera poses, hand landmarks, object poses, intrinsics

The ground truth is the point. Without it "the pipeline ran" is all you can
say. With it you can say how far off it was.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.synthetic_render import render_frame, transform_quads  # noqa: E402
from tools.synthetic_scene import (  # noqa: E402
    HAND_BONES,
    Scene,
    bone_radius,
    build_room,
    canonical_hand,
    curl_hand,
    hand_to_world,
)

TABLE_TOP_Z = 0.75
OBJECT_HALF = np.array([0.0225, 0.0225, 0.030])
OBJECT_START = np.array([0.10, 0.02, TABLE_TOP_Z + OBJECT_HALF[2]])
OBJECT_GOAL = np.array([-0.22, -0.10, TABLE_TOP_Z + OBJECT_HALF[2]])


def look_at_pose(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world pose, OpenCV axes: x right, y down, z forward."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, dtype=np.float64)
    if abs(float(forward @ up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)
    pose[:3, 3] = eye
    return pose


def scan_trajectory(
    count: int, rng: np.random.Generator, close_fraction: float = 0.45
) -> list[np.ndarray]:
    """A wide orbit followed by a close pass over the working area.

    Two phases, and the second one is not optional. A scan that only orbits
    wide cannot localize demo clips shot close: the first real capture had a
    1.55 m orbit against demos shot at 0.6 m, roughly a sixfold scale
    difference, and the best demo-to-scan feature match gave 126 features
    where scan-to-scan neighbours gave over 700. Stage 2 could not recover a
    pose from that.

    So the second phase comes in to the demo camera's own distance and height
    and covers the working area from there. That is also the advice given to
    the operator in `wristview-videos/README.md`.

    Height varies within each phase because a single-height orbit leaves the
    vertical scale poorly constrained.
    """
    poses = []
    target_base = np.array([0.0, 0.0, TABLE_TOP_Z])
    wide_count = int(round(count * (1.0 - close_fraction)))

    for i in range(count):
        if i < wide_count:
            # Wide orbit: establishes the room and the table's surroundings.
            phase = i / max(wide_count, 1)
            angle = phase * 2.0 * np.pi * 1.25
            radius = 1.55 + 0.28 * np.sin(phase * 2.0 * np.pi * 2.0)
            height = 1.18 + 0.42 * np.sin(phase * 2.0 * np.pi * 1.5 + 0.6)
            target = target_base
        else:
            # Close pass: the demo camera sits about 0.6 m out at 1.38 m high,
            # so cover that band and sweep the working area under it.
            phase = (i - wide_count) / max(count - wide_count, 1)
            angle = -np.pi / 2 + (phase - 0.5) * 2.4
            radius = 0.62 + 0.16 * np.sin(phase * 2.0 * np.pi * 1.5)
            height = 1.34 + 0.16 * np.sin(phase * 2.0 * np.pi * 2.0 + 0.3)
            target = target_base + np.array(
                [0.10 * np.sin(phase * 2 * np.pi), 0.08 * np.cos(phase * 2 * np.pi), 0.0]
            )

        eye = np.array([radius * np.cos(angle), radius * np.sin(angle), height])
        # Hand-held jitter. A perfectly smooth orbit is not what real footage
        # looks like, and the reconstruction should survive the real thing.
        eye = eye + rng.normal(0, 0.006, 3)
        poses.append(look_at_pose(eye, target + rng.normal(0, 0.012, 3)))
    return poses


def demo_camera_trajectory(
    count: int, rng: np.random.Generator, seed_phase: float
) -> list[np.ndarray]:
    """A head-mounted camera: standing at the table, looking down at it."""
    poses = []
    for i in range(count):
        phase = i / count
        stand_angle = -np.pi / 2 + seed_phase
        distance = 0.60 - 0.05 * np.sin(phase * np.pi)
        eye = np.array(
            [
                distance * np.cos(stand_angle) + 0.05 * np.sin(phase * 2 * np.pi * 0.7),
                distance * np.sin(stand_angle),
                1.38 + 0.035 * np.sin(phase * 2 * np.pi * 1.3),
            ]
        )
        eye = eye + rng.normal(0, 0.004, 3)
        # The head follows the hands, so the look-at point drifts with the task.
        target = np.array([0.0, 0.0, TABLE_TOP_Z]) + np.array(
            [0.06 * np.sin(phase * 2 * np.pi * 0.9), 0.05 * phase, 0.0]
        )
        poses.append(look_at_pose(eye, target + rng.normal(0, 0.006, 3)))
    return poses


def task_schedule(phase: float) -> tuple[np.ndarray, float, bool]:
    """The pick-and-place script.

    Returns the target position for the grasp point, the finger curl amount,
    and whether the object is held.
    """
    approach_end, close_end, carry_end, open_end = 0.28, 0.38, 0.72, 0.82

    lift = np.array([0.0, 0.0, 0.14])
    start = OBJECT_START.copy()
    goal = OBJECT_GOAL.copy()

    if phase < approach_end:
        # Reach in from the near edge, dropping onto the object.
        travel = phase / approach_end
        entry = start + np.array([0.10, -0.34, 0.20])
        position = entry + (start - entry) * (travel**0.75)
        return position, 0.05 + 0.10 * travel, False

    if phase < close_end:
        travel = (phase - approach_end) / (close_end - approach_end)
        return start, 0.15 + 0.30 * travel, travel > 0.55

    if phase < carry_end:
        travel = (phase - close_end) / (carry_end - close_end)
        # Lift, traverse, lower. A sine arc, which is what a person does.
        arc = np.sin(np.clip(travel, 0, 1) * np.pi)
        position = start + (goal - start) * travel + lift * arc
        return position, 0.45, True

    if phase < open_end:
        travel = (phase - open_end + (open_end - carry_end)) / (open_end - carry_end)
        return goal, 0.45 - 0.30 * travel, travel < 0.45

    travel = (phase - open_end) / (1.0 - open_end)
    exit_point = goal + np.array([0.12, -0.32, 0.22])
    return goal + (exit_point - goal) * (travel**0.8), 0.15, False


def hand_rotation(pitch_deg: float = -20.0, yaw_deg: float = -32.0) -> np.ndarray:
    """Orientation for a top-down grasp.

    Local frame is +y toward the fingertips, +x toward the thumb, +z out of
    the back of the hand.

    Two angles, and both matter for whether a hand estimator can see anything.
    Pitch tilts the fingers down toward the table. Yaw swings the hand so the
    fingers point across the camera's view rather than along it. With no yaw
    the fingers point almost straight away from a head-mounted camera, which
    foreshortens them into the back of the hand. MediaPipe loses the hand
    entirely in that pose, which was measured, not assumed.
    """
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rot_z @ rot_x


def grasp_point_local(hand_local: np.ndarray) -> np.ndarray:
    """The thumb-index tip midpoint, which is where a gripper's jaws would be."""
    return 0.5 * (hand_local[4] + hand_local[8])


def build_demo(
    scene: Scene,
    frame_count: int,
    rng: np.random.Generator,
    seed_phase: float,
) -> dict:
    """Generate one demo episode: camera poses, hand landmarks, object poses."""
    camera_poses = demo_camera_trajectory(frame_count, rng, seed_phase)
    rotation = hand_rotation()
    canonical = canonical_hand()

    hands, object_poses, held_flags, widths = [], [], [], []
    object_position = OBJECT_START.copy()

    for i in range(frame_count):
        phase = i / max(frame_count - 1, 1)
        target, curl, held = task_schedule(phase)

        hand_local = curl_hand(canonical, curl)
        # Place the wrist so the grasp point lands on the target.
        wrist = target - rotation @ grasp_point_local(hand_local)
        hand_world = hand_to_world(hand_local, rotation, wrist)

        grasp_world = 0.5 * (hand_world[4] + hand_world[8])
        width = float(np.linalg.norm(hand_world[4] - hand_world[8]))

        if held:
            object_position = grasp_world.copy()
        object_pose = np.eye(4)
        object_pose[:3, 3] = object_position

        hands.append(hand_world)
        object_poses.append(object_pose)
        held_flags.append(bool(held))
        widths.append(width)

    return {
        "camera_poses": np.stack(camera_poses),
        "hand_landmarks": np.stack(hands),
        "object_poses": np.stack(object_poses),
        "held": np.array(held_flags),
        "grasp_width_m": np.array(widths),
    }


def hand_capsules(landmarks: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Turn 21 landmarks into the capsules the renderer draws."""
    capsules = []
    for a, b in HAND_BONES:
        capsules.append((landmarks[a], landmarks[b], bone_radius(a, b)))
    # A forearm stub, so the hand does not float in mid-air. It runs back from
    # the wrist, away from the knuckles. Kept short and slim: an oversized
    # forearm crowds the hand and costs detections.
    direction = landmarks[0] - landmarks[9]
    direction = direction / max(np.linalg.norm(direction), 1e-6)
    capsules.append((landmarks[0], landmarks[0] + direction * 0.11, 0.023))
    return capsules


def render_clip(
    out_path: Path,
    frames_dir: Path,
    poses: np.ndarray,
    scene: Scene,
    intrinsics: dict,
    device: str,
    fps: float,
    hand_landmarks: np.ndarray | None = None,
    object_poses: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    label: str = "clip",
) -> None:
    """Render every frame, then encode to mp4."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.png"):
        stale.unlink()

    import cv2

    total = len(poses)
    for i, pose in enumerate(poses):
        quads = list(scene.quads)
        if object_poses is not None:
            quads = quads + transform_quads(scene.object_quads, object_poses[i])
        capsules = hand_capsules(hand_landmarks[i]) if hand_landmarks is not None else []

        image, _ = render_frame(
            pose, quads, capsules,
            intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"],
            intrinsics["width"], intrinsics["height"],
            device=device, rng=rng,
        )
        cv2.imwrite(str(frames_dir / f"{i:05d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if (i + 1) % 25 == 0 or i == total - 1:
            print(f"  {label}: {i + 1}/{total} frames", flush=True)

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%05d.png"),
        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(command, check=True)
    print(f"  wrote {out_path} ({total} frames at {fps} fps)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="fixtures/room01", help="output directory")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--scan-frames", type=int, default=240)
    parser.add_argument("--demo-frames", type=int, default=180)
    parser.add_argument("--demos", type=int, default=3)
    parser.add_argument("--scan-fps", type=float, default=4.0,
                        help="Encoded scan frame rate. Stage 0 samples the scan at 4 fps, "
                             "so 4 here keeps every rendered frame.")
    parser.add_argument("--demo-fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_frames"

    rng = np.random.default_rng(args.seed)
    print("building room ...")
    scene = build_room(seed=args.seed)
    scene.object_quads = _resize_object(scene, OBJECT_HALF)
    print(f"  {len(scene.quads)} room quads, {len(scene.object_quads)} object quads")

    # A 60 degree horizontal field of view, close to an iPhone main wide.
    fov_deg = 60.0
    focal = (args.width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    intrinsics = {
        "width": args.width, "height": args.height,
        "fx": focal, "fy": focal,
        "cx": args.width / 2.0, "cy": args.height / 2.0,
        "horizontal_fov_deg": fov_deg,
    }
    print(f"  intrinsics: {args.width}x{args.height}, f={focal:.1f}px, fov={fov_deg} deg")

    print(f"rendering scan ({args.scan_frames} frames) ...")
    scan_poses = np.stack(scan_trajectory(args.scan_frames, rng))
    # The scan shows the static room. No hand, and the object is present so
    # the splat contains it, which Stage 5 relies on for background.
    scan_object_poses = np.repeat(np.eye(4)[None], len(scan_poses), axis=0)
    scan_object_poses[:, :3, 3] = OBJECT_START
    render_clip(
        out_dir / "scan.mp4", work_dir / "scan", scan_poses, scene, intrinsics,
        device, args.scan_fps, object_poses=scan_object_poses, rng=rng, label="scan",
    )

    groundtruth = {
        "intrinsics": intrinsics,
        "world": {
            "up_axis": "z",
            "table_top_z_m": TABLE_TOP_Z,
            "object_half_extents_m": OBJECT_HALF.tolist(),
            "object_start_m": OBJECT_START.tolist(),
            "object_goal_m": OBJECT_GOAL.tolist(),
        },
        "scan": {"camera_poses": scan_poses.tolist(), "fps": args.scan_fps},
        "demos": {},
    }

    for index in range(args.demos):
        name = f"demo_{index}"
        print(f"rendering {name} ({args.demo_frames} frames) ...")
        demo = build_demo(scene, args.demo_frames, rng, seed_phase=0.22 * (index - 1))
        render_clip(
            out_dir / f"{name}.mp4", work_dir / name, demo["camera_poses"], scene,
            intrinsics, device, args.demo_fps,
            hand_landmarks=demo["hand_landmarks"], object_poses=demo["object_poses"],
            rng=rng, label=name,
        )
        groundtruth["demos"][name] = {
            "fps": args.demo_fps,
            "camera_poses": demo["camera_poses"].tolist(),
            "hand_landmarks": demo["hand_landmarks"].tolist(),
            "object_poses": demo["object_poses"].tolist(),
            "held": demo["held"].tolist(),
            "grasp_width_m": demo["grasp_width_m"].tolist(),
        }

    # ARKit trajectories. Real metres, which is what gives Stage 1 its metric
    # scale and Stage 2 its fallback when registration is poor.
    def write_arkit(name: str, poses: np.ndarray, fps: float) -> None:
        payload = {
            "frames": [
                {"timestamp": i / fps, "transform": pose.flatten().tolist()}
                for i, pose in enumerate(poses)
            ]
        }
        (out_dir / f"arkit_{name}.json").write_text(json.dumps(payload))

    write_arkit("scan", scan_poses, args.scan_fps)
    for name, demo in groundtruth["demos"].items():
        write_arkit(name, np.asarray(demo["camera_poses"]), args.demo_fps)
    (out_dir / "groundtruth.json").write_text(json.dumps(groundtruth))

    if not args.keep_frames and work_dir.exists():
        shutil.rmtree(work_dir)

    print(f"\nfixture written to {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name}  {path.stat().st_size / 1e6:.1f} MB")
    return 0


def _resize_object(scene: Scene, half: np.ndarray):
    """Rebuild the object quads at the configured size, centred on the origin."""
    from tools.synthetic_scene import _box_quads

    texture = scene.object_quads[0].texture
    return _box_quads(np.zeros(3), half, [texture])


if __name__ == "__main__":
    raise SystemExit(main())
