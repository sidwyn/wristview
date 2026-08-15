"""Stages 4 and 5 end to end over a synthetic run directory.

Builds the artifacts Stage 3 would have produced, then runs retargeting and
rendering for real. This catches the wiring faults that unit tests miss: a
stage reading a key another stage never wrote, an index mapping between the
video rate and the control rate, a missing file on a fallback path.

No splat here, so Stage 5 renders geometry against a flat background. The
splat path is covered in test_splat.py.
"""

import json

import numpy as np
import pytest

from wristview.config import Config
from wristview.runctx import RunContext, write_json
from wristview.stages import s04_retarget, s05_render

FPS = 60.0
FRAMES = 120


def build_hand_sequence() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A reach, a grasp, a carry, and a release, in world metres."""
    landmarks = np.zeros((FRAMES, 21, 3))
    object_positions = np.zeros((FRAMES, 3))
    held = np.zeros(FRAMES, dtype=bool)

    start = np.array([0.10, 0.0, 0.80])
    goal = np.array([-0.20, -0.10, 0.80])

    for i in range(FRAMES):
        phase = i / (FRAMES - 1)
        if phase < 0.25:            # reach
            centre = start + np.array([0.0, -0.25, 0.18]) * (1 - phase / 0.25)
            gap, carrying = 0.085, False
        elif phase < 0.35:          # close
            centre, carrying = start, phase > 0.30
            gap = 0.085 - 0.05 * (phase - 0.25) / 0.10
        elif phase < 0.75:          # carry
            travel = (phase - 0.35) / 0.40
            centre = start + (goal - start) * travel
            centre = centre + np.array([0.0, 0.0, 0.12 * np.sin(travel * np.pi)])
            gap, carrying = 0.035, True
        else:                       # release and retract
            travel = (phase - 0.75) / 0.25
            centre = goal + np.array([0.0, -0.2, 0.15]) * travel
            gap, carrying = 0.035 + 0.05 * travel, False

        # Thumb and index straddle the grasp centre; the palm sits behind.
        landmarks[i, 4] = centre + np.array([gap / 2, 0.0, 0.0])
        landmarks[i, 8] = centre - np.array([gap / 2, 0.0, 0.0])
        landmarks[i, 0] = centre + np.array([0.0, -0.10, 0.02])
        landmarks[i, 5] = centre + np.array([0.02, -0.03, 0.01])
        landmarks[i, 9] = centre + np.array([0.0, -0.03, 0.01])
        landmarks[i, 13] = centre + np.array([-0.02, -0.03, 0.01])
        landmarks[i, 17] = centre + np.array([-0.04, -0.03, 0.01])

        held[i] = carrying
        object_positions[i] = centre if carrying else (goal if phase >= 0.75 else start)

    return landmarks, object_positions, held


@pytest.fixture
def run(tmp_path) -> RunContext:
    """A run directory holding everything Stages 0 to 3 would have written."""
    config = Config.load(
        overrides={
            "render": {"width": 96, "height": 72, "write_preview_video": False},
            "retarget": {"control_rate_hz": 15.0},
        }
    )
    ctx = RunContext.create(tmp_path / "runs", "integration", config)

    write_json(ctx.root / "sources.json", {
        "scan": str(tmp_path / "scan.mov"), "demos": [], "arkit": {},
        "instruction": "a drinking glass",
    })

    write_json(ctx.stage_dir(0) / "manifest.json", {
        "clips": {
            "scan": {"kind": "scan", "frame_names": [], "frame_count": 0,
                     "frames_dir": "00_ingest/scan/frames",
                     "video_info": {"fps": FPS}, "effective_fps": FPS},
            "demo_0": {"kind": "demo", "frame_names": [f"demo_0_{i:05d}.jpg" for i in range(FRAMES)],
                       "frame_count": FRAMES, "frames_dir": "00_ingest/demo_0/frames",
                       "video_info": {"fps": FPS}, "effective_fps": FPS},
        },
        "arkit": None,
    })

    landmarks, object_positions, held = build_hand_sequence()

    estimate_dir = ctx.episode_dir(3, "demo_0")
    np.savez_compressed(
        estimate_dir / "hand.npz",
        landmarks_world=landmarks,
        landmarks_cam=landmarks,
        landmarks_px=np.zeros((FRAMES, 21, 2)),
        valid=np.ones(FRAMES, dtype=bool),
        confidence=np.ones(FRAMES),
    )
    object_poses = np.repeat(np.eye(4)[None], FRAMES, axis=0)
    object_poses[:, :3, 3] = object_positions
    np.save(estimate_dir / "object_pose.npy", object_poses)
    np.save(estimate_dir / "object_valid.npy", np.ones(FRAMES, dtype=bool))

    # A small cube of points standing in for the fitted object model.
    rng = np.random.default_rng(0)
    np.save(estimate_dir / "object_model.npy", rng.uniform(-0.03, 0.03, (200, 3)))

    write_json(ctx.stage_dir(3) / "summary.json", {
        "demo_0": {"clip_id": "demo_0", "hand_detection_rate": 1.0,
                   "object_track_rate": 1.0, "hand_backend": "synthetic"}
    })
    del held
    return ctx


class TestStage4:
    def test_produces_a_trajectory(self, run):
        statuses = s04_retarget.run(run)
        assert statuses["demo_0"]["rejected"] is False

        path = run.episode_dir(4, "demo_0", create=False) / "ee_trajectory.npy"
        assert path.exists()
        poses = np.load(path)
        assert poses.ndim == 3 and poses.shape[1:] == (4, 4)

    def test_resamples_to_the_control_rate(self, run):
        s04_retarget.run(run)
        payload = np.load(run.episode_dir(4, "demo_0", create=False) / "ee_trajectory.npz")
        duration = (FRAMES - 1) / FPS
        expected = int(np.floor(duration * 15.0)) + 1
        assert len(payload["poses"]) == expected
        assert len(payload["width_m"]) == expected
        assert len(payload["closed"]) == expected

    def test_detects_exactly_one_grasp_and_one_release(self, run):
        statuses = s04_retarget.run(run)
        grasp = statuses["demo_0"]["grasp"]
        assert grasp["closed_frames"] > 0, "the gripper never closed"
        assert grasp["transitions"] == 2, f"expected close then open, got {grasp['transitions']}"
        assert grasp["grasp_onset_frame"] is not None
        assert grasp["release_frame"] is not None
        assert grasp["grasp_onset_frame"] < grasp["release_frame"]

    def test_rotations_stay_valid(self, run):
        s04_retarget.run(run)
        poses = np.load(run.episode_dir(4, "demo_0", create=False) / "ee_trajectory.npy")
        for pose in poses:
            rotation = pose[:3, :3]
            assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)

    def test_widths_stay_inside_the_gripper_limits(self, run):
        s04_retarget.run(run)
        payload = np.load(run.episode_dir(4, "demo_0", create=False) / "ee_trajectory.npz")
        spec = run.config.get("retarget.gripper")
        assert payload["width_m"].min() >= spec["min_width_m"] - 1e-9
        assert payload["width_m"].max() <= spec["max_width_m"] + 1e-9

    def test_writes_the_meta_contract(self, run):
        s04_retarget.run(run)
        meta = json.loads((run.stage_dir(4, create=False) / "meta.json").read_text())
        assert meta["status"] == "ok"
        assert meta["backends"]["gripper"] in {"config", "urdf"}
        assert "demo_0_trajectory" in meta["outputs"]

    def test_writes_a_urdf_when_none_was_given(self, run):
        s04_retarget.run(run)
        assert (run.stage_dir(4, create=False) / "gripper.urdf").exists()


class TestStage5:
    def test_renders_wrist_frames(self, run):
        s04_retarget.run(run)
        statuses = s05_render.run(run)
        assert "demo_0" in statuses

        wrist_dir = run.episode_dir(5, "demo_0", create=False) / "wrist"
        frames = sorted(wrist_dir.glob("*.png"))
        assert len(frames) == statuses["demo_0"]["frames"]
        assert len(frames) > 0

    def test_renders_at_the_control_rate_by_default(self, run):
        s04_retarget.run(run)
        statuses = s05_render.run(run)
        control = len(np.load(
            run.episode_dir(4, "demo_0", create=False) / "ee_trajectory.npz"
        )["poses"])
        assert statuses["demo_0"]["frames"] == control

    def test_frames_are_the_configured_size(self, run):
        import cv2

        s04_retarget.run(run)
        s05_render.run(run)
        frame = cv2.imread(
            str(sorted((run.episode_dir(5, "demo_0", create=False) / "wrist").glob("*.png"))[0])
        )
        assert frame.shape[:2] == (72, 96)

    def test_the_gripper_is_actually_drawn(self, run):
        import cv2

        s04_retarget.run(run)
        s05_render.run(run)
        # With no splat the background is flat, so any structure in the image
        # is the gripper and the object.
        frames = sorted((run.episode_dir(5, "demo_0", create=False) / "wrist").glob("*.png"))
        middle = cv2.imread(str(frames[len(frames) // 2]))
        assert middle.std() > 1.0, "the frame is a flat background; nothing was drawn"

    def test_writes_a_contact_sheet(self, run):
        s04_retarget.run(run)
        s05_render.run(run)
        assert (run.episode_dir(5, "demo_0", create=False) / "contact_sheet.png").exists()

    def test_records_which_renderer_ran(self, run):
        s04_retarget.run(run)
        s05_render.run(run)
        meta = json.loads((run.stage_dir(5, create=False) / "meta.json").read_text())
        assert meta["status"] == "ok"
        assert meta["backends"]["splat_renderer"] == "wristview_mps_rasterizer"
        assert meta["backends"]["camera_model"] in {"pinhole", "fisheye"}

    def test_max_frames_caps_the_render(self, run):
        run.config.data["render"]["max_frames"] = 5
        s04_retarget.run(run)
        statuses = s05_render.run(run)
        assert statuses["demo_0"]["frames"] == 5


def test_stage_5_refuses_without_stage_4(run):
    """A stage must fail loudly when its input is missing, not invent one."""
    with pytest.raises(FileNotFoundError):
        s05_render.run(run)
