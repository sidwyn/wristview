"""Write a run's episodes as one LeRobotDataset.

One dataset serves every training run in the experiment. Each run selects the
cameras it wants through the policy's `input_features`, so runs A, B, A', B'
and C read the same rows and differ only in which image keys they consume.
That is what makes the paired comparison in EXPERIMENT-PLAN 5.3 mechanically
airtight rather than merely intended: A' and B' cannot drift apart, because
they are the same frames.

Three facts from the LeRobot 0.4.4 source shape this file.

`modeling_diffusion.py:130` stacks every visual feature, so the diffusion
policy takes any number of cameras.

`configuration_diffusion.py:241` raises unless every image feature has the
SAME shape. Our renders are 640x360 and the ego frames are 1920x1080, so the
ego view is downscaled HERE, at export. Doing it at training time would mean
every run repeating the work and disagreeing about how.

`utils/constants.py:23` fixes the key names: `observation.state`,
`observation.images.<camera>`, `action`.

The actions come from Stage 4. The retarget already produces an end-effector
pose per frame, and the action is the change from one to the next plus the
gripper, which is what a policy emits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .logging_setup import get

log = get(__name__)

# Every camera must share this. See configuration_diffusion.py:241.
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 360

EGO_KEY = "observation.images.ego"
WRIST_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

# 6 pose numbers plus the gripper. Translation in metres, rotation as a
# rotation vector in radians, which is what scipy's as_rotvec gives and what
# avoids the sign ambiguity a quaternion carries.
STATE_DIM = 7
ACTION_DIM = 7


def pose_to_state(pose: np.ndarray, gripper: float) -> np.ndarray:
    """Flatten a 4x4 pose plus a gripper width into the state vector."""
    from scipy.spatial.transform import Rotation

    translation = np.asarray(pose, dtype=np.float32)[:3, 3]
    rotvec = Rotation.from_matrix(np.asarray(pose)[:3, :3]).as_rotvec()
    return np.concatenate([translation, rotvec, [gripper]]).astype(np.float32)


def relative_action(current: np.ndarray, following: np.ndarray, gripper: float) -> np.ndarray:
    """The action taken at `current`: how the effector moves to reach `following`.

    Relative, not absolute. A policy that predicts absolute pose has to learn
    the workspace's coordinate frame, which does not transfer between sessions
    and is not what the robot's controller accepts.
    """
    from scipy.spatial.transform import Rotation

    current = np.asarray(current, dtype=np.float64)
    following = np.asarray(following, dtype=np.float64)
    delta_t = following[:3, 3] - current[:3, 3]
    delta_r = Rotation.from_matrix(current[:3, :3].T @ following[:3, :3]).as_rotvec()
    return np.concatenate([delta_t, delta_r, [gripper]]).astype(np.float32)


def features_spec() -> dict:
    """The dataset schema. Both cameras carry the SAME shape, deliberately."""
    image = {"dtype": "video", "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
             "names": ["height", "width", "channels"]}
    return {
        EGO_KEY: dict(image),
        WRIST_KEY: dict(image),
        STATE_KEY: {"dtype": "float32", "shape": (STATE_DIM,),
                    "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]},
        ACTION_KEY: {"dtype": "float32", "shape": (ACTION_DIM,),
                     "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]},
    }


def load_image(path: Path, width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT) -> np.ndarray:
    """Read an image as RGB uint8 at the dataset's one size.

    INTER_AREA for the downscale. The ego frames come in at 1920x1080 and area
    averaging is the right filter for a 3x reduction; bilinear aliases.
    """
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"cannot read {path}")
    if (image.shape[1], image.shape[0]) != (width, height):
        interp = cv2.INTER_AREA if image.shape[1] > width else cv2.INTER_LINEAR
        image = cv2.resize(image, (width, height), interpolation=interp)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# --- the alignment gate -------------------------------------------------------
#
# The fault this catches is image[i] paired with action[j], i != j. It is
# invisible in every summary statistic: frame counts match, shapes match, the
# dataset loads, and a policy trains on it happily while learning the wrong
# association. The tap sync had the same shape and the same fix, a lag search.
#
# The signal is the wrist camera's optical flow against the action's
# translation magnitude. The wrist camera is RIGIDLY attached to the
# end-effector, so when the effector moves the whole image flows. That makes
# the two series the same physical quantity measured two ways, one from pixels
# and one from the pose stream, and it depends on nothing else the pipeline
# produced. No segmentation, no colour threshold, no object model.
#
# Measured on real26bm demo_0, 350 exported frames:
#     lag -2  r +0.2819
#     lag -1  r +0.3606
#     lag  0  r +0.4196   <-- peak
#     lag +1  r +0.4172
#     lag +2  r +0.3821
#
# Two things that table says. The peak is at zero, which is the pass. And the
# margin over lag +1 is 0.0024, so this gate resolves a lag of 2 frames or
# more and CANNOT reliably distinguish a 1-frame error from none. That limit
# is stated rather than hidden: a gate whose resolution is unknown is not
# evidence.
#
# The ego camera is head-mounted and is not attached to the effector, so it
# does not carry this signal. Measured, it peaks at lag +10 with r -0.245.
# Using it would fail a correct dataset.
MAX_LAG = 10
MIN_ABS_CORRELATION = 0.15


def alignment_lag(dataset, camera: str = WRIST_KEY, max_lag: int = MAX_LAG) -> dict:
    """Cross-correlate wrist optical flow against action magnitude."""
    import cv2

    def grey(index: int) -> np.ndarray:
        frame = dataset[index][camera].numpy().transpose(1, 2, 0)
        return cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    actions = np.stack([
        dataset[i]["action"].numpy().astype(np.float64) for i in range(len(dataset))
    ])
    speed = np.linalg.norm(actions[:, :3], axis=1)

    previous = grey(0)
    flow_magnitude = []
    for index in range(1, len(dataset)):
        current = grey(index)
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        flow_magnitude.append(float(np.median(np.linalg.norm(flow, axis=2))))
        previous = current

    image = np.asarray(flow_magnitude)
    action = speed[:len(image)]
    curve = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = image[-lag:], action[:len(action) + lag]
        elif lag > 0:
            a, b = image[:len(image) - lag], action[lag:]
        else:
            a, b = image, action
        count = min(len(a), len(b))
        curve[lag] = (float(np.corrcoef(a[:count], b[:count])[0, 1])
                      if count > 20 else float("nan"))

    finite = {k: v for k, v in curve.items() if np.isfinite(v)}
    peak = max(finite, key=lambda k: finite[k]) if finite else None
    return {
        "camera": camera, "frames": len(dataset),
        "peak_lag": peak,
        "peak_r": finite.get(peak),
        "r_at_zero": curve.get(0),
        "curve": {str(k): round(v, 4) for k, v in curve.items() if np.isfinite(v)},
    }


def verify_alignment(dataset, camera: str = WRIST_KEY) -> dict:
    """Raise unless the exported images and actions share an instant."""
    report = alignment_lag(dataset, camera=camera)
    lag, r_zero = report["peak_lag"], report["r_at_zero"]

    if lag is None or r_zero is None or not np.isfinite(r_zero):
        raise ValueError(
            "the alignment gate could not measure anything: too few frames "
            "to correlate. That is not a pass."
        )
    if abs(r_zero) < MIN_ABS_CORRELATION:
        raise ValueError(
            f"the alignment gate is inconclusive: correlation at lag 0 is "
            f"{r_zero:+.3f}, under {MIN_ABS_CORRELATION}. The wrist view and "
            f"the actions do not covary strongly enough to show alignment "
            f"either way, so this dataset is UNVERIFIED, not verified. A "
            f"near-static episode does this."
        )
    if lag != 0:
        raise ValueError(
            f"the exported images and actions are misaligned by {lag:+d} "
            f"frames. Cross-correlating wrist optical flow against action "
            f"magnitude peaks at lag {lag:+d} (r {report['peak_r']:+.3f}), not "
            f"at 0 (r {r_zero:+.3f}). Every image in this dataset is paired "
            f"with the action from {abs(lag)} frame(s) "
            f"{'later' if lag > 0 else 'earlier'}. A policy trained on it "
            f"learns the wrong association and nothing downstream can detect "
            f"that."
        )
    log.info("alignment gate PASSED: peak at lag 0, r %+.4f (lag +1 r %+.4f)",
             r_zero, report["curve"].get("1", float("nan")))
    return report


def build_dataset(
    episodes: list[dict],
    out_root: Path,
    repo_id: str,
    fps: int,
    task: str,
    robot_type: str = "wristview-rendered",
    verify: bool = True,
):
    """Write one LeRobotDataset from a list of episodes.

    Each episode is a dict with `ego`, `wrist` (lists of image paths, equal
    length), `poses` (N,4,4) and `gripper` (N,). The action at frame i moves
    the effector from pose i to pose i+1, so the last frame of an episode has
    no action and is dropped: a frame whose action is invented is not data.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_root = Path(out_root)
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features_spec(),
        root=out_root, robot_type=robot_type, use_videos=True,
    )

    written = 0
    for index, episode in enumerate(episodes):
        ego, wrist = episode["ego"], episode["wrist"]
        poses = np.asarray(episode["poses"], dtype=np.float64)
        gripper = np.asarray(episode["gripper"], dtype=np.float32)
        count = min(len(ego), len(wrist), len(poses), len(gripper))
        if count < 2:
            log.warning("episode %d has %d frames, too few for an action", index, count)
            continue
        if not (len(ego) == len(wrist) == len(poses) == len(gripper)):
            raise ValueError(
                f"episode {index} has mismatched lengths: ego {len(ego)}, "
                f"wrist {len(wrist)}, poses {len(poses)}, gripper {len(gripper)}. "
                f"A dataset built from misaligned streams pairs each image with "
                f"the wrong action and nothing downstream can detect it."
            )

        # The last frame has no successor, so it has no action.
        for i in range(count - 1):
            dataset.add_frame({
                EGO_KEY: load_image(Path(ego[i])),
                WRIST_KEY: load_image(Path(wrist[i])),
                STATE_KEY: pose_to_state(poses[i], gripper[i]),
                ACTION_KEY: relative_action(poses[i], poses[i + 1], gripper[i]),
                "task": task,
            })
        # Encode in this process. The parallel encoder's worker pool dies on
        # macOS with BrokenProcessPool, and LeRobot's own docstring says
        # parallel encoding defaults off there because it already saturates
        # the CPU. Passing it explicitly rather than relying on the default.
        dataset.save_episode(parallel_encoding=False)
        written += 1
        log.info("episode %d written, %d frames", index, count - 1)

    # Flush. Episode metadata is buffered and only written when the buffer
    # fills, which defaults to 10 episodes. A 3-episode dataset therefore
    # leaves meta/episodes empty, and loading it then falls through to the
    # Hugging Face Hub and fails with a 401 on a repo_id that was never meant
    # to exist remotely. The error names authentication and the cause is an
    # unflushed local buffer.
    dataset.finalize()

    episodes_dir = out_root / "meta" / "episodes"
    parquet = sorted(episodes_dir.rglob("*.parquet")) if episodes_dir.exists() else []
    if not parquet:
        raise RuntimeError(
            f"{episodes_dir} holds no parquet file after finalize(), so this "
            f"dataset cannot be loaded from disk. Writing it and not checking "
            f"would hand over a dataset that only fails at training time."
        )
    log.info("wrote %d episodes to %s, %d episode metadata file(s)",
             written, out_root, len(parquet))

    # Gate the dataset that was actually written, by reading it back. Checking
    # the buffers before they are serialised would test the wrong artifact.
    if verify:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        written_back = LeRobotDataset(repo_id=repo_id, root=out_root)
        report = verify_alignment(written_back)
        (out_root / "alignment.json").write_text(json.dumps(report, indent=2))
    return dataset
