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


def build_dataset(
    episodes: list[dict],
    out_root: Path,
    repo_id: str,
    fps: int,
    task: str,
    robot_type: str = "wristview-rendered",
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
    return dataset
