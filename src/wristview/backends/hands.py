"""Hand pose estimation.

**WiLoR is the primary backend.** It returns a MANO hand: 21 joints in metres
plus a camera translation, so the hand lands in the camera frame directly with
no PnP step. Measured on the ten verification stills in
`wristview-videos/a3/frames`, it detects on 10 of 10, including every
wrapped-grasp contact frame where the fingers occlude themselves. About 170 ms
a frame on MPS.

HaMeR, which the build plan originally named, does not install here. It pins
`mmcv==1.3.9`, a 2021 release that fails to build under Python 3.11, and it
pulls detectron2 and ViTPose. Attempted with and without build isolation. Its
Hugging Face Space is also down. See BUILD-LOG.md.

MediaPipe Hands stays as a fallback. It gives 21 landmarks and a metric
hand-centred model, which PnP lifts into the camera frame. It is markedly
weaker than WiLoR exactly where this task lives, on fingers wrapped around an
object, so it runs only when WiLoR is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..camera import Intrinsics
from ..logging_setup import get

log = get(__name__)

# MediaPipe landmark indices, named so the retargeting code reads clearly.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
PINKY_MCP = 17
PINKY_TIP = 20
NUM_LANDMARKS = 21


@dataclass
class HandFrame:
    """One frame's hand estimate."""

    detected: bool
    landmarks_px: np.ndarray      # (21, 2) pixels
    landmarks_cam: np.ndarray     # (21, 3) metres, camera frame
    confidence: float
    handedness: str


def hamer_available() -> tuple[bool, str]:
    """Report whether HaMeR can run, and why not when it cannot."""
    try:
        import hamer  # noqa: F401
    except ImportError as exc:
        return False, f"hamer not importable: {exc}"
    try:
        import detectron2  # noqa: F401
    except ImportError as exc:
        return False, f"hamer present but detectron2 missing: {exc}"
    return True, "available"


def wilor_available() -> tuple[bool, str]:
    """Report whether WiLoR can run, and why not when it cannot."""
    from . import mano_compat

    ok, reason = mano_compat.available()
    if not ok:
        return False, f"MANO stack unusable: {reason}"
    try:
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (  # noqa: F401
            WiLorHandPose3dEstimationPipeline,
        )
    except ImportError as exc:
        return False, f"wilor_mini not importable: {exc}"
    return True, "available"


class WiLoRHands:
    """WiLoR hand pose. 21 MANO joints in metres, in the camera frame.

    WiLoR predicts joints in a hand-centred frame plus a camera translation
    `pred_cam_t_full`, computed against its own focal length. That focal is
    derived from the crop, not from our calibrated camera, so the translation
    is rescaled onto our intrinsics rather than trusted as-is. Without that
    correction the hand sits at the wrong depth and every retargeted distance
    inherits the error.
    """

    def __init__(self, device: str = "mps", dtype=None):
        from . import mano_compat

        mano_compat.apply()

        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )

        self.device = device
        self._pipeline = WiLorHandPose3dEstimationPipeline(
            device=torch.device(device),
            dtype=dtype or torch.float32,
            verbose=False,
        )
        log.info("WiLoR ready on %s", device)

    def close(self) -> None:
        self._pipeline = None

    def __enter__(self) -> WiLoRHands:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def process(self, image_bgr: np.ndarray, intrinsics: Intrinsics) -> HandFrame:
        """Estimate one frame."""
        empty = HandFrame(
            detected=False,
            landmarks_px=np.zeros((NUM_LANDMARKS, 2)),
            landmarks_cam=np.zeros((NUM_LANDMARKS, 3)),
            confidence=0.0,
            handedness="",
        )
        rgb = image_bgr[:, :, ::-1]
        try:
            detections = self._pipeline.predict(rgb)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the clip
            log.debug("WiLoR failed on a frame: %s", exc)
            return empty
        if not detections:
            return empty

        # The manipulating hand is the one with the largest box. A second hand
        # steadying the scene, which C005 has, is smaller and further away.
        best = max(
            detections,
            key=lambda d: (d["hand_bbox"][2] - d["hand_bbox"][0])
            * (d["hand_bbox"][3] - d["hand_bbox"][1]),
        )
        preds = best["wilor_preds"]

        joints = np.asarray(preds["pred_keypoints_3d"][0], dtype=np.float64)
        translation = np.asarray(preds["pred_cam_t_full"][0], dtype=np.float64)
        pixels = np.asarray(preds["pred_keypoints_2d"][0], dtype=np.float64)

        # Rescale WiLoR's translation from its own focal length onto ours.
        # Depth scales with focal length under a fixed projected size.
        wilor_focal = float(np.asarray(preds["scaled_focal_length"]))
        if wilor_focal > 1e-6:
            translation = translation * (intrinsics.fx / wilor_focal)

        cam = joints + translation[None, :]
        if cam[:, 2].min() < 0.02 or cam[:, 2].max() > 5.0:
            return empty

        return HandFrame(
            detected=True,
            landmarks_px=pixels,
            landmarks_cam=cam,
            confidence=1.0,
            handedness="Right" if float(best.get("is_right", 1.0)) > 0.5 else "Left",
        )


class MediaPipeHands:
    """MediaPipe Hands, lifted into the camera frame with PnP."""

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This MediaPipe build has no `solutions` module. Version 1.0.x removed "
                "it, and its replacement crashes on macOS arm64 inside "
                "DrishtiMetalHelper. Pin mediapipe==0.10.21."
            )
        self._module = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        log.info("MediaPipe Hands ready (version %s)", mp.__version__)

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> MediaPipeHands:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def process(self, image_bgr: np.ndarray, intrinsics: Intrinsics) -> HandFrame:
        """Estimate one frame. Returns a HandFrame with `detected` set."""
        height, width = image_bgr.shape[:2]
        result = self._hands.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        empty = HandFrame(
            detected=False,
            landmarks_px=np.zeros((NUM_LANDMARKS, 2)),
            landmarks_cam=np.zeros((NUM_LANDMARKS, 3)),
            confidence=0.0,
            handedness="",
        )
        if not result.multi_hand_landmarks:
            return empty

        # Take the most confident hand. v1 is single-hand manipulation.
        scores = [
            handedness.classification[0].score for handedness in (result.multi_handedness or [])
        ]
        best = int(np.argmax(scores)) if scores else 0
        landmarks = result.multi_hand_landmarks[best]

        pixels = np.array(
            [[lm.x * width, lm.y * height] for lm in landmarks.landmark], dtype=np.float64
        )

        world = None
        if getattr(result, "multi_hand_world_landmarks", None):
            world = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.multi_hand_world_landmarks[best].landmark],
                dtype=np.float64,
            )

        cam = self._lift_to_camera(pixels, world, intrinsics)
        if cam is None:
            return empty

        return HandFrame(
            detected=True,
            landmarks_px=pixels,
            landmarks_cam=cam,
            confidence=float(scores[best]) if scores else 1.0,
            handedness=(
                result.multi_handedness[best].classification[0].label
                if result.multi_handedness
                else ""
            ),
        )

    @staticmethod
    def _lift_to_camera(
        pixels: np.ndarray, world: np.ndarray | None, intrinsics: Intrinsics
    ) -> np.ndarray | None:
        """Place the metric hand model in the camera frame by solving PnP.

        MediaPipe's world landmarks are metric but hand-centred, with no
        absolute position. PnP against the pixel positions recovers where the
        hand actually is.
        """
        if world is None:
            return None
        try:
            success, rvec, tvec = cv2.solvePnP(
                world.astype(np.float64),
                pixels.astype(np.float64),
                intrinsics.matrix,
                np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error as exc:
            log.debug("hand PnP failed: %s", exc)
            return None
        if not success:
            return None

        rotation, _ = cv2.Rodrigues(rvec)
        cam = world @ rotation.T + tvec.reshape(3)
        # A hand behind the camera, or 5 metres away, is a bad solve.
        if cam[:, 2].min() < 0.02 or cam[:, 2].max() > 5.0:
            return None
        return cam


class GroundTruthHands:
    """Read hand landmarks from a fixture's ground truth file.

    Only for synthetic fixtures. The renderer draws smooth capsules, which are
    out of distribution for a detector trained on photographs: MediaPipe finds
    the fixture hand on about 17 percent of frames, measured. That is a
    property of the fixture, not of the pipeline, and it would otherwise stop
    Stages 4 and 5 from ever being exercised end to end.

    Any run using this backend records `hand=synthetic_groundtruth` in
    meta.json. It must never be mistaken for an estimate.
    """

    def __init__(self, groundtruth_path: Path, clip_id: str):
        import json

        payload = json.loads(Path(groundtruth_path).read_text())
        demo = payload["demos"][clip_id]
        self.landmarks_world = np.asarray(demo["hand_landmarks"])
        self.count = len(self.landmarks_world)
        log.warning(
            "using SYNTHETIC GROUND TRUTH hand landmarks for %s (%d frames). "
            "This is fixture data, not an estimate.",
            clip_id, self.count,
        )

    def frame(self, index: int, camera_pose: np.ndarray, intrinsics: Intrinsics) -> HandFrame:
        """Return the true landmarks, expressed in the camera frame."""
        from ..geometry import invert_pose, transform_points

        if index >= self.count:
            return HandFrame(False, np.zeros((21, 2)), np.zeros((21, 3)), 0.0, "")
        world = self.landmarks_world[index]
        cam = transform_points(invert_pose(camera_pose), world)
        pixels, _ = intrinsics.project(cam)
        return HandFrame(True, pixels, cam, 1.0, "Right")


def palm_normal(landmarks: np.ndarray) -> np.ndarray:
    """Unit normal of the palm, pointing out of the back of the hand.

    Built from the wrist and the index and pinky knuckles, which are the three
    most stable landmarks on the hand. Fingertips move; the palm does not.
    """
    origin = landmarks[WRIST]
    across = landmarks[INDEX_MCP] - landmarks[PINKY_MCP]
    along = landmarks[MIDDLE_MCP] - origin
    normal = np.cross(across, along)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


def grasp_center(landmarks: np.ndarray) -> np.ndarray:
    """Midpoint of the thumb and index fingertips. The gripper's jaw centre."""
    return 0.5 * (landmarks[THUMB_TIP] + landmarks[INDEX_TIP])


def grasp_axis(landmarks: np.ndarray) -> np.ndarray:
    """Unit vector from index tip to thumb tip. The gripper's closing axis."""
    axis = landmarks[THUMB_TIP] - landmarks[INDEX_TIP]
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return axis / norm


def grasp_width(landmarks: np.ndarray) -> float:
    """Thumb tip to index tip distance, in metres."""
    return float(np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP]))


def approach_direction(landmarks: np.ndarray) -> np.ndarray:
    """Direction the gripper reaches along, from the palm toward the grasp.

    Not the palm normal on its own. A gripper approaches along the line from
    the knuckles out through the point where the fingers meet.
    """
    palm = (landmarks[INDEX_MCP] + landmarks[PINKY_MCP] + landmarks[WRIST]) / 3.0
    direction = grasp_center(landmarks) - palm
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return palm_normal(landmarks)
    return direction / norm
