"""Object segmentation and tracking.

The build plan asks for Grounding DINO to turn the task's text description
into a box, and SAM 2 to turn that box into a mask tracked across frames.
Both are attempted. When either is unavailable the fallback runs: seed a mask
from where the fingertips converge, then track it by appearance and flow.

The fallback is not a substitute for the models on real footage. It assumes
one rigid object near the grasp point, which is exactly the v1 scope the plan
sets: pick-and-place and tool use, no cloth, no liquids, no articulation.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..logging_setup import get

log = get(__name__)


def grounding_dino_available() -> tuple[bool, str]:
    try:
        from transformers import AutoModelForZeroShotObjectDetection  # noqa: F401
    except ImportError as exc:
        return False, f"transformers missing: {exc}"
    return True, "available"


def sam2_available() -> tuple[bool, str]:
    try:
        import sam2  # noqa: F401
    except ImportError as exc:
        return False, f"sam2 not importable: {exc}"
    return True, "available"


# A detection wider than this share of the frame is not an object, it is the
# image. See the raise in `detect`.
MAX_DETECTION_FRAME_FRACTION = 0.60


class GroundingDinoDetector:
    """Text-prompted box detection."""

    def __init__(self, device: str, model_id: str = "IDEA-Research/grounding-dino-tiny"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        log.info("Grounding DINO ready on %s (%s)", device, model_id)

    def detect(
        self, image_bgr: np.ndarray, prompt: str, box_threshold: float, text_threshold: float
    ) -> np.ndarray | None:
        """Return the highest-scoring box as (x0, y0, x1, y1), or None."""
        import torch
        from PIL import Image

        # Grounding DINO wants lowercase phrases ending in a period.
        text = prompt.lower().strip()
        if not text.endswith("."):
            text += "."

        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil.size[::-1]],
        )[0]
        if len(results["boxes"]) == 0:
            return None
        best = int(torch.argmax(results["scores"]))
        box = results["boxes"][best].cpu().numpy()

        # A box covering nearly the whole frame is argmax latching onto
        # nothing, not a detection. Grounding DINO always returns its
        # highest-scoring box, and when the phrase matches nothing in
        # particular that box is the image.
        #
        # Checking real27's scan for the tea box returned [6, 3, 1912, 1075] on
        # a 1920x1080 frame, 98 per cent of the area, on 5 of 14 sampled
        # frames. Read as detections they said the carton was present in every
        # scan frame. It was not on the mat at all.
        #
        # Refuse rather than return it. A caller cannot tell a real box from
        # this one, and the failure is silent: the same defect family as
        # `_overflow`, a number that describes a failure and is never read.
        height, width = image_bgr.shape[:2]
        area = float(max(box[2] - box[0], 0) * max(box[3] - box[1], 0))
        fraction = area / float(width * height)
        if fraction > MAX_DETECTION_FRAME_FRACTION:
            raise ValueError(
                f"Grounding DINO returned a box covering {fraction * 100:.0f} "
                f"per cent of the frame for {prompt!r}, over "
                f"{MAX_DETECTION_FRAME_FRACTION * 100:.0f}. That is argmax with "
                f"nothing to latch onto, not a detection. Either the object is "
                f"absent from this frame, or the prompt names something the "
                f"detector cannot find."
            )
        return box


class Sam2Segmenter:
    """SAM 2 image segmentation from a box or point prompt."""

    def __init__(self, device: str, model_id: str = "facebook/sam2.1-hiera-tiny"):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        log.info("SAM 2 ready on %s (%s)", device, model_id)

    def segment(
        self, image_bgr: np.ndarray, box: np.ndarray | None = None,
        point: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return a boolean mask, or None."""
        import torch

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            self.predictor.set_image(rgb)
            masks, scores, _ = self.predictor.predict(
                box=box[None] if box is not None else None,
                point_coords=point[None] if point is not None else None,
                point_labels=np.array([1]) if point is not None else None,
                multimask_output=True,
            )
        if masks is None or len(masks) == 0:
            return None
        best = int(np.argmax(scores))
        return masks[best].astype(bool)


def seed_mask_from_point(
    image_bgr: np.ndarray, point: np.ndarray, window: int = 90
) -> np.ndarray | None:
    """Fallback segmentation: grow a region from the grasp point.

    GrabCut inside a window centred on where the fingertips converge. The
    object being grasped is, by construction, at that point.
    """
    height, width = image_bgr.shape[:2]
    cx, cy = int(round(point[0])), int(round(point[1]))
    if not (0 <= cx < width and 0 <= cy < height):
        return None

    x0, y0 = max(0, cx - window), max(0, cy - window)
    x1, y1 = min(width, cx + window), min(height, cy + window)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None

    patch = image_bgr[y0:y1, x0:x1]
    mask = np.full(patch.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    # The border is background, a small core around the point is foreground.
    mask[:4, :] = mask[-4:, :] = mask[:, :4] = mask[:, -4:] = cv2.GC_BGD
    local = (cy - y0, cx - x0)
    radius = max(6, window // 8)
    cv2.circle(mask, (local[1], local[0]), radius, cv2.GC_FGD, -1)

    try:
        cv2.grabCut(
            patch, mask, None,
            np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
            4, cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as exc:
        log.debug("grabCut failed: %s", exc)
        return None

    local_mask = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    full = np.zeros((height, width), dtype=bool)
    full[y0:y1, x0:x1] = local_mask
    return full


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected blob. Drops speckle."""
    if not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def clean_mask(mask: np.ndarray, min_pixels: int, max_fraction: float) -> np.ndarray | None:
    """Reject a mask that is too small to be real or too big to be one object."""
    if mask is None or not mask.any():
        return None
    mask = largest_component(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    count = int(mask.sum())
    if count < min_pixels:
        return None
    if count > max_fraction * mask.size:
        return None
    return mask


def propagate_mask(
    previous_gray: np.ndarray, current_gray: np.ndarray, previous_mask: np.ndarray
) -> np.ndarray | None:
    """Carry a mask forward with dense optical flow.

    Used when the segmenter loses the object for a frame or two. Cheap, and it
    keeps a track alive across a brief occlusion by the hand.
    """
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray, current_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0
    )
    ys, xs = np.nonzero(previous_mask)
    if len(ys) == 0:
        return None
    dx = flow[ys, xs, 0]
    dy = flow[ys, xs, 1]
    new_x = np.clip(np.round(xs + dx).astype(int), 0, previous_mask.shape[1] - 1)
    new_y = np.clip(np.round(ys + dy).astype(int), 0, previous_mask.shape[0] - 1)
    out = np.zeros_like(previous_mask)
    out[new_y, new_x] = True
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
