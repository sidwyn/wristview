"""Monocular depth, made metric.

The build plan wants scale-consistent depth for object placement. WARPED uses
SpatialTrackerV2; Depth-Anything V2 is the named substitute and it runs on MPS
at about 59 ms a frame.

Depth-Anything returns *relative* inverse depth. That is not directly usable:
placing an object needs metres. The fix uses what Stage 1 already produced.
Render the splat from the same camera pose and it gives metric depth for the
static scene. Fit an affine map from the model's relative depth to the splat's
metric depth over the background pixels only, then apply that map everywhere.
The moving object, which the splat has in the wrong place, gets metric depth
from the background's calibration.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..logging_setup import get

log = get(__name__)


def depth_anything_available() -> tuple[bool, str]:
    try:
        from transformers import AutoModelForDepthEstimation  # noqa: F401
    except ImportError as exc:
        return False, f"transformers missing: {exc}"
    return True, "available"


class DepthAnythingV2:
    """Relative monocular depth on MPS."""

    def __init__(self, device: str, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        log.info("Depth-Anything V2 ready on %s (%s)", device, model_id)

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return relative inverse depth at the input resolution.

        Larger values are nearer. The units are arbitrary, which is why
        `fit_metric_depth` exists.
        """
        import torch
        from PIL import Image

        height, width = image_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            prediction = self.model(**inputs).predicted_depth

        prediction = torch.nn.functional.interpolate(
            prediction[:, None], size=(height, width), mode="bicubic", align_corners=False
        )[0, 0]
        return prediction.float().cpu().numpy()


def fit_metric_depth(
    relative: np.ndarray,
    metric_reference: np.ndarray,
    reference_valid: np.ndarray,
    exclude: np.ndarray | None = None,
    min_samples: int = 500,
) -> tuple[np.ndarray | None, dict]:
    """Map relative inverse depth onto metres using a metric reference.

    The model predicts inverse depth up to an affine transform, so the fit is
    `1 / metric ≈ a * relative + b`, solved by least squares on pixels where
    the reference is valid.

    `exclude` masks out pixels the reference gets wrong, meaning the hand and
    the moving object. Fitting on those would drag the whole calibration.
    """
    valid = reference_valid.copy()
    if exclude is not None:
        valid &= ~exclude
    # Drop the far plane: those pixels are background the splat never modelled.
    valid &= metric_reference > 1e-3
    valid &= np.isfinite(relative)

    if int(valid.sum()) < min_samples:
        return None, {"fit": "failed", "samples": int(valid.sum())}

    x = relative[valid].astype(np.float64)
    y = 1.0 / metric_reference[valid].astype(np.float64)

    # Trim the worst decile each side. Splat depth is noisy at silhouettes.
    lo, hi = np.percentile(y, [5, 95])
    keep = (y >= lo) & (y <= hi)
    x, y = x[keep], y[keep]
    if len(x) < min_samples:
        return None, {"fit": "failed", "samples": int(len(x))}

    design = np.stack([x, np.ones_like(x)], axis=1)
    solution, residuals, _, _ = np.linalg.lstsq(design, y, rcond=None)
    slope, intercept = float(solution[0]), float(solution[1])

    inverse = slope * relative + intercept
    # Anything at or behind the camera plane is not a valid depth.
    metric = np.where(inverse > 1e-4, 1.0 / np.maximum(inverse, 1e-4), 0.0)

    predicted = design @ solution
    correlation = float(np.corrcoef(predicted, y)[0, 1]) if len(y) > 2 else 0.0
    diagnostics = {
        "fit": "ok",
        "samples": int(len(x)),
        "slope": slope,
        "intercept": intercept,
        "correlation": round(correlation, 4),
        "residual_rmse_inv_m": float(np.sqrt(np.mean((predicted - y) ** 2))),
    }
    return metric.astype(np.float32), diagnostics


def backproject(
    depth: np.ndarray, mask: np.ndarray, intrinsics, max_points: int = 4000
) -> np.ndarray:
    """Lift masked pixels into camera-frame 3D points, subsampled."""
    ys, xs = np.nonzero(mask & (depth > 1e-4))
    if len(ys) == 0:
        return np.zeros((0, 3))
    if len(ys) > max_points:
        picked = np.random.default_rng(0).choice(len(ys), max_points, replace=False)
        ys, xs = ys[picked], xs[picked]
    values = depth[ys, xs].astype(np.float64)
    pixels = np.stack([xs, ys], axis=1).astype(np.float64)
    return intrinsics.unproject(pixels, values)
