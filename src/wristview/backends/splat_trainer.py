"""Train a Gaussian splat from posed images, on MPS.

The loss is the 3DGS default: L1 plus a D-SSIM term at weight 0.2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..logging_setup import get
from .splat_mps import Densifier, GaussianModel, render

log = get(__name__)


@dataclass
class TrainCamera:
    """One posed training image."""

    name: str
    view_matrix: torch.Tensor  # (4, 4) world to camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    image: torch.Tensor        # (H, W, 3) in [0, 1]


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel[:, None] @ kernel[None, :]


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural similarity over an (H, W, 3) pair. Returns a scalar."""
    device, dtype = pred.device, pred.dtype
    window = _gaussian_window(window_size, 1.5, device, dtype)
    window = window.expand(3, 1, window_size, window_size).contiguous()

    a = pred.permute(2, 0, 1)[None]
    b = target.permute(2, 0, 1)[None]
    pad = window_size // 2

    mu_a = F.conv2d(a, window, padding=pad, groups=3)
    mu_b = F.conv2d(b, window, padding=pad, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a2 = F.conv2d(a * a, window, padding=pad, groups=3) - mu_a2
    sigma_b2 = F.conv2d(b * b, window, padding=pad, groups=3) - mu_b2
    sigma_ab = F.conv2d(a * b, window, padding=pad, groups=3) - mu_ab

    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return (numerator / denominator).mean()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = float(((pred - target) ** 2).mean())
    return float(10.0 * np.log10(1.0 / max(mse, 1e-10)))


def load_cameras(
    entries: list[dict], images_root: Path, device: str, long_side: int
) -> list[TrainCamera]:
    """Load posed images, downscaling to `long_side` for training speed."""
    cameras: list[TrainCamera] = []
    for entry in entries:
        path = images_root / entry["name"]
        image = cv2.imread(str(path))
        if image is None:
            log.warning("could not read training image %s", path)
            continue
        full_h, full_w = image.shape[:2]

        scale = min(1.0, long_side / max(full_w, full_h))
        width = max(8, int(round(full_w * scale)))
        height = max(8, int(round(full_h * scale)))
        if (width, height) != (full_w, full_h):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        rgb = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(
            device=device, dtype=torch.float32
        ) / 255.0

        sx = width / entry["width"]
        sy = height / entry["height"]
        cameras.append(
            TrainCamera(
                name=entry["name"],
                view_matrix=torch.tensor(
                    np.asarray(entry["view_matrix"]), device=device, dtype=torch.float32
                ),
                fx=entry["fx"] * sx,
                fy=entry["fy"] * sy,
                cx=entry["cx"] * sx,
                cy=entry["cy"] * sy,
                width=width,
                height=height,
                image=rgb,
            )
        )
    return cameras


def train_splat(
    points: np.ndarray,
    colors: np.ndarray,
    cameras: list[TrainCamera],
    config: dict,
    device: str,
    progress_dir: Path | None = None,
) -> tuple[GaussianModel, dict]:
    """Fit a Gaussian splat to the posed images. Returns the model and metrics."""
    if not cameras:
        raise ValueError("no training cameras; Stage 1 cannot train a splat")

    rng = np.random.default_rng(0)
    init_points = int(config.get("init_points", 60000))
    if len(points) > init_points:
        idx = rng.choice(len(points), init_points, replace=False)
        points, colors = points[idx], colors[idx]
    elif len(points) < 1000:
        # A thin sparse cloud starves the optimizer. Jitter each point into a
        # small cluster so densification has somewhere to start.
        repeats = max(2, 1000 // max(len(points), 1))
        spread = float(np.linalg.norm(points.std(axis=0))) * 0.02 + 1e-3
        points = np.repeat(points, repeats, axis=0) + rng.normal(0, spread, (len(points) * repeats, 3))
        colors = np.repeat(colors, repeats, axis=0)

    centre = points.mean(axis=0)
    scene_extent = float(np.percentile(np.linalg.norm(points - centre, axis=1), 95))
    scene_extent = max(scene_extent, 1e-3)
    log.info(
        "splat init: %d points, scene extent %.3f, %d training views",
        len(points), scene_extent, len(cameras),
    )

    model = GaussianModel(
        means=torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)),
        colors=torch.from_numpy(np.ascontiguousarray(colors, dtype=np.float32)),
        sh_degree=int(config.get("sh_degree", 2)),
        device=device,
    )
    model.train()

    # Position learning rate scales with the scene, exactly as in 3DGS.
    optimizer = torch.optim.Adam(
        [
            {"params": [model.means], "lr": float(config["position_lr"]) * scene_extent, "name": "means"},
            {"params": [model.sh], "lr": float(config["feature_lr"]), "name": "sh"},
            {"params": [model.opacity_logit], "lr": float(config["opacity_lr"]), "name": "opacity"},
            {"params": [model.log_scales], "lr": float(config["scaling_lr"]), "name": "scales"},
            {"params": [model.quats], "lr": float(config["rotation_lr"]), "name": "quats"},
        ],
        eps=1e-15,
    )

    grad_percentile = config.get("densify_grad_percentile")
    densifier = Densifier(
        grad_threshold=float(config["densify_grad_threshold"]),
        grad_percentile=float(grad_percentile) if grad_percentile is not None else None,
        prune_opacity=float(config["prune_opacity_threshold"]),
        max_gaussians=int(config["max_gaussians"]),
        scene_extent=scene_extent,
    )

    iterations = int(config["iterations"])
    tile_size = int(config.get("tile_size", 16))
    max_per_tile = int(config.get("max_per_tile", 128))
    background = torch.zeros(3, device=device)

    history: list[dict] = []
    started = time.perf_counter()
    order = rng.permutation(len(cameras)).tolist()
    cursor = 0

    for step in range(1, iterations + 1):
        if cursor >= len(order):
            order = rng.permutation(len(cameras)).tolist()
            cursor = 0
        camera = cameras[order[cursor]]
        cursor += 1

        # Raise the SH degree every 500 steps, so colour learns before view
        # dependence. Fitting all bands from step one overfits the input views.
        if step % 500 == 0:
            model.raise_sh_degree()

        result = render(
            model,
            camera.view_matrix,
            camera.fx, camera.fy, camera.cx, camera.cy,
            camera.width, camera.height,
            background=background,
            tile_size=tile_size,
            max_per_tile=max_per_tile,
            near=0.01,
            far=1e4,
        )

        l1 = torch.abs(result.rgb - camera.image).mean()
        d_ssim = 1.0 - ssim(result.rgb, camera.image)
        loss = 0.8 * l1 + 0.2 * d_ssim

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        densifier.accumulate(model, result)
        optimizer.step()

        with torch.no_grad():
            # Keep quaternions on the unit sphere. Adam drifts them otherwise.
            model.quats.data = F.normalize(model.quats.data, dim=-1)

        if (
            int(config["densify_from_iter"]) <= step <= int(config["densify_until_iter"])
            and step % int(config["densify_interval"]) == 0
        ):
            stats = densifier.step(model, optimizer)
            log.info(
                "  step %5d densify: +%d cloned, +%d split, -%d pruned, now %d "
                "(grad threshold %.2e)",
                step, stats["cloned"], stats["split"], stats["pruned"], stats["count"],
                stats["threshold"],
            )

        reset_interval = int(config.get("opacity_reset_interval", 0))
        if reset_interval and step % reset_interval == 0 and step < int(config["densify_until_iter"]):
            densifier.reset_opacity(model, optimizer)
            log.info("  step %5d opacity reset", step)

        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                quality = psnr(result.rgb, camera.image)
            history.append({"step": step, "loss": float(loss), "psnr": quality, "count": model.count})
            log.info(
                "  step %5d  loss %.4f  psnr %.2f dB  gaussians %d  (%.0fs)",
                step, float(loss), quality, model.count, time.perf_counter() - started,
            )

        if progress_dir is not None and step % max(iterations // 4, 1) == 0:
            _dump_preview(model, cameras[0], progress_dir / f"train_{step:05d}.png",
                          tile_size, max_per_tile)

    model.eval()
    elapsed = time.perf_counter() - started

    # Held-out quality is not available with this few views, so report the
    # training-view reconstruction quality and say so.
    with torch.no_grad():
        scores = []
        for camera in cameras[:: max(1, len(cameras) // 8)]:
            result = render(
                model, camera.view_matrix, camera.fx, camera.fy, camera.cx, camera.cy,
                camera.width, camera.height, background=background,
                tile_size=tile_size, max_per_tile=max_per_tile, near=0.01, far=1e4,
            )
            scores.append(psnr(result.rgb, camera.image))

    metrics = {
        "iterations": iterations,
        "train_seconds": round(elapsed, 1),
        "gaussians": model.count,
        "scene_extent": round(scene_extent, 4),
        "train_psnr_mean_db": round(float(np.mean(scores)), 2) if scores else None,
        "train_psnr_min_db": round(float(np.min(scores)), 2) if scores else None,
        "history": history,
        "note": "PSNR is measured on training views. There are too few views to hold any out.",
    }
    log.info(
        "splat done: %d gaussians, %.1fs, train PSNR %.2f dB",
        model.count, elapsed, metrics["train_psnr_mean_db"] or 0.0,
    )
    return model, metrics


def _dump_preview(
    model: GaussianModel, camera: TrainCamera, path: Path, tile_size: int, max_per_tile: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        result = render(
            model, camera.view_matrix, camera.fx, camera.fy, camera.cx, camera.cy,
            camera.width, camera.height, tile_size=tile_size, max_per_tile=max_per_tile,
            near=0.01, far=1e4,
        )
        image = (result.rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
