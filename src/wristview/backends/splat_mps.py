"""3D Gaussian splatting on Apple Silicon.

Why this file exists. The build plan names `gsplat`, which is CUDA-only, and
suggests Brush or gsplat-mlx instead. On this machine gsplat-mlx fails to
compile its Metal extension against the current MLX, and Brush needs a Rust
toolchain plus a CLI that cannot render from an arbitrary camera pose. Stage 5
must render from wrist poses the operator never walked through, so the
renderer has to be callable per pose from Python. That rules out driving an
external binary and leaves a native implementation.

So this is a tile-based differentiable rasterizer written in plain PyTorch. It
runs on MPS. It trains and it renders, and Stage 5 calls the same `render`
function that Stage 1 trains against.

The rasterizer follows Kerbl et al. 2023:
  project each Gaussian to a 2D conic, sort by depth, bucket into tiles,
  alpha-composite front to back.

The one structural difference from the CUDA original is the compositing loop.
CUDA walks each tile's list sequentially with early termination. Here the tile
lists are padded to a fixed depth and composited with an exclusive cumulative
product, because that is one parallel kernel instead of a serial loop and MPS
rewards it. The cost is a cap on Gaussians per tile, which is
`max_per_tile` below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..logging_setup import get

log = get(__name__)

# Real spherical harmonic basis constants, bands 0 to 2.
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)


def sh_bands(degree: int) -> int:
    """Number of SH coefficients for a degree."""
    return (degree + 1) ** 2


def eval_sh(sh: torch.Tensor, degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate spherical harmonics.

    `sh` is (N, bands, 3), `dirs` is (N, 3) unit view directions. Returns
    (N, 3) linear RGB before the 0.5 offset and clamp.
    """
    result = SH_C0 * sh[:, 0]
    if degree >= 1:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
        if degree >= 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + SH_C2[0] * xy * sh[:, 4]
                + SH_C2[1] * yz * sh[:, 5]
                + SH_C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
                + SH_C2[3] * xz * sh[:, 7]
                + SH_C2[4] * (xx - yy) * sh[:, 8]
            )
    return result


def quat_to_rotmat_torch(quat: torch.Tensor) -> torch.Tensor:
    """(N, 4) `(w, x, y, z)` quaternions to (N, 3, 3) rotation matrices."""
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


@dataclass
class RenderResult:
    """One rendered view."""

    rgb: torch.Tensor      # (H, W, 3) in [0, 1]
    alpha: torch.Tensor    # (H, W) accumulated coverage
    depth: torch.Tensor    # (H, W) alpha-weighted mean depth in scene units
    visible: torch.Tensor  # (N,) bool, which Gaussians survived culling


class GaussianModel(torch.nn.Module):
    """A set of 3D Gaussians with the standard 3DGS parameterization."""

    def __init__(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        scales: torch.Tensor | None = None,
        sh_degree: int = 2,
        device: str = "mps",
    ):
        super().__init__()
        self.sh_degree = int(sh_degree)
        self.active_sh_degree = 0  # Raised during training, one band at a time.
        self.device_str = device

        num = means.shape[0]
        means = means.to(device=device, dtype=torch.float32)

        if scales is None:
            scales = _knn_scale_init(means)
        scales = scales.to(device=device, dtype=torch.float32)

        bands = sh_bands(self.sh_degree)
        sh = torch.zeros(num, bands, 3, device=device, dtype=torch.float32)
        # Store the DC band as an SH coefficient, not as raw colour.
        sh[:, 0] = (colors.to(device=device, dtype=torch.float32) - 0.5) / SH_C0

        self.means = torch.nn.Parameter(means)
        self.log_scales = torch.nn.Parameter(torch.log(scales.clamp_min(1e-7)))
        quats = torch.zeros(num, 4, device=device, dtype=torch.float32)
        quats[:, 0] = 1.0
        self.quats = torch.nn.Parameter(quats)
        self.opacity_logit = torch.nn.Parameter(
            torch.logit(torch.full((num,), 0.1, device=device, dtype=torch.float32))
        )
        self.sh = torch.nn.Parameter(sh)

        # Densification statistics, reset each cycle.
        self.register_buffer("grad_accum", torch.zeros(num, device=device))
        self.register_buffer("grad_count", torch.zeros(num, device=device))
        self.register_buffer("max_radii", torch.zeros(num, device=device))

    @property
    def count(self) -> int:
        return self.means.shape[0]

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logit)

    def raise_sh_degree(self) -> None:
        if self.active_sh_degree < self.sh_degree:
            self.active_sh_degree += 1

    def state(self) -> dict:
        """Serializable state. Kept plain so a `.pt` file loads without this class."""
        return {
            "means": self.means.detach().cpu(),
            "log_scales": self.log_scales.detach().cpu(),
            "quats": self.quats.detach().cpu(),
            "opacity_logit": self.opacity_logit.detach().cpu(),
            "sh": self.sh.detach().cpu(),
            "sh_degree": self.sh_degree,
            "active_sh_degree": self.active_sh_degree,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state(), path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "mps") -> GaussianModel:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        num = payload["means"].shape[0]
        model = cls(
            means=payload["means"],
            colors=torch.zeros(num, 3),
            scales=torch.ones(num, 3),
            sh_degree=int(payload["sh_degree"]),
            device=device,
        )
        with torch.no_grad():
            model.means.copy_(payload["means"].to(device))
            model.log_scales.copy_(payload["log_scales"].to(device))
            model.quats.copy_(payload["quats"].to(device))
            model.opacity_logit.copy_(payload["opacity_logit"].to(device))
            model.sh.copy_(payload["sh"].to(device))
        model.active_sh_degree = int(payload.get("active_sh_degree", model.sh_degree))
        return model

    def export_ply(self, path: str | Path, max_points: int | None = None) -> Path:
        """Write the Gaussian centres as a coloured point cloud.

        This is the artifact a human looks at to check Stage 1. It is not a
        full 3DGS ply, it is the centres with their DC colour.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            means = self.means.detach().cpu().numpy()
            rgb = (SH_C0 * self.sh[:, 0].detach().cpu().numpy() + 0.5).clip(0, 1)
            opacity = self.opacity.detach().cpu().numpy()

        keep = opacity > 0.02
        means, rgb = means[keep], rgb[keep]
        if max_points and len(means) > max_points:
            idx = np.random.default_rng(0).choice(len(means), max_points, replace=False)
            means, rgb = means[idx], rgb[idx]

        colors = (rgb * 255).astype(np.uint8)
        with open(path, "w") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(means)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            handle.write("end_header\n")
            for point, color in zip(means, colors, strict=True):
                handle.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{color[0]} {color[1]} {color[2]}\n"
                )
        return path


def _knn_scale_init(means: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Initialize each Gaussian's size from the distance to its neighbours.

    Chunked, because the full pairwise distance matrix is quadratic and a
    60k-point cloud would need 14 GB.
    """
    num = means.shape[0]
    device = means.device
    out = torch.empty(num, device=device)
    chunk = 2048
    for start in range(0, num, chunk):
        stop = min(start + chunk, num)
        dist = torch.cdist(means[start:stop], means)
        # The first neighbour is the point itself, at distance zero.
        knn = dist.topk(min(k + 1, num), largest=False).values[:, 1:]
        out[start:stop] = knn.mean(dim=1)
    scale = out.clamp(1e-4, 1.0)
    return scale[:, None].repeat(1, 3)


def _project_gaussians(
    model: GaussianModel,
    view_matrix: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float,
    far: float,
):
    """Project every Gaussian to a 2D conic. Returns the visible subset."""
    rot_wc = view_matrix[:3, :3]
    trans_wc = view_matrix[:3, 3]

    means_cam = model.means @ rot_wc.T + trans_wc
    depth = means_cam[:, 2]

    inside = (depth > near) & (depth < far)
    if not bool(inside.any()):
        return None

    # 3D covariance in world space, then rotated into the camera.
    rotation = quat_to_rotmat_torch(model.quats)
    scale = model.scales
    scaled = rotation * scale[:, None, :]           # R @ diag(s)
    cov3d = scaled @ scaled.transpose(1, 2)          # R S S^T R^T
    cov_cam = rot_wc @ cov3d @ rot_wc.T

    # Affine approximation of the perspective projection, Zwicker et al. 2001.
    safe_depth = depth.clamp_min(near)
    jac = torch.zeros(model.count, 2, 3, device=means_cam.device, dtype=means_cam.dtype)
    jac[:, 0, 0] = fx / safe_depth
    jac[:, 0, 2] = -fx * means_cam[:, 0] / (safe_depth * safe_depth)
    jac[:, 1, 1] = fy / safe_depth
    jac[:, 1, 2] = -fy * means_cam[:, 1] / (safe_depth * safe_depth)

    cov2d = jac @ cov_cam @ jac.transpose(1, 2)
    # Dilate by a third of a pixel. Without this, Gaussians smaller than a
    # pixel alias badly and their gradients go to zero.
    cov2d[:, 0, 0] = cov2d[:, 0, 0] + 0.3
    cov2d[:, 1, 1] = cov2d[:, 1, 1] + 0.3

    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    inside = inside & (det > 1e-9)
    det_safe = det.clamp_min(1e-9)

    # Inverse of a 2x2, written out. This is the conic.
    conic = torch.stack(
        [
            cov2d[:, 1, 1] / det_safe,
            -cov2d[:, 0, 1] / det_safe,
            cov2d[:, 0, 0] / det_safe,
        ],
        dim=-1,
    )

    uv = torch.stack(
        [fx * means_cam[:, 0] / safe_depth + cx, fy * means_cam[:, 1] / safe_depth + cy],
        dim=-1,
    )

    # Three sigma along the major axis bounds the visible footprint.
    trace = cov2d[:, 0, 0] + cov2d[:, 1, 1]
    gap = torch.sqrt((0.25 * (cov2d[:, 0, 0] - cov2d[:, 1, 1]) ** 2 + cov2d[:, 0, 1] ** 2).clamp_min(0))
    eig_max = (0.5 * trace + gap).clamp_min(1e-9)
    radius = 3.0 * torch.sqrt(eig_max)

    inside = (
        inside
        & (uv[:, 0] + radius > 0)
        & (uv[:, 0] - radius < width)
        & (uv[:, 1] + radius > 0)
        & (uv[:, 1] - radius < height)
    )
    if not bool(inside.any()):
        return None

    return uv, conic, radius, depth, means_cam, inside


def render(
    model: GaussianModel,
    view_matrix: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    background: torch.Tensor | None = None,
    tile_size: int = 16,
    max_per_tile: int = 128,
    near: float = 0.01,
    far: float = 100.0,
    max_pixels_per_chunk: int = 262144,
) -> RenderResult:
    """Rasterize the Gaussians into one image.

    `view_matrix` is the 4x4 world-to-camera transform. Camera axes follow
    OpenCV: x right, y down, z forward.

    Differentiable with respect to every model parameter, so Stage 1 trains
    through it and Stage 5 calls it with `torch.no_grad`.
    """
    device = model.means.device
    dtype = model.means.dtype
    if background is None:
        background = torch.zeros(3, device=device, dtype=dtype)

    projected = _project_gaussians(
        model, view_matrix, fx, fy, cx, cy, width, height, near, far
    )
    if projected is None:
        return RenderResult(
            rgb=background.expand(height, width, 3).clone(),
            alpha=torch.zeros(height, width, device=device, dtype=dtype),
            depth=torch.full((height, width), far, device=device, dtype=dtype),
            visible=torch.zeros(model.count, dtype=torch.bool, device=device),
        )

    uv, conic, radius, depth, means_cam, inside = projected
    keep_idx = torch.nonzero(inside, as_tuple=False).squeeze(1)

    uv_v = uv[keep_idx]
    conic_v = conic[keep_idx]
    radius_v = radius[keep_idx]
    depth_v = depth[keep_idx]
    opacity_v = model.opacity[keep_idx]

    # View-dependent colour. The direction runs from the camera to the Gaussian.
    cam_center = -view_matrix[:3, :3].T @ view_matrix[:3, 3]
    dirs = F.normalize(model.means[keep_idx] - cam_center[None, :], dim=-1)
    colors_v = (eval_sh(model.sh[keep_idx], model.active_sh_degree, dirs) + 0.5).clamp(0.0, 1.0)

    # Retain the screen-space gradient. Densification uses its magnitude to
    # decide which Gaussians are under-fitting their region of the image.
    if model.training and uv_v.requires_grad:
        uv_v.retain_grad()

    # Sort front to back once. Every tile inherits this order.
    order = torch.argsort(depth_v)
    uv_s, conic_s = uv_v[order], conic_v[order]
    radius_s, depth_s = radius_v[order], depth_v[order]
    opacity_s, colors_s = opacity_v[order], colors_v[order]
    num_visible = uv_s.shape[0]

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    num_tiles = tiles_x * tiles_y

    with torch.no_grad():
        tx0 = ((uv_s[:, 0] - radius_s) / tile_size).floor().clamp(0, tiles_x - 1).to(torch.int64)
        tx1 = ((uv_s[:, 0] + radius_s) / tile_size).floor().clamp(0, tiles_x - 1).to(torch.int64)
        ty0 = ((uv_s[:, 1] - radius_s) / tile_size).floor().clamp(0, tiles_y - 1).to(torch.int64)
        ty1 = ((uv_s[:, 1] + radius_s) / tile_size).floor().clamp(0, tiles_y - 1).to(torch.int64)

        span_x = (tx1 - tx0 + 1)
        span_y = (ty1 - ty0 + 1)
        span = span_x * span_y

        # A Gaussian that covers most of the frame is nearly always a
        # degenerate one mid-optimization. Clamping its span keeps the pair
        # list bounded instead of letting one blob cost a gigabyte.
        max_span = max(4, num_tiles // 4)
        too_wide = span > max_span
        if bool(too_wide.any()):
            tx1 = torch.where(too_wide, tx0, tx1)
            ty1 = torch.where(too_wide, ty0, ty1)
            span_x = (tx1 - tx0 + 1)
            span_y = (ty1 - ty0 + 1)
            span = span_x * span_y

        gauss_of_pair = torch.repeat_interleave(
            torch.arange(num_visible, device=device), span
        )
        within = torch.arange(span.sum(), device=device) - torch.repeat_interleave(
            torch.cumsum(span, 0) - span, span
        )
        offset_x = within % span_x[gauss_of_pair]
        offset_y = within // span_x[gauss_of_pair]
        tile_of_pair = (ty0[gauss_of_pair] + offset_y) * tiles_x + (tx0[gauss_of_pair] + offset_x)

        # One sort on a composite key keeps depth order inside each tile
        # without relying on a stable sort, which MPS does not guarantee.
        key = tile_of_pair * num_visible + gauss_of_pair
        pair_order = torch.argsort(key)
        tile_sorted = tile_of_pair[pair_order]
        gauss_sorted = gauss_of_pair[pair_order]

        counts = torch.bincount(tile_sorted, minlength=num_tiles)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(tile_sorted.shape[0], device=device) - starts[tile_sorted]

        # Pad every tile's list to `max_per_tile`. Slot -1 means empty.
        slots = torch.full((num_tiles, max_per_tile), -1, dtype=torch.int64, device=device)
        fits = rank < max_per_tile
        slots[tile_sorted[fits], rank[fits]] = gauss_sorted[fits]

        overflow = int((counts.clamp_min(max_per_tile) - max_per_tile).sum().item())

    # Pixel coordinates, grouped by tile.
    tile_ids = torch.arange(num_tiles, device=device)
    tile_col = (tile_ids % tiles_x) * tile_size
    tile_row = (tile_ids // tiles_x) * tile_size
    local = torch.arange(tile_size, device=device, dtype=dtype)
    local_y, local_x = torch.meshgrid(local, local, indexing="ij")
    pix_x = tile_col[:, None].to(dtype) + local_x.reshape(1, -1) + 0.5
    pix_y = tile_row[:, None].to(dtype) + local_y.reshape(1, -1) + 0.5

    rgb_flat = torch.zeros(num_tiles, tile_size * tile_size, 3, device=device, dtype=dtype)
    alpha_flat = torch.zeros(num_tiles, tile_size * tile_size, device=device, dtype=dtype)
    depth_flat = torch.zeros(num_tiles, tile_size * tile_size, device=device, dtype=dtype)

    # Chunk over tiles so peak memory stays bounded regardless of resolution.
    pixels_per_tile = tile_size * tile_size
    tiles_per_chunk = max(1, max_pixels_per_chunk // (pixels_per_tile * max(max_per_tile // 32, 1)))

    for start in range(0, num_tiles, tiles_per_chunk):
        stop = min(start + tiles_per_chunk, num_tiles)
        chunk_slots = slots[start:stop]                     # (T, K)
        valid = chunk_slots >= 0
        if not bool(valid.any()):
            continue
        safe_slots = chunk_slots.clamp_min(0)

        mean_c = uv_s[safe_slots]                            # (T, K, 2)
        conic_c = conic_s[safe_slots]                        # (T, K, 3)
        op_c = opacity_s[safe_slots]                         # (T, K)
        col_c = colors_s[safe_slots]                         # (T, K, 3)
        dep_c = depth_s[safe_slots]                          # (T, K)

        dx = pix_x[start:stop, :, None] - mean_c[:, None, :, 0]   # (T, P, K)
        dy = pix_y[start:stop, :, None] - mean_c[:, None, :, 1]

        power = -0.5 * (
            conic_c[:, None, :, 0] * dx * dx
            + 2.0 * conic_c[:, None, :, 1] * dx * dy
            + conic_c[:, None, :, 2] * dy * dy
        )
        alpha = op_c[:, None, :] * torch.exp(power.clamp(max=0.0))
        alpha = alpha * valid[:, None, :].to(dtype)
        # Cap at 0.99 so a single Gaussian can never fully occlude and kill
        # the gradient path to everything behind it.
        alpha = alpha.clamp(0.0, 0.99)

        one_minus = 1.0 - alpha
        transmittance = torch.cat(
            [
                torch.ones_like(one_minus[:, :, :1]),
                torch.cumprod(one_minus, dim=2)[:, :, :-1],
            ],
            dim=2,
        )
        weight = alpha * transmittance                       # (T, P, K)

        rgb_flat[start:stop] = torch.einsum("tpk,tkc->tpc", weight, col_c)
        alpha_flat[start:stop] = weight.sum(dim=2)
        depth_flat[start:stop] = torch.einsum("tpk,tk->tp", weight, dep_c)

    # Un-tile back to an image.
    rgb_img = _untile(rgb_flat, tiles_x, tiles_y, tile_size, height, width, channels=3)
    alpha_img = _untile(alpha_flat[..., None], tiles_x, tiles_y, tile_size, height, width, 1)[..., 0]
    depth_img = _untile(depth_flat[..., None], tiles_x, tiles_y, tile_size, height, width, 1)[..., 0]

    rgb_img = rgb_img + (1.0 - alpha_img)[..., None] * background[None, None, :]
    depth_img = torch.where(alpha_img > 1e-4, depth_img / alpha_img.clamp_min(1e-6),
                            torch.full_like(depth_img, far))

    visible_mask = torch.zeros(model.count, dtype=torch.bool, device=device)
    visible_mask[keep_idx] = True

    result = RenderResult(rgb=rgb_img, alpha=alpha_img, depth=depth_img, visible=visible_mask)
    # Stashed for the densifier, which needs the screen-space gradient and the
    # index mapping back to model parameters.
    result.__dict__["_uv"] = uv_v
    result.__dict__["_keep_idx"] = keep_idx
    result.__dict__["_radius"] = radius_v
    result.__dict__["_overflow"] = overflow
    return result


def _untile(
    flat: torch.Tensor, tiles_x: int, tiles_y: int, tile_size: int,
    height: int, width: int, channels: int,
) -> torch.Tensor:
    """(num_tiles, tile_size^2, C) back to (H, W, C), cropping the padding."""
    grid = flat.reshape(tiles_y, tiles_x, tile_size, tile_size, channels)
    grid = grid.permute(0, 2, 1, 3, 4).reshape(tiles_y * tile_size, tiles_x * tile_size, channels)
    return grid[:height, :width]


class Densifier:
    """Adaptive density control: clone, split, prune, and reset opacity.

    Straight from Kerbl et al. Section 5.2. Gaussians whose screen-space
    position gradient is large are under-fitting their part of the image, so
    small ones are cloned and large ones are split.
    """

    def __init__(
        self,
        grad_threshold: float = 0.0004,
        prune_opacity: float = 0.005,
        max_gaussians: int = 400000,
        percent_dense: float = 0.01,
        scene_extent: float = 1.0,
    ):
        self.grad_threshold = grad_threshold
        self.prune_opacity = prune_opacity
        self.max_gaussians = max_gaussians
        self.percent_dense = percent_dense
        self.scene_extent = scene_extent

    def accumulate(self, model: GaussianModel, result: RenderResult) -> None:
        """Record screen-space gradient magnitude for the visible Gaussians."""
        uv = result.__dict__.get("_uv")
        keep_idx = result.__dict__.get("_keep_idx")
        radius = result.__dict__.get("_radius")
        if uv is None or uv.grad is None:
            return
        with torch.no_grad():
            grad_norm = uv.grad.norm(dim=-1)
            model.grad_accum.index_add_(0, keep_idx, grad_norm)
            model.grad_count.index_add_(0, keep_idx, torch.ones_like(grad_norm))
            model.max_radii[keep_idx] = torch.maximum(model.max_radii[keep_idx], radius)

    def step(self, model: GaussianModel, optimizer: torch.optim.Optimizer) -> dict:
        """Run one densify-and-prune cycle. Returns counts for the log."""
        with torch.no_grad():
            avg_grad = model.grad_accum / model.grad_count.clamp_min(1.0)
            scales = model.scales
            max_scale = scales.max(dim=1).values

            headroom = max(0, self.max_gaussians - model.count)
            selected = (avg_grad >= self.grad_threshold) & (model.grad_count > 0)

            size_limit = self.percent_dense * self.scene_extent
            clone_mask = selected & (max_scale <= size_limit)
            split_mask = selected & (max_scale > size_limit)

            # Respect the cap. Split costs two Gaussians and returns one.
            budget = headroom
            if int(clone_mask.sum()) > budget:
                clone_mask = _truncate_mask(clone_mask, budget)
            budget -= int(clone_mask.sum())
            if int(split_mask.sum()) > budget:
                split_mask = _truncate_mask(split_mask, max(budget, 0))

            new_tensors = []
            if bool(clone_mask.any()):
                new_tensors.append(self._clone(model, clone_mask))
            if bool(split_mask.any()):
                new_tensors.append(self._split(model, split_mask))

            n_clone = int(clone_mask.sum())
            n_split = int(split_mask.sum())

            if new_tensors:
                _extend_model(model, optimizer, new_tensors)
                # Split replaces its parents with two smaller children.
                keep = torch.ones(model.count, dtype=torch.bool, device=model.means.device)
                padded_split = torch.zeros_like(keep)
                padded_split[: split_mask.shape[0]] = split_mask
                keep &= ~padded_split
                _prune_model(model, optimizer, keep)

            prune_mask = model.opacity < self.prune_opacity
            oversized = model.scales.max(dim=1).values > 0.5 * self.scene_extent
            prune_mask |= oversized
            n_prune = int(prune_mask.sum())
            if n_prune and n_prune < model.count:
                _prune_model(model, optimizer, ~prune_mask)

            model.grad_accum.zero_()
            model.grad_count.zero_()
            model.max_radii.zero_()

        return {"cloned": n_clone, "split": n_split, "pruned": n_prune, "count": model.count}

    def _clone(self, model: GaussianModel, mask: torch.Tensor) -> dict:
        """Copy small under-fitting Gaussians. The copy drifts under gradient."""
        return {
            "means": model.means[mask].clone(),
            "log_scales": model.log_scales[mask].clone(),
            "quats": model.quats[mask].clone(),
            "opacity_logit": model.opacity_logit[mask].clone(),
            "sh": model.sh[mask].clone(),
        }

    def _split(self, model: GaussianModel, mask: torch.Tensor, factor: int = 2) -> dict:
        """Replace a large Gaussian with `factor` smaller ones, sampled inside it."""
        count = int(mask.sum())
        scales = model.scales[mask].repeat(factor, 1)
        rotation = quat_to_rotmat_torch(model.quats[mask]).repeat(factor, 1, 1)
        samples = torch.randn(count * factor, 3, device=scales.device) * scales
        means = model.means[mask].repeat(factor, 1) + torch.bmm(
            rotation, samples[:, :, None]
        ).squeeze(-1)
        return {
            "means": means,
            "log_scales": torch.log(scales / (0.8 * factor)),
            "quats": model.quats[mask].repeat(factor, 1),
            "opacity_logit": model.opacity_logit[mask].repeat(factor),
            "sh": model.sh[mask].repeat(factor, 1, 1),
        }

    def reset_opacity(self, model: GaussianModel, optimizer: torch.optim.Optimizer) -> None:
        """Knock opacity down periodically so floaters get pruned next cycle."""
        with torch.no_grad():
            new_value = torch.logit(
                torch.minimum(model.opacity, torch.full_like(model.opacity, 0.01)).clamp(1e-4, 1 - 1e-4)
            )
            _replace_param_(model, optimizer, "opacity_logit", new_value)


def _truncate_mask(mask: torch.Tensor, budget: int) -> torch.Tensor:
    """Keep at most `budget` set entries of a boolean mask."""
    if budget <= 0:
        return torch.zeros_like(mask)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)[:budget]
    out = torch.zeros_like(mask)
    out[idx] = True
    return out


def _optimizer_state_for(optimizer: torch.optim.Optimizer, param: torch.nn.Parameter):
    return optimizer.state.get(param, None)


def _replace_param_(
    model: GaussianModel, optimizer: torch.optim.Optimizer, name: str, value: torch.Tensor
) -> None:
    """Swap a parameter's data and reset its Adam moments to match."""
    param = getattr(model, name)
    state = _optimizer_state_for(optimizer, param)
    new_param = torch.nn.Parameter(value.contiguous())

    for group in optimizer.param_groups:
        for i, p in enumerate(group["params"]):
            if p is param:
                group["params"][i] = new_param
    if state is not None:
        optimizer.state.pop(param, None)
        optimizer.state[new_param] = {
            "step": state.get("step", torch.tensor(0.0)),
            "exp_avg": torch.zeros_like(new_param),
            "exp_avg_sq": torch.zeros_like(new_param),
        }
    setattr(model, name, new_param)


def _extend_model(
    model: GaussianModel, optimizer: torch.optim.Optimizer, additions: list[dict]
) -> None:
    """Append new Gaussians, keeping the optimizer state aligned."""
    for name in ("means", "log_scales", "quats", "opacity_logit", "sh"):
        extra = torch.cat([add[name] for add in additions], dim=0)
        merged = torch.cat([getattr(model, name).data, extra], dim=0)
        _replace_param_(model, optimizer, name, merged)

    added = sum(add["means"].shape[0] for add in additions)
    device = model.means.device
    for buffer_name in ("grad_accum", "grad_count", "max_radii"):
        old = getattr(model, buffer_name)
        setattr(
            model, buffer_name,
            torch.cat([old, torch.zeros(added, device=device, dtype=old.dtype)], dim=0),
        )


def _prune_model(
    model: GaussianModel, optimizer: torch.optim.Optimizer, keep: torch.Tensor
) -> None:
    """Drop Gaussians, keeping the optimizer state aligned."""
    for name in ("means", "log_scales", "quats", "opacity_logit", "sh"):
        _replace_param_(model, optimizer, name, getattr(model, name).data[keep])
    for buffer_name in ("grad_accum", "grad_count", "max_radii"):
        setattr(model, buffer_name, getattr(model, buffer_name)[keep])
