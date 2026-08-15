"""Build a dense metric point cloud from posed scan frames.

Why this exists. COLMAP's sparse cloud is far too thin to render from: on a
desk scan it gives 13,667 points, and a wrist camera 20 cm from the surface
sees 0.1 percent pixel coverage. COLMAP's dense stage needs CUDA. A Gaussian
splat covers the frame properly but is slow to train and, on Apple Silicon,
rests on a hand-written rasterizer.

So: run monocular depth on each scan frame, anchor it to metric using the
sparse points that frame actually observes, and back-project every pixel. The
result is geometrically correct where the sparse cloud agrees, and dense
enough to render.

The anchoring is the part that matters. Depth-Anything returns relative
inverse depth, defined up to an affine transform. Each scan frame already has
tens to hundreds of triangulated points with known metric depth, so the
transform is over-determined and solved per frame by least squares. No frame
inherits another frame's scale.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..camera import Intrinsics
from ..geometry import transform_points
from ..logging_setup import get

log = get(__name__)


def _fit_affine_depth(
    relative: np.ndarray,
    pixels: np.ndarray,
    metric_depth: np.ndarray,
    min_points: int = 20,
) -> tuple[np.ndarray, dict] | None:
    """Map relative inverse depth to metres using known 3D points.

    Solves `1 / metric ~= a * relative + b` on the pixels where a triangulated
    point lands, then applies it everywhere.
    """
    height, width = relative.shape
    xs = np.clip(np.round(pixels[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.round(pixels[:, 1]).astype(int), 0, height - 1)

    sampled = relative[ys, xs].astype(np.float64)
    target = 1.0 / np.maximum(metric_depth, 1e-6)

    good = np.isfinite(sampled) & np.isfinite(target) & (metric_depth > 1e-3)
    if int(good.sum()) < min_points:
        return None
    sampled, target = sampled[good], target[good]

    # Trim the worst decile each side: triangulated points include outliers,
    # and one bad anchor drags the whole frame's depth.
    keep = (target >= np.percentile(target, 5)) & (target <= np.percentile(target, 95))
    if int(keep.sum()) >= min_points:
        sampled, target = sampled[keep], target[keep]

    design = np.stack([sampled, np.ones_like(sampled)], axis=1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    slope, intercept = float(solution[0]), float(solution[1])

    predicted = design @ solution
    correlation = float(np.corrcoef(predicted, target)[0, 1]) if len(target) > 2 else 0.0

    inverse = slope * relative + intercept
    metric = np.where(inverse > 1e-4, 1.0 / np.maximum(inverse, 1e-4), 0.0)
    return metric.astype(np.float32), {
        "anchors": int(len(target)),
        "correlation": round(correlation, 4),
        "slope": slope,
        "intercept": intercept,
    }


def build(
    frames_dir: Path,
    frame_names: list[str],
    poses: dict[str, np.ndarray],
    observations: dict[str, tuple[np.ndarray, np.ndarray]],
    intrinsics: Intrinsics,
    device: str,
    work_long_side: int = 640,
    max_frames: int = 60,
    points_per_frame: int = 25000,
    min_correlation: float = 0.5,
    near_m: float = 0.05,
    far_m: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fuse posed monocular depth into one metric world point cloud.

    `observations` maps a frame name to the pixels and metric depths of the
    triangulated points it sees. Those are the per-frame anchors.
    """
    from .depth import DepthAnythingV2

    usable = [n for n in frame_names if n in poses and n in observations]
    if not usable:
        raise ValueError("no scan frame has both a pose and triangulated observations")
    if len(usable) > max_frames:
        picked = np.linspace(0, len(usable) - 1, max_frames).astype(int)
        usable = [usable[i] for i in picked]

    model = DepthAnythingV2(device)
    scale = min(1.0, work_long_side / max(intrinsics.width, intrinsics.height))
    width = max(32, int(round(intrinsics.width * scale)))
    height = max(32, int(round(intrinsics.height * scale)))
    work = intrinsics.scaled(width, height)

    rng = np.random.default_rng(0)
    all_points, all_colors = [], []
    kept, skipped = 0, 0
    correlations = []

    for index, name in enumerate(usable):
        image = cv2.imread(str(frames_dir / name))
        if image is None:
            continue
        small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        relative = model.predict(small)

        pixels, depths = observations[name]
        fit = _fit_affine_depth(relative, pixels * scale, depths)
        if fit is None:
            skipped += 1
            continue
        metric, stats = fit
        correlations.append(stats["correlation"])
        if stats["correlation"] < min_correlation:
            # A weak fit means the monocular depth disagrees with the geometry
            # this frame actually observed, so its pixels are not trustworthy.
            skipped += 1
            continue

        valid = (metric > near_m) & (metric < far_m)
        ys, xs = np.nonzero(valid)
        if len(ys) == 0:
            skipped += 1
            continue
        if len(ys) > points_per_frame:
            choose = rng.choice(len(ys), points_per_frame, replace=False)
            ys, xs = ys[choose], xs[choose]

        cam = work.unproject(
            np.stack([xs, ys], axis=1).astype(np.float64), metric[ys, xs].astype(np.float64)
        )
        all_points.append(transform_points(poses[name], cam))
        all_colors.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)[ys, xs] / 255.0)
        kept += 1

        if (index + 1) % 10 == 0:
            log.info("  dense cloud: %d/%d frames, %d points so far",
                     index + 1, len(usable), sum(len(p) for p in all_points))

    if not all_points:
        raise ValueError("no scan frame produced a usable depth fit")

    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    report = {
        "frames_used": kept,
        "frames_skipped": skipped,
        "points": int(len(points)),
        "median_depth_correlation": round(float(np.median(correlations)), 4) if correlations else 0.0,
        "min_correlation_required": min_correlation,
    }
    log.info(
        "dense cloud: %d points from %d frames (%d skipped), median depth "
        "correlation %.3f",
        len(points), kept, skipped, report["median_depth_correlation"],
    )
    return points, colors, report


def observations_from_reconstruction(reconstruction) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per frame, the pixels and metric depths of its triangulated points."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        cam_from_world = np.eye(4)
        cam_from_world[:3, :4] = np.asarray(image.cam_from_world().matrix())

        pixels, depths = [], []
        for point2d in image.points2D:
            if not point2d.has_point3D():
                continue
            xyz = reconstruction.points3D[point2d.point3D_id].xyz
            depth = float((cam_from_world[:3, :3] @ xyz + cam_from_world[:3, 3])[2])
            if depth <= 1e-3:
                continue
            pixels.append(point2d.xy)
            depths.append(depth)
        if len(pixels) >= 20:
            out[image.name] = (np.asarray(pixels), np.asarray(depths))
    return out


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """One point per voxel. Overlapping frames pile up points in the same place."""
    if voxel_m <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[index], colors[index]


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    with open(path, "w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, rgb, strict=True):
            handle.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )
    return path


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = Path(path).read_text().split("\n")
    start = lines.index("end_header") + 1
    data = np.array(
        [[float(v) for v in line.split()] for line in lines[start:] if line.strip()]
    )
    if data.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return data[:, :3], data[:, 3:6] / 255.0
