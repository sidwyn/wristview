"""Feature matching, reconstruction, and localization.

The build plan calls for hloc plus LightGlue rather than plain COLMAP SIFT,
because LightGlue is markedly better on low-texture indoor scenes.

One change from stock hloc: hloc hardcodes `cuda if available else cpu`, so on
this machine it runs SuperPoint on the CPU at 876 ms per image against 20 ms
on MPS. That is a 44x penalty, measured. So extraction and matching run here
on MPS, and the results are written in hloc's own HDF5 layout. Reconstruction
and geometric verification are then handed to unmodified hloc and pycolmap.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from ..logging_setup import get

log = get(__name__)

_EXTRACTOR = None
_MATCHER = None


def names_to_pair(name0: str, name1: str) -> str:
    """hloc's pair key. Must match exactly or hloc cannot find the matches."""
    return "/".join((name0.replace("/", "-"), name1.replace("/", "-")))


def get_extractor(device: str, max_keypoints: int = 2048):
    """SuperPoint, cached across calls."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        from lightglue import SuperPoint

        _EXTRACTOR = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        log.info("SuperPoint ready on %s, max %d keypoints", device, max_keypoints)
    return _EXTRACTOR


def get_matcher(device: str):
    """LightGlue, cached across calls.

    Adaptive depth and width pruning are switched on. LightGlue ships them off
    by default, and they cut matching cost roughly fourfold.

    Cost is strongly data-dependent, which is worth knowing before optimizing
    against a benchmark. On this footage at 1024 keypoints on MPS, a pair of
    neighbouring frames costs about 155 ms because the pruning exits early on
    an easy match. A distant loop-closure pair costs about 630 ms because
    confidence never rises and every layer runs. Benchmarking with one
    repeated easy pair understates the real cost by four times.
    """
    global _MATCHER
    if _MATCHER is None:
        from lightglue import LightGlue

        _MATCHER = (
            LightGlue(features="superpoint", depth_confidence=0.95, width_confidence=0.99)
            .eval()
            .to(device)
        )
        log.info("LightGlue ready on %s with adaptive pruning", device)
    return _MATCHER


def _load_image(path: Path, device: str) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


def extract_features(
    image_dir: Path,
    names: list[str],
    out_path: Path,
    device: str,
    max_keypoints: int = 1024,
    resize_max: int = 1024,
) -> Path:
    """Run SuperPoint over every image and write hloc's feature HDF5."""
    extractor = get_extractor(device, max_keypoints)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    with h5py.File(str(out_path), "a", libver="latest") as handle, torch.no_grad():
        for index, name in enumerate(names):
            image = _load_image(image_dir / name, device)
            feats = extractor.extract(image[None], resize=resize_max)

            keypoints = feats["keypoints"][0].cpu().numpy().astype(np.float32)
            descriptors = feats["descriptors"][0].cpu().numpy().astype(np.float32).T
            scores = feats["keypoint_scores"][0].cpu().numpy().astype(np.float32)

            group = handle.create_group(name)
            dataset = group.create_dataset("keypoints", data=keypoints)
            # hloc reads this attribute when building the COLMAP database.
            dataset.attrs["uncertainty"] = 1.0
            group.create_dataset("descriptors", data=descriptors)
            group.create_dataset("scores", data=scores)
            group.create_dataset(
                "image_size", data=np.array([image.shape[2], image.shape[1]], dtype=np.int32)
            )
            if (index + 1) % 50 == 0 or index == len(names) - 1:
                log.info("  features: %d/%d images", index + 1, len(names))
    return out_path


def global_descriptors(image_dir: Path, names: list[str], size: int = 20) -> np.ndarray:
    """Cheap image-retrieval descriptors: contrast-normalized tiny images.

    NetVLAD would be stronger, but it is another model on the critical path.
    For a video orbit and demo clips shot in the same room, viewpoint
    similarity tracks appearance closely enough to pick candidate pairs.
    """
    out = np.zeros((len(names), size * size), dtype=np.float32)
    for index, name in enumerate(names):
        image = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = small - small.mean()
        norm = np.linalg.norm(small)
        out[index] = (small / norm).ravel() if norm > 1e-6 else small.ravel()
    return out


def build_pairs(
    names: list[str],
    descriptors: np.ndarray | None = None,
    mode: str = "exhaustive",
    exhaustive_limit: int = 120,
    seq_window: int = 10,
    retrieval_k: int = 15,
) -> list[tuple[str, str]]:
    """Choose which image pairs to match.

    Exhaustive is correct but quadratic. Above `exhaustive_limit` images it
    switches to a temporal window plus retrieval, which is what makes a
    240-frame scan match in minutes rather than an hour.
    """
    count = len(names)
    if mode == "exhaustive" and count <= exhaustive_limit:
        return [(names[i], names[j]) for i in range(count) for j in range(i + 1, count)]

    if descriptors is None:
        raise ValueError("retrieval pairing needs global descriptors")

    pairs: set[tuple[int, int]] = set()
    # Temporal neighbours. Consecutive video frames always overlap.
    for i in range(count):
        for j in range(i + 1, min(i + seq_window + 1, count)):
            pairs.add((i, j))

    # Loop closures. Without these an orbit never closes and the
    # reconstruction drifts into two disconnected halves.
    similarity = descriptors @ descriptors.T
    np.fill_diagonal(similarity, -np.inf)
    for i in range(count):
        scores = similarity[i].copy()
        # Temporal neighbours are already paired, so do not spend budget there.
        low = max(0, i - seq_window)
        high = min(count, i + seq_window + 1)
        scores[low:high] = -np.inf
        for j in np.argsort(-scores)[:retrieval_k]:
            if np.isfinite(scores[j]):
                pairs.add((min(i, int(j)), max(i, int(j))))

    return [(names[i], names[j]) for i, j in sorted(pairs)]


def build_query_pairs(
    query_names: list[str],
    reference_names: list[str],
    query_desc: np.ndarray,
    reference_desc: np.ndarray,
    top_k: int = 20,
) -> list[tuple[str, str]]:
    """For each demo frame, pick the most similar scan frames to match against."""
    similarity = query_desc @ reference_desc.T
    pairs = []
    for i, name in enumerate(query_names):
        for j in np.argsort(-similarity[i])[:top_k]:
            pairs.append((name, reference_names[int(j)]))
    return pairs


def match_pairs(
    pairs: list[tuple[str, str]],
    features_path: Path,
    out_path: Path,
    device: str,
    features_ref_path: Path | None = None,
    cache_size: int = 48,
    empty_cache_every: int = 200,
) -> Path:
    """Run LightGlue over the pair list and write hloc's match HDF5."""
    matcher = get_matcher(device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    ref_path = features_ref_path or features_path

    # A strict LRU, kept small on purpose. Holding every image's descriptors
    # resident on MPS made matching get progressively slower, from 448 ms a
    # pair to 1472 ms over twelve minutes, as device memory grew and the
    # machine started swapping. In isolation the same match costs 155 ms.
    # The pair list is sorted, so a small window captures nearly every reuse.
    cache: OrderedDict[tuple[str, str], dict] = OrderedDict()

    def load(path: Path, name: str) -> dict:
        key = (str(path), name)
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            return hit

        with h5py.File(str(path), "r", libver="latest") as handle:
            group = handle[name]
            entry = {
                "keypoints": torch.from_numpy(group["keypoints"][()]).to(device)[None],
                # hloc stores descriptors as (dim, count). LightGlue wants
                # (count, dim), so transpose back on the way in.
                "descriptors": torch.from_numpy(
                    np.ascontiguousarray(group["descriptors"][()].T)
                ).to(device)[None],
                "image_size": torch.from_numpy(
                    group["image_size"][()].astype(np.float32)
                ).to(device)[None],
            }
        cache[key] = entry
        while len(cache) > cache_size:
            cache.popitem(last=False)
        return entry

    # Matching results are buffered and written in one pass at the end.
    # Creating two small HDF5 datasets per pair inside the loop cost about
    # 285 ms a pair on this machine, against 57 ms for the match itself.
    # h5py's per-dataset metadata write dominated everything.
    buffer: list[tuple[str, np.ndarray, np.ndarray]] = []
    seen: set[str] = set()
    started = time.perf_counter()

    with torch.no_grad():
        for index, (name0, name1) in enumerate(pairs):
            key = names_to_pair(name0, name1)
            if key in seen:
                continue
            seen.add(key)

            feats0 = load(features_path, name0)
            feats1 = load(ref_path, name1)
            if feats0["keypoints"].shape[1] < 2 or feats1["keypoints"].shape[1] < 2:
                continue

            result = matcher({"image0": feats0, "image1": feats1})
            matches0 = result["matches0"][0].cpu().short().numpy()
            scores0 = result["matching_scores0"][0].cpu().numpy().astype(np.float16)
            buffer.append((key, matches0, scores0))

            # Release the transient allocations from the attention layers.
            # Without this the MPS caching allocator keeps growing.
            if empty_cache_every and (index + 1) % empty_cache_every == 0:
                if device == "mps":
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()

            if (index + 1) % 250 == 0 or index == len(pairs) - 1:
                elapsed = time.perf_counter() - started
                rate = (index + 1) / max(elapsed, 1e-6)
                log.info(
                    "  matches: %d/%d pairs (%.0f ms/pair, %.0fs left)",
                    index + 1, len(pairs), 1000.0 / max(rate, 1e-9),
                    (len(pairs) - index - 1) / max(rate, 1e-9),
                )

    with h5py.File(str(out_path), "a", libver="latest") as handle:
        for key, matches0, scores0 in buffer:
            group = handle.create_group(key)
            group.create_dataset("matches0", data=matches0)
            group.create_dataset("matching_scores0", data=scores0)

    log.info("wrote %d pair matches to %s", len(buffer), out_path.name)
    return out_path


def write_pairs_file(pairs: list[tuple[str, str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{a} {b}" for a, b in pairs))
    return path


def run_reconstruction(
    sfm_dir: Path,
    image_dir: Path,
    pairs_path: Path,
    features_path: Path,
    matches_path: Path,
    intrinsics: dict | None = None,
    image_list: list[str] | None = None,
):
    """Incremental mapping through hloc and pycolmap.

    When the focal length came from EXIF it is trusted and fixed, which
    removes a class of silent focal drift. When it is only a guess, the camera
    is handed to COLMAP as a SIMPLE_RADIAL model with the guess as a starting
    point and refined during bundle adjustment. Freezing an unmeasured focal
    is worse than refining it: it bakes the error into every pose.
    """
    import pycolmap
    from hloc import reconstruction as hloc_reconstruction

    image_options: dict = {}
    if intrinsics is not None:
        if intrinsics.get("self_calibrate"):
            image_options = {
                "camera_model": intrinsics.get("colmap_camera_model", "SIMPLE_RADIAL"),
                # A single focal prior, no distortion prior. COLMAP refines it.
                "camera_params": ",".join(
                    str(float(v))
                    for v in (intrinsics["fx"], intrinsics["cx"], intrinsics["cy"], 0.0)
                ),
            }
        else:
            image_options = {
                "camera_model": "PINHOLE",
                "camera_params": ",".join(
                    str(float(intrinsics[key])) for key in ("fx", "fy", "cx", "cy")
                ),
            }

    sfm_dir.mkdir(parents=True, exist_ok=True)
    return hloc_reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=image_dir,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_list=image_list,
        image_options=image_options or None,
        verbose=False,
    )


def refined_camera(reconstruction) -> dict:
    """Read the camera COLMAP actually converged on.

    Returns pinhole intrinsics plus whatever distortion the model carried, so
    the caller can see how far the self-calibration moved from the prior and
    whether the pinhole approximation downstream is safe.
    """
    cameras = list(reconstruction.cameras.values())
    if not cameras:
        return {}
    camera = cameras[0]
    params = {
        name: float(value)
        for name, value in zip(camera.params_info.split(", "), camera.params, strict=False)
    }

    focal_x = params.get("fx", params.get("f", 0.0))
    focal_y = params.get("fy", params.get("f", focal_x))
    distortion = {k: v for k, v in params.items() if k.startswith("k") or k.startswith("p")}

    return {
        "model": camera.model.name if hasattr(camera.model, "name") else str(camera.model),
        "width": int(camera.width),
        "height": int(camera.height),
        "fx": focal_x,
        "fy": focal_y,
        "cx": params.get("cx", camera.width / 2.0),
        "cy": params.get("cy", camera.height / 2.0),
        "distortion": distortion,
        "params": params,
    }


def reconstruction_poses(reconstruction) -> dict[str, np.ndarray]:
    """Camera-to-world pose per registered image name."""
    from ..geometry import invert_pose

    poses: dict[str, np.ndarray] = {}
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        # `cam_from_world` is a method in pycolmap 4.1, while `has_pose`,
        # `name`, and `points2D` alongside it are properties. Reading it as a
        # property yields the bound method and fails on `.matrix()`.
        cam_from_world = np.asarray(image.cam_from_world().matrix())
        world_from_cam = np.eye(4)
        world_from_cam[:3, :4] = cam_from_world
        poses[image.name] = invert_pose(world_from_cam)
    return poses


def reconstruction_points(reconstruction) -> tuple[np.ndarray, np.ndarray]:
    """All 3D points and their colors, as (N, 3) float and (N, 3) in [0, 1]."""
    points, colors = [], []
    for point in reconstruction.points3D.values():
        points.append(point.xyz)
        colors.append(np.asarray(point.color, dtype=np.float64) / 255.0)
    if not points:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.asarray(points), np.asarray(colors)


def build_keypoint_to_point3d(reconstruction) -> dict[str, dict[int, int]]:
    """Map image name and keypoint index to the 3D point id it observes.

    This is what turns a 2D-2D match against a scan frame into the 2D-3D
    correspondence that absolute pose estimation needs.
    """
    lookup: dict[str, dict[int, int]] = {}
    for image in reconstruction.images.values():
        table: dict[int, int] = {}
        for index, point2d in enumerate(image.points2D):
            if point2d.has_point3D():
                table[index] = int(point2d.point3D_id)
        lookup[image.name] = table
    return lookup


def localize_frame(
    query_name: str,
    query_keypoints: np.ndarray,
    reference_names: list[str],
    matches_path: Path,
    lookup: dict[str, dict[int, int]],
    reconstruction,
    camera,
    ransac_max_error_px: float,
    min_inliers: int,
) -> tuple[np.ndarray | None, dict]:
    """Estimate one demo frame's pose against the scan reconstruction.

    Returns the camera-to-world pose and diagnostics, or None if it failed.
    """
    import pycolmap

    from ..geometry import invert_pose

    points2d: list[np.ndarray] = []
    points3d: list[np.ndarray] = []
    seen: set[int] = set()

    with h5py.File(str(matches_path), "r", libver="latest") as handle:
        for reference in reference_names:
            key = names_to_pair(query_name, reference)
            if key not in handle:
                continue
            matches0 = handle[key]["matches0"][()]
            table = lookup.get(reference, {})
            if not table:
                continue
            query_indices = np.nonzero(matches0 != -1)[0]
            for query_index in query_indices:
                point3d_id = table.get(int(matches0[query_index]))
                # One 3D point per query keypoint. A keypoint matched in five
                # scan frames must not vote five times.
                if point3d_id is None or int(query_index) in seen:
                    continue
                seen.add(int(query_index))
                points2d.append(query_keypoints[query_index])
                points3d.append(reconstruction.points3D[point3d_id].xyz)

    diagnostics = {"correspondences": len(points2d), "inliers": 0}
    if len(points2d) < min_inliers:
        return None, diagnostics

    options = pycolmap.AbsolutePoseEstimationOptions()
    options.ransac.max_error = float(ransac_max_error_px)

    result = pycolmap.estimate_and_refine_absolute_pose(
        np.asarray(points2d, dtype=np.float64),
        np.asarray(points3d, dtype=np.float64),
        camera,
        estimation_options=options,
    )
    if result is None:
        return None, diagnostics

    inliers = int(np.sum(result["inliers"]))
    diagnostics["inliers"] = inliers
    if inliers < min_inliers:
        return None, diagnostics

    world_from_cam = np.eye(4)
    world_from_cam[:3, :4] = result["cam_from_world"].matrix()
    return invert_pose(world_from_cam), diagnostics


def load_keypoints(features_path: Path, name: str) -> np.ndarray:
    with h5py.File(str(features_path), "r", libver="latest") as handle:
        return handle[name]["keypoints"][()]


def features_cover(path: Path, names: list[str]) -> bool:
    """True when an existing feature file already holds every image.

    Matching a 172-frame scan costs about twenty minutes, so re-running a
    later part of Stage 1, retraining the splat for instance, must not throw
    that away.
    """
    if not path.exists():
        return False
    try:
        with h5py.File(str(path), "r", libver="latest") as handle:
            return all(name in handle for name in names)
    except OSError:
        return False


def matches_cover(path: Path, pairs: list[tuple[str, str]]) -> bool:
    """True when an existing match file already holds every pair."""
    if not path.exists():
        return False
    try:
        with h5py.File(str(path), "r", libver="latest") as handle:
            for name0, name1 in pairs:
                if names_to_pair(name0, name1) in handle:
                    continue
                if names_to_pair(name1, name0) in handle:
                    continue
                return False
        return True
    except OSError:
        return False
