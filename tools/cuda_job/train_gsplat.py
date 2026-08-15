"""Train a Gaussian splat with gsplat on a CUDA machine, then render wrist views.

Runs on the rented box. Needs only gsplat, torch, numpy and opencv: it reads
the exported folder and writes a splat plus one wrist-view video per clip.

    python train_gsplat.py --data export/ --out result/ --iterations 30000

The reference implementation is gsplat's own simple_trainer. This wraps it so
the same wrist trajectories the local pipeline produced are rendered from the
trained splat, which is the only way to compare the two backends fairly.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------
# COLMAP binary readers. Small and dependency-free on purpose: the remote box
# should not need pycolmap installed to read a model this repo wrote.
# --------------------------------------------------------------------------

def _read_next(fid, num_bytes, format_char_sequence):
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)


def read_cameras_binary(path: Path) -> dict:
    cameras = {}
    with open(path, "rb") as fid:
        count = _read_next(fid, 8, "Q")[0]
        for _ in range(count):
            camera_id, model_id, width, height = _read_next(fid, 24, "iiQQ")
            # 0 SIMPLE_PINHOLE, 1 PINHOLE, 2 SIMPLE_RADIAL, 3 RADIAL
            num_params = {0: 3, 1: 4, 2: 4, 3: 5}.get(model_id, 4)
            params = _read_next(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = {
                "model_id": model_id, "width": width, "height": height,
                "params": np.array(params),
            }
    return cameras


def read_images_binary(path: Path) -> dict:
    images = {}
    with open(path, "rb") as fid:
        count = _read_next(fid, 8, "Q")[0]
        for _ in range(count):
            props = _read_next(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])       # w, x, y, z
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = ""
            char = _read_next(fid, 1, "c")[0]
            while char != b"\x00":
                name += char.decode("utf-8")
                char = _read_next(fid, 1, "c")[0]
            num_points = _read_next(fid, 8, "Q")[0]
            fid.read(24 * num_points)
            images[image_id] = {
                "qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": name,
            }
    return images


def read_points3d_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    with open(path, "rb") as fid:
        count = _read_next(fid, 8, "Q")[0]
        for _ in range(count):
            props = _read_next(fid, 43, "QdddBBBd")
            xyz.append(props[1:4])
            rgb.append(props[4:7])
            track_len = _read_next(fid, 8, "Q")[0]
            fid.read(8 * track_len)
    return np.array(xyz), np.array(rgb) / 255.0


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# --------------------------------------------------------------------------

def load_scene(data_dir: Path, device: str):
    sparse = data_dir / "sparse" / "0"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    points, colors = read_points3d_binary(sparse / "points3D.bin")

    views = []
    for image in images.values():
        camera = cameras[image["camera_id"]]
        params = camera["params"]
        if camera["model_id"] in (0, 2):        # SIMPLE_PINHOLE, SIMPLE_RADIAL
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]

        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = qvec_to_rotmat(image["qvec"])
        world_to_cam[:3, 3] = image["tvec"]
        views.append({
            "name": image["name"],
            "world_to_cam": world_to_cam,
            "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            "width": camera["width"], "height": camera["height"],
        })
    views.sort(key=lambda v: v["name"])
    return points, colors, views


def main() -> int:
    import cv2
    from gsplat import rasterization

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=1600,
                        help="long side used for training images")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    data = Path(args.data).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    export = json.loads((data / "export.json").read_text())
    print(f"run {export['run']}, scale {export['metric_scale_m_per_unit']:.6f} m/unit")

    points, colors, views = load_scene(data, device)
    print(f"{len(points)} points, {len(views)} views")

    # --- model -----------------------------------------------------------
    means = torch.tensor(points, dtype=torch.float32, device=device)
    rgb = torch.tensor(colors, dtype=torch.float32, device=device)

    distances = torch.cdist(means[:, None, :].squeeze(1), means)
    knn = distances.topk(4, largest=False).values[:, 1:].mean(dim=1)
    scales = torch.log(knn.clamp(1e-4, 1.0))[:, None].repeat(1, 3)

    quats = torch.zeros(len(means), 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((len(means),), 0.1, device=device))

    sh_bands = (args.sh_degree + 1) ** 2
    sh = torch.zeros(len(means), sh_bands, 3, device=device)
    sh[:, 0] = (rgb - 0.5) / 0.28209479177387814

    params = {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh": torch.nn.Parameter(sh),
    }
    extent = float(np.percentile(np.linalg.norm(points - points.mean(0), axis=1), 95))
    optimizer = torch.optim.Adam([
        {"params": [params["means"]], "lr": 1.6e-4 * extent},
        {"params": [params["scales"]], "lr": 5e-3},
        {"params": [params["quats"]], "lr": 1e-3},
        {"params": [params["opacities"]], "lr": 5e-2},
        {"params": [params["sh"]], "lr": 2.5e-3},
    ], eps=1e-15)

    # --- training images -------------------------------------------------
    cache = []
    for view in views:
        image = cv2.imread(str(data / "images" / view["name"]))
        if image is None:
            continue
        scale = min(1.0, args.resolution / max(image.shape[1], image.shape[0]))
        width = int(round(image.shape[1] * scale))
        height = int(round(image.shape[0] * scale))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        K = view["K"].copy()
        K[0] *= width / view["width"]
        K[1] *= height / view["height"]
        cache.append({
            "rgb": torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0,
                                dtype=torch.float32, device=device),
            "viewmat": torch.tensor(view["world_to_cam"], dtype=torch.float32, device=device),
            "K": torch.tensor(K, dtype=torch.float32, device=device),
            "width": width, "height": height,
        })
    print(f"cached {len(cache)} training images at {cache[0]['width']}x{cache[0]['height']}")

    def render(view, sh_degree):
        out_img, out_alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=params["sh"],
            viewmats=view["viewmat"][None],
            Ks=view["K"][None],
            width=view["width"], height=view["height"],
            sh_degree=sh_degree,
            packed=True,          # far lower memory on large scenes
        )
        return out_img[0], out_alpha[0], info

    if not args.render_only:
        rng = np.random.default_rng(0)
        peak = 0.0
        for step in range(1, args.iterations + 1):
            view = cache[int(rng.integers(len(cache)))]
            sh_degree = min(args.sh_degree, step // (args.iterations // 4 + 1))
            rendered, _, _ = render(view, sh_degree)

            loss = (rendered - view["rgb"]).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
                psnr = -10 * math.log10(max(float(((rendered - view["rgb"]) ** 2).mean()), 1e-10))
                print(f"  step {step:6d}  loss {float(loss):.4f}  psnr {psnr:.2f} dB  "
                      f"gaussians {len(params['means'])}  peak {peak:.1f} GiB", flush=True)

        torch.save({k: v.detach().cpu() for k, v in params.items()}, out / "splat.pt")
        print(f"saved splat.pt, peak CUDA memory {peak:.1f} GiB")

    # --- render the wrist trajectories ------------------------------------
    traj_dir = data / "trajectories"
    for path in sorted(traj_dir.glob("*.npz")):
        traj = np.load(path)
        poses = traj["poses_video_rate"]
        clip = path.stem
        frames_out = out / clip
        frames_out.mkdir(parents=True, exist_ok=True)

        width, height, fov = 640, 480, 90.0
        focal = (width / 2) / math.tan(math.radians(fov) / 2)
        K = torch.tensor([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                         dtype=torch.float32, device=device)

        offset = np.eye(4)
        offset[:3, 3] = [0.0, -0.04, -0.08]
        pitch = math.radians(-25.0)
        offset[:3, :3] = np.array([[1, 0, 0],
                                   [0, math.cos(pitch), -math.sin(pitch)],
                                   [0, math.sin(pitch), math.cos(pitch)]])

        with torch.no_grad():
            for index, ee in enumerate(poses):
                cam = ee @ offset
                viewmat = torch.tensor(np.linalg.inv(cam), dtype=torch.float32, device=device)
                image, _, _ = render(
                    {"viewmat": viewmat, "K": K, "width": width, "height": height},
                    args.sh_degree,
                )
                frame = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(str(frames_out / f"{index:05d}.png"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"rendered {clip}: {len(poses)} frames")

    print(f"done. results in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
