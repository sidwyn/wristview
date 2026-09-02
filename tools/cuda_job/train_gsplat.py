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
    """Return points, colours, views and the directory the images live in.

    If `undistort_export.py` has been run, its pinhole intrinsics and
    undistorted images are used instead of the COLMAP camera. gsplat
    rasterises a pure pinhole model, so a radial term in the camera is a
    systematic disagreement between the images and the model fitted through
    them, worst at the frame edge where the wrist camera often looks.
    """
    sparse = data_dir / "sparse" / "0"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    points, colors = read_points3d_binary(sparse / "points3D.bin")

    pinhole = {}
    image_dir = data_dir / "images"
    pinhole_path = data_dir / "cameras_pinhole.json"
    if pinhole_path.exists() and (data_dir / "undistorted").is_dir():
        pinhole = json.loads(pinhole_path.read_text())
        image_dir = data_dir / "undistorted"
        print(f"using undistorted images and pinhole intrinsics from {pinhole_path.name}")

    views = []
    for image in images.values():
        camera = cameras[image["camera_id"]]
        params = camera["params"]
        override = pinhole.get(str(image["camera_id"]))
        if override:
            fx, fy = override["fx"], override["fy"]
            cx, cy = override["cx"], override["cy"]
            camera = dict(camera, width=override["width"], height=override["height"])
        elif camera["model_id"] in (0, 2):       # SIMPLE_PINHOLE, SIMPLE_RADIAL
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
    return points, colors, views, image_dir


def _gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim(a: torch.Tensor, b: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM between two HxWx3 images in [0, 1]."""
    import torch.nn.functional as F

    x = a.permute(2, 0, 1)[None]
    y = b.permute(2, 0, 1)[None]
    channels = x.shape[1]
    w = _gaussian_window(window, sigma, x.device).expand(channels, 1, window, window)
    pad = window // 2

    mu_x = F.conv2d(x, w, padding=pad, groups=channels)
    mu_y = F.conv2d(y, w, padding=pad, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = F.conv2d(x * x, w, padding=pad, groups=channels) - mu_x2
    sigma_y = F.conv2d(y * y, w, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=pad, groups=channels) - mu_xy

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator).mean()


def main() -> int:
    import cv2
    from gsplat import rasterization
    try:
        from gsplat.strategy import DefaultStrategy
    except ImportError as exc:  # pragma: no cover - depends on the installed gsplat
        raise SystemExit(
            "this gsplat build has no gsplat.strategy.DefaultStrategy, so the splat "
            f"cannot densify. Install gsplat 1.0 or newer. ({exc})"
        ) from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=1600,
                        help="long side used for training images")
    parser.add_argument("--probe-at", type=int, default=2500,
                        help="survey PSNR over a spread of training views at "
                             "this step, so a short run reports whether the "
                             "long one is worth starting")
    parser.add_argument("--max-gaussians", type=int, default=7_000_000,
                        help="stop densifying above this count. Memory scales "
                             "with the count and the trainer holds no "
                             "checkpoint, so an out-of-memory crash loses the "
                             "whole run. real27full reached 5.4 M by step 6000 "
                             "at 14.6 GiB, on a path to about 14 M by the time "
                             "refinement stops. 7 M is about 19 GiB on a 24 GB "
                             "card.")
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                        help="write splat_latest.pt this often, so a crash "
                             "late in a long run does not lose everything.")
    parser.add_argument("--holdout", default=None,
                        help="JSON list of image names to EXCLUDE from "
                             "training and evaluate on afterwards. The poses "
                             "stay in the reconstruction, so the held-out "
                             "views are real photographs from known cameras "
                             "that the splat has never seen.")
    parser.add_argument("--holdout-out", default=None,
                        help="directory for the held-out comparisons")
    parser.add_argument("--wrist-fov-deg", type=float, default=62.1,
                        help="second set of intrinsics to render the held-out "
                             "poses through, matching the wrist camera")
    parser.add_argument("--wrist-size", default="640x360")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--load-splat", default=None,
                        help="load a trained splat.pt into the parameters "
                             "instead of starting from the COLMAP seed. "
                             "`--render-only` on its own renders the UNTRAINED "
                             "seed cloud, which the README wrongly described as "
                             "rendering an existing splat.")
    parser.add_argument("--ssim-weight", type=float, default=0.2,
                        help="weight of the structural term, 0 for pure L1")
    args = parser.parse_args()

    ssim_weight = float(args.ssim_weight)
    device = "cuda"
    data = Path(args.data).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    export = json.loads((data / "export.json").read_text())
    print(f"run {export['run']}, scale {export['metric_scale_m_per_unit']:.6f} m/unit")

    points, colors, views, image_dir = load_scene(data, device)
    print(f"{len(points)} points, {len(views)} views")

    # --- model -----------------------------------------------------------
    means = torch.tensor(points, dtype=torch.float32, device=device)
    rgb = torch.tensor(colors, dtype=torch.float32, device=device)

    # Mean distance to the three nearest neighbours, which sets each Gaussian's
    # initial size.
    #
    # This was one `torch.cdist` over every point against every other, which
    # allocates N^2 floats. real27full seeded 36,962 points and needed 5.5 GiB,
    # so it fit and nobody noticed. real28scanb seeds 78,961 and asked for
    # 23.23 GiB on a 23.53 GiB card, and the run died before step 1. The cost
    # is quadratic in a number that grows with every better scan, which is the
    # wrong way round: a scan good enough to want is a scan too big to start.
    #
    # Chunking makes the peak linear in the chunk and independent of N.
    chunk = max(1, int(2 ** 28 // max(len(means), 1)))
    neighbour = torch.empty(len(means), device=device)
    for start in range(0, len(means), chunk):
        block = means[start:start + chunk]
        distances = torch.cdist(block, means)
        # The nearest is the point itself, at distance zero, so take four and
        # drop the first.
        neighbour[start:start + chunk] = distances.topk(
            4, largest=False).values[:, 1:].mean(dim=1)
        del distances
    print(f"  neighbour distances in chunks of {chunk} rows, "
          f"peak {chunk * len(means) * 4 / 2**30:.2f} GiB")
    scales = torch.log(neighbour.clamp(1e-4, 1.0))[:, None].repeat(1, 3)

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

    # One optimizer per parameter, keyed by name. This is not a style choice.
    # The densification strategy has to rewrite optimizer state whenever it
    # clones, splits or prunes Gaussians, and it looks that state up by
    # parameter name. A single Adam over several param groups gives it nothing
    # to key on, so densification cannot be attached to it.
    learning_rates = {
        "means": 1.6e-4 * extent,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh": 2.5e-3,
    }
    optimizers = {
        name: torch.optim.Adam(
            [{"params": [params[name]], "lr": rate, "name": name}], eps=1e-15
        )
        for name, rate in learning_rates.items()
    }

    # Adaptive density control. Without this the splat can only ever refine the
    # points COLMAP already found, which is most of what 3DGS does and all of
    # what the 30,000 step run was missing.
    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=2e-4,
        grow_scale3d=0.01,
        prune_scale3d=0.1,
        refine_start_iter=500,
        refine_stop_iter=args.iterations // 2,
        reset_every=3000,
        refine_every=100,
        absgrad=False,  # not supported alongside packed=True
        verbose=True,
    )
    if args.load_splat:
        loaded = torch.load(args.load_splat, map_location=device, weights_only=True)
        missing = set(params) - set(loaded)
        if missing:
            raise KeyError(f"{args.load_splat} has no {sorted(missing)}")
        params = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(loaded[k].to(device)) for k in params}
        ).to(device)
        optimizers = {
            name: torch.optim.Adam(
                [{"params": [params[name]], "lr": rate, "name": name}], eps=1e-15
            )
            for name, rate in learning_rates.items()
        }
        print(f"loaded {len(params['means'])} Gaussians from {args.load_splat}")

    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=extent)

    # Decay the position learning rate by 100x over the run, as in the
    # reference 3DGS. Without it the means keep taking full-size steps to the
    # last iteration, so Gaussians jitter around their optimum instead of
    # settling into it, and fine detail never resolves. Only the positions
    # decay; the appearance parameters do not.
    means_schedule = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max(args.iterations, 1))
    )

    # --- training images -------------------------------------------------
    # Match by stem, not by full name. The undistorted copies may be PNG while
    # the reconstruction records the original JPEG names.
    on_disk = {p.stem: p for p in sorted(image_dir.iterdir()) if p.is_file()}

    holdout_names = set()
    if args.holdout:
        holdout_names = {Path(n).stem for n in json.loads(Path(args.holdout).read_text())}
        print(f"holding out {len(holdout_names)} views from training")

    cache, held = [], []
    for view in views:
        path = image_dir / view["name"]
        if not path.exists():
            path = on_disk.get(Path(view["name"]).stem)
        image = cv2.imread(str(path)) if path else None
        if image is None:
            # This used to `continue`. A wrong image directory then trained on
            # zero views and reported nothing. Refuse instead.
            raise FileNotFoundError(
                f"the reconstruction registers {view['name']} but {image_dir} "
                f"holds no readable image with that name or stem. It holds "
                f"{len(on_disk)} files."
            )
        scale = min(1.0, args.resolution / max(image.shape[1], image.shape[0]))
        width = int(round(image.shape[1] * scale))
        height = int(round(image.shape[0] * scale))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        K = view["K"].copy()
        K[0] *= width / view["width"]
        K[1] *= height / view["height"]
        if Path(view["name"]).stem in holdout_names:
            held.append({"name": view["name"], "image": image, "K": K,
                         "world_to_cam": view["world_to_cam"],
                         "width": width, "height": height})
            continue
        if args.render_only:
            # Nothing reads the training cache in this mode, and building it
            # costs 33 minutes for 1,051 views at 1920.
            continue
        # Keep the pixels in host memory, not on the GPU.
        #
        # Caching every training image on the card costs width x height x 3 x 4
        # bytes each. At 1920x1080 that is 24.9 MB per view: fine for the 298
        # views of real26/a at 7.4 GiB, fatal for the 586 views of the merged
        # scan at 14.6 GiB, which left too little for the Gaussians and ran a
        # 24 GB card out of memory at step 3000 with 901k Gaussians.
        #
        # One view moves to the GPU per step. That is 24.9 MB over PCIe per
        # step, about 75 seconds across a 30000 step run, against a ceiling
        # that otherwise scales with the number of training views. Pin the
        # memory so the copy can overlap compute.
        # Cache as uint8, not float32. The source is 8-bit: a JPEG or PNG
        # decodes to uint8, so /255.0 in float32 stores four bytes to carry one
        # byte of information. The division happens on the GPU after the copy
        # and the result is identical.
        #
        # This is what the ceiling actually is. real31's scan is 1077 views at
        # 1920x1080: 26.8 GB as float32, which with torch and the CUDA context
        # exceeded a 41 GB cgroup and the kernel killed the process right after
        # "cached 1077 training images". As uint8 the same cache is 6.7 GB.
        # It also makes the pinned allocation four times cheaper.
        cache.append({
            "rgb": torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                dtype=torch.uint8).pin_memory(),
            "viewmat": torch.tensor(view["world_to_cam"], dtype=torch.float32, device=device),
            "K": torch.tensor(K, dtype=torch.float32, device=device),
            "width": width, "height": height,
        })
    if cache:
        print(f"cached {len(cache)} training images at "
              f"{cache[0]['width']}x{cache[0]['height']}")
    elif not args.render_only:
        raise ValueError(
            "no training images were cached, so there is nothing to train on"
        )
    if held:
        print(f"held out {len(held)} views at "
              f"{held[0]['width']}x{held[0]['height']}")

    def render(view, sh_degree, render_mode="RGB"):
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
            render_mode=render_mode,
        )
        return out_img[0], out_alpha[0], info

    if not args.render_only:
        rng = np.random.default_rng(0)
        peak = 0.0
        def survey(sample: int = 12) -> list[float]:
            """Return PSNR over evenly spaced cached views, in dB.

            The per-step PSNR printed below comes from the single view that
            step happened to draw, so it swings by several dB on its own. This
            samples a fixed spread instead, which is the number that can be
            compared against a target.
            """
            picks = np.linspace(0, len(cache) - 1, min(sample, len(cache))).astype(int)
            scores = []
            with torch.no_grad():
                for index in sorted(set(int(i) for i in picks)):
                    drawn, _, _ = render(cache[index], sh_degree)
                    target_probe = cache[index]["rgb"].to(device).float().div_(255.0)
                    mse = float(((drawn - target_probe) ** 2).mean())
                    scores.append(-10 * math.log10(max(mse, 1e-10)))
            return scores

        seed_count = len(params["means"])
        growth_checked = False
        capped = False
        for step in range(1, args.iterations + 1):
            view = cache[int(rng.integers(len(cache)))]
            sh_degree = min(args.sh_degree, step // 1000)
            rendered, _, info = render(view, sh_degree)

            # L1 plus a structural term, the reference 3DGS objective. L1
            # alone is indifferent to whether an edge lands in the right place
            # as long as the average is right, which is what a splat gets
            # wrong first.
            target = view["rgb"].to(device, non_blocking=True).float().div_(255.0)
            l1 = (rendered - target).abs().mean()
            loss = (1.0 - ssim_weight) * l1 + ssim_weight * (1.0 - ssim(rendered, target))
            # Hands the strategy the screen-space means so it can retain their
            # gradients. Densification is driven by that gradient, so this must
            # happen before backward or nothing is ever selected to split.
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for optimizer in optimizers.values():
                optimizer.step()
            # A count ceiling. Densification here is exponential early on and
            # nothing bounded it: real26bm finished at 5.0 M and 15.0 GiB, and
            # real27full passed 5.4 M by step 6000 on the same view count.
            # Above the cap the strategy still accumulates gradients but is
            # told a step past `refine_stop_iter`, so it refines no further.
            if len(params["means"]) >= args.max_gaussians:
                if not capped:
                    capped = True
                    print(f"  CAP: {len(params['means'])} Gaussians at step "
                          f"{step} reached --max-gaussians "
                          f"{args.max_gaussians}. Densification stops here; "
                          f"optimisation continues.", flush=True)
                refine_step = strategy.refine_stop_iter + 1
            else:
                refine_step = step
            strategy.step_post_backward(
                params, optimizers, strategy_state, refine_step, info, packed=True
            )
            means_schedule.step()

            # A trainer that silently never densifies produces plausible
            # numbers and the wrong result. The 30,000 step run held at exactly
            # the COLMAP seed count for every step and nobody noticed until the
            # count was read. Fail loudly instead.
            if step >= 2000 and not growth_checked:
                growth_checked = True
                if len(params["means"]) <= seed_count:
                    raise RuntimeError(
                        f"densification is not running: {len(params['means'])} Gaussians "
                        f"at step {step}, unchanged from the seed count of {seed_count}. "
                        "Expected growth well before this point. Check that the strategy "
                        "is stepped both before and after backward, and that "
                        "refine_start_iter is below this step."
                    )
                print(f"  densification confirmed: {seed_count} -> {len(params['means'])} "
                      f"Gaussians by step {step}", flush=True)

            # Report quality early, so a probe run says whether the long run is
            # worth starting. Waiting for step 30000 to find out costs 40
            # minutes of rented GPU.
            if step in (args.probe_at, args.iterations):
                scores = survey()
                scores.sort()
                middle = scores[len(scores) // 2]
                print(
                    f"  SURVEY at step {step}: median {middle:.2f} dB over "
                    f"{len(scores)} views, worst {scores[0]:.2f} dB, best "
                    f"{scores[-1]:.2f} dB, {len(params['means'])} Gaussians",
                    flush=True,
                )
                if step == args.probe_at:
                    # Reference points, not a threshold. The first is measured
                    # on this repo's own splats; the second is the target the
                    # gate applies to the finished splat.
                    print(
                        "  reference: the cuda_job README records 20 dB or "
                        "better by step 3000 on an earlier capture. The gate "
                        "on the finished splat needs 25 dB.",
                        flush=True,
                    )
                    if middle < 15.0:
                        print(
                            "  ON TRACK: NO. Under 15 dB this late is the "
                            "failure mode this probe exists to catch. Stop and "
                            "look before starting the long run.",
                            flush=True,
                        )
                    else:
                        print(
                            f"  ON TRACK: {middle:.2f} dB at step {step}. "
                            "Judge against the reference above.",
                            flush=True,
                        )

            if args.checkpoint_every and step % args.checkpoint_every == 0:
                torch.save({k: v.detach().cpu() for k, v in params.items()},
                           out / "splat_latest.pt")
                print(f"  checkpoint at step {step}: splat_latest.pt, "
                      f"{len(params['means'])} Gaussians", flush=True)

            if step % 1000 == 0:
                peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
                with torch.no_grad():
                    error = float(((rendered - target) ** 2).mean().detach())
                    reported = float(loss.detach())
                    position_lr = optimizers["means"].param_groups[0]["lr"]
                psnr = -10 * math.log10(max(error, 1e-10))
                print(f"  step {step:6d}  loss {reported:.4f}  psnr {psnr:.2f} dB  "
                      f"gaussians {len(params['means'])}  lr {position_lr:.2e}  "
                      f"peak {peak:.1f} GiB", flush=True)

        torch.save({k: v.detach().cpu() for k, v in params.items()}, out / "splat.pt")
        print(f"saved splat.pt, {len(params['means'])} Gaussians from a seed of "
              f"{seed_count}, peak CUDA memory {peak:.1f} GiB")

    # --- held-out views, at two different intrinsics -----------------------
    #
    # The question this answers: is the splat itself soft at wrist-camera
    # range, or is the softness downstream of it?
    #
    # Each held-out view is a real photograph from a pose the reconstruction
    # knows and training never saw. Rendering it at the SCAN intrinsics
    # compares like with like against the photograph. Rendering the SAME pose
    # at the WRIST intrinsics, and comparing against the photograph resampled
    # to those intrinsics, isolates the effect of the camera model alone: same
    # rays, same splat, different pixel grid.
    #
    # The resampling is a pure intrinsic change at a fixed pose, so it is the
    # homography K_wrist @ K_scan^-1 and involves no depth and no assumption
    # about the scene.
    if held:
        holdout_dir = Path(args.holdout_out) if args.holdout_out else out / "holdout"
        holdout_dir.mkdir(parents=True, exist_ok=True)
        wide, high = (int(v) for v in args.wrist_size.lower().split("x"))
        focal = (wide / 2.0) / math.tan(math.radians(args.wrist_fov_deg) / 2.0)
        K_wrist = torch.tensor([[focal, 0.0, wide / 2.0],
                                [0.0, focal, high / 2.0],
                                [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        print(f"\nheld-out views at two intrinsics. wrist: {wide}x{high}, "
              f"f={focal:.1f} px, {args.wrist_fov_deg} deg")

        rows = []
        with torch.no_grad():
            for item in held:
                truth = torch.tensor(
                    cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB) / 255.0,
                    dtype=torch.float32, device=device)
                K_scan = torch.tensor(item["K"], dtype=torch.float32, device=device)
                viewmat = torch.tensor(item["world_to_cam"],
                                       dtype=torch.float32, device=device)

                scan_render, _, _ = rasterization(
                    means=params["means"],
                    quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
                    scales=torch.exp(params["scales"]),
                    opacities=torch.sigmoid(params["opacities"]),
                    colors=params["sh"], viewmats=viewmat[None], Ks=K_scan[None],
                    width=item["width"], height=item["height"],
                    sh_degree=args.sh_degree, packed=True)
                scan_render = scan_render[0].clamp(0, 1)

                wrist_render, _, _ = rasterization(
                    means=params["means"],
                    quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
                    scales=torch.exp(params["scales"]),
                    opacities=torch.sigmoid(params["opacities"]),
                    colors=params["sh"], viewmats=viewmat[None], Ks=K_wrist[None],
                    width=wide, height=high,
                    sh_degree=args.sh_degree, packed=True)
                wrist_render = wrist_render[0].clamp(0, 1)

                # The photograph, resampled onto the wrist camera's grid.
                homography = (K_wrist @ torch.linalg.inv(K_scan)).cpu().numpy()
                truth_wrist = cv2.warpPerspective(
                    cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB) / 255.0,
                    homography, (wide, high), flags=cv2.INTER_AREA)
                truth_wrist = torch.tensor(truth_wrist, dtype=torch.float32, device=device)

                def score(a, b):
                    mse = float(((a - b) ** 2).mean())
                    psnr = -10 * math.log10(max(mse, 1e-10))
                    # ssim takes HxWx3 and permutes internally. Passing it
                    # pre-permuted 4D tensors is what killed the first attempt,
                    # after 30,000 steps had already run.
                    value = float(ssim(a, b))
                    return psnr, value

                scan_psnr, scan_ssim = score(scan_render, truth)
                wrist_psnr, wrist_ssim = score(wrist_render, truth_wrist)
                rows.append({"name": item["name"],
                             "scan_psnr_db": round(scan_psnr, 2),
                             "scan_ssim": round(scan_ssim, 4),
                             "wrist_psnr_db": round(wrist_psnr, 2),
                             "wrist_ssim": round(wrist_ssim, 4)})
                print(f"  {item['name']}: scan {scan_psnr:5.2f} dB / SSIM "
                      f"{scan_ssim:.4f}   wrist {wrist_psnr:5.2f} dB / SSIM "
                      f"{wrist_ssim:.4f}", flush=True)

                stem = Path(item["name"]).stem
                def save(tag, tensor, stem=stem):
                    array = (tensor.cpu().numpy()[:, :, ::-1] * 255).astype("uint8")
                    cv2.imwrite(str(holdout_dir / f"{stem}_{tag}.png"), array)
                save("scan_truth", truth)
                save("scan_render", scan_render)
                save("wrist_truth", truth_wrist)
                save("wrist_render", wrist_render)

        scans = [r["scan_psnr_db"] for r in rows]
        wrists = [r["wrist_psnr_db"] for r in rows]
        summary = {
            "views": len(rows),
            "scan_psnr_median_db": round(float(np.median(scans)), 2),
            "wrist_psnr_median_db": round(float(np.median(wrists)), 2),
            "scan_ssim_median": round(float(np.median([r["scan_ssim"] for r in rows])), 4),
            "wrist_ssim_median": round(float(np.median([r["wrist_ssim"] for r in rows])), 4),
            "per_view": rows,
            "note": ("held-out low-pass views, never trained on. The scan "
                     "columns compare against the photograph; the wrist "
                     "columns compare against the same photograph resampled "
                     "to the wrist camera's intrinsics at the same pose."),
        }
        (holdout_dir / "holdout.json").write_text(json.dumps(summary, indent=2))
        print(f"\nHELD-OUT MEDIAN: scan {summary['scan_psnr_median_db']:.2f} dB "
              f"(SSIM {summary['scan_ssim_median']:.4f}), wrist "
              f"{summary['wrist_psnr_median_db']:.2f} dB "
              f"(SSIM {summary['wrist_ssim_median']:.4f})")

    # --- render the wrist trajectories ------------------------------------
    traj_dir = data / "trajectories"
    if not traj_dir.is_dir():
        print("no trajectories/ in the payload, so nothing to render. Training done.")
        return 0
    for path in sorted(traj_dir.glob("*.npz")):
        traj = np.load(path)
        # Control rate, not video rate. A wrist frame is only useful paired
        # with the action taken at that instant, and the action stream is at
        # the control rate. s05_render.py reads `poses` for the same reason, so
        # rendering `poses_video_rate` here produced a layer whose frame count
        # could never match the episode it was supposed to composite with:
        # 268 against 209 on real31's demo_1.
        poses = traj["poses"] if "poses" in traj else traj["poses_video_rate"]
        clip = path.stem
        frames_out = out / clip
        frames_out.mkdir(parents=True, exist_ok=True)

        # The camera comes from the shipped config, not from constants here.
        #
        # These were hardcoded to 640x480 at 90 degrees with no pitch, which is
        # the mount as it stood BEFORE the real27 correction. That correction
        # narrowed the lens to the recorded camera's 62.1 degrees and pitched
        # the mount 12.02 degrees down to keep the finger tips in frame, and it
        # moved measured alpha from 0.635 to 0.939. Nothing linked the two
        # files, so this renderer silently kept drawing the old camera while
        # --wrist-fov-deg and --wrist-size, which look like they control it,
        # are read only by the holdout path.
        wc = dict(export.get("wrist_camera") or {})
        width, height = (int(v) for v in args.wrist_size.lower().split("x"))
        fov = float(wc.get("fov_deg", args.wrist_fov_deg))
        focal = (width / 2) / math.tan(math.radians(fov) / 2)
        K = torch.tensor([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                         dtype=torch.float32, device=device)

        standoff = float(wc["standoff_m"])
        back = float(wc.get("mount_back_m", 0.12))
        up = float(wc.get("mount_up_m", 0.07))
        aim_ahead = float(wc.get("aim_ahead_m", 0.06))
        current = float(np.hypot(back, up))
        if current > 1e-9:
            factor = standoff / current
            back, up, aim_ahead = back * factor, up * factor, aim_ahead * factor
        grasp_offset = float(wc.get("grasp_offset_m", 0.02))

        fingertips = np.array([0.0, 0.0, grasp_offset])
        eye = fingertips + np.array([0.0, -up, -back])
        forward = (fingertips + np.array([0.0, 0.0, aim_ahead])) - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, -1.0, 0.0]))
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        offset = np.eye(4)
        offset[:3, :3] = np.stack([right, down, forward], axis=1)
        offset[:3, 3] = eye
        pitch = float(wc.get("pitch_down_deg", 0.0))
        if pitch:
            angle = math.radians(-pitch)
            rot = np.array([[1.0, 0.0, 0.0],
                            [0.0, math.cos(angle), -math.sin(angle)],
                            [0.0, math.sin(angle), math.cos(angle)]])
            spin = np.eye(4); spin[:3, :3] = rot
            offset = offset @ spin
        if path == sorted(traj_dir.glob("*.npz"))[0]:
            print(f"  wrist camera from config: {width}x{height}, {fov} deg, "
                  f"pitch {pitch} deg down, standoff {standoff} m")

        # Stage 5 composites the gripper and the object against this layer
        # with a depth test, so colour alone is useless to it: it needs the
        # per-pixel depth to decide what is in front, and the alpha to know
        # which pixels hold a surface at all. Coverage taken from depth alone
        # reads 100 per cent on a view that is largely empty.
        #
        # This wrote a flat directory of colour PNGs, which Stage 5 rejects
        # with "the external splat layer has no frame 00000".
        colour_dir = frames_out / "color"
        depth_dir = frames_out / "depth"
        alpha_dir = frames_out / "alpha"
        for d in (colour_dir, depth_dir, alpha_dir):
            d.mkdir(parents=True, exist_ok=True)
        DEPTH_SCALE_MM = 10000.0   # metres -> uint16, 0.1 mm resolution to 6.5 m

        with torch.no_grad():
            for index, ee in enumerate(poses):
                cam = ee @ offset
                viewmat = torch.tensor(np.linalg.inv(cam), dtype=torch.float32, device=device)
                image, alpha, _ = render(
                    {"viewmat": viewmat, "K": K, "width": width, "height": height},
                    args.sh_degree, render_mode="RGB+ED",
                )
                depth_m = image[..., 3].clamp(min=0).cpu().numpy()
                image = image[..., :3]
                a8 = (alpha[..., 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(str(alpha_dir / f"{index:05d}.png"), a8)
                d16 = np.clip(depth_m * DEPTH_SCALE_MM, 0, 65535).astype(np.uint16)
                cv2.imwrite(str(depth_dir / f"{index:05d}.png"), d16)
                frame = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(str(colour_dir / f"{index:05d}.png"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # The manifest Stage 5 reads. depth_scale_mm is not optional: without
        # it the uint16 depth is unitless and the depth test is meaningless.
        (frames_out / "render.json").write_text(json.dumps({
            "clip": clip,
            "width": width, "height": height,
            "frames": len(poses),
            "depth_scale_mm": DEPTH_SCALE_MM,
            "source_index": list(range(len(poses))),
            "fov_deg": fov, "pitch_down_deg": pitch, "standoff_m": standoff,
            "rate": "control",
            "note": ("control-rate poses; camera from export.json wrist_camera, "
                     "not from literals in this file"),
        }, indent=1))
        print(f"rendered {clip}: {len(poses)} frames, colour+depth+alpha")

    print(f"done. results in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
