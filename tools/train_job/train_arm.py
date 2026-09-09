"""Train one arm of the wristview experiment and score it in millimetres.

One dataset, four arms. The arms differ ONLY in `input_features`; the rows,
the actions, the held-out episodes, the hyperparameters and the seed schedule
are identical. That is what makes the paired comparison exact.

The score is EXPERIMENT-PLAN section 6.1: show the policy the observation,
hide the recorded action, take its predicted action, and measure the
translation error in millimetres against what the operator did. Mean over
every frame of every held-out episode.

NULL sees no camera at all. It is the sanity floor: a policy that can only
read proprioception. If A' and B' sit near NULL, nothing was learned from
pixels and every later number is arithmetic on noise.

NULL crashed on 1 September in three minutes with "You must provide at least
one image or the environment state among the inputs". That is
`configuration_diffusion.py:230`, a CONFIG assertion, and the model underneath
does not share it: `_prepare_global_conditioning` opens with
`global_cond_feats = [batch[OBS_STATE]]`, unconditionally, and appends images
and env-state only if the config declares them. A state-only diffusion policy
runs; lerobot simply refuses to configure one.

So NULL declares the state a second time, as the env-state feature, and is fed
a copy of it. The conditioning vector becomes the same 7 proprioceptive numbers
twice. That adds no information, which is the point: NULL must see no pixels,
and it now does not.
"""
from __future__ import annotations

import argparse, json, math, os, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

EGO = "observation.images.ego"
WRIST = "observation.images.wrist"
WRIST_REAL = "observation.images.wrist_real"
STATE, ACTION = "observation.state", "action"
ENV_STATE = "observation.environment_state"

ARMS = {
    "NULL": [],
    "A_prime": [EGO],
    "B_prime": [EGO, WRIST],
    "C": [EGO, WRIST_REAL],
}


# What ImageNet-pretrained and R3M weights were both trained under, from RGB
# in [0, 1]. lerobot normalises VISUAL features with NormalizationMode.MEAN_STD
# using THIS DATASET's statistics, so without the override below a pretrained
# encoder sees the statistics of one desk under one light, not the ones its
# features were built on.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _freeze_batchnorm(module) -> int:
    """Hold every BatchNorm2d at its pretrained running statistics.

    `use_group_norm=False` is mandatory with pretrained weights, and it
    reintroduces exactly the problem GroupNorm was there to solve: a diffusion
    batch mixes arbitrary timesteps, so BatchNorm re-estimates its running mean
    and variance against noisy mixed-timestep activations and walks the
    pretrained feature distribution away within a few hundred steps. The run
    then lands on the scratch number for a reason that has nothing to do with
    the representation, which is a false negative that looks exactly like a
    real one.

    Convolution weights still train. Only the running statistics are held.
    """
    import torch.nn as nn

    count = 0
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            count += 1
    return count


def _batchnorm_all_eval(module) -> bool:
    """True when no BatchNorm2d in `module` is in training mode."""
    import torch.nn as nn

    return all(not m.training for m in module.modules()
               if isinstance(m, nn.BatchNorm2d))


def _load_pretrained_into_backbone(backbone, checkpoint: str, arch: str,
                                   prefix: str, inner_key: str | None = None) -> dict:
    """Load R3M or VIP weights into lerobot's stripped-resnet backbone.

    Both ship a state dict whose names do not match lerobot's positional
    Sequential. R3M is resnet18 under `convnet.`; VIP is resnet50 under
    `module.convnet.`, nested inside a `vip` key beside a `global_step`.

    lerobot builds the backbone as `nn.Sequential(*resnet18().children()[:-2])`,
    so the module names are positional ("0.weight", "1.running_mean", ...) and
    do not match R3M's ("convnet.conv1.weight"). Rather than remap indices by
    hand, this loads R3M into a real torchvision resnet18 and then rebuilds the
    same Sequential lerobot builds. The index mapping is therefore torchvision's
    own, not ours, and cannot drift.

    A silent partial load would read exactly like "R3M does not help", so every
    step is checked and the checks are returned for the caller to assert on.
    """
    import torch
    import torch.nn as nn
    import torchvision

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if inner_key:
        raw = raw[inner_key]
    convnet = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
    if not convnet:
        raise SystemExit(f"no tensors under prefix {prefix!r} in {checkpoint}")

    # Drop the classifier head. lerobot builds its backbone from
    # `children()[:-2]`, which discards avgpool and fc, so fc never reaches the
    # model. R3M simply omits it. VIP ships one, reshaped to its 1024-dim
    # embedding, and loading it would fail on a size mismatch against
    # torchvision's 1000-class default. Neither case touches the backbone.
    dropped_head = sorted(k for k in convnet if k.startswith("fc."))
    for k in dropped_head:
        convnet.pop(k)

    build = getattr(torchvision.models, arch)
    model = build()
    fresh = nn.Sequential(*list(build().children())[:-2])
    report = model.load_state_dict(convnet, strict=False)

    # `fc` is the classifier head. lerobot strips it along with avgpool, so its
    # absence is expected and is the ONLY absence that is.
    unexpected_missing = [k for k in report.missing_keys if not k.startswith("fc.")]
    loaded = nn.Sequential(*list(model.children())[:-2])
    backbone.load_state_dict(loaded.state_dict(), strict=True)

    # Prove the weights actually moved, rather than trusting that they did.
    a = loaded.state_dict()["0.weight"]
    b = fresh.state_dict()["0.weight"]
    return {
        "checkpoint": str(checkpoint),
        "arch": arch,
        "convnet_keys": len(convnet),
        "dropped_head_keys": dropped_head,
        "backbone_tensors": len(loaded.state_dict()),
        "missing_keys": list(report.missing_keys),
        "unexpected_keys": list(report.unexpected_keys),
        "unexpected_missing": unexpected_missing,
        "differs_from_fresh_init": bool(not torch.allclose(a, b)),
        "conv1_absmean_loaded": float(a.abs().mean()),
        "conv1_absmean_fresh": float(b.abs().mean()),
    }


def _hardware() -> dict:
    """Record what produced the numbers. Two pods, two things that can differ."""
    import torch

    info = {"torch_version": torch.__version__}
    try:
        import torchvision
        info["torchvision_version"] = torchvision.__version__
    except Exception:  # noqa: BLE001
        info["torchvision_version"] = "unknown"
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        try:
            import subprocess
            info["driver_version"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            info["driver_version"] = "unknown"
    else:
        info["gpu_name"] = "cpu"
    return info


def _lerobot_version() -> str:
    try:
        from importlib.metadata import version
        return version("lerobot")
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_r3m_into_backbone(backbone, checkpoint: str) -> dict:
    return _load_pretrained_into_backbone(backbone, checkpoint, "resnet18", "convnet.")


def working_video_backend() -> str:
    """Return a decode backend that actually decodes here.

    lerobot's `get_safe_default_codec` asks `find_spec` whether torchcodec is
    INSTALLED and says yes when it is. torchcodec is a wrapper over a dylib
    linked against a specific libavutil, so it can be installed and unable to
    open a single frame. Load it instead of looking for it.

    Duplicated from `wristview.lerobot_export` on purpose: this file is copied
    to a rented pod on its own and must not need the package.
    """
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any failure means it cannot decode
        print(f"  torchcodec cannot decode here, using pyav: "
              f"{str(exc).splitlines()[0]}", flush=True)
        return "pyav"
    return "torchcodec"


class _WithEnvState(torch.utils.data.Dataset):
    """A dataset that also serves `observation.state` under the env-state key.

    A copy, not a view: the policy normalises the two features separately and
    must not be able to write through one into the other.
    """

    def __init__(self, inner):
        self.inner = inner
        self.meta = inner.meta

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> dict:
        item = self.inner[index]
        item[ENV_STATE] = item[STATE].clone()
        return item


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="Backbone weights held fixed. Trains only the action head. "
                         "R3M and CLIP publish their results in this configuration, "
                         "so a pretrained probe that only fine-tunes cannot tell a "
                         "useless representation from one training destroyed.")
    ap.add_argument("--backbone-weights", default="none",
                    choices=["none", "imagenet", "r3m", "vip"],
                    help="none: random init, which is what Phase 1 ran. "
                         "imagenet: torchvision ResNet18_Weights.IMAGENET1K_V1. "
                         "r3m: the Ego4D-trained R3M resnet18 from surajnair/r3m-18. "
                         "vip: the Ego4D-trained VIP resnet50, Ma et al. ICLR 2023, "
                         "a value-implicit objective rather than R3M's.")
    ap.add_argument("--vip-checkpoint", default=None,
                    help="local path to the VIP model.pt. Required for "
                         "--backbone-weights vip so the pod needs no hub access.")
    ap.add_argument("--r3m-checkpoint", default=None,
                    help="local path to R3M pytorch_model.bin. Required for "
                         "--backbone-weights r3m so the pod needs no hub access.")
    ap.add_argument("--pod-id", default=None, help="recorded in the result JSON")
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything, run the three checks and one training "
                         "step, then stop. Proves a configuration on a laptop "
                         "before it is paid for on a GPU.")
    ap.add_argument("--image-stats", default="imagenet",
                    choices=["imagenet", "dataset"],
                    help="imagenet: every camera normalised with ImageNet mean and "
                         "std, which is what the pretrained weights expect and what "
                         "Phases 1B, 2 and 3 used. dataset: every camera normalised "
                         "with its OWN mean and std from the patched stats.json. The "
                         "rendered wrist channel is darker and bluer than a real "
                         "camera, so under `imagenet` it reaches the backbone at mean "
                         "[-1.201, -0.658, -0.175] against the real channel's "
                         "[0.096, 0.293, 0.207]. `dataset` is the test of whether "
                         "that offset is what costs the render 0.14 mm.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-schedule", default="constant",
                    choices=["constant", "cosine"],
                    help="cosine: linear warmup to --lr over --warmup-steps, then "
                         "cosine decay to 0 at --steps. This is lerobot's own "
                         "diffusion default shape. `constant` is the default so "
                         "the 4 September runs stay reproducible.")
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--down-dims", default=None,
                    help="U-Net channel widths, e.g. 128,256,512. The default "
                         "512,1024,2048 is ~250 M parameters, which is where "
                         "the model size actually lives: freezing ResNet18 "
                         "removes only 11 M of 263 M.")
    ap.add_argument("--holdout", default="13,14,15")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.diffusion.processor_diffusion import (
        make_diffusion_pre_post_processors,
    )
    from lerobot.utils.constants import OBS_IMAGES

    root = Path(args.dataset).resolve()
    repo_id = f"wristview/{root.parents[1].name}"
    holdout = [int(x) for x in args.holdout.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cams = ARMS[args.arm]
    IMG = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640))
    inputs = {c: IMG for c in cams}
    inputs[STATE] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
    # See the module docstring. Only the camera-free arm needs this.
    if not cams:
        inputs[ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(7,))
    # `pretrained_backbone_weights` defaults to None in lerobot 0.4.4, so the
    # Phase 1 arms really were randomly initialised. An earlier comment in this
    # file claimed the opposite; it was wrong and is removed.
    #
    # `use_group_norm` MUST be False whenever the backbone is pretrained.
    # `modeling_diffusion.py:481` raises otherwise, and it is right to: the
    # GroupNorm substitution replaces every BatchNorm2d with a freshly
    # initialised GroupNorm, which discards the pretrained normalisation
    # parameters. See `_freeze_batchnorm` for what that costs and how it is paid.
    pretrained = None
    if args.backbone_weights == "imagenet":
        pretrained = "ResNet18_Weights.IMAGENET1K_V1"
    # VIP is a resnet50. lerobot derives the U-Net conditioning width from a
    # dummy forward through whatever backbone it builds, so naming the right
    # architecture is all that is needed; 2048 channels instead of 512.
    vision_backbone = "resnet50" if args.backbone_weights == "vip" else "resnet18"
    cfg = DiffusionConfig(
        input_features=inputs,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        n_obs_steps=2, horizon=16, n_action_steps=8,
        crop_shape=(340, 600) if cams else None,
        vision_backbone=vision_backbone,
        device=device,
        pretrained_backbone_weights=pretrained,
        use_group_norm=(args.backbone_weights == "none"),
        **({"down_dims": tuple(int(x) for x in args.down_dims.split(","))}
           if args.down_dims else {}),
    )

    base = LeRobotDataset(repo_id=repo_id, root=root,
                          video_backend=working_video_backend())
    stats = {k: {kk: (v.numpy() if hasattr(v, "numpy") else v) for kk, v in d.items()}
             for k, d in base.meta.stats.items()}
    if not cams:
        stats[ENV_STATE] = dict(stats[STATE])

    # THE IMAGE STATISTICS IN THIS DATASET ARE WRONG, AND NOT BY A LITTLE.
    #
    # lerobot writes video-channel stats from `RunningQuantileStats`, whose
    # `update` computes `np.mean(batch**2)` on a uint8 array.  255**2 wraps to
    # 1 mod 256, so the mean of squares is garbage, the variance comes out
    # negative, and `np.sqrt(np.maximum(0, variance))` clamps it to near zero.
    # Measured on this dataset's ego channel: stored std [0.0171, 0.0127,
    # 0.0111] against a true per-pixel std of [0.242, 0.215, 0.194], so
    # MEAN_STD normalisation was dividing by a number 14 to 20 times too small
    # and handing the encoder values in [-46, +44] instead of [-2, +2].
    # The mean is unaffected and is correct.
    #
    # Overriding with ImageNet statistics fixes both problems at once: it is
    # what the pretrained weights expect, and this dataset's TRUE per-pixel std
    # of [0.242, 0.215, 0.194] is close to ImageNet's [0.229, 0.224, 0.225]
    # anyway. The normalisation MODE is left alone, so there is one code path,
    # not two.
    image_stats_note = None
    if cams:
        before = {c: np.asarray(stats[c]["std"]).ravel().round(4).tolist() for c in cams}
        applied = {}
        for c in cams:
            stats[c] = dict(stats[c])
            if args.image_stats == "imagenet":
                m = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
                sd = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
            else:
                # Each channel by its own statistics, from the patched
                # stats.json. Both wrist channels then arrive at the backbone
                # on the same scale, which is the point of the comparison.
                m = np.asarray(stats[c]["mean"], dtype=np.float32).reshape(3, 1, 1)
                sd = np.asarray(stats[c]["std"], dtype=np.float32).reshape(3, 1, 1)
            stats[c]["mean"], stats[c]["std"] = m, sd
            applied[c] = {"mean": [round(float(v), 4) for v in m.ravel()],
                          "std": [round(float(v), 4) for v in sd.ravel()]}
        image_stats_note = {"mode": args.image_stats,
                            "dataset_std_before": before,
                            "applied": applied}
        print(f"  image stats mode '{args.image_stats}'. applied {json.dumps(applied)}",
              flush=True)

    fps = base.meta.fps
    delta = {**{c: [i / fps for i in cfg.observation_delta_indices] for c in cams},
             STATE: [i / fps for i in cfg.observation_delta_indices],
             ACTION: [i / fps for i in cfg.action_delta_indices]}
    full = LeRobotDataset(repo_id=repo_id, root=root, delta_timestamps=delta,
                          video_backend=working_video_backend())
    if not cams:
        full = _WithEnvState(full)

    # `full[i]` decodes every camera for that row. Reading the episode index
    # that way costs 3 video decodes per row, 28,236 on this dataset, before a
    # single training step, and every decoded frame is discarded. The column
    # lives in parquet.
    base._ensure_hf_dataset_loaded()
    epi = np.asarray(base.hf_dataset["episode_index"], dtype=np.int64)
    train_idx = np.nonzero(~np.isin(epi, holdout))[0]
    test_idx = np.nonzero(np.isin(epi, holdout))[0]
    print(f"{args.arm} seed {args.seed}: {len(train_idx)} train rows, "
          f"{len(test_idx)} held-out rows from episodes {holdout}", flush=True)

    # lerobot 0.4 moved normalisation OUT of the policy and into a processor
    # pipeline, and `DiffusionPolicy.__init__` now swallows `dataset_stats`
    # through `**kwargs`. Passing it, as this file used to, therefore trains on
    # raw metres: actions of order 1e-3 against a unit-variance noise schedule.
    # Nothing raises. The policy just learns nothing and the arm reads as a
    # null result. Normalisation is explicit here for that reason.
    policy = DiffusionPolicy(cfg).to(device)
    preprocess, postprocess = make_diffusion_pre_post_processors(cfg, dataset_stats=stats)

    r3m_report = None
    bn_frozen = 0
    if cams and args.backbone_weights == "r3m":
        if not args.r3m_checkpoint:
            raise SystemExit("--backbone-weights r3m needs --r3m-checkpoint")
        r3m_report = _load_r3m_into_backbone(
            policy.diffusion.rgb_encoder.backbone, args.r3m_checkpoint)
        print("  R3M load:", json.dumps(r3m_report, indent=2), flush=True)
        # A silent partial load reads exactly like "R3M does not help".
        assert not r3m_report["unexpected_missing"], (
            f"R3M load left {r3m_report['unexpected_missing']} unfilled")
        assert not r3m_report["unexpected_keys"], "R3M load had unexpected keys"
        assert r3m_report["differs_from_fresh_init"], (
            "R3M conv1 is identical to a fresh init, so nothing was loaded")
        print("  R3M load VERIFIED: 120 tensors, no unexpected or missing keys "
              "outside fc, conv1 differs from fresh init", flush=True)

    vip_report = None
    if cams and args.backbone_weights == "vip":
        if not args.vip_checkpoint:
            raise SystemExit("--backbone-weights vip needs --vip-checkpoint")
        vip_report = _load_pretrained_into_backbone(
            policy.diffusion.rgb_encoder.backbone, args.vip_checkpoint,
            "resnet50", "module.convnet.", inner_key="vip")
        print("  VIP load:", json.dumps(vip_report, indent=2), flush=True)
        assert not vip_report["unexpected_missing"], (
            f"VIP load left {vip_report['unexpected_missing']} unfilled")
        assert not vip_report["unexpected_keys"], "VIP load had unexpected keys"
        assert vip_report["differs_from_fresh_init"], (
            "VIP conv1 is identical to a fresh init, so nothing was loaded")
        print(f"  VIP load VERIFIED: {vip_report['backbone_tensors']} tensors, no "
              f"unexpected or missing keys outside fc, conv1 differs from fresh init",
              flush=True)

    if cams and args.backbone_weights != "none":
        bn_frozen = _freeze_batchnorm(policy.diffusion.rgb_encoder.backbone)
        print(f"  froze running statistics on {bn_frozen} BatchNorm2d modules",
              flush=True)

    frozen = trainable = 0
    if args.freeze_encoder and cams:
        for name, param in policy.named_parameters():
            if "rgb_encoder" in name:
                param.requires_grad_(False)
    for param in policy.parameters():
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    print(f"  parameters: {trainable/1e6:.2f} M trainable, {frozen/1e6:.2f} M frozen", flush=True)

    def set_train() -> None:
        """`policy.train()` recurses and puts BatchNorm back into training mode,
        so the freeze has to be re-applied every single time, not once at
        construction."""
        policy.train()
        if bn_frozen:
            _freeze_batchnorm(policy.diffusion.rgb_encoder.backbone)

    set_train()
    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],
                           lr=args.lr)

    # A constant 1e-4 never lets the validation curve settle, so the last-6
    # score carries learning-rate jitter rather than the model's final quality.
    # On 4 September every arm was still bouncing 0.8 to 1.1 mm between
    # adjacent checks at the cutoff, which is five to ten times the difference
    # between the arms.
    def lr_at(step: int) -> float:
        """Multiplier on `args.lr` at a given step. Warmup then cosine to 0."""
        if args.lr_schedule == "constant":
            return 1.0
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full, train_idx.tolist()),
        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed))

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full, test_idx.tolist()),
        batch_size=args.batch_size, shuffle=False, num_workers=2)

    # WHICH STEPS ARE COMPARED. `generate_actions` returns the horizon sliced
    # `[n_obs_steps - 1 : n_obs_steps - 1 + n_action_steps]`, so with
    # action_delta_indices [-1, 0, 1, ... 14] the prediction is for deltas
    # 0 to 7. The recorded chunk starts at delta -1. This file used to compare
    # the prediction against `truth[:, :8]`, which is deltas -1 to 6: every arm
    # was scored against ground truth shifted one frame, 1/15 s, into the past.
    # The offset is applied once, here, and `phase0_floors.py` scores the same
    # slice so the arms and the floors are comparable.
    PRED_START = cfg.n_obs_steps - 1
    PRED_STOP = PRED_START + cfg.n_action_steps

    def score() -> np.ndarray:
        """Predicted action against the recorded one, in millimetres."""
        policy.eval()
        errs = []
        with torch.no_grad():
            for batch in test_loader:
                truth = batch[ACTION][:, PRED_START:PRED_STOP, :3].to(device)
                prepared = preprocess(dict(batch))
                if cfg.image_features:
                    prepared[OBS_IMAGES] = torch.stack(
                        [prepared[key] for key in cfg.image_features], dim=-4)
                pred = policy.diffusion.generate_actions(prepared)
                pred = postprocess(pred).to(device)
                if pred.shape[1] != truth.shape[1]:
                    raise RuntimeError(
                        f"the policy returned {pred.shape[1]} action steps and "
                        f"{truth.shape[1]} were sliced from the recorded chunk. "
                        f"These must match or the score compares different "
                        f"instants."
                    )
                errs.append(torch.linalg.norm(pred[..., :3] - truth, dim=-1)
                            .flatten().cpu().numpy())
        set_train()
        return np.concatenate(errs) * 1000.0

    # WHAT THE BACKBONE ACTUALLY RECEIVES. Take one real batch, push it through
    # the real preprocessing, and look. A pretrained encoder fed the wrong
    # normalisation is out of distribution on the first forward pass and lands
    # on the scratch number for a reason that has nothing to do with the
    # representation, so this is checked rather than assumed.
    #
    # EVERY camera, not just the first. This checked `cams[0]` only, which is
    # always the ego view, so on B_prime and C the wrist channel went
    # unverified and the check still printed PASSED. The wrist channels are
    # the worst-corrupted ones in this dataset, 15.4x and 9.7x against ego's
    # 14.1x, so they are exactly the ones an ego-only assertion must not be
    # trusted for.
    norm_check = None
    if cams:
        probe = next(iter(test_loader))
        prepared_probe = preprocess(dict(probe))
        norm_check = {}
        pm_worst, ps_all = 0.0, []
        for cam in cams:
            img = prepared_probe[cam].detach().float()
            ch = img.reshape(-1, 3, *img.shape[-2:]).permute(1, 0, 2, 3).reshape(3, -1)
            cm = [round(float(v), 3) for v in ch.mean(dim=1).cpu().numpy()]
            cs = [round(float(v), 3) for v in ch.std(dim=1).cpu().numpy()]
            norm_check[cam] = {"per_channel_mean": cm, "per_channel_std": cs}
            pm_worst = max(pm_worst, max(abs(v) for v in cm))
            ps_all.extend(cs)
            print(f"  normalised batch reaching the backbone, {cam}: "
                  f"mean {cm} std {cs}", flush=True)
        pm, ps = [pm_worst], ps_all
        # Under ImageNet statistics this must sit near zero mean, unit variance.
        # The bar is loose because one batch of one desk is not ImageNet, but it
        # is far tighter than the factor of 14 to 20 the broken dataset std gave.
        if not (max(abs(v) for v in pm) < 1.5 and 0.3 < min(ps) and max(ps) < 3.0):
            raise SystemExit(
                f"preprocessing check FAILED across {len(cams)} camera(s): "
                f"{norm_check}. The backbone is not "
                f"receiving ImageNet-normalised input, so this probe would "
                f"measure the normalisation, not the representation. Stopping "
                f"rather than training."
            )
        print("  normalisation check PASSED", flush=True)

    # The validation curve, not just the endpoint. 2,000 steps is short for a
    # diffusion policy, and without the curve an UNDERTRAINED run and a NULL
    # result are the same number. They call for opposite responses: one needs
    # more steps, the other needs a different experiment.
    if args.dry_run:
        batch = preprocess(dict(next(iter(loader))))
        out_ = policy.forward(batch)
        loss = out_[0] if isinstance(out_, tuple) else out_["loss"]
        loss.backward()
        if bn_frozen:
            assert _batchnorm_all_eval(policy.diffusion.rgb_encoder.backbone)
        grads = sum(1 for n, q in policy.named_parameters()
                    if "rgb_encoder" in n and q.grad is not None and q.grad.abs().sum() > 0)
        print(f"  DRY RUN ok: loss {float(loss):.4f}, "
              f"{grads} encoder tensors received a non-zero gradient "
              f"({'expected 0 when frozen' if args.freeze_encoder else 'expected > 0'})",
              flush=True)
        # The schedule the run will actually follow, read off the same function
        # the optimiser uses rather than described. A schedule nobody printed
        # is a schedule nobody checked.
        trace = [(st, args.lr * lr_at(st)) for st in (0, 250, 500, 5000, 9999)
                 if st <= args.steps or args.steps >= 10000]
        print(f"  lr schedule '{args.lr_schedule}', base {args.lr:g}, "
              f"warmup {args.warmup_steps}, steps {args.steps}", flush=True)
        for st, v in trace:
            print(f"    step {st:5d}  lr {v:.3e}", flush=True)
        return 0

    started = time.time(); step = 0; losses = []; curve = []
    while step < args.steps:
        for batch in loader:
            # Asserted every step, printed once. `.train()` is called again on
            # every eval, and a freeze that silently lapses is the exact failure
            # this guard exists to catch.
            if bn_frozen:
                assert _batchnorm_all_eval(policy.diffusion.rgb_encoder.backbone), (
                    f"a BatchNorm2d went back into training mode at step {step}")
                if step == 0:
                    print(f"  BatchNorm freeze ASSERTED in the training loop: "
                          f"all {bn_frozen} modules still in eval mode", flush=True)
            batch = preprocess(dict(batch))
            out_ = policy.forward(batch)
            loss = out_[0] if isinstance(out_, tuple) else out_["loss"]
            loss.backward(); opt.step(); opt.zero_grad(); scheduler.step()
            losses.append(float(loss)); step += 1
            if step % args.eval_every == 0 or step >= args.steps:
                e = score()
                curve.append({"step": step, "val_mm_mean": float(e.mean()),
                              "val_mm_median": float(np.median(e)),
                              "train_loss": float(np.mean(losses[-args.eval_every:]))})
                print(f"  step {step:5d}  loss {curve[-1]['train_loss']:.4f}  "
                      f"val {curve[-1]['val_mm_mean']:7.2f} mm  "
                      f"{(time.time()-started)/60:.1f} min", flush=True)
            if step >= args.steps:
                break
    train_min = (time.time() - started) / 60
    err_mm = score()

    # Was it still improving when we stopped?
    tail = [c["val_mm_mean"] for c in curve[-4:]]
    still_falling = len(tail) >= 4 and tail[-1] < tail[0] * 0.97
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    result = {
        "arm": args.arm, "seed": args.seed, "steps": args.steps,
        "freeze_encoder": bool(args.freeze_encoder),
        "down_dims": list(cfg.down_dims),
        "trainable_params_m": trainable / 1e6, "frozen_params_m": frozen / 1e6,
        "batch_size": args.batch_size, "holdout": holdout,
        "cameras": cams, "train_rows": len(train_idx), "test_rows": len(test_idx),
        "action_error_mm_mean": float(err_mm.mean()),
        "action_error_mm_median": float(np.median(err_mm)),
        "action_error_mm_p90": float(np.percentile(err_mm, 90)),
        "final_train_loss": float(np.mean(losses[-250:])),
        "val_curve": curve,
        "still_falling_at_end": bool(still_falling),
        "last4_val_mm": tail,
        "train_minutes": train_min, "peak_vram_gb": peak, "device": device,
        # A result with no library version beside it cannot be reproduced or
        # compared. real31's nine runs recorded twenty fields and not this one.
        "lerobot_version": _lerobot_version(),
        "torch_version": torch.__version__,
        "scored_action_deltas": cfg.action_delta_indices[PRED_START:PRED_STOP],
        "image_stats": args.image_stats,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": args.warmup_steps,
        "eval_every": args.eval_every,
        "backbone_weights": args.backbone_weights,
        "vision_backbone": vision_backbone,
        "vip_load": vip_report,
        "use_group_norm": cfg.use_group_norm,
        "batchnorm_modules_frozen": bn_frozen,
        "r3m_load": r3m_report,
        "image_stats_override": image_stats_note,
        "normalised_batch_check": norm_check,
        "pod_id": args.pod_id,
        "hardware": _hardware(),
    }
    # The name has to carry what varied, or two runs of the same arm on one pod
    # overwrite each other and the second silently wins.
    tag = args.arm
    if args.backbone_weights != "none":
        tag += f"_{args.backbone_weights}"
        tag += "_frozen" if args.freeze_encoder else "_finetuned"
    result["run_tag"] = tag
    (out / f"{tag}_seed{args.seed}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
