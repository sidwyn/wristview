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
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

EGO = "observation.images.ego"
WRIST = "observation.images.wrist"
WRIST_REAL = "observation.images.wrist_real"
STATE, ACTION = "observation.state", "action"

ARMS = {
    "NULL": [],
    "A_prime": [EGO],
    "B_prime": [EGO, WRIST],
    "C": [EGO, WRIST_REAL],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="ImageNet weights, no gradient. Trains only the action head.")
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

    root = Path(args.dataset).resolve()
    repo_id = "wristview/real31full"
    holdout = [int(x) for x in args.holdout.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cams = ARMS[args.arm]
    IMG = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640))
    inputs = {c: IMG for c in cams}
    inputs[STATE] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
    cfg = DiffusionConfig(
        input_features=inputs,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        n_obs_steps=2, horizon=16, n_action_steps=8,
        crop_shape=(340, 600) if cams else None,
        device=device,
        **({"down_dims": tuple(int(x) for x in args.down_dims.split(","))}
           if args.down_dims else {}),
    )

    base = LeRobotDataset(repo_id=repo_id, root=root)
    stats = {k: {kk: (v.numpy() if hasattr(v, "numpy") else v) for kk, v in d.items()}
             for k, d in base.meta.stats.items()}
    fps = base.meta.fps
    delta = {**{c: [i / fps for i in cfg.observation_delta_indices] for c in cams},
             STATE: [i / fps for i in cfg.observation_delta_indices],
             ACTION: [i / fps for i in cfg.action_delta_indices]}
    full = LeRobotDataset(repo_id=repo_id, root=root, delta_timestamps=delta)

    epi = np.array([int(full[i]["episode_index"]) for i in range(len(full))])
    train_idx = np.nonzero(~np.isin(epi, holdout))[0]
    test_idx = np.nonzero(np.isin(epi, holdout))[0]
    print(f"{args.arm} seed {args.seed}: {len(train_idx)} train rows, "
          f"{len(test_idx)} held-out rows from episodes {holdout}", flush=True)

    policy = DiffusionPolicy(cfg, dataset_stats=stats).to(device)

    # `pretrained_backbone_weights` already defaults to ImageNet ResNet18, so
    # the encoder starts pretrained either way. Freezing it right-sizes the
    # model: batch 1 fine-tuned ~11 M encoder parameters on 1,493 training
    # rows and reached training loss 0.0067 while validation sat flat from
    # step 750, which is memorisation, not undertraining.
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

    policy.train()
    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full, train_idx.tolist()),
        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed))

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full, test_idx.tolist()),
        batch_size=args.batch_size, shuffle=False, num_workers=2)

    def score() -> np.ndarray:
        """Predicted action against the recorded one, in millimetres."""
        policy.eval()
        errs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
                pred = policy.predict_action_chunk(batch) if hasattr(policy, "predict_action_chunk") \
                    else policy.select_action(batch)
                truth = batch[ACTION]
                if pred.ndim == 3 and truth.ndim == 3:
                    n = min(pred.shape[1], truth.shape[1])
                    a, b = pred[:, :n, :3], truth[:, :n, :3]
                else:
                    a, b = pred[..., :3], truth[..., :3]
                errs.append(torch.linalg.norm(a - b, dim=-1).flatten().cpu().numpy())
        policy.train()
        return np.concatenate(errs) * 1000.0

    # The validation curve, not just the endpoint. 2,000 steps is short for a
    # diffusion policy, and without the curve an UNDERTRAINED run and a NULL
    # result are the same number. They call for opposite responses: one needs
    # more steps, the other needs a different experiment.
    started = time.time(); step = 0; losses = []; curve = []
    while step < args.steps:
        for batch in loader:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out_ = policy.forward(batch)
            loss = out_[0] if isinstance(out_, tuple) else out_["loss"]
            loss.backward(); opt.step(); opt.zero_grad()
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
    }
    (out / f"{args.arm}_seed{args.seed}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
