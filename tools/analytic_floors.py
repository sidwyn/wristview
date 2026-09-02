"""Two analytic baselines, scored exactly as the trained arms are.

A trained policy has to beat something. NULL as specified, a policy with no
camera, cannot be built: LeRobot's diffusion policy raises "You must provide
at least one image or the environment state among the inputs". So the floor is
computed rather than trained.

FLOOR-MEAN         predict the mean of the TRAINING actions for every frame.
FLOOR-PERSISTENCE  predict action[t] = action[t-1].

Persistence is the honest bar. At 15 Hz on a smooth reach, repeating the last
action is a strong predictor, so an arm that cannot beat it has learned
nothing useful. Mean-action is easy to beat and flatters the arms; it is here
only to show how easy.

Scored with the same rule as `train_arm.py`: the recorded action chunk against
the prediction, translation only, L2 in millimetres, every element of every
chunk of every held-out row.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

STATE, ACTION = "observation.state", "action"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--holdout", default="13,14,15")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    root = Path(args.dataset).resolve()
    holdout = [int(x) for x in args.holdout.split(",")]

    # The SAME chunking the arms see, so the numbers are comparable.
    cfg = DiffusionConfig(
        input_features={STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        n_obs_steps=2, horizon=16, n_action_steps=8, crop_shape=None, device="cpu")
    base = LeRobotDataset(repo_id="wristview/real31full", root=root)
    fps = base.meta.fps
    delta = {STATE: [i / fps for i in cfg.observation_delta_indices],
             ACTION: [i / fps for i in cfg.action_delta_indices]}
    ds = LeRobotDataset(repo_id="wristview/real31full", root=root, delta_timestamps=delta)

    epi = np.array([int(ds[i]["episode_index"]) for i in range(len(ds))])
    train_idx = np.nonzero(~np.isin(epi, holdout))[0]
    test_idx = np.nonzero(np.isin(epi, holdout))[0]

    # Mean of the TRAINING actions. Chunk mean, so it matches what is scored.
    tr = np.stack([ds[int(i)][ACTION].numpy()[:, :3] for i in train_idx])
    mean_action = tr.reshape(-1, 3).mean(axis=0)

    err_mean, err_persist = [], []
    for i in test_idx:
        row = ds[int(i)]
        truth = row[ACTION].numpy()[:, :3]              # (horizon, 3)
        err_mean.append(np.linalg.norm(truth - mean_action[None, :], axis=-1))
        # action[t-1]: the previous row's first action, within the same episode.
        prev = i - 1
        if prev < 0 or epi[prev] != epi[i]:
            last = truth[0]                              # episode start: no prior
        else:
            last = ds[int(prev)][ACTION].numpy()[0, :3]
        err_persist.append(np.linalg.norm(truth - last[None, :], axis=-1))

    out = {}
    for name, e in (("FLOOR_MEAN", err_mean), ("FLOOR_PERSISTENCE", err_persist)):
        v = np.concatenate(e) * 1000.0
        out[name] = {"action_error_mm_mean": float(v.mean()),
                     "action_error_mm_median": float(np.median(v)),
                     "action_error_mm_p90": float(np.percentile(v, 90)),
                     "rows": int(len(test_idx))}
        print(f"{name:20s} mean {v.mean():7.2f} mm   median {np.median(v):7.2f}   p90 {np.percentile(v,90):7.2f}")
    out["holdout"] = holdout
    out["mean_action_mm"] = (mean_action * 1000).tolist()
    out["train_rows"] = int(len(train_idx))
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nheld-out rows {len(test_idx)}, training rows {len(train_idx)}")
    print(f"mean training action: {np.round(mean_action*1000,3).tolist()} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
