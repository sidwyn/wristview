"""Observe the tensor the policy actually stacks, not one computed alongside."""
import json
import os
from pathlib import Path

# CPU only, and modestly. This runs while a long MPS job owns the machine.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import OBS_IMAGES

torch.set_num_threads(2)

# The dataset to check comes from the command line. This was pinned to
# "wristview/real27-test" under a scratchpad path, which is both a dataset
# that no longer exists and a directory the 30 August reboot deleted. A tool
# that can only check one vanished dataset cannot check the next one.
#
# HF_HUB_OFFLINE matters: LeRobotDataset resolves its version against the Hub
# even for a purely local root, and a repo id the Hub has never heard of comes
# back as a 401 that reads like an auth problem rather than a lookup for a
# dataset that was never pushed.
import sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/real31full/06_export/lerobot").resolve()
REPO_ID = json.loads((ROOT / "meta" / "info.json").read_text()).get("repo_id") or f"wristview/{ROOT.parent.parent.name}"
EGO, WRIST = "observation.images.ego", "observation.images.wrist"
WRIST_REAL = "observation.images.wrist_real"
STATE, ACTION = "observation.state", "action"
print(f"checking {ROOT}\n  repo_id {REPO_ID}")
base = LeRobotDataset(repo_id=REPO_ID, root=ROOT)
print(f"  {base.meta.total_episodes} episodes, {base.meta.total_frames} frames, {base.meta.fps} fps")
IMG = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640))

# The three arms of set 2, named as EXPERIMENT-PLAN section 5.3 names them.
# Every arm reads the SAME rows; only the cameras differ. That is what makes
# the paired error cancel and the fraction-closed number meaningful:
#     fraction closed = (A' - B') / (A' - C)
ARMS = {
    "A' ego only            ": [EGO],
    "B' ego + RENDERED wrist": [EGO, WRIST],
    "C  ego + REAL wrist    ": [EGO, WRIST_REAL],
    "B-only rendered wrist  ": [WRIST],
}
for label, cams in ARMS.items():
    inputs = {c: IMG for c in cams}
    inputs[STATE] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
    cfg = DiffusionConfig(input_features=inputs,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        n_obs_steps=2, horizon=16, n_action_steps=8, crop_shape=(340, 600), device="cpu")
    stats = {k: {kk: (v.numpy() if hasattr(v, "numpy") else v) for kk, v in d.items()}
             for k, d in base.meta.stats.items()}
    policy = DiffusionPolicy(cfg, dataset_stats=stats).to("cpu")
    policy.train()

    seen = {}
    original = policy.diffusion._prepare_global_conditioning
    def spy(batch, _o=original, _s=seen):
        if OBS_IMAGES in batch:
            _s["stacked"] = tuple(batch[OBS_IMAGES].shape)
        return _o(batch)
    policy.diffusion._prepare_global_conditioning = spy

    ds = LeRobotDataset(repo_id=REPO_ID, root=ROOT,
        delta_timestamps={**{c: [i / base.meta.fps for i in cfg.observation_delta_indices] for c in cams},
                          STATE: [i / base.meta.fps for i in cfg.observation_delta_indices],
                          ACTION: [i / base.meta.fps for i in cfg.action_delta_indices]})
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)))
    out = policy.forward(batch)
    loss = out[0] if isinstance(out, tuple) else out["loss"]
    loss.backward()
    n_grad = sum(1 for p in policy.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"{label:18s} OBSERVED OBS_IMAGES {seen.get('stacked')}   "
          f"cams={len(cams)}  loss {float(loss):.4f}  grads {n_grad}")
