"""Observe the tensor the policy actually stacks, not one computed alongside."""
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

S = Path("/tmp/claude-501/-Users-sidwyn-Documents-Documents-Personal-Projects-atlas/4c7d9149-910a-457c-9850-0c4d7292cfb6/scratchpad")
EGO, WRIST = "observation.images.ego", "observation.images.wrist"
STATE, ACTION = "observation.state", "action"
base = LeRobotDataset(repo_id="wristview/real27-test", root=S / "lrdataset")
IMG = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640))

for label, cams in {"(a) ego only": [EGO], "(b) ego + wrist": [EGO, WRIST],
                    "(c) wrist only": [WRIST]}.items():
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

    ds = LeRobotDataset(repo_id="wristview/real27-test", root=S / "lrdataset",
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
