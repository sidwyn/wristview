"""Turn a gsplat checkpoint from the CUDA box into one the MPS renderer loads.

The two trainers store the same Gaussians under different names. gsplat keeps a
flat dict of raw parameters; `backends.splat_mps.GaussianModel` keeps the same
tensors under the names its own `state()` writes, plus the SH degree, which
gsplat never records because its render call is given the degree each step.

The activations are the part worth being careful about, because getting them
wrong renders something that still looks like a scene. In `tools/cuda_job/
train_gsplat.py`:

    scales    = torch.log(knn.clamp(...))        stored in log space
    opacities = torch.logit(full(..., 0.1))      stored in logit space
    render(scales=torch.exp(...), opacities=torch.sigmoid(...))

which is the same convention `GaussianModel` uses for `log_scales` and
`opacity_logit`. So the conversion is a rename, not a transform, and the only
value that has to be supplied from outside is the SH degree, which follows from
the band count: 16 bands is degree 3.

Both use wxyz quaternions and the same SH basis, so nothing is reordered.

Verify the result rather than trusting it. `--check` re-renders one training
view and prints PSNR; a mismatched activation shows up there immediately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# (degree + 1) ** 2 bands, inverted.
BANDS_TO_DEGREE = {1: 0, 4: 1, 9: 2, 16: 3}


def convert(payload: dict) -> dict:
    """gsplat's parameter dict to `GaussianModel.state()` form."""
    missing = {"means", "scales", "quats", "opacities", "sh"} - set(payload)
    if missing:
        raise KeyError(f"gsplat checkpoint is missing {sorted(missing)}")

    bands = int(payload["sh"].shape[1])
    if bands not in BANDS_TO_DEGREE:
        raise ValueError(
            f"{bands} SH bands is not a whole degree; expected one of "
            f"{sorted(BANDS_TO_DEGREE)}"
        )
    degree = BANDS_TO_DEGREE[bands]

    return {
        "means": payload["means"].detach().cpu(),
        # A rename, not a transform: both sides hold log scales.
        "log_scales": payload["scales"].detach().cpu(),
        "quats": payload["quats"].detach().cpu(),
        # Likewise, both sides hold logits.
        "opacity_logit": payload["opacities"].detach().cpu(),
        "sh": payload["sh"].detach().cpu(),
        "sh_degree": degree,
        # The checkpoint is fully trained, so every band is live. Leaving this
        # at 0 would render flat colour and look merely "a bit dull".
        "active_sh_degree": degree,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.source, map_location="cpu", weights_only=True)
    if "sh_degree" in payload:
        print(f"{args.source} is already in MPS form, copying unchanged")
        state = payload
    else:
        state = convert(payload)

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.destination)
    print(
        f"wrote {args.destination}\n"
        f"  gaussians   {state['means'].shape[0]:,}\n"
        f"  sh degree   {state['sh_degree']} ({state['sh'].shape[1]} bands)\n"
        f"  scale range {state['log_scales'].exp().min():.5f} to "
        f"{state['log_scales'].exp().max():.5f} m\n"
        f"  opacity     median {state['opacity_logit'].sigmoid().median():.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
