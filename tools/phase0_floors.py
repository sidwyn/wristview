"""Choose the holdout, then compute both trivial floors on it.

`analytic_floors.py` does the same arithmetic through `LeRobotDataset.
__getitem__`, which decodes every video channel for every row it touches.
That is 3 channels x 9,412 rows on this dataset and it is all thrown away:
the floors read `action` only, which lives in parquet. This calls the dataset's
own `_get_query_indices` and `_query_hf_dataset` instead, so the action chunk
and its episode-boundary padding are produced by exactly the code
`__getitem__` would have used, without opening a video file.

It is checked against `analytic_floors.py`'s published real31full numbers
before it is trusted. See `--verify-real31`.

    FLOOR-MEAN         predict the mean TRAINING action on every frame
    FLOOR-PERSISTENCE  repeat the previous action on every frame

Scored as `train_arm.py` scores an arm: translation only, L2 in millimetres,
over every element of every chunk of every held-out row.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

STATE, ACTION = "observation.state", "action"


def build(root: Path, repo_id: str):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    from wristview.lerobot_export import working_video_backend

    cfg = DiffusionConfig(
        input_features={STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        n_obs_steps=2, horizon=16, n_action_steps=8, crop_shape=None, device="cpu")
    base = LeRobotDataset(repo_id=repo_id, root=root,
                          video_backend=working_video_backend())
    fps = base.meta.fps
    delta = {STATE: [i / fps for i in cfg.observation_delta_indices],
             ACTION: [i / fps for i in cfg.action_delta_indices]}
    return LeRobotDataset(repo_id=repo_id, root=root, delta_timestamps=delta,
                          video_backend=working_video_backend())


def action_chunks(ds, indices) -> np.ndarray:
    """(N, horizon, 3) translation chunks, via the dataset's own query path."""
    ds._ensure_hf_dataset_loaded()
    out = []
    for i in indices:
        item = ds.hf_dataset[int(i)]
        query, _pad = ds._get_query_indices(item["index"].item(),
                                            item["episode_index"].item())
        out.append(ds._query_hf_dataset(query)[ACTION].numpy()[:, :3])
    return np.stack(out)


def stats(errors_mm: np.ndarray, rows: int) -> dict:
    return {"action_error_mm_mean": float(errors_mm.mean()),
            "action_error_mm_median": float(np.median(errors_mm)),
            "action_error_mm_p90": float(np.percentile(errors_mm, 90)),
            "rows": rows}


def floors(ds, holdout: list[int]) -> dict:
    ds._ensure_hf_dataset_loaded()
    epi = np.asarray(ds.hf_dataset["episode_index"], dtype=np.int64)
    train_idx = np.nonzero(~np.isin(epi, holdout))[0]
    test_idx = np.nonzero(np.isin(epi, holdout))[0]

    mean_action = action_chunks(ds, train_idx).reshape(-1, 3).mean(axis=0)
    truth = action_chunks(ds, test_idx)                       # (N, H, 3)

    err_mean = np.linalg.norm(truth - mean_action[None, None, :], axis=-1)

    # action[t-1] is the PREVIOUS row's first action, and an episode's first
    # row has no predecessor, so it repeats its own. That is the same rule
    # analytic_floors.py uses.
    last = np.empty((len(test_idx), 3))
    for n, i in enumerate(test_idx):
        prev = int(i) - 1
        if prev < 0 or epi[prev] != epi[i]:
            last[n] = truth[n, 0]
        else:
            last[n] = action_chunks(ds, [prev])[0, 0]
    err_persist = np.linalg.norm(truth - last[:, None, :], axis=-1)

    out = {"FLOOR_MEAN": stats(err_mean.ravel() * 1000.0, len(test_idx)),
           "FLOOR_PERSISTENCE": stats(err_persist.ravel() * 1000.0, len(test_idx)),
           "holdout": [int(h) for h in holdout],
           "mean_action_mm": (mean_action * 1000).tolist(),
           "train_rows": int(len(train_idx)),
           "test_rows": int(len(test_idx))}
    for name in ("FLOOR_MEAN", "FLOOR_PERSISTENCE"):
        s = out[name]
        print(f"  {name:20s} mean {s['action_error_mm_mean']:7.2f} mm   "
              f"median {s['action_error_mm_median']:7.2f}   "
              f"p90 {s['action_error_mm_p90']:7.2f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--repo-id", default="wristview/sept02_final")
    ap.add_argument("--holdout", default=None,
                    help="comma separated episode ids. Default: every fifth, "
                         "offset to land on exactly ten")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify-real31", default=None,
                    help="path to runs/real31full/06_export/lerobot; reproduces "
                         "the published 12.64 / 8.79 / 28.67 on episodes 13,14,15 "
                         "before this tool is trusted on anything else")
    args = ap.parse_args()

    if args.verify_real31:
        print("--- verification against analytic_floors.py on real31full ---")
        ds = build(Path(args.verify_real31).resolve(), "wristview/real31full")
        got = floors(ds, [13, 14, 15])
        want = {"mean": 12.64, "median": 8.79, "p90": 28.67}
        m = got["FLOOR_MEAN"]
        ok = all(abs(m[f"action_error_mm_{k}"] - v) < 0.02 for k, v in want.items())
        print(f"  published FLOOR-MEAN  mean {want['mean']} median {want['median']} "
              f"p90 {want['p90']}   -> {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            print("  STOPPING. This tool does not reproduce the known answer, so "
                  "its numbers on a new dataset mean nothing.")
            return 1
        print()

    root = Path(args.dataset).resolve()
    ds = build(root, args.repo_id)
    ds._ensure_hf_dataset_loaded()
    episodes = sorted(set(int(e) for e in ds.hf_dataset["episode_index"]))

    if args.holdout:
        holdout = [int(x) for x in args.holdout.split(",")]
    else:
        # Every fifth, not a consecutive block: adjacent takes share lighting,
        # cube placement and the operator's fatigue. `len % 5` shifts the start
        # so the count lands on ten rather than eleven, and keeps the picks off
        # both ends rather than biasing to one.
        holdout = episodes[len(episodes) % 5 :: 5][:10]

    print(f"--- {root} ---")
    print(f"  {len(episodes)} episodes, ids {episodes[0]} to {episodes[-1]}")
    print(f"  holdout ({len(holdout)}): {holdout}")
    out = floors(ds, holdout)
    out["episodes_total"] = len(episodes)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
