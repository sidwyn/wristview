# wristview

Rebuild of the [WARPED](https://arxiv.org/html/2604.10809v1) pipeline in our own code.

**In:** head-mounted egocentric video of a person doing a task, plus a scan of the room.
**Out:** robot wrist-camera views with end-effector trajectories, ready to train on.

WARPED has released no code. This is a reimplementation.

## Run it

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .

wristview run \
  --scan  wristview-videos/overview.mov \
  --demos wristview-videos/A001_08141521_C005.mov \
          wristview-videos/A001_08141521_C006.mov \
          wristview-videos/A001_08141521_C007.mov \
  --out runs --instruction "a drinking glass"
```

One command, scan plus demos in, rendered wrist views out. Re-run a single
stage over an existing run without redoing the rest:

```bash
wristview stage 5 --run runs/<run_id>              # re-render only
wristview stage 3-5 --run runs/<run_id>            # re-estimate onward
wristview stage 1 --run runs/<run_id> --set scene.splat.iterations=6000
```

## Files

| File | What it is |
|---|---|
| `RUNBOOK.md` | **Start here.** What to do, in order. |
| `BUILD-PLAN.md` | The spec. Seven stages, data contracts, risks. |
| `BUILD-LOG.md` | What actually runs on Apple Silicon, what does not, and what was substituted. |
| `configs/default.yaml` | Every knob, with the reasoning next to it. |

## Architecture

Six stages. Each is a pure function over a run directory: it reads files,
writes files, and never calls the next one. The driver in `cli.py` sequences
them. That is what lets any stage be re-run, swapped, or replaced on its own.

```
runs/<run_id>/
  00_ingest/     frames, intrinsics.json, manifest.json
  01_scene/      colmap/, scene.ply, splat.pt, scale.json
  02_localize/   demo_<n>/camera_poses.npy
  03_estimate/   demo_<n>/hand.npz, object_masks/, object_pose.npy
  04_retarget/   demo_<n>/ee_trajectory.npz
  05_render/     demo_<n>/wrist/%05d.png
```

Every stage writes a `meta.json` with its inputs, outputs, metrics, timings,
git SHA, config hash, and **which backend actually ran**. That last field is
the one to check when a batch looks wrong: it says whether the stage used the
preferred implementation or fell back.

Stages 6 and 7, export and QC, are deliberately not built. The plan says to
stop at the first rendered wrist view and get buyer judgment before building
the export format.

## Platform

Targets macOS on Apple Silicon. Three substitutions were forced, and
`BUILD-LOG.md` gives the errors in full:

| Component | Plan | What runs | Why |
|---|---|---|---|
| Gaussian splatting | gsplat | native PyTorch tile rasterizer on MPS | gsplat is CUDA-only; gsplat-mlx will not build; Brush cannot render from an arbitrary pose through a Python call |
| Hand pose | HaMeR | WiLoR, via `WiLoR-mini` | HaMeR pins `mmcv==1.3.9`, which will not build under Python 3.11 |
| Feature matching | hloc | hloc's formats, MPS execution | hloc hardcodes cuda-or-cpu; SuperPoint is 44x slower on the CPU path |

## Tests

```bash
python -m pytest tests/ -q     # 128 tests
ruff check src/ tools/ tests/
```

`tools/` holds a synthetic scene generator used only by the tests. It renders
a textured room, a scripted pick-and-place, and ground-truth camera and object
trajectories, which is what makes the geometry testable without footage. It is
not a pipeline input.

## Why this matters

Raw egocentric video is worth $2 to $5 an hour with a 90 percent reject rate.
Data carrying pose labels sells for roughly 100 times that. This pipeline is
the difference between the two, using a camera that costs nothing because you
already own it.
