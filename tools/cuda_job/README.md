# Splat training on a rented CUDA box

Everything needed to train a Gaussian splat with gsplat and render the wrist
trajectories from it. Nothing here imports the wristview package, so the remote
machine needs only gsplat and its dependencies.

## Which instance to rent

**A 24 GB card is comfortable. 16 GB works. 12 GB is tight.**

Measured and estimated for this capture, 193 scan images at 2160x1214:

| What | Memory |
|---|---|
| Training images, cached on GPU at 1600 px long side | **3.1 GiB** |
| Training images at full 2160 px | 5.7 GiB |
| 1 M Gaussians, parameters plus Adam state | 0.7 GiB |
| 2 M Gaussians, parameters plus Adam state | 1.3 GiB |
| Rasteriser working set, `packed=True`, 1600 px | 2 to 4 GiB |
| **Total at 1600 px, 1 M Gaussians** | **7 to 9 GiB** |
| **Total at full resolution, 2 M Gaussians** | **12 to 15 GiB** |

So:

- **RTX 4090, 24 GB** · roughly $0.40 to $0.80 an hour. The safe default.
- **A10 or L4, 24 GB** · similar, often cheaper, slower.
- **RTX 3090, 24 GB** · fine.
- **T4, 16 GB** · works at 1600 px. Slow, perhaps 3 to 4 hours for 30k steps.
- **Anything with 12 GB** · only at 1600 px with `--resolution 1200`, and watch it.

The image cache is the largest single item, and it is the one to cut first:
`--resolution 1200` roughly halves it. `packed=True` is already set in the
trainer, which is what keeps the rasteriser's working set from scaling with
the number of Gaussians times tiles.

**Expected wall clock for 30,000 steps on a 4090:** 25 to 45 minutes. Budget
an hour of GPU time including setup, so about $1.

## Setup on the box

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gsplat opencv-python numpy
```

`gsplat` compiles CUDA kernels on first import, which takes a few minutes.

## Run it

```bash
# 1. On the laptop: package the run
python -m tools.export_for_gsplat --run runs/real03 --out exports/real03

# 2. Ship it. About 100 MB, dominated by the scan images.
rsync -avz exports/real03/ user@box:~/wristview-export/
rsync -avz tools/cuda_job/ user@box:~/wristview-job/

# 3. On the box
cd ~/wristview-job
python train_gsplat.py --data ~/wristview-export --out ~/result --iterations 30000

# 4. Bring back the splat and the rendered frames
rsync -avz user@box:~/result/ ./result/
```

`--render-only` skips training and renders from an existing `splat.pt`, which
is what to use when iterating on the wrist camera offset.

## Check the Gaussian count before letting it run for an hour

The first 30,000 step run held at **exactly 13,667 Gaussians**, the COLMAP seed
count, for every step. PSNR reached 28 dB and was flat from step 2,000, so about
28,000 steps did nothing. There was no densification strategy in the trainer at
all, and nothing failed.

The trainer now raises at step 2,000 if the count has not grown, so this cannot
repeat silently. **Confirm it cheaply first:**

```bash
python train_gsplat.py --data ~/wristview-export --out /tmp/probe --iterations 2500
```

Growth should appear by about step 600, since `refine_start_iter` is 500 and
`refine_every` is 100. Expect a line reading
`densification confirmed: 13667 -> N Gaussians by step 2000`. If it raises
instead, do not start the long run.

Expect a few hundred thousand to about 2 M Gaussians by the end, so use the
1 M and 2 M rows in the memory table above rather than the seed count. If the
card runs out, lower `--resolution` before touching the strategy.

**What the count being stuck cost, measured on the returned `splat.pt`:** 9.8%
of the Gaussians finished below the 0.005 opacity that pruning removes, and 5.0%
were wider than a tenth of the scene. 20 of them were wider than the whole
0.59 m scene, the largest 31.3 m across. Together 14.6% of the splat was dead or
degenerate. Growth is the visible half of adaptive density control; pruning is
the half that keeps floaters out of the wrist camera's near plane.

## What comes back

```
result/
  splat.pt          the trained Gaussians
  demo_0/           wrist-view frames, one PNG per video-rate sample
  demo_1/ ...
```

Encode locally with the same side-by-side tool:

```bash
python -m tools.render_wrist_videos --run runs/real03 --out deliverables/wrist-gsplat \
    --cloud result/splat.pt      # once the tool learns to read a splat
```

## What to check when it finishes

The point of running this is to find out whether the local Metal rasterizer is
telling the truth. Compare against the numbers in `BUILD-LOG.md`:

| Quantity | Metal, measured | gsplat, expect |
|---|---|---|
| Train PSNR at 3,000 steps | 19.1 dB | 20 dB or better |
| Train PSNR at 30,000 steps | never completed, OOM at step 2,650 | 25 to 32 dB |
| Gaussians after densification | 60,709 at 3,000 steps | several hundred thousand |
| Peak memory | 8.5 GiB at 17k Gaussians, then OOM | printed each 1,000 steps |

**If gsplat reaches 25 dB or better and the Metal path does not, the Metal
path is wrong**, and `GSPLAT-PORT.md` describes how to make gsplat the default
on CUDA while keeping Metal as the local fallback. That document also lists the
three conventions that have to match: screen-space gradient units, near-plane
semantics, and the per-tile Gaussian cap.
