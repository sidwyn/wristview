# The MPS rasteriser: three faults, deferred

Deferred, not abandoned. The GPU path renders this splat correctly, so nothing
is blocked on fixing these. One of the three is unexplained, and unexplained
faults are open-ended, so this is not the moment to start.

Written 2026-08-26, from the real26 GPU splat: 3,912,607 Gaussians, 298
registered scan views, scene extent 0.66 m.

## The disagreement

gsplat renders that splat from a registered scan pose at **30.46 dB** against
the photograph. The MPS rasteriser, given the same splat, the same pose and the
same photograph, returns **7.21 dB**.

The conversion is not the cause. `tools/convert_gsplat_checkpoint.py` is a
rename: every tensor's shape, minimum and maximum are identical before and
after, and both trainers store scales in log space, opacities in logit space,
and wxyz quaternions.

## Fault 1: the per-tile cap discards most of the splat, silently

`render()` pads every tile to `max_per_tile` entries and drops the rest.
Measured on one view at 960x540, 2040 tiles:

| `max_per_tile` | dropped pairs | mean alpha | PSNR |
|---|---|---|---|
| 128 (the default) | 11,076,570 | 0.185 | 6.39 dB |
| 512 | 10,294,350 | 0.408 | 7.93 dB |
| 2048 | 7,411,593 | 0.764 | 12.30 dB |
| 8192 | 1,244,683 | 0.995 | 20.32 dB |
| 16384 | 15,224 | 0.998 | 20.63 dB |

This splat needs about 5,500 Gaussians per tile. The default is 128. Stage 5
used 256.

The count was already computed. `splat_mps.py:469` calculates it and
`splat_mps.py:584` stores it on the result as `_overflow`, and no caller read
it. That is defect row 15.

**Done:** scoring a truncated render now raises. It never returns a number.

## Fault 2: a 10 dB gap survives with truncation eliminated

At `max_per_tile=16384` only 15,224 pairs are dropped and alpha is 0.998, so
truncation is no longer material. The score is still **20.63 dB at 960x540**
against gsplat's **30.46 dB at 1920x1080**.

Resolution is a confound that could not be removed, because of fault 3. Some of
the gap is real: a splat fitted at 1920 has Gaussians sized for 1920, and
rendering at half that resolution is not the same measurement. Whether that
accounts for 10 dB is unknown.

Candidates, untested: the SH basis or band ordering, the 2D covariance
convention, the low-pass filter gsplat applies to small Gaussians, and the
alpha compositing order within a depth block.

**This one is why the work is deferred.** Faults 1 and 3 have known shapes.
This does not.

## Fault 3: it crashes above 960 px on a splat this size

At 1440x810 and 1920x1080 with 3.9 M Gaussians:

```
RuntimeError: bincount only supports 1-d non-negative integral inputs
```

`splat_mps.py:460` calls `torch.bincount(tile_sorted, ...)`. The input goes
negative or non-integral, which points at an index overflow in the
gaussian-tile pair construction at a pair count this large. It does not appear
at 960x540, and it did not appear at any size with the 10,455-Gaussian local
splat.

## What was done instead

The gate moved to where the renderer is verified. `tools/cuda_job/
measure_splat.py` renders registered scan poses with gsplat and writes
measurements; `tools/check_splat.py --measurements` applies the thresholds. The
verdict logic stays in `wristview.splatqc`, single-sourced.

The MPS path is now a local preview tool and says so when it runs. Stage 5
takes its splat layer from `render.splat_layer_dir` when set, and composites
the lens, the gripper and the object itself.

## Where to start, when this is picked up

1. Remove the resolution confound first. Fix fault 3, then re-measure at 1920
   with an adequate cap. If the gap closes, faults 1 and 3 were the whole story
   and fault 2 does not exist.
2. If the gap survives, render a single Gaussian of known colour, position and
   scale through both rasterisers and difference the images. A basis or
   convention error shows up on one primitive; it does not need a scene.
3. Only then consider the tile-cap architecture. Padding every tile to a fixed
   width cannot scale, and a sorted-by-tile compaction is the usual answer.
