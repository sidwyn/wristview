# Overnight run · 15 August

Read this first. Everything below was measured, not assumed.

## The headline

The splat port is fixed and is now clearly worth having.

| | before | after |
|---|---|---|
| Gaussians | 13,667, unchanged for 30,000 steps | **1,530,082** |
| Mean PSNR over all 193 views | 28.93 dB | **35.21 dB** |
| Wrist-view scene coverage | 44 to 66% (point cloud) | **94 to 99%** |
| Peak GPU memory | 3.9 GiB | 6.5 GiB of 24 |
| Wall clock | about 9 min | about 9 min |

**Tuesday's videos are in `deliverables/wrist-gsplat/`,** five clips, source on
the left and synthesised wrist camera on the right. Read the README beside
them: it says what is measured, what is approximated and what is wrong.

## What you asked for, and where it is

| # | Task | State |
|---|---|---|
| 1 | Fix densification | done, `tools/cuda_job/train_gsplat.py` |
| 2 | Fix the requires_grad warning | done, and three more like it in the tests |
| 3 | Re-export trajectories after the roll fix | done, shipped and verified by hash |
| 4 | Ship and run 30,000 steps | done, `/workspace/result2`, `/workspace/train2.log` |
| 5 | Verify the run properly | done, see below |
| 6 | Bring back splat.pt and the frames | done, `artifacts/splat_30k.pt`, hash verified both ends |
| 7 | Render the five wrist videos | done, `deliverables/wrist-gsplat/` |
| 8 | Compare against the point cloud | done, in that README and below |
| 9 | Angular velocity gate on hand pose | done, `src/wristview/qc.py` |
| 10 | Allocator curve, ground-truth re-measure | curve done and it changes the story; re-measure running |
| 11 | CAPTURE-SOP and the evidence table | done, two new capture rules, sixteen numbered defects |

## Bug 12 was three bugs

Densification really was missing: no strategy object existed at all. Fixing it
alone moved mean PSNR from 28.93 to only **30.25 dB**, for 51 times the
Gaussians. That poor return was the useful part, because it meant something
else was capping quality.

**The optimizer had to be restructured first.** gsplat's `DefaultStrategy`
rewrites optimizer state whenever it clones, splits or prunes, and it looks that
state up by parameter name. The trainer used one Adam over five parameter
groups, which gives it no key. `check_sanity` requires `params` and `optimizers`
to hold identical keys. Densification cannot be bolted onto the old structure.

**Radial distortion was being ignored.** COLMAP solved this camera as
SIMPLE_RADIAL with k1 = 0.1088, which moves a pixel 26 px at the image corner.
gsplat rasterises a pinhole model and has nowhere to put that term. The
signature was clear once looked for: reconstruction error grew monotonically
from the image centre outward, 31.8 dB at the centre against 29.0 dB at the
edge. Images are now undistorted before training.

**The position learning rate never decayed.** Reference 3DGS decays it 100x
across the run. This trainer held it constant to the last step.

With all three fixed, plus a structural term in the loss, mean PSNR is
**35.21 dB** and the centre-to-edge falloff drops from 3.7 dB to 2.2 dB.

`train_gsplat.py` now raises at step 2,000 if the Gaussian count has not grown,
so this cannot recur silently. Growth appears by step 600.

## Verification, item 5

Every number re-measured by rendering all 193 views, not read off the training
log. **The per-step PSNR in the log is one random view and swings 28 to 38 dB;
it cannot show a plateau.** That is why the original run looked flat.

| step | 1k | 5k | 10k | 20k | 30k |
|---|---|---|---|---|---|
| Gaussians | 42,863 | 726,053 | 1,241,973 | 1,530,082 | 1,530,082 |
| PSNR (single view) | 27.06 | 33.39 | 33.24 | 34.06 | 33.45 |

Growth stops at 15,000 by design, `refine_stop_iter`. Peak memory 6.5 GiB.

Pixel statistics on the rendered wrist frames, twelve sampled per clip: frame
means 27 to 90 of 255, max 255, **zero near-black frames** in the raw renders.

## The Metal out-of-memory was not densification

You told me not to guess at this one. Instrumented over 500 steps at a
near-constant 60,000 Gaussians:

| step | Gaussians | allocated | driver |
|---|---|---|---|
| 1 | 60,000 | 1.02 GiB | 3.59 GiB |
| 200 | 60,000 | 1.02 GiB | 4.68 GiB |
| 500 | 61,214 | 1.03 GiB | 6.34 GiB |

Tensor memory is flat. Driver memory climbs about 5.5 MiB per step and is still
climbing at step 500, roughly 20 GiB by step 3,000. The MPS caching allocator
keeps every block it ever used. `torch.mps.empty_cache()` every 50 steps holds
driver memory at 3.5 GiB for identical work. Now on by default as
`scene.splat.empty_cache_interval`.

## Splat against point cloud, item 8

**Coverage is the real gain,** 44 to 66% becomes 94 to 99%. The desk, keyboard
and monitor read as surfaces rather than scattered dots. The held mug is now
carried as its own Gaussians selected out of the splat, so it is a continuous
surface instead of a speckled overlay.

**One measurement contradicts the obvious reading.** Laplacian variance is 6.7
times higher for the point cloud, 7192 against 1071. That does not mean it is
sharper. An unfilled point cloud is nothing but high-frequency contrast, and
that metric rewards it. It is measuring speckle. By eye the splat resolves
keyboard keys and the monitor bezel, which the point cloud never does.

**Coverage overstates usefulness.** It counts finite depth, and a dim Gaussian
tail counts. demo_3 reads 94% coverage on frames that are visibly almost black.
The near-black count is the companion number to read beside it.

**The splat is worse in one respect.** 65 frames of 877 go near-black, 43 of
them in demo_3, where the point cloud merely goes sparse. A Gaussian field falls
off to background outside its fitted volume; stray points persist. 79% of
demo_3's dark frames and 73% of demo_2's land on frames where the hand was never
detected and the trajectory was already holding a stale pose. Same capture
fault, now a rule in `CAPTURE-SOP.md`.

**Rendered with view-dependent colour off** (SH degree 0, trained at 3). The
wrist camera sits 0.10 m from surfaces scanned from 0.5 m and up, so the
view-dependent terms are evaluated far outside their fitted angles and produce
iridescent smears. Degree 0 trades highlights for legible geometry. This was the
single largest visual improvement of the night.

## Open, and why

- **Close range is soft, and no amount of training fixes it.** The scan was shot
  from 0.5 m and up; the wrist camera renders from 0.10 m. That is
  extrapolation. It needs a capture change: a very close pass, or accept it.
- **demo_0 and demo_1 keep one roll flip each.** The wrist is steady while the
  fingers reconfigure, so the thumb-index axis that sets gripper roll swings
  alone. Fixing it means taking roll from the palm, which changes the documented
  pose mapping. That is your call, not mine.
- **216,181 Gaussians, 14%, finished below the 0.005 opacity that pruning
  removes.** Pruning stops at `refine_stop_iter`, so the last 15,000 steps let
  them decay with nothing to collect them. A final prune pass would cut the file
  14% at no visual cost.
- **The wrist mount is still a guess** and needs a real robot to fix.
- **The ground-truth re-measure is still running.** The first attempt died at
  Stage 1 after 842 seconds with `SQLite error: database is locked`: iCloud had
  the COLMAP database open, because the run directory sat under `~/Documents`.
  Re-running outside the synced tree. `CAPTURE-SOP.md` now names this, since the
  error says SQLite and not iCloud.

## The box

Left running and idle, GPU at 0%, nothing queued. **I did not stop or restart
it.** Two full training runs used, as budgeted, plus several 20-second probes.

Persistent on `/workspace`: `result2/` (the good splat), `result2_distorted/`
(densified but pre-undistortion, kept for the comparison), `result/` (the
original broken run), both training logs, `wrist_splat_frames/` (227 MB of
rendered frames and videos), and the three scripts written tonight.

Transfer note: the direct TCP route refused the key, and the proxy gives an
interactive shell that ignores a remote command, so everything went over the
SSH stdin channel as base64 with a hash check on both ends. Helpers are in the
scratchpad. `runpodctl` was never needed.

`artifacts/splat_30k.pt` is 361,101,827 bytes,
sha256 `d0af10d4577570b72fd50c483939917df218a857e85bd5dfdb1ba1e8b2b309ea`,
identical on both ends. It is gitignored: a 361 MB tensor is not source, and
this repo has been to 1.1 GB once already.
