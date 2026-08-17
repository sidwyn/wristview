# wristview · Build plan

**Goal:** rebuild the WARPED pipeline in our own code. Egocentric head-mounted video in, robot wrist-view training data out.

**Decisions made:**

| Question | Answer |
|---|---|
| v1 scope | Core. Through to rendered wrist views. |
| Validation | Buyer judgment. Send renders to a buyer and ask if they would train on it. |
| Product shape | Internal CLI. Batches on our own footage. No UI, no accounts. |
| Gripper | Generic parallel jaw, configurable by URDF path. |

**Reference:** [WARPED](https://arxiv.org/html/2604.10809v1). We are reimplementing, not forking. They have released no code.

## Evidence so far

| Test | Result | Date |
|---|---|---|
| **Gate 0** · COLMAP on a hand-held iPhone room scan | **182 of 182 frames registered, 0.909 px mean reprojection error.** Single SIMPLE_RADIAL camera, 20,635 points, mean track length 9.3, one connected model. | 14 Aug 2026 |
| **A3** · Hand pose on contact and wrap frames | **WiLoR succeeded on all frames**, including full finger wrap around the glass. | 14 Aug 2026 |

**Caveat on A3:** the test object was a transparent glass, so fingers stayed partly visible through it. Occlusion was partial. An opaque object is the harder case and is untested.

### Defects, and how each was caught

Every row below shares one property: **plausible output, no crash, caught only
by looking.**
None of them raised an error.
Each produced a result that looked reasonable until it was measured against
something independent.
This is the dominant failure mode in this pipeline, and it is why the QC gates
in `src/wristview/qc.py` exist and why each one records its evidence.

| # | Defect | What it looked like | What exposed it |
|---|---|---|---|
| 1 | hloc stores descriptors (D, N), LightGlue wants (N, D) | matching ran and returned matches | match counts far below the expected range |
| 2 | splat tile clamp anchored to the Gaussian's corner | near geometry vanished behind far geometry | reading it as "splatting is hard at close range" until the clamp was inspected |
| 3 | densification never fired, gradient in pixels against an NDC threshold | training ran, loss fell | Gaussian count never grew, `+0 cloned, +0 split` |
| 4 | whole (pixels, slots) tensor retained for backward | trained, then ran out of memory | memory scaled with density, not with scene |
| 5 | compositing chunk of 1024 tiles meant early exit never fired | correct images, high memory | 30k to 120k Gaussians moved 1.4 to 1.5 GiB only after the chunk dropped to 64 |
| 6 | absolute densification threshold | worked at 0.30 m scene scale | selected nothing at 2.57 m; replaced by a percentile |
| 7 | opacity reset disabled by my own earlier fix | no floater culling | floaters 2 cm from the camera pinned depth at the near plane |
| 8 | Stage 1 wrote the untransformed reconstruction | poses looked ordinary | error was a constant 2.5 m and 147 degrees |
| 9 | ARKit matched to COLMAP by index | scale fitted, RMSE 0.1235 m | matching on timestamp gave 0.0007 m |
| 10 | pycolmap mixes properties and methods | ran for 26 minutes first | `cam_from_world` read as a property yields a bound method |
| 11 | **gripper roll taken from the hand, so the wrist camera flipped** | **videos rendered, geometry correct, camera upside down** | **median angle from world up was 73 to 161 degrees on five clips, four of them past 90** |
| 12 | **hand pose jumped 124 degrees in one frame and held** | **a plausible fast reach** | **2400 deg/s against a human limit near 900** |
| 13 | **the gsplat trainer never densified** | **30,000 steps, loss fell, PSNR 28 dB** | **Gaussian count sat at exactly the 13,667 seed for every step** |
| 14 | radial distortion ignored by the rasteriser | a splat that trained fine | error grew from 31.8 dB at the image centre to 29.0 dB at the edge |
| 15 | position learning rate never decayed | loss fell, looked converged | training PSNR capped near 30 dB with 705 k Gaussians |
| 16 | MPS allocator memory read as a densification leak | an out-of-memory blamed on model growth | allocated held at 1.02 GiB while driver climbed 3.59 to 6.34 GiB |
| 17 | **`max_reproj_error_px` was never read by any code** | **a configured gate that looked like a guarantee** | **Stage 1 reported `ok` at 1.5169 px against its own 1.5 limit** |
| 18 | **pre-flight ceiling normalised by time, not baseline** | **session 4 scored a better ratio than session 3** | **its absolute match count was half session 3's, 214 against 436** |
| 19 | **the marker reference discarded its own valid mask** | **a plausible marker world pose, used by every marker check ever run** | **bootstrapping over random halves of the detections gave exactly zero spread** |

Also caught the same way, outside the numbered set: two ArUco markers averaged
into one scale (46% error from a spurious 8-frame detection), Stage 1 re-scaling
its own cached output, and the marker check using the Stage 0 prior focal length
of 1836 instead of the solved 2819, which produced an identical 21 cm error on
five independent clips.
Constancy across independent inputs is itself a signal.

**Bug 11 in detail.**
A parallel jaw is symmetric about its approach axis, so two orientations describe
the same physical grasp.
The roll was taken from the thumb-to-index axis, which crosses over mid-episode,
so the wrist camera turned upside down partway through a clip.
Nothing failed, and the trajectory stayed correct.

The fix resolves the ambiguity explicitly, in two passes, but not in the order
that first suggests itself.
Applying the canonical test per frame and then enforcing continuity does not
work: where the gripper is near vertical, both orientations are almost equally
aligned with world up, so the per-frame test is decided by noise.
Measured, that ordering re-inverted 2 to 12 frames per clip that the canonical
pass had just corrected.
Continuity runs first and chains each frame to its neighbour by full rotation
distance, which leaves exactly one unknown, the sign of the chain.
A single vote against world up then settles that sign for the whole clip.
Mid-episode flips become impossible by construction.

World up comes from the ArUco marker plane in Stage 1, not from a convention,
because COLMAP's world orientation is arbitrary.

Measured on the raw per-frame gripper poses, before and after both fixes:

| clip | flips before | median up before | flips after | median up after |
|---|---|---|---|---|
| demo_0 | 3 | 162.4° | **1** | 16.3° |
| demo_1 | 5 | 71.1° | **1** | 35.1° |
| demo_2 | 4 | 160.8° | 0 | 19.2° |
| demo_3 | 2 | 153.0° | 0 | 27.0° |
| demo_4 | 9 | 118.8° | 0 | 30.2° |

Both columns are measured on the same input, the raw per-frame gripper poses
with velocity outliers already removed, so the pair is comparable. An earlier
version of this table quoted a "before" measured without that removal, which
made two of the rows disagree with the code that produced the "after".

Four of the five clips were mostly upside down before, with a median past 90
degrees. All five now sit between 16 and 35 degrees of world up.

demo_0 and demo_1 keep one flip each. Those are not wrist motion and the
velocity gate below does not flag them: the palm is steady while the fingers
reconfigure, so the thumb-index axis swings on its own. Removing them means
taking the gripper roll from the palm rather than from the thumb-index axis,
which contradicts the pose mapping in Stage 4, so it is a design change and
not a fix. Recorded, not hidden.

**The frames that read as inverted are not the same defect.**
Of 145 inverted frames across the five clips, none coincide with a
velocity-flagged outlier. They split two ways:

| clip | inverted | on a real detection | on a held pose after tracking stopped |
|---|---|---|---|
| demo_0 | 36 | 36 | 0 |
| demo_1 | 5 | 5 | 0 |
| demo_2 | 59 | 1 | 58 |
| demo_3 | 37 | 4 | 33 |
| demo_4 | 8 | 8 | 0 |

demo_0, demo_1 and demo_4 are genuine hand turnover. The operator rolls their
hand during the place, and the camera follows, which is what it should do.

demo_2 and demo_3 are a different problem: the hand leaves the frame before the
clip ends, and `slerp_fill` holds the last measured pose. demo_2 renders a
frozen gripper for its last 59 frames, 42 per cent of the clip. That is a
capture fault, and `CAPTURE-SOP.md` needs a rule to keep the hand in shot until
after the release.

A flip inside an episode is a defect, so `ROLL_FLIPS_MAX` is 0, not a tolerance.

**Bug 12: hand tracking claimed motion no hand can make.**
The residual flips traced to a different defect.
At the failing step the hand rotation jumps 124 and 101 degrees in one frame and
then holds, with the next step at 2 and 3 degrees.
At 20 fps that is 2400 deg/s.
A human wrist and forearm peak near 700 to 900 deg/s, so the pose was wrong,
not fast.

The gate measures the **palm** frame, not the gripper frame.
This distinction decides whether the threshold means anything.
The gripper roll is taken from the thumb-index axis, so it carries finger
articulation, which is faster than the wrist and does not share its limit.
The palm, built from the wrist and the index and middle knuckles, moves as one
piece and is the thing a human limit applies to.
The same 900 deg/s threshold flags 5 to 18 steps per clip on the gripper frame
against 2 to 4 on the palm, and demo_4 alone drops from 18 to 3.
Gating the gripper frame would have rejected all five episodes for motion that
was mostly real finger movement.

| clip | frames checked | flagged | fraction | longest run | peak |
|---|---|---|---|---|---|
| demo_0 | 123 | 4 | 3.2% | 1 | 2369 °/s |
| demo_1 | 160 | 3 | 1.9% | 1 | 1094 °/s |
| demo_2 | 73 | 2 | 2.7% | 1 | 1355 °/s |
| demo_3 | 144 | 3 | 2.1% | 1 | 1146 °/s |
| demo_4 | 108 | 3 | 2.8% | 1 | 1808 °/s |

Every flagged frame is isolated, so all five episodes are repaired rather than
rejected.
A repaired frame is filled from its neighbours and recorded as filled.
`ee_trajectory.npz` carries `hand_measured` and `hand_filled` per frame, so the
export never presents a filled frame as a measured one.

**Why the reject rules are what they are.**
`HAND_OUTLIER_MAX_RUN` is 1.
Two consecutive bad frames span 100 ms at 20 fps, and real motion happens inside
that window, so filling a run invents a trajectory instead of recovering one.
One frame is 50 ms with measurement on both sides, which interpolation can carry.
`HAND_OUTLIER_MAX_FRACTION` is 0.05.
Above that, more than one control sample in twenty is filled rather than
measured, and describing the trajectory as a measurement stops being true.
Both numbers come from the fill argument, not from the observed 1.9 to 3.2 per
cent, which is why they would still reject a worse capture.

**Bug 13: the CUDA splat trainer had no densification strategy.**
A 30,000 step run on a 4090 reached about 28 dB PSNR, against 19.1 dB from the
Metal path at 3,000 steps, so the port looked justified and the numbers looked
healthy.
The Gaussian count stayed at exactly 13,667 for all 30,000 steps, which is the
COLMAP seed count, and PSNR was flat from step 2,000 onward.
Roughly 28,000 steps did nothing.

Adaptive density control is most of what makes 3DGS work, and `train_gsplat.py`
never created a strategy object at all.
The loop refined the points COLMAP already found and could never add one.

The fix needed a second change that is easy to miss.
gsplat's `DefaultStrategy` rewrites optimizer state whenever it clones, splits
or prunes, and it looks that state up **by parameter name**.
The trainer used a single Adam over five parameter groups, which gives the
strategy no key to look up, and `check_sanity` requires `params` and
`optimizers` to hold identical keys.
So the optimizer had to become one Adam per parameter first.

The trainer now raises at step 2,000 if the Gaussian count has not grown.
This is the third defect in this table whose only symptom was a number that
never moved, after bug 3 and bug 5.

*Numbered 13 rather than 12: the hand velocity defect above took 12 first.*

**What fixing bug 13 then exposed.**
Densification alone moved mean PSNR over all 193 views from 28.93 to 30.25 dB.
That is a poor return for 51 times the Gaussians, and the gap was the finding.

Two more omissions came out of chasing it, both measured rather than guessed.

*Bug 14.*
COLMAP solved the camera as SIMPLE_RADIAL with k1 = 0.1088, which displaces a
pixel by 26 px at the image corner.
gsplat rasterises a pure pinhole model and has nowhere to put that term.
Reconstruction error rose monotonically from the principal point outward,
31.8 dB at the centre against 29.0 dB at the edge, which is the signature.
The images are now undistorted before training.

*Bug 15.*
The reference 3DGS decays the position learning rate by 100x across the run.
This trainer held it constant to the last step, so the Gaussians kept taking
full-size steps and never settled.

Together with a structural term in the loss, the three fixes give:

| splat | Gaussians | mean PSNR over 193 views | centre to edge |
|---|---|---|---|
| original, no densification | 13,667 | 28.93 dB | 31.10 to 27.38 |
| densification only | 704,979 | 30.25 dB | 31.94 to 29.04 |
| plus undistortion, lr decay, SSIM | 1,530,082 | **35.21 dB** | **35.35 to 33.19** |

Peak GPU memory 6.5 GiB of 24, and the run takes about 9 minutes.
Pruning also started working: the largest Gaussian is 0.46 m across a 0.75 m
scene, against 31.3 m before, and oversized Gaussians fell from 680 to 180.

**Bug 19: the marker reference ignored which frames saw the marker.**
`estimate_marker_world_pose` thins its input when more than 40 frames are
valid:

```python
indices = [i for i in range(len(frame_names)) if valid[i]]
if len(indices) > max_frames:
    indices = list(np.linspace(0, len(indices) - 1, max_frames).astype(int).tolist())
```

The second line builds a linspace over **positions** and then uses those
numbers as **frame indices**. The mask is discarded, and the function reads the
first `len(indices)` frames of the clip instead of the frames that actually saw
the marker. It should index back through `indices`.

This sat in every marker agreement check the project has run.

It was found by accident. A hypothesis needed testing, that session 4's marker
reference was weakly constrained because all 77 of its detections are
far-range and confined to two of eight azimuth sectors. Bootstrapping the
reference over random halves of the detections returned **exactly zero spread**
across twelve trials, which is not a stable estimate but a stuck one.

With the bug fixed the test ran properly and **refuted the hypothesis**: the
reference is stable to 0.04 cm and 0.09 degrees, two orders of magnitude below
the 1.79 to 3.29 cm and 1.72 to 7.02 degree disagreement it is used to measure.
Far-range and one-sided did not make it shaky. The disagreement is in the demo
poses or the per-frame planar solve.

Worth recording plainly: fixing this changed the measured agreement by less
than 0.02 cm. The bug was real and long-standing, and on this data the frames
it wrongly chose happened to be similar to the right ones. A real defect with
no observable effect here is still a defect, because the next capture would not
be so lucky.

**Bug 17: a QC gate that was never wired in.**
`max_reproj_error_px: 1.5` sat in `configs/default.yaml` from the start,
documented and plausible, and no code ever read it.
Only `min_registration_rate` was enforced.
Sessions 1 to 3 all measured well under the limit, so nothing revealed it.
Session 4 measured **1.5169 px** and Stage 1 reported `ok`.

A gate nobody enforces is worse than no gate. An unset threshold is visibly
missing; a threshold sitting in the config reads like a guarantee, and every
later decision is taken on the assumption that it held.
Both reconstruction gates are now evaluated together in `qc.reconstruction_gates`
and recorded per run, pass or fail.
The 1.5 was left where it was.

**Bug 18: the pre-flight ceiling moved with the thing it measured.**
The pre-flight ratio divides demo-to-scan matches by the scan's own self-match
ceiling, and that ceiling was measured between scan frames about one second
apart.
One second is a proxy for baseline, and it only holds if the camera moves at a
constant speed.

Session 4's scan sweeps three passes at three speeds in 69 seconds, so its
one-second pairs span a much wider baseline than session 3's.
The ceiling fell from 572 to 232, and the ratio rose from 0.76 to between 0.86
and 1.02 across five clips, including one above 1.0 which should have been
impossible to read as good news.
Absolute matchability had **halved**, 436 to 214.
The metric reported an improvement where there was a regression, because its
denominator degraded faster than its numerator.

The ceiling is now measured between scan pairs whose features travel about as
far across the image as the demo-to-scan pairs do, which compares like with
like without needing a reconstruction that pre-flight runs before there is one.
The baseline used is reported in pixels.
A second gate on the absolute count now sits beside the ratio, because either
alone can mislead: a ratio is blind to a uniformly poor scan, and a count is
blind to texture, resolution and keypoint budget.

**Bug 16: the Metal out-of-memory was not what it looked like.**
The Metal splat ran out of memory and the cause was assumed to be
densification.
Instrumented over 500 steps, logging the allocator every 50, at a near-constant
60,000 Gaussians:

| step | Gaussians | allocated | driver |
|---|---|---|---|
| 1 | 60,000 | 1.02 GiB | 3.59 GiB |
| 100 | 60,000 | 1.02 GiB | 4.20 GiB |
| 200 | 60,000 | 1.02 GiB | 4.68 GiB |
| 300 | 61,170 | 1.03 GiB | 5.20 GiB |
| 400 | 61,704 | 1.03 GiB | 5.78 GiB |
| 500 | 61,214 | 1.03 GiB | 6.34 GiB |

Tensor memory is flat.
Driver memory climbs about 5.5 MiB per step and is still climbing at step 500,
which extrapolates to roughly 20 GiB by step 3,000.
The MPS caching allocator keeps every block it has ever used.
The model was never the problem.

Calling `torch.mps.empty_cache()` every 50 steps holds driver memory between
3.48 and 3.66 GiB over the same 500 steps, with the same Gaussian count.
That is now `scene.splat.empty_cache_interval`, on by default.

### Capture sessions

Three sessions, and the difference between them is entirely capture technique.
The pipeline code that processed session 1 and session 3 is the same code.

| | Session 1 · 14 Aug | Session 2 · 15 Aug, 10am | Session 3 · 15 Aug, 11am |
|---|---|---|---|
| Scan | wide oblique orbit only | 40 s, 15 s close pass (37%) | 30 s, 18 s close pass (60%) |
| Demo-to-scan raw matches | **126** | **212** | **436** |
| Scan self-match ceiling | 700+ | 549 | 572 |
| Pre-flight ratio | — (below 0.25) | **0.39** | **0.76** |
| Verdict | fail | marginal, under the 0.40 mark | **pass, five of five clips** |
| ArUco in scan | none in scene | 124 of 224 (55%) | 114 of 193 (59%) |
| ArUco in demo | none in scene | **0 of 143 (0%)** | **199 of 205 (97%)** |
| Stage 1 registration | 172 of 172 | 224 of 224 | 193 of 193 |
| Stage 2 outcome | all episodes rejected | not run | see run summary |

**What changed between them, and what it bought:**

- **1 → 2.** Added a close top-down pass at demo framing, and a marker for
  metric scale. Raw matches 126 → 212. Still marginal, because the demo saw
  almost nothing but repetitive wood grain.
- **2 → 3.** Added static texture to the workspace, moved the marker to within
  5 to 10 cm of the object so it stays inside the demo crop, and raised the
  close-pass fraction from 37% to 60%. Raw matches 212 → 436, ratio 0.39 → 0.76.

**Metric scale verified twice over, session 4.** The marker gives
0.10095 m per reconstruction unit. Independently, the task object is a cube
measured with a ruler at 76.2 mm; reconstructed under that scale its three
sides come out **81.5, 80.3 and 72.7 mm**, mean 78.2 mm, a **+2.6 per cent**
disagreement.

Two independent sources agreeing to 2.6 per cent is the strongest verification
in the project. The marker is the one to trust, and the two are not averaged:
it is a rigid printed plane with sharp corners solved over 77 frames, while the
cube is soft, rounded and reconstructed from a partial shell, and the spread
across its own three sides is 8.8 mm, larger than the disagreement being
measured. The cube can confirm the marker to within its own precision. It
cannot overturn it.

Measuring the cube needs care. A first attempt fitted an oriented box by PCA
over every cube point and reported 100 x 91 x 71 mm, which is wrong for a
reason worth recording: the cube sits on the desk, so only three faces
reconstruct, and principal axes fitted to a partial shell mix the vertical
extent with a face diagonal. Fitting the desk plane and measuring the vertical
edge against it, then the top face's own extents, gives the numbers above.

**Two measurements worth keeping, both from session 3:**

Of the 64 scan frames the demo matched best, **0% came from the wide orbit and
100% from the close pass**, median position 76% into the scan timeline. The
wide orbit earns its place by giving the splat a background, not by helping
localization.

Of the demo features that matched, **63% came from added structure**: 41.4%
from the marker sheet, 22.0% from the keyboard, 36.6% from desk and wood
grain. Removing the marker sheet alone would drop the ratio to about 0.45.

**Why session 1 was dangerous rather than merely bad.** Stage 2 reported
**100% of frames registered** on it, and produced camera trajectories of 85 m
across a 0.69 m desk at 9.7 m/s. Registration rate is not a correctness
signal. Both checks that now catch this, the pre-flight ratio and the
physical-plausibility test, exist because of that session. See `src/wristview/qc.py`
for the thresholds and the evidence recorded beside each one, and
`CAPTURE-SOP.md` for the procedure that avoids it.

### Pre-flight, all four sessions re-scored

Under the corrected metric, with the ceiling taken at a matched baseline rather
than a matched time:

| Session | ceiling | baseline | demo to scan | ratio | verdict |
|---|---|---|---|---|---|
| 1 | 524 | 304 px | **90** | 0.17 | fail |
| 2 | 448 | 353 px | **212** | 0.47 | fail on the absolute gate |
| 3 | 622 | 96 px | **436** | 0.70 | **pass** |
| 4 | 95 | 737 px | **214** | 2.26 | fail on the absolute gate |

**The trend is 90, 212, 436, 214.** Session 4 regressed to roughly session 2's
matchability while every other thing about it improved: better object, better
marker coverage in the demos, a genuine close pass, 99.7 per cent registration.

**The ratio is still not trustworthy, and the fix only half worked.** Matching
the ceiling on feature displacement made session 4 worse, 1.02 to 2.26, because
displacement conflates a change of scale with a change of baseline. Session 4's
demos sit about 0.3 m from the surface while much of its scan is at 1.1 m, so
features appear at very different image sizes and the displacement reads 737 px
even where the match is good. The scan pairs then selected to match that
"baseline" are genuinely poor ones, the ceiling collapses to 95, and the ratio
inflates further.

So the absolute count is the gate to trust. It reproduces all four known
outcomes with no tuning. The ratio is kept as a diagnostic, because a ratio
needs a trustworthy denominator and no denominator tried so far has been one.

**Sequencing:** see `wristview-runbook.md`. Path A there is two hours and produces the evidence needed at Actuate on 18 August. This build is Path B, and it can run unattended in parallel.

## Environment

**Target macOS on Apple Silicon first.** Do not pre-emptively rent a GPU.

| Component | Apple Silicon |
|---|---|
| COLMAP, ffmpeg | Fine. CPU only. |
| hloc, LightGlue | Fine on MPS |
| SAM 2, Depth-Anything V2 | Fine on MPS |
| **Gaussian splatting** | **`gsplat` is CUDA-only.** Use [Brush](https://github.com/ArthurBrussee/brush) (wgpu, trains and renders), [gsplat-mlx](https://a1091150.github.io/gsplat-mlx/) (closest to a drop-in), or [splat-apple](https://github.com/ghif/splat-apple). |
| **WiLoR / HaMeR** | **The likeliest wall.** Both pull in detectron2-family deps that are painful on Apple Silicon. WiLoR is the target. |

If a stage genuinely blocks, rent one Linux box with an NVIDIA GPU for that stage only. RunPod, Lambda or Vast.ai. A 4090 or A10, roughly $0.40 to $0.80 an hour, shut down after.

**Note on where the difficulty lives:** compute is not the constraint. Stage 3 object pose and Stage 4 retargeting are hard problems, not hard compute. A GPU makes them faster to iterate on, not easier to solve.

---

## Gate 0 · Run before writing anything

Extract frames from the room-scan video. Run COLMAP. Look at the result.

```bash
ffmpeg -i scan.mov -vf fps=4 frames/%05d.jpg
colmap automatic_reconstructor --workspace_path ws --image_path frames
```

**Pass:** COLMAP registers 80 percent or more of frames into one model, and the sparse cloud looks like the room.
**Fail:** stop. Fix capture first. Almost always iPhone stabilization still enabled, or too little viewpoint overlap.

Nothing below matters until this passes. One hour.

---

## Architecture

Seven stages. Each stage is a pure function over a run directory. Each reads files, writes files, and never calls the next stage. This matters: it lets any stage be re-run, swapped, or replaced without touching the others.

```
runs/<run_id>/
  00_ingest/     frames, intrinsics.json, manifest.json
  01_scene/      colmap/, scene.ply, scale.json
  02_localize/   demo_<n>/camera_poses.npy
  03_estimate/   demo_<n>/hand.npz, object_masks/, object_pose.npy
  04_retarget/   demo_<n>/ee_trajectory.npy
  05_render/     demo_<n>/wrist/%05d.png
  06_export/     lerobot/
  qc/            report.json, report.html
```

**Data contract rule:** every stage writes a `meta.json` with its inputs, git SHA, config hash, and timings. Non-negotiable. It is how we debug a bad batch three weeks later.

---

## Stage 0 · Ingest

**In:** `scan.mov`, `demo_*.mov`, optional ARKit trajectory JSON.
**Out:** deduplicated frames, camera intrinsics.

- Extract frames. Scan at 4 fps, demos at full rate.
- Drop blurred frames by variance of Laplacian, with a configurable threshold.
- Intrinsics: read from EXIF or ARKit if present, otherwise let COLMAP self-calibrate. **Use the pinhole model. Reject ultra-wide footage.**
- If ARKit poses exist, store them. They become our metric scale reference and our fallback pose source.

---

## Stage 1 · Scene reconstruction

**In:** scan frames.
**Out:** camera poses, sparse cloud, Gaussian splat, metric scale factor.

- Feature matching with **hloc + LightGlue**, not plain COLMAP SIFT. WARPED names LightGlue and it is markedly better on low-texture indoor scenes.
- COLMAP mapper for poses and sparse points.
- Train the splat with **gsplat**.

### The metric scale problem. Do not skip this.

COLMAP reconstructions are scale-ambiguous. WARPED needs metric units or the retargeted trajectory is meaningless.

**Our advantage over WARPED: the iPhone gives metric scale for free.** ARKit's trajectory is in real metres. Fit a Sim(3) transform between the ARKit trajectory and the COLMAP trajectory over the scan, and the scale factor falls out.

If ARKit is unavailable, fall back to a printed ArUco marker of known size placed in the scene during the scan.

**Write `scale.json` and assert it is populated before any later stage runs.**

---

## Stage 2 · Demo localization

**In:** demo frames, Stage 1 model.
**Out:** per-frame 6DoF camera pose in scene coordinates.

- Register each demo frame against the scan reconstruction using hloc.
- **Fallback:** if fewer than 70 percent of frames register, use the ARKit trajectory transformed by the Sim(3) from Stage 1. Log which path was used, per episode.
- Smooth the trajectory. Reject an episode if tracking is lost for more than 2 percent of frames. That threshold comes from our own QC spec.

---

## Stage 3 · Hand and object estimation

**In:** demo frames, camera poses.
**Out:** hand pose per frame in world coordinates, object mask and pose.

- **Hand:** use **[WiLoR](https://github.com/rolpotamias/WiLoR)**, not HaMeR. Verified working on our own contact and wrap frames in A3, including full finger wrap. The HaMeR Hugging Face Space is dead and WiLoR is newer and better at in-the-wild detection. Both output a MANO-family mesh in camera frame, so transform to world using the Stage 2 poses.
- **Evaluate [HaWoR](https://github.com/ThunderVVV/HaWoR) as an alternative.** It reconstructs hand motion in **world space** directly from egocentric video, which would collapse the camera-frame-to-world transform into one step. Worth a day of evaluation before committing to the WiLoR path.
- **Object:** [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) takes the text description and returns a box. [SAM 2](https://github.com/facebookresearch/sam2) turns the box into a mask and tracks it across frames.
- **Depth:** a monocular depth model for scale-consistent object placement. WARPED uses SpatialTrackerV2. Depth-Anything V2 or UniDepth are viable substitutes.
- **Object pose:** fit a rigid transform per frame from the masked points against the splat geometry. **This is the hardest part of Stage 3 and the most likely thing to need iteration.**

---

## Stage 4 · Retargeting

**In:** hand pose, object pose.
**Out:** end-effector trajectory, 6DoF plus gripper width, at video framerate.

- **Grasp detection:** find grasp onset and release from fingertip-to-object proximity plus motion coupling between hand and object. Emit a binary open/closed signal.
- **Pose mapping:** do **not** put the gripper where the wrist was. Derive it from the grasp. Gripper origin from the thumb-index midpoint. Approach direction from the palm normal. Roll from the thumb-index axis.
- **Width:** thumb-index distance, clamped to the gripper's limits from the URDF.
- Smooth, then resample to a fixed control rate. Make it configurable. Published datasets span 3 Hz to 50 Hz.

**Expect to iterate here.** The pose mapping is a design choice, not a solved formula, and it is where output quality lives.

---

## Stage 5 · Wrist-view rendering

**In:** splat, end-effector trajectory, object pose, gripper URDF.
**Out:** wrist-camera image sequence.

- Place a virtual wrist camera at a fixed offset from the end-effector. Configurable.
- Render the splat from that pose. **Key property: the splat was built from the static scene, so it contains no human. The arm disappears for free.** No inpainting needed.
- Composite the posed object mesh and the gripper mesh with correct depth ordering.
- Apply the target camera model, including fisheye distortion, so images match the deployment camera.

---

## Stage 6 · Export

LeRobot v3. Per episode: RGB wrist frames, action as absolute end-effector pose plus width, language instruction, success label.

**Assert the timestamp tolerance of 1e-4 seconds.** A dataset outside it fails to load, and that is a silent, expensive failure.

---

## Stage 7 · QC · our differentiator

This is the part WARPED does not have and the part that makes the tool ours.

Port the Control 1 checks from Section 10 of the operating plan, and run them automatically on every batch:

- Timestamp consistency, 1e-4 seconds
- Dropped frames, reject above 0.5 percent
- Localization loss, reject above 2 percent of the episode
- Gripper logic: reject any success-labeled episode where the gripper never closed
- Blur and exposure thresholds
- Perceptual-hash duplicate detection
- Retarget confidence: flag frames where hand pose confidence is low

Emit `report.json` and a one-page HTML report per batch. **The HTML report is what goes to the buyer alongside the data.**

---

## Milestones

| # | Deliverable | Gate |
|---|---|---|
| **M0** | Gate 0 passes | COLMAP registers the room scan |
| **M1** | Stages 0 to 1 | A splat you can fly through, with a verified metric scale factor |
| **M2** | Stage 2 | Demo trajectory drawn inside the scan reconstruction, visually correct |
| **M3** | Stage 3 | Hand mesh and object mask overlaid on demo video, tracking through the grasp |
| **M4** | Stage 4 | Gripper trajectory plotted in 3D. Grasp opens and closes at the right moments. |
| **M5** | Stage 5 | **First rendered wrist view. This is the screenshot you send a buyer.** |
| **M6** | Stages 6 to 7 | LeRobot dataset that loads, plus a QC report |

**Stop at M5 and go get buyer judgment before building M6.** If the renders are not convincing, the export format is wasted work.

---

## Stack

- Python 3.11, `uv`
- COLMAP, hloc, LightGlue
- gsplat
- HaMeR, Grounding DINO, SAM 2, Depth-Anything V2
- PyTorch, CUDA. **Rent a GPU. This will not run on the MacBook.** An A10 or 4090 is enough.
- Config in YAML. One config per run, hashed into `meta.json`.

---

## Honest risk assessment

| Risk | Severity |
|---|---|
| **Object pose estimation (Stage 3)** | Highest. WARPED does a joint hand-object optimization we are approximating. Budget the most time here. |
| **Retargeting quality (Stage 4)** | High. A design choice, not a formula. Will need several attempts. |
| **Render realism (Stage 5)** | Medium. Splat quality falls off fast away from the scanned viewpoints, and the wrist camera looks from angles you never walked through. |
| **Metric scale** | Medium, but ARKit largely solves it. If scale is wrong, everything downstream is silently wrong. |
| **Splat and localization** | Low. Well-trodden. |

**The scope inside the scope:** WARPED assumes rigid objects, tabletop scenes, minimal scene change. Stay inside that for v1. Pick-and-place and tool use. Not cloth, not liquids, not articulated objects.

---

## Handoff to Claude Code

One prompt. Runs unattended through Stage 5.

> Read `wristview-build-plan.md`. Build Stages 0 through 5, so I can run one command on a scan video plus demo videos and get rendered robot wrist-camera views out.
>
> Target platform is macOS on Apple Silicon. Python 3.11 with `uv`. Use Metal and MPS where possible. **For Gaussian splatting use Brush or gsplat-mlx, not gsplat, since gsplat is CUDA-only.**
>
> **Run all stages through to Stage 5 without waiting for my approval.** Write each stage's artifact to disk and log what you produced and where. If a dependency cannot be installed on Apple Silicon, say so clearly in the log, skip that stage if the pipeline can continue without it, and keep going.
>
> Do not build Stage 6 or 7. Do not scaffold future stages.

Read the log in the morning. The milestone table above says what a correct artifact looks like.

**Two rules to enforce:** every stage writes files and reads files, and no stage calls the next. If it starts chaining stages, stop it.
