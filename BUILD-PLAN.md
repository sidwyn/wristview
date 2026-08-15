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
