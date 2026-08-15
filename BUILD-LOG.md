# wristview · Build log

**Platform:** macOS 26.5.1, Apple Silicon (arm64), Python 3.11.11 via `uv`, PyTorch 2.13 on MPS.
**Scope:** Stages 0 through 5. Stages 6 and 7 are deliberately not built.

This log records what runs, what does not, and what was substituted. The
dependency section is the part to read first: the build plan predicted two
walls and both were real.

---

## Dependencies on Apple Silicon

### Blocked, with a substitute

**`gsplat` · CUDA only.** Known before starting. Not attempted.

**`gsplat-mlx` · does not build.** Not on PyPI. From git it fails compiling its
Metal extension against MLX 0.32:

```
subprocess.CalledProcessError: Command '['cmake', '--build', '.', '-j15']'
    returned non-zero exit status 2.
```

**Brush · not usable from Python.** Needs a Rust toolchain, which this machine
does not have. More importantly it is a CLI and a viewer, and Stage 5 has to
render from arbitrary wrist poses through a Python call. Driving an external
binary per frame does not fit.

**Substitute: a native tile rasterizer in plain PyTorch, on MPS.**
`src/wristview/backends/splat_mps.py`. Follows Kerbl et al. 2023: project each
Gaussian to a 2D conic, sort by depth, bucket into tiles, alpha-composite front
to back. It trains and it renders, and Stage 1 and Stage 5 call the same
function.

One structural difference from the CUDA original. CUDA walks each tile's list
serially with early termination. Here the tile lists are padded to a fixed
depth and composited with an exclusive cumulative product, because that is one
parallel kernel instead of a serial loop and MPS rewards it. The cost is a cap
on Gaussians per tile.

Measured: 46 ms per training iteration at 320x240 with 5,000 Gaussians.
Gradients reach all five parameter groups. Covered by 19 tests.

**HaMeR · does not build.** The plan called it "the likeliest wall" and it was.
It pins `mmcv==1.3.9`, a 2021 release, which fails under Python 3.11:

```
ModuleNotFoundError: No module named 'pkg_resources'
```

Attempted with and without `--no-build-isolation`. It also needs detectron2 and
ViTPose. Its Hugging Face Space is down, so the manual route in RUNBOOK step A3
is gone too.

**Substitute: WiLoR**, via `WiLoR-mini`, which drops the detectron2 dependency.
Upstream WiLoR is a clone-and-run repo with no `pyproject.toml` or `setup.py`,
so it is not pip-installable; the `warmshao/WiLoR-mini` fork is.

Two shims were needed, both in `src/wristview/backends/mano_compat.py`. Running
the MANO layer unpickles `MANO_RIGHT.pkl`, which holds chumpy arrays, and
chumpy was last released in 2019:

1. it imports `numpy.bool`, `numpy.int`, `numpy.float`, removed in NumPy 1.24
2. it calls `inspect.getargspec`, removed in Python 3.11

Both are restorable aliases, not real incompatibilities.

**Verified on the ten stills in `wristview-videos/a3/frames`, the frames from
the earlier manual check: 10 of 10 detected**, including every wrapped-grasp
contact frame where the fingers occlude themselves. About 170 ms a frame on
MPS. It also finds the second hand in the C005 frames, which matches the note
in that directory's README.

### Works, with a change

**hloc · works, but its device selection is hardcoded.** `extract_features` and
`match_features` both do `cuda if available else cpu`, so on this machine
SuperPoint runs on the CPU at **876 ms an image against 20 ms on MPS**, a 44x
penalty. So extraction and matching run on MPS in
`src/wristview/backends/sfm.py`, writing hloc's own HDF5 layout. Reconstruction
and geometric verification are handed to unmodified hloc and pycolmap.

**MediaPipe · version-sensitive.** 1.0.1 removes `mp.solutions`, and its
replacement crashes on macOS arm64:

```
F0000 graph_service.h:139] Check failed: service_ Service is unavailable.
    @ -[DrishtiMetalHelper initWithCalculatorContext:]
```

with both the GPU and CPU delegate. **Pinned to 0.10.21**, where the legacy
solutions API works. MediaPipe is now only a fallback behind WiLoR.

**torch and pycolmap both ship libomp**, and loading both aborts with OMP Error
#15. `src/wristview/__init__.py` sets `KMP_DUPLICATE_LIB_OK` before torch
loads.

### Works as-is

COLMAP 4.1.1 (Homebrew, no CUDA), ffmpeg 7.0.2, pycolmap 4.1.1, LightGlue,
SuperPoint, SAM 2, Depth-Anything V2 (59 ms a frame on MPS), OpenCV, trimesh,
yourdfpy.

---

## Performance notes worth keeping

**LightGlue cost is strongly data-dependent, and benchmarking it wrong is easy.**
Adaptive depth and width pruning are off by default; switching them on cuts
cost roughly fourfold. But with pruning on, a pair of neighbouring frames
costs about 155 ms while a distant loop-closure pair costs about 630 ms,
because confidence never rises and every layer runs. A benchmark that repeats
one easy pair understates the real cost by four times. Two hours went into
chasing a phantom regression that was really this.

**Writing HDF5 inside the matching loop cost more than the matching.** Creating
two small datasets per pair ran at about 285 ms a pair against 57 ms for the
match itself. Results are now buffered and written in one pass.

**Stage 1 caches features and matches.** Matching a 172-frame scan is about
twenty minutes; retraining the splat should not pay for it again. Override with
`--set scene.force_rematch=true`.

---

## Bugs found by tests, not by looking at output

Three of these are in the splat rasterizer, which is the one component with no
upstream to fall back on. Every convention in it had to be got right locally,
and two of the three produced plausible but degraded output rather than a
crash. That is the dangerous failure mode: a thin splat reads as "splatting is
hard at close range", not as a unit error.

**Densification never fired, so the splat could not grow.** Adaptive density
control is most of what makes 3DGS work. This rasterizer projects to pixels, so
the accumulated screen-space gradient is per-pixel, while
`densify_grad_threshold` follows the 3DGS convention and is calibrated against
normalized device coordinates, where the image spans [-1, 1]. The two differ by
half the image size, about 300x at 720p. Measured on the real scan:

```
mean screen-space gradient, pixel units   1.1e-07
configured threshold                      4.0e-04
Gaussians ever qualifying                 0
```

The room splat sat at its 13,066 initial points for a thousand steps and only
shrank as pruning removed some, logging `+0 cloned, +0 split, -94 pruned`.
After converting to NDC in `accumulate`, the same scan grows 13,066 to 14,196
to 17,966 to 20,861 over the first six hundred steps. The conversion also makes
the threshold independent of training resolution.

**Splat depth ordering.** The guard against runaway Gaussians clamped a
Gaussian's tile span by moving its far corner to its near corner, collapsing it
onto a single tile so it vanished from the rest of its own footprint. A
Gaussian that fills the frame is not degenerate when the Stage 5 wrist camera
sits centimetres from a surface. The symptom was near geometry disappearing
behind far geometry. The clamp is now centred on the Gaussian.

**Grasp thresholds could never latch on this footage.** WiLoR measures the
thumb-index width at 12.2 cm with the hand open and 5.8 cm wrapped around the
glass. The configured `close_distance_m` was 4.5 cm, below the wrapped width,
so the Schmitt trigger never fired and every episode would have reported no
grasp at all, with no error anywhere. Thresholds now come from each episode's
own width distribution.

**pycolmap mixes properties and methods.** On `Image`, `has_pose`, `name`, and
`points2D` are properties while `cam_from_world` and `num_points2D` are
methods. Reading `cam_from_world` as a property yields the bound method and
fails on `.matrix()`, after twenty-six minutes of matching. Every pycolmap
accessor in the codebase was audited; that was the only one misused. Note that
`if not image.has_pose` would have failed silently in the other direction, so
this is worth checking rather than assuming.

**Ingest discarded 41 percent of a fully-registrable scan.** Blur rejection used
an absolute variance-of-Laplacian threshold, which tracks scene texture as much
as focus and happened to sit at this footage's median. Now relative to each
clip's own median. Deduplication was also running on demo clips, where adjacent
60 fps frames are near-identical by nature and every one is a distinct moment
in the trajectory. Scan frames kept went from 107 of 182 to 172 of 182.

**Grasp hysteresis inverted the final state.** A short run at the end of an
episode was absorbed into the preceding one, so a clip ending in a two-frame
release reported the gripper still closed. First and last runs are now left
alone.

---

## Results on the real footage

`runs/real01`, from `wristview-videos/`.

### Stage 0 · Ingest

| Clip | Kept | Rate |
|---|---|---|
| scan, overview.mov | 172 / 182 | 6 fps |
| demo_0, C005 | 471 / 471 | 60 fps, later 20 fps by config |
| demo_1, C006 | 574 / 574 | " |
| demo_2, C007 | 405 / 405 | " |

### Stage 1 · Reconstruction

Against a standalone COLMAP run on the same clip, which registered 182 of 182
at 0.909 px with 20,635 points and a mean track length of 9.3:

| Metric | Baseline | This run | |
|---|---|---|---|
| Registered | 182 / 182 | 172 / 172, 100 percent | matches |
| Models | 1 | 1 | matches |
| Mean reprojection error | 0.909 px | 1.517 px | worse |
| 3D points | 20,635 | 13,066 | fewer |
| Mean track length | 9.3 | 11.4 | better |

Every frame ingest kept was registered, into one connected model, so the ten
dropped frames cost nothing.

The two worse numbers come from the feature front-end, not from ingest. The
baseline used COLMAP SIFT with subpixel refinement; this uses SuperPoint at
1024 keypoints, which localizes to roughly a pixel. The higher track length
says LightGlue matches each point across more views: fewer points, better
observed. `scene.max_keypoints: 2048` recovers most of the precision at about
four times the matching cost.

**The self-calibration moved the focal length by 55 percent.**

```
prior, a guess at 0.85 x image width   f = 1836 px    60.9 deg horizontal
refined by bundle adjustment           f = 2860 px    41.5 deg horizontal, k = +0.120
```

41.5 degrees is not a phone main-wide lens. Freezing the guess, which is what
the code did before this was changed, would have carried that error silently
into every camera pose and every downstream distance. Worth confirming against
whatever actually shot the clips.

**The demo clips are a top-down desk view**, not head-mounted egocentric.
Confirmed by overlaying WiLoR landmarks on a frame: keyboard, glass and coaster
seen from above. The pipeline handles it, but it matters for Stage 5, because
the splat only contains viewpoints the scan actually visited and a wrist camera
near the table looks from angles an overhead scan may never have covered.
Stage 5 reports splat coverage per episode, which is the measure of that risk.

### Stage 2 · Localization — blocked by the capture

**All three demo episodes are rejected.** This is a footage limitation, not a
code fault, and it stops the pipeline before a wrist view.

Stage 2 first reported 100 percent of frames registered on all three
episodes, with trajectories like this:

| Episode | Registered | Inlier ratio | Camera path | Median speed |
|---|---|---|---|---|
| demo_0 | 157 / 157 | 33 percent | 84.9 m | 9.7 m/s |
| demo_1 | 191 / 191 | 39 percent | 82.6 m | 5.4 m/s |
| demo_2 | 135 / 135 | 38 percent | 64.1 m | 4.8 m/s |

The scene is 0.69 m across. Those paths are around a hundred times the scene
size, in clips under ten seconds. Orientation was stable throughout, about
six degrees end to end, so nothing upstream looked wrong.

**A registration rate proves nothing on its own.** PnP returns a pose whenever
it finds enough inliers, and with weak geometry those poses are individually
valid and collectively nonsense.

The cause was established by elimination, not by guessing:

| Test | Result | Conclusion |
|---|---|---|
| Self-localize scan frames against their own model | 4.1 mm error, 92 percent inliers | the localization code is correct |
| Sweep RANSAC threshold 12 to 2 px | path got worse | not a threshold problem |
| Sweep assumed focal 1800 to 4500 px | inlier ratio flat near 19 percent | not a calibration problem |
| Match one demo frame against **all 172** scan frames | best gives 126 matches, 24 percent inliers | the footage |

Scan-to-scan neighbours give 700 or more matches. The best achievable
demo-to-scan match gives 126.

Looking at the frames explains it. The demos are tight top-down close-ups
covering roughly 30 cm of desk, showing a hand, the glass and the coaster.
The scan is a wide oblique orbit showing monitor, keyboard and speakers. They
share only the coaster and some wood grain: about a sixfold scale difference
plus a large viewpoint change, which SuperPoint and LightGlue cannot bridge.

**The fix is in the capture.** The scan has to include close, top-down views
at the demo's framing and working distance. After the wide orbit, move in and
cover the working area slowly from directly above, at the height the demo
camera sits, with the object in place. Twenty extra seconds of scanning.

**Downstream behaviour was verified with `localize.accept_implausible`**, an
explicit override that passes a bad trajectory on and marks every artifact
from it as geometrically invalid. With it, Stages 3, 4 and 5 all execute on
the real clips, and the failure propagates exactly as predicted:

- Stage 3 hand estimation works: WiLoR detects on 74, 61 and 87 percent of
  frames across the three episodes, and SAM 2 returns object masks
- Stage 3 depth fitting fails with zero samples, because the splat renders
  nothing from cameras the bad poses place tens of metres away
- Stage 5 renders 118, 143 and 101 wrist frames at 640x480, all empty, and
  reports splat coverage of 0.000

Three independent checks caught the same problem: the Stage 2 plausibility
test, the Stage 3 depth fit finding no valid reference pixels, and the Stage 5
coverage warning. That redundancy is the point.

## Known gaps

**No metric scale in the current footage.** There is no ARKit trajectory and no
ArUco marker in `wristview-videos/`, so Stage 1 has no scale source. It refuses
to continue by default, because a wrong scale fails silently and every
downstream distance inherits it. Fix by measuring one real distance and setting
`scene.scale.manual.known_distance_m` with
`reconstruction_distance_units`; Stage 1 logs the reconstruction's span and
camera path length to compare against.

**The synthetic fixture cannot validate hand estimation.** It renders the hand
as smooth capsules, which is out of distribution for detectors trained on
photographs. MediaPipe found it on 17 percent of frames, measured. The fixture
is a unit-test asset for the geometry: projection, trajectories, rigid
transforms, rendering. Hand estimation is validated on the real clips.

---

## Artifacts

Every stage writes a `meta.json` with its inputs, outputs, metrics, timings,
git SHA, config hash, and which backend actually ran. That last field is the
one to check: it says whether a stage used the preferred implementation or a
fallback.

```
runs/<run_id>/
  config.yaml         the resolved config, hashed into every meta.json
  sources.json        input videos and any ARKit trajectories
  wristview.log       the full run log
  run_summary.json    per-stage status and timing
  00_ingest/          frames, intrinsics.json, manifest.json
  01_scene/           colmap/, scene.ply, splat.pt, scale.json, camera_refined.json
  02_localize/        demo_<n>/camera_poses.npy, status.json
  03_estimate/        demo_<n>/hand.npz, object_masks/, object_pose.npy, overlay.mp4
  04_retarget/        demo_<n>/ee_trajectory.npz, trajectory.png, gripper.urdf
  05_render/          demo_<n>/wrist/%05d.png, wrist.mp4, contact_sheet.png
```

## Tests

181 tests, `ruff` clean.

```bash
uv run --python 3.11 python -m pytest tests/ -q
```
