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

**Splat depth ordering.** The guard against runaway Gaussians clamped a
Gaussian's tile span by moving its far corner to its near corner, collapsing it
onto a single tile so it vanished from the rest of its own footprint. A
Gaussian that fills the frame is not degenerate when the Stage 5 wrist camera
sits centimetres from a surface. The symptom was near geometry disappearing
behind far geometry. The clamp is now centred on the Gaussian.

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

128 tests, `ruff` clean.

```bash
uv run --python 3.11 python -m pytest tests/ -q
```
