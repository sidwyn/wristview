# Plan · replace the Metal rasterizer with gsplat

**Status: not started. This is a plan, not work in progress.**

## Why

`src/wristview/backends/splat_mps.py` is a differentiable 3D Gaussian
rasterizer written from scratch in PyTorch, because gsplat is CUDA-only,
gsplat-mlx will not build against current MLX, and Brush cannot be called per
pose from Python. It works. It is also the single riskiest file in the
codebase, and the bug record says so plainly:

| Bug | Symptom | How it was found |
|---|---|---|
| Densification threshold in pixels, not NDC | splat never grew, 13k points for 1000 steps | reading a log line that said `+0 cloned, +0 split` |
| Tile-span clamp anchored to a corner | near geometry vanished behind far geometry | a unit test written for depth ordering |
| Whole tile list materialised for backward | OOM at 51k Gaussians, PSNR fell as the splat grew | a training run dying |
| Compositing chunk too large for early exit | OOM again at 59k | measuring peak memory directly |
| Absolute densification threshold | worked at 0.30 m scene scale, selected nothing at 2.57 m | comparing two scenes |
| Opacity reset disabled to protect PSNR | floaters 2 cm from the camera, object placed at the camera | ground-truth accuracy harness |

Six defects, and every one produced **plausible but degraded output rather
than a crash**. A thin splat reads as "splatting is hard at close range", not
as a unit error. The cost of owning this file is not the code, it is that
every convention in 3DGS has to be independently rediscovered and verified.

gsplat is maintained, widely used, and its conventions are the reference ones.
The moment a CUDA machine is in the loop, the Metal rasterizer should stop
being load-bearing and become a fallback.

## What must not happen

The Metal rasterizer must not be deleted. Apple Silicon is the development
machine, and a pipeline that cannot run locally is a pipeline nobody iterates
on. The goal is **two interchangeable backends behind one interface**, chosen
by device, not a migration away from one.

## The interface boundary

Two call sites use the rasterizer, and they are the entire surface area.

```
Stage 1  backends/splat_trainer.py     trains: render, backward, densify
Stage 5  stages/s05_render.py          renders: one image from one pose
```

Both go through `render()` and the `GaussianModel` parameters. So the boundary
is already close to right. Formalise it as a protocol:

```python
class SplatBackend(Protocol):
    def create(self, points, colors, sh_degree, device) -> GaussianParams: ...
    def render(self, params, view_matrix, intrinsics, near, far,
               background) -> RenderResult: ...          # differentiable
    def densify(self, params, optimizer, stats, config) -> DensifyStats: ...
    def save(self, params, path) -> None: ...
    def load(self, path, device) -> GaussianParams: ...
```

`RenderResult` already carries `rgb`, `alpha`, `depth`, `visible`. Stage 3 and
Stage 5 both depend on `depth` being metric and on `alpha` marking coverage, so
those are contract, not implementation detail.

Three things are currently implicit and must become explicit in the contract,
because each one was a bug:

1. **Screen-space gradient units.** gsplat reports NDC. The Metal backend
   reports pixels and converts in `Densifier.accumulate`. The protocol must
   state NDC and each backend converts internally.
2. **Near-plane semantics.** Stage 3 passes `near=0.3` to cull floaters, not
   to clip geometry. Both backends must honour it identically.
3. **Gaussians per tile.** The Metal backend caps at `max_per_tile`; gsplat does
   not. A capped backend must report when it saturates, so quality loss is
   visible rather than silent.

## Migration steps

1. **Extract the protocol** and make `splat_mps` implement it. No behaviour
   change. Tests stay green. This is worth doing even if the port never
   happens, because it names the contract.
2. **Add a golden-image test.** Fix a small Gaussian set and a set of poses,
   render, and store the images. Both backends must agree to a tolerance. This
   is the whole safety net for the port: without it, "gsplat works" means
   nothing.
3. **Add the gsplat backend** behind `scene.splat.backend: gsplat`, importable
   only when CUDA is present. Keep `mps` the default on Apple Silicon.
4. **Run both on the same reconstruction** and compare PSNR, Gaussian count,
   training time, and the Stage 5 wrist renders. Record the numbers in
   BUILD-LOG.md. If they disagree, the Metal backend is wrong until proven
   otherwise, since gsplat is the reference.
5. **Re-run the ground-truth accuracy harness** under both. Object position and
   grasp agreement are the rows that exercise splat depth, so they are the ones
   that would catch a divergence Stage 5 renders would hide.
6. **Switch the default by device.** CUDA gets gsplat, MPS keeps the local
   rasterizer, and `meta.json` records which one ran, as it already does for
   every other backend.

## What this costs

Steps 1 and 2 are a day and are worth doing regardless: they turn the current
implicit contract into a tested one. Steps 3 to 6 need a CUDA machine and
should not start before one is in the loop.

## What would change the plan

If a Metal-native 3DGS library becomes installable and can render from an
arbitrary pose through a Python call, that is a better answer than either
option here, because it removes the CUDA dependency without keeping a
hand-written rasterizer. gsplat-mlx was the closest candidate and does not
currently build; it is worth re-checking before starting step 3.
