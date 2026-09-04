# Phase 1 result

Pod `veo3ib6djk3fyc`, RTX 4090, created 17:14 UTC, deleted 17:52 UTC.
**38 minutes from creation, $0.47 at $0.74/hr.** Nothing left running.

    lerobot 0.4.4, torch 2.10.0+cu128, seed 1, 2000 steps, batch 32
    dataset  53 episodes, 9,412 frames, alignment gate passed at lag 0, r +0.674
    holdout  3, 8, 13, 18, 23, 28, 33, 38, 43, 48   (1,860 rows)
    scored   action deltas 0 to 7, the window the policy actually predicts

## The gate

**A_prime does NOT beat FLOOR-MEAN. The gate FAILS.**

| | mean | median | p90 | last-6 val mean |
|---|---|---|---|---|
| **FLOOR-MEAN**, the bar | **6.14** | 4.41 | 12.01 | - |
| FLOOR-PERSISTENCE, reference | 3.72 | 2.15 | 8.16 | - |
| NULL, no camera | 6.63 | 5.18 | 12.48 | **7.07** |
| A_prime, ego camera | 6.69 | 5.07 | 12.53 | **6.97** |

Read as the protocol requires: the last-6 validation mean, **6.97 mm against a
bar of 6.14 mm**. It also loses on the final full evaluation, 6.69 against
6.14, so the reading does not depend on which of the two statistics is used.

Per CC-PHASE-0-AND-1.md: **STOP.** No B_prime, no C, no seeds 2 and 3, and no
adjustment of steps, config or holdout to get across the bar.

## The stronger finding: the camera is doing nothing

NULL now runs. It crashed in three minutes on 1 September and every run since
had no floor under it. With it, the two curves can be put side by side:

    step        250    500    750   1000   1250   1500   1750   2000
    NULL       8.49   8.50   7.10   7.50   7.15   7.24   6.90   6.54
    A_prime    8.74   8.56   7.08   7.13   6.72   7.27   7.06   6.57

They are the same line. Final full evaluation 6.63 mm for NULL against 6.69 mm
for A_prime, and **NULL is marginally the better of the two**. Both reach an
identical training loss of 0.0391.

A_prime carries 248.55 M trainable parameters, a ResNet18 encoder and a
640x360 ego view. NULL carries proprioception and nothing else. On this
dataset the ego camera contributes nothing distinguishable from noise.

That is a cleaner statement of the problem than the gate itself. The gate says
A' did not beat a trivial predictor. The NULL comparison says why: **no
information is reaching the policy from the pixels.** Running B' or C, which
add more cameras to the same harness, would have measured the same nothing at
three times the cost.

## What this does and does not establish

It establishes that with 53 episodes, 9,412 frames, 2,000 steps and this
configuration, an offline diffusion policy learns nothing from the ego view
that it does not already get from the state.

It does not establish that the render is bad, because B' never ran. It does
not establish that the data is bad. And it does not separate "not enough data"
from "not enough training" from "wrong architecture" - one seed, one
configuration, and a curve that was still drifting down at step 2000 for both
arms cannot separate those.

**Both arms were still improving when they stopped.** NULL fell 7.24 to 6.54
over its last three checks and A_prime 7.27 to 6.57. `still_falling_at_end` is
not read here, per the protocol, but the curves are stated so the reader can
see it. 2,000 steps was fixed by the real31 curves, where learning finished by
step 750; that is not what these curves look like. Whether that matters is a
question for Sidwyn, not a reason to quietly run longer.

## Both floors, again, because the ordering matters

FLOOR-PERSISTENCE scores **3.72 mm**, far below both arms and below the bar.
On this holdout, repeating the previous action is a much better predictor than
either trained policy, on every statistic.

The bar was set at FLOOR-MEAN on the argument that persistence "loses the
mean" - true on real31 at 14.71 against 12.64, and false here at 3.72 against
6.14. The gate was applied as instructed and would have failed against either
floor, so nothing turns on it this time. It will matter the moment an arm
lands between 3.72 and 6.14.
