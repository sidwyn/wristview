# Phase 2: the three-arm comparison

Pod `h0zms9f1y5vsd3`, RTX 4090, created 13:34 PT, deleted 15:08 PT.
About 93 minutes from creation, about **$1.15** against a $2.00 cap.
B_prime and C ran sequentially on ONE pod so the experiment's own comparison
shares hardware.

    dataset      runs/sept02_final/06_export/lerobot, 53 episodes, 9,412 frames
    holdout      3, 8, 13, 18, 23, 28, 33, 38, 43, 48   (1,860 rows)
    encoder      R3M resnet18 (Ego4D), fine-tuned, selected by Phase 2a
    seed 1, 2000 steps, batch 32, lerobot 0.4.4, ImageNet normalisation override

## Every arm, one dataset, one holdout

| arm | mean | median | p90 | last-6 | 2nd-half slope |
|---|---|---|---|---|---|
| FLOOR-PERSISTENCE, reference | 3.72 | 2.15 | 8.16 | - | - |
| FLOOR-MEAN, the bar | 6.14 | 4.41 | 12.01 | - | - |
| NULL, no camera | 6.63 | 5.18 | 12.48 | 7.07 | -0.217 |
| A' scratch, broken normalisation | 6.69 | 5.07 | 12.53 | 6.97 | -0.067 |
| A' scratch, normalisation FIXED | 6.56 | 4.98 | 12.13 | 7.05 | -0.193 |
| **A' ego only, R3M ft** | **5.61** | 4.12 | 10.41 | 6.20 | -0.254 |
| **B' ego + RENDERED wrist, R3M ft** | **5.70** | 4.17 | 10.77 | 6.22 | -0.274 |
| **C ego + REAL wrist, R3M ft** | **5.50** | 4.10 | 10.05 | 6.02 | -0.132 |

    A' ego, R3M ft    8.12 8.04 6.55 6.45 6.15 6.63 5.87 5.56
    B' ego+rendered   7.94 7.04 6.39 6.42 6.26 6.77 5.83 5.66
    C  ego+real       9.46 7.48 6.56 6.20 5.75 6.30 5.80 5.48

## The recovery fraction is VOID

                    A' - C      A' - B'     recovery
    on mean         +0.109      -0.089      -0.815
    on last-6       +0.184      -0.021      -0.111

    within-run spread, last 4 checks:  A' 1.06   B' 1.12   C 0.82 mm

**The denominator is 0.11 to 0.18 mm. The noise inside a single run is 0.82 to
1.12 mm.** The noise is five to ten times the entire prize.

The rule that voids the fraction when the denominator sits inside the noise
floor was recorded before these numbers existed. It fires. **Neither -0.815
nor -0.111 is a recovery figure and neither should be quoted as one.**

Sidwyn wrote the same caveat in the essay before any arm had run: "If the
denominator A - C is too small where a real wrist camera turns out to not be
of much help, a tiny change in B swings it wildly." It was correct.

## Three things that did resolve

**1. Pretraining works.** NULL 6.63 to A' 5.61, a gap of 1.02 mm on a
1-mm-scale problem. It is the only gap in this project larger than the noise.
Phase 2a established it is pretraining and not the normalisation repair: a
scratch encoder with the overflow fixed scored 6.56 mean and 7.05 on last-6,
against NULL's 7.07.

**2. The rendered wrist view adds nothing.** B' 5.70 against A' 5.61. Worse,
by less than the noise.

**3. The real wrist view adds almost nothing either.** C 5.50 against A' 5.61.
Better, by less than the noise.

**Point 3 is the finding.** C is the ceiling: a real camera on a real arm
filming the real grasp. It moves the metric by 0.11 mm. **There was never a
gap for a render to close.** No improvement to B' could have produced a
meaningful recovery figure, because the instrument cannot resolve what the
experiment was built to measure.

That is a statement about the measurement, not about the render. **The render
was never on trial.**

The ordering does hold, C < A' < B' on the mean and C < A' on every statistic
including p90 and last-6. The direction is what the experiment predicted. The
magnitude is not measurable.

## Caveats that travel with these numbers

- **One seed. No error bar.** A 0.20 mm spread across three arms cannot be
  separated from run-to-run variation.
- **Every arm still descending at step 2000**, slopes -0.13 to -0.27. These
  are un-converged runs compared at a common cutoff. Fair, and weak.
- **FLOOR-PERSISTENCE at 3.72 mm still beats every arm by 1.8 mm.** Repeating
  the previous action remains the best predictor on this dataset.

## The follow-up worth running, written and not run

Not another arm. **Seeds 2 and 3 for A', B' and C**: six runs, about 3 GPU
hours, about $2.50. Every conclusion above turns on a 0.20 mm spread with no
error bar. If that spread is inside seed noise, the honest headline is that
this experiment cannot answer its question at 53 episodes with this metric,
which is a real result and cheap to establish.

## What the staged gates bought

The corruption was not confined to the ego channel: ego 14.1x, wrist 15.4x,
wrist_real 9.7x. Had B' and C been funded on the original schedule, all three
arms would have trained through destroyed normalisation, the comparison would
have returned null, and the write-up would have concluded that rendered wrist
views do not work.

The stop rule that kept B' and C unfunded until A' had been understood is the
only reason that did not happen.
