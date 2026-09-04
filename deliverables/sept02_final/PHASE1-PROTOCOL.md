# Phase 1 protocol, recorded before the result

Written while the runs are in flight, so the reading of the answer cannot be
chosen after the answer is known.

## What is running

Pod `veo3ib6djk3fyc`, RTX 4090 24564 MiB, created 2026-09-04 10:14 PDT.
Payload sha256 `c4f47b3c260afe04a9f15e5eada00f012a1c152cbb6a87f1684d74f868957930`,
251,862,955 bytes, hash verified on both ends before anything ran.

    lerobot   0.4.4, pinned, the version verified on CPU here first
    torch     2.10.0+cu128 on the pod
    dataset   runs/sept02_final/06_export/lerobot, 53 episodes, 9,412 frames
    holdout   3, 8, 13, 18, 23, 28, 33, 38, 43, 48
    arms      NULL then A_prime, seed 1
    steps     2000, batch 32, eval every 250
    down_dims default, encoder trained

One configuration. No sweep. No second seed.

## The gate

**Does A_prime beat FLOOR-MEAN on the MEAN?  Bar: 6.14 mm.**

- **No** -> STOP. No B_prime, no C, no seeds 2 and 3, and no adjustment of
  steps, config or holdout to get across the bar. The finding is that 53
  episodes is still not enough to learn this task offline.
- **Yes** -> stop anyway. Sidwyn authorises Phase 2 separately.

## How the number is read

`train_arm.py` writes a full `val_curve`. The reported figure is the **mean of
the last 6 validation checks**, not the final one.

`still_falling_at_end` is ignored. It reads four endpoints at
`train_arm.py:176`, and on real31 it fired on a 0.004 mm margin and produced a
false undertrained conclusion. The curve is read instead.

Median and p90 are diagnostics. **Losing the median to FLOOR-PERSISTENCE does
not trip the gate** and is expected at 15 Hz.

## What is already known and must not be forgotten when the number lands

FLOOR-PERSISTENCE scores 3.72 mm mean on this holdout, better than FLOOR-MEAN's
6.14 on every statistic. That is the reverse of real31, where persistence lost
the mean, and it is the empirical fact the choice of bar was argued from. The
gate stays at 6.14 mm because that is the instruction. The persistence number
is reported beside every arm so the reading is not lost.

An arm between 3.72 and 6.14 passes the gate as written and still loses to a
zero-order hold. If that happens it is reported as exactly that, and the
question of what it means goes to Sidwyn.

## Prediction, on record, before the result

`runs/real31full/PREDICTION-arms.md` predicted that if the rendered wrist view
does real work, B' improves the MEAN much more than the MEDIAN relative to A'.
Phase 1 does not run B', so it cannot test that. What Phase 1 answers is the
prior question that prediction assumes: whether ANY arm beats a trivial
predictor on this data.

On real31 none did. All nine runs lost to FLOOR-MEAN on both statistics.
