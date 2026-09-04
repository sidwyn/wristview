# Phase 0: the holdout, the floors, and the bar

Written before any GPU started, as CC-PHASE-0-AND-1.md requires.

Dataset `runs/sept02_final/06_export/lerobot`, 53 episodes, 9,412 frames,
15 Hz, three camera channels.
Stage 6 status **ok**, `alignment gate PASSED: peak at lag 0, r +0.6740`.

## 1. The holdout

Ten episodes, every fifth of the sorted ids, not a consecutive block:

    3, 8, 13, 18, 23, 28, 33, 38, 43, 48

    training  43 episodes, 7,552 rows
    held out  10 episodes, 1,860 rows

53 is not divisible by 5, so "every fifth" yields eleven picks from offset 0
and ten from offsets 3 or 4.
Offset 3 was used, which gives the ten the instruction asks for and keeps the
picks off both ends of the session rather than biasing to one.

This list is now fixed.
Nothing below or after it may change the holdout.

## 2. The two trivial floors

Scored as an arm is scored: translation only, L2 in millimetres, over every
element of every predicted chunk of every held-out row.

| baseline | mean | median | p90 |
|---|---|---|---|
| **FLOOR-MEAN** | **6.14 mm** | 4.41 | 12.01 |
| FLOOR-PERSISTENCE | 3.72 mm | 2.15 | 8.16 |

Mean training action `[1.450, 0.009, -0.265]` mm.

### The scoring window, which changed and had to

The policy does not predict the whole recorded chunk.
`generate_actions` returns the horizon sliced
`[n_obs_steps - 1 : n_obs_steps - 1 + n_action_steps]`, which with
`action_delta_indices` of `[-1, 0, 1, ... 14]` is **deltas 0 to 7**.

`analytic_floors.py` scored the floors over all sixteen deltas, out to +14.
`train_arm.py` scored the arms over eight.
The floors carried the far end of the horizon, where every predictor is worse,
and the arms did not, so the floors were handicapped and the comparison was
never like for like.
Both now score deltas 0 to 7.

For comparison with real31's published numbers, the same holdout over the old
sixteen-delta window:

| baseline | mean | median | p90 |
|---|---|---|---|
| FLOOR-MEAN | 6.25 mm | 4.42 | 12.27 |
| FLOOR-PERSISTENCE | 5.31 mm | 3.09 | 12.10 |

The window costs FLOOR-MEAN almost nothing here, 6.25 against 6.14, and costs
persistence a great deal, 5.31 against 3.72.
That is the expected shape: repeating the last action tracks the next frame
well and decays over eight more.

### The tool was checked before it was believed

`tools/phase0_floors.py` reproduces every published real31full number from
`analytic_floors.py`, on real31's own episodes 13, 14, 15, over the window
that tool used:

    FLOOR-MEAN         12.64 mm   median 8.79   p90 28.67    published, matched
    FLOOR-PERSISTENCE  14.71 mm   median 7.94   p90 34.01    published, matched

It reaches those numbers by a different route: `analytic_floors.py` reads
rows through `LeRobotDataset.__getitem__`, which decodes three video channels
per row and discards them, and this one calls the dataset's own
`_get_query_indices` and `_query_hf_dataset` and touches no video at all.

## 3. The bar

**The Phase 1 pass bar is FLOOR-MEAN on the MEAN: 6.14 mm.**

A_prime must score below 6.14 mm mean.
Median and p90 are diagnostics.

### One thing that must be read before the result is

**The two floors have swapped order relative to real31, and the reasoning that
set the bar does not survive the swap.**

| | FLOOR-MEAN | FLOOR-PERSISTENCE | which wins the mean |
|---|---|---|---|
| real31, episodes 13-15 | 12.64 | 14.71 | FLOOR-MEAN |
| **sept02_final, this holdout** | **6.14** | **3.72** | **FLOOR-PERSISTENCE** |

CC-PHASE-0-AND-1.md sets the bar at FLOOR-MEAN because "persistence fails
where it matters. It cannot turn a corner, so it loses the mean, 14.71 against
FLOOR-MEAN's 12.64. FLOOR-MEAN has no such excuse."

On this dataset persistence does not lose the mean.
It wins it, by a wide margin, and it wins the median and the p90 as well.
So the empirical fact the bar was chosen on is not true here, and
FLOOR-MEAN 6.14 mm is now the EASIER of the two floors rather than the harder
one, on every statistic.

This is recorded, not acted on.
The instruction is explicit that the bar is FLOOR-MEAN on the mean, strictly,
and that persistence stays a reference benchmark.
Moving the bar is a change to a measurement and belongs to Sidwyn alone.
Phase 1 will be gated at 6.14 mm as instructed, and the persistence number
will be reported beside it so the reading is not lost.

The likely cause is the motion itself.
The same trivial predictor scores 12.64 mm on real31 and 6.14 mm here, so
sept02's actions are about half the size, and a smaller, smoother action
stream is exactly where a zero-order hold does best.

## 4. Not yet answered

`FLOOR-PERSISTENCE` at 3.72 mm mean is a hard number for any policy to beat at
15 Hz, and it is not the gate.
Whether an arm that beats 6.14 and loses to 3.72 has learned anything useful
is a question this run will have to put to Sidwyn, not answer on its own.
