# Phase 3: five seeds, 10,000 steps, three arms

Fifteen runs on five pods, one seed per pod, A_prime then B_prime then C
sequentially so all three arms of a seed share hardware. Clock started
2026-09-08T18:04:12Z and every pod was deleted by 22:19:53Z, six hours inside
the ten-hour cap. **$14.47 against a $35 cap.**

    encoder    R3M resnet18 (Ego4D), fine-tuned, 120 tensors verified per run
    steps      10,000, batch 32, eval every 500
    lr         1e-4, cosine, 500-step warmup
    holdout    3, 8, 13, 18, 23, 28, 33, 38, 43, 48
    score      mean of the last 6 validation checks, steps 7500 to 10000

## The seed table

    arm            s1      s2      s3      s4      s5    mean      sd
    A_prime      3.64    3.74    3.65    3.78    3.81    3.72   0.079
    B_prime      3.90    3.86    3.86    3.80    3.75    3.84   0.059
    C            3.77    3.68    3.72    3.70    3.61    3.70   0.062

## The paired differences

Per seed, so the shared dice cancel.

    pair            s1      s2      s3      s4      s5    mean     SE    2xSE  signs  verdict
    A' - C      -0.138  +0.060  -0.073  +0.080  +0.207  +0.027  0.061   0.121   3/5   NOT ESTABLISHED
    A' - B'     -0.266  -0.123  -0.217  -0.021  +0.062  -0.113  0.060   0.121   4/5   NOT ESTABLISHED
    C  - B'     -0.128  -0.184  -0.144  -0.101  -0.145  -0.140  0.013   0.027   5/5   **ESTABLISHED**

## What this establishes

**1. The real wrist camera adds nothing. `A' - C` is +0.027 mm and its sign
splits 3/5.**

This is the finding. C is a physical camera bolted to the arm, filming the
actual grasp. Against the ego view alone it is worth nothing, and it does not
even lean consistently in one direction across seeds.

Phase 2 could not distinguish `A' - C` from noise: the difference was 0.11 mm
against a within-run bounce of 0.82 to 1.12 mm. **Phase 3 measured it with a
noise floor twelve times tighter and found no effect at all.** The ceiling is
not merely hard to see. There is no ceiling.

**2. The recovery fraction is VOID, for a stronger reason than in Phase 2.**

Phase 2 voided it because the denominator sat inside the noise floor. Here the
denominator is +0.027 mm and NOT ESTABLISHED, so the quantity
`(A' - B') / (A' - C)` divides by something indistinguishable from zero. **No
improvement to the render could have produced a meaningful recovery figure**,
because there was never a gap for it to close.

**3. The real wrist view beats the rendered one. `C - B'` is -0.140 mm, 5 of 5
seeds, ESTABLISHED.**

The only established difference in the phase, and it is clean: every seed
agrees, SE 0.013, and the mean clears 2 x SE five times over. The render is
measurably worse than the real thing.

It is also a difference between two things that each add nothing over the ego
view. Both B_prime and C are inside the noise of A_prime. So this says the
render is an imperfect copy of a camera that was not helping either.

A candidate cause is on record from before any of these runs: the rendered
wrist channel reaches the backbone at mean [-1.201, -0.658, -0.175] against
the real channel's [0.096, 0.293, 0.207]. It is further out of ImageNet
distribution. That is not established as the cause; it is the first thing to
check if anyone tries to close the 0.140 mm.

**4. Every arm has drawn level with FLOOR-PERSISTENCE.**

    FLOOR-PERSISTENCE  3.72        FLOOR-MEAN  6.14
    A_prime            3.72   +0.00   2 of 5 seeds below
    B_prime            3.84   +0.12   0 of 5 seeds below
    C                  3.70   -0.02   4 of 5 seeds below

The essay's second finding, that repeating your last move beats a neural
network, was measured at 2,000 steps with a constant learning rate where every
arm lost to persistence by 1.8 mm. **At 10,000 steps with a decaying rate that
margin is gone**, and C is nominally below it. That section does not survive
this phase as written.

## The caveat that must travel with these numbers

**All fifteen runs were still descending at the cutoff.** Second-half slopes
run -0.064 to -0.122 mm per check, and every run's minimum falls at step 8,500
to 10,000. The score window, steps 7,500 to 10,000, sits on a curve that has
flattened but not stopped. A longer budget would move every arm down. Whether
it would move them differently is not known and this phase does not test it.

One seed each, five seeds, one holdout, one encoder, one dataset of 53
episodes.

## What the phase actually bought

The arms did not separate. The instrument did.

    A_prime, Phase 2   5.61 mm    2,000 steps, constant lr, one seed
    A_prime, Phase 3   3.72 mm   10,000 steps, cosine lr, five seeds

**1.89 mm from step count and a learning-rate schedule alone**, on the same
arm, the same encoder and the same data. That is ten times the entire
three-arm spread the earlier phases were straining to read.

    within-run bounce, Phase 2    0.82 to 1.12 mm
    between-seed sd, Phase 3      0.059 to 0.079 mm

The decision threshold went from unusable to 0.027 mm on the tightest pair.

## The near-miss, recorded because it nearly published

At three seeds, `A' - B'` read -0.266, -0.123, -0.217. Mean -0.202, SE 0.042,
2 x SE 0.083. **It cleared the threshold by more than double, with the sign
agreeing 3 of 3.** On that evidence the rendered wrist view hurts by 0.2 mm.

Seeds 4 and 5 then returned -0.021 and **+0.062, the opposite sign**. The mean
halved, the sd nearly doubled, and the difference stopped clearing 2 x SE.

**Three seeds would have published a false positive on the arm this project
cares about most.** The five-seed rule and the requirement that the sign agree
in 4 of 5 both existed before any number did, and between them they caught it.
