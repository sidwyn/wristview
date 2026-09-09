# Phase 4a: is the render's deficit colour?

Ten runs, five pods, one seed per pod, B_ds then C_ds, both wrist channels
normalised by their own statistics instead of ImageNet's. Clock started
2026-09-09T03:48:07Z, last pod deleted 09:19:24Z, **$18.21 of a $25 cap**,
inside the 6-hour wall clock.

**Phase 4b returned nothing. See the section at the end.**

## The table

    arm             s1      s2      s3      s4      s5    mean      sd
    B_ds          3.84    3.84    3.86    3.84    3.79    3.83   0.027
    C_ds          3.75    3.74    3.84    3.77    3.74    3.77   0.042

    pair              s1      s2      s3      s4      s5    mean     SE    2xSE  signs  verdict
    C_ds - B_ds   -0.086  -0.097  -0.019  -0.069  -0.050  -0.064  0.014   0.028   5/5   ESTABLISHED
    B_ds - B_p3   -0.062  -0.022  -0.006  +0.034  +0.034  -0.004  0.018   0.036   3/5   NOT ESTABLISHED
    C_ds - C_p3   -0.020  +0.064  +0.119  +0.066  +0.129  +0.072  0.027   0.053   4/5   ESTABLISHED

The last two rows are seed-matched but **not hardware-matched**: the Phase 3
pods were gone before these ran.

Phase 3 reference, both channels on ImageNet statistics:
`C - B_prime = -0.140`, 5 of 5, ESTABLISHED.

## The reading, by the rule committed before the numbers

The plan gave three branches. This result is the third.

    C_ds - B_ds not established, and B_ds - B_p3 established negative
        -> paragraph A.  DOES NOT HOLD. C_ds - B_ds is established and
           B_ds - B_p3 is not.

    C_ds - B_ds still established near -0.14
        -> paragraph B.  DOES NOT HOLD. It is established at -0.064, less
           than half of -0.14.

    anything else
        -> report the table, use neither paragraph, one plain sentence.

**One plain sentence: normalising each wrist channel by its own statistics
halved the real-versus-rendered gap, from -0.140 mm to -0.064 mm, and the gap
remains established on 5 of 5 seeds, so the measurement did not resolve
whether colour is the cause.**

Neither pre-written paragraph goes into the essay. Writing a third one now
would be a paragraph composed after seeing the numbers, which is the thing the
two pre-written ones exist to prevent.

## Why the gap narrowed, which is the part that matters

The halving did not come from the render getting better.

                        B       C      gap
    Phase 3 (imagenet)  3.84    3.70   -0.140
    Phase 4 (dataset)   3.83    3.77   -0.064

    the render moved   -0.004 mm    NOT ESTABLISHED
    the real camera    +0.072 mm    ESTABLISHED, and worse

**Fixing the rendered channel's normalisation did nothing for the rendered
channel.** `B_ds - B_p3` is -0.004 mm with the sign splitting 3/2. The gap
closed only because the same change made the REAL camera worse by 0.072 mm,
established on 4 of 5 seeds.

So the essay's published guess, that the render is worse because its colours
sit off the scale the encoder expects, is not supported. The render was given
exactly the normalisation the guess says it needed and did not improve. What
changed was the real camera, in the wrong direction.

That is a stronger statement than "did not resolve", and it is why the plain
sentence above is deliberately narrower than what these numbers appear to say.
The rule was written to be applied, not argued with. **The essay paragraph is
Sidwyn's call**, and it should be made knowing that its current claim has been
tested and did not survive.

## What is still true

The real wrist view beats the rendered one, on every seed, under both
normalisations. That is now 10 of 10 seed-pairs across two phases. The size of
the gap depends on how the channels are normalised, from 0.140 mm to 0.064 mm.

Both arms remain inside the noise of the ego view alone. Phase 3 measured
`A' - C` at +0.027 mm, not established, so this is still a difference between
two things that each add nothing over a head camera.

## Phase 4b: VIP. Failed, and not for the reason forecast.

A_vip fails 13 seconds after it starts, on every pod:

    VIP load VERIFIED: 318 tensors, no unexpected or missing keys outside fc
    froze running statistics on 53 BatchNorm2d modules
    normalisation check PASSED
    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 106.00 MiB.
      GPU 0 has a total capacity of 23.53 GiB of which 103.69 MiB is free.
      This process has 23.42 GiB in use.

Every assertion the plan asked for passed. The weights load, the architecture
is right, the feature map is 2048 channels, the gradients flow. Then the first
forward pass at batch 32 does not fit on a 4090. Phase 3's resnet18 arms
peaked near 13 GiB.

**The dry run could not have caught it.** It ran on CPU at batch 4, because
that is what a laptop can do. Neither the plan's assertion list nor mine
included a peak-VRAM check at the real batch size on the real card. A plan
that adds a backbone should measure that on one pod before it buys five.

No A_vip run reached step 1, so there is no partial curve. This is a failure,
not a truncation, and 4b has no result.

**If 4b is wanted it needs its own phase and a decision first**: batch 16 for
VIP with a matching batch-16 R3M re-run so the pair is comparable, or gradient
accumulation to hold the effective batch at 32. Both are experiments, not a
flag change.
