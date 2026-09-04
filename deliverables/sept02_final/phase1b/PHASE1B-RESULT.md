# Phase 1b result: the pretrained encoder probe

Pods `btwdzn6ov4mbx2` (ImageNet) and `0bnskekoikmoix` (R3M), both RTX 4090.
**Both deleted. `runpodctl pod list` returns `[]`.**

All four runs: A_prime, ego camera only, seed 1, 2000 steps, batch 32, eval
every 250, holdout 3/8/13/18/23/28/33/38/43/48, lerobot 0.4.4. Nothing else
moved.

## The numbers

The reported figure is the mean of the last 6 validation checks, as the
protocol requires. `still_falling_at_end` is not read.

| run | mean | median | p90 | last-6 |
|---|---|---|---|---|
| FLOOR-PERSISTENCE | 3.72 | 2.15 | 8.16 | reference, not a gate |
| FLOOR-MEAN | 6.14 | 4.41 | 12.01 | the Phase 1 bar |
| **NULL, no camera** | **6.63** | 5.18 | 12.48 | **7.07** |
| A_prime scratch (Phase 1) | 6.69 | 5.07 | 12.53 | 6.97 |
| ImageNet FROZEN | 6.51 | 5.00 | 12.06 | 6.80 |
| ImageNet fine-tuned | 6.38 | 4.82 | 12.07 | 6.77 |
| R3M FROZEN | 6.42 | 4.86 | 12.25 | 6.90 |
| **R3M fine-tuned** | **5.61** | **4.12** | **10.41** | **6.20** |

**All four beat NULL and all four beat A_prime scratch, on every statistic.**
R3M fine-tuned is the best by a clear margin: 5.61 mm against NULL's 6.63, and
it is the only run to beat FLOOR-MEAN's 6.14 on the final evaluation.

Against the brief's decision table:

- **any run beats 6.63**: yes, all four.
- **frozen beats fine-tuned**: NO, the reverse, for both backbones. Fine-tuning
  did not overwrite the representation on this data.
- **R3M beats ImageNet**: yes. 5.61 against 6.38 fine-tuned, 6.42 against 6.51
  frozen. The Ego4D domain match is doing real work on head-mounted video,
  which is the interesting reading the brief flagged in advance.

## THE RESULT IS CONFOUNDED, AND THE CONFOUND IS MINE

**Phase 1b changed two things at once, not one.**

It changed the backbone from random to pretrained, which is the experiment. It
also fixed the input normalisation, which the brief required and which turned
out to be far more broken than the brief knew: lerobot writes video statistics
through `RunningQuantileStats`, which squares a uint8 frame, so 255 becomes 1
mod 256, the variance goes negative and the sqrt guard clamps it to near zero.
This dataset's ego std is [0.0171, 0.0127, 0.0111] where the true per-pixel std
is [0.242, 0.215, 0.194]. `MEAN_STD` was dividing by a number 14 to 20 times
too small and handing the encoder values in [-46, +44] instead of [-2, +2].

Phase 1 ran with that. Phase 1b did not.

So the comparison against NULL 6.63 and A_prime scratch 6.69 is not clean.
Some part of the gain from 6.69 to 5.61 belongs to pretraining and some belongs
to giving the encoder images in range at all. **This run cannot say how much
of each**, because the control that would separate them was not run.

**The missing control is one run and about $0.25: A_prime, scratch weights,
with the normalisation fix.** Until that exists, the honest claim is "pixels
help once the images are correctly normalised and the encoder is pretrained",
not "pretraining was the bottleneck". I did not run it: it is outside what the
brief authorised, and Phase 2 is Sidwyn's call.

## The three mandatory assertions, all of which ran and printed

**BatchNorm frozen, and still frozen inside the loop.** Every run:

    froze running statistics on 20 BatchNorm2d modules
    BatchNorm freeze ASSERTED in the training loop: all 20 modules still in eval mode

`use_group_norm=False` on all four, as required with pretrained weights. The
assertion runs every step, not once at construction, because `policy.train()`
recurses and is called again after every evaluation.

**R3M state dict verified, not assumed.** Both R3M runs:

    R3M: 120 keys, missing ['fc.weight', 'fc.bias'], unexpected [],
         differs from fresh init True

120 of 120 backbone tensors filled. The only missing keys are the classifier
head, which lerobot strips along with avgpool, so its absence is expected and
is the only absence that is. `conv1` was compared against a freshly
initialised resnet18 and differs, so the weights actually moved.

The load goes through a real `torchvision.models.resnet18` and then rebuilds
the same `nn.Sequential(*children[:-2])` lerobot builds, so the positional
index mapping is torchvision's own and cannot drift.

**What the backbone actually receives.** One real batch, through the real
preprocessing, on every run:

    mean [0.096, 0.257, 0.351]   std [1.056, 0.966, 0.839]

Near zero mean and unit variance under ImageNet statistics. Before the
override the same batch would have arrived at roughly 15x that scale. The run
refuses to train if this check fails.

## Hardware

Identical except the driver:

| | pod A, ImageNet | pod B, R3M |
|---|---|---|
| GPU | RTX 4090 | RTX 4090 |
| **driver** | **570.195.03** | **580.159.04** |
| CUDA | 12.8 | 12.8 |
| cuDNN | 91002 | 91002 |
| torch | 2.10.0+cu128 | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 | 0.25.0+cu128 |

A driver minor version is not a plausible cause of a 0.77 mm difference, and
the frozen/fine-tuned pair that matters most sits WITHIN each pod, so the
comparison the brief cared about is unaffected. Recorded because it was asked
for and because "same pod type" turned out not to mean "same driver".

## Read the curves, not the flag

    ImageNet FROZEN       8.46 9.05 6.87 7.08 6.55 7.27 6.64 6.41
    ImageNet fine-tuned   8.14 9.28 7.04 6.95 6.47 7.46 6.39 6.32
    R3M FROZEN            8.21 8.53 6.93 7.17 6.72 7.20 7.08 6.33
    R3M fine-tuned        8.12 8.04 6.55 6.45 6.15 6.63 5.87 5.56

**R3M fine-tuned is still falling at step 2000** and is the only curve whose
last three checks are all below every earlier one. Its final training loss,
0.0381, is the lowest of the four but not by much, so this is not memorisation
showing up as a validation gain.

The step budget was not extended, as instructed.

## Caveats a reader must carry

One seed. One configuration. No error bar. A 0.77 mm gap between R3M
fine-tuned and ImageNet fine-tuned is not established as larger than seed
noise, and Phase 1 has no repeat measurement to estimate that noise from.

R3M fine-tuned's last-6 mean of 6.20 is marginally ABOVE FLOOR-MEAN's 6.14. It
beats the floor on the final evaluation, 5.61, and loses to it on the
smoothed statistic. Both are reported here rather than whichever is friendlier.

Every arm still loses to FLOOR-PERSISTENCE at 3.72 mm.
