# Defect table

Each row records a check that reported a wrong answer, and the reason.

The rows share a family. A measurement stood in for the thing it was meant to
test. The substitute agreed with the real quantity most of the time, so nothing
looked wrong until it disagreed.

**One sub-family is the most repeated defect in this project, and it is worth
naming on its own: the code computed the exact number that described the
failure, and then no code read it.** Not an approximation of that number, not a
proxy. The number itself, correct, in memory, discarded.

Rows 7, 9, 10, 11, 12, 15, 40, 43, 44 and 48 are all this. The trainer measured 17.51 dB on its
own training views and nothing compared it to anything. The rasteriser counted
11,076,570 dropped Gaussian-tile pairs, stored the count on the result object as
`_overflow`, and no caller ever looked. `train_gsplat.py` printed a Gaussian
count every 1,000 steps while the count sat unchanged for 30,000 of them.
Stage 5 recorded splat alpha coverage of 100 per cent while drawing the wrong
Gaussians at every pixel.

The lesson is not "measure more". The measurement was always there. It is that a
computed diagnostic with no reader is not a check, and writing one costs the
same as writing a check while buying nothing. **If a number is worth computing,
something must fail on it, or it must not be computed.**

| # | Defect | What it measured | What it should have measured | Cost |
|---|---|---|---|---|
| 1 | Hand lift scaled all three translation components by `fx / wilor_focal` | the hand at the right depth | the hand at the right depth AND the right pixel | 25x lateral collapse; grasp never worked on any session; 614 px reprojection |
| 2 | Object height gate checked the centre sat at half the object height above the desk | a quantity the solver computes directly | whether the tracked height ever changes | passed at zero error on every frame of a real defect |
| 3 | Hand identity guard counted label changes | the label of the selected hand | whether the selector crossed between two hands | rejected a single-hand clip; under label selection the counter is vacuous |
| 4 | Stage 5 coverage warning read the splat's alpha | that a Gaussian was drawn at a pixel | whether a camera was ever positioned to constrain it | 87-94% alpha while every frame extrapolated 40 cm |
| 5 | Pre-flight reported the median of 6 samples | the middle of the distribution | the worst case and its run length | passed a clip at 389 matches that Stage 2 then rejected at 73.4% |
| 6 | Workspace detection keyed on marker visibility | whether the marker was in frame | whether the frame can localize against the scan | the hand covered the marker for 4.25 s, longer than either 2.3 s and 3.0 s sync segment; no bridge length separates the two |
| 7 | Audio presence checked with `ffprobe ... \| head -1` | the first line of output | whether an audio stream exists | reported "no audio" for two of three files |
| 8 | `--keep-going` let Stage 4 run on stale Stage 3 output | that Stage 4 completed | that its inputs were current | four clean-looking trajectories from a crashed upstream stage |
| 9 | Palm-frame conditioning check | a constant of the MANO hand model | whether the observation constrains the palm frame | could never detect what it was asked to detect |
| 10 | Velocity gate divided a frame-index gap by a scalar fps | frame index treated as a clock | the real interval between kept frames | Stage 0 drops frames, so gaps of 1.25 s read as 0.059 s |
| 11 | Splat quality judged by alpha coverage and train PSNR at 540 px | that Gaussians covered the frame | whether the right Gaussians were there | a splat scoring 17.4 dB on its own training views shipped as a render |
| 12 | `init_points: 60000` applied as a ceiling | the largest cloud allowed | the cloud the optimiser starts from | real26 seeded 8810 and nothing said the target was missed |
| 13 | `train_gsplat.py` skipped an unreadable training image with `continue` | that the loop finished | that every registered view was loaded | a wrong image directory would train on zero views in silence |
| 14 | `train_gsplat.py` read `undistorted/` and `cameras_pinhole.json` from a script that was never written | that undistortion was handled | whether it ran | every GPU run so far fitted a pinhole model to radially distorted images |
| 15 | MPS rasteriser dropped Gaussians past a per-tile cap and stored the count in an unread `_overflow` | a render | a render of the whole splat | scored the 30.46 dB GPU splat at 7.21 dB and called it broken |
| 16 | Splat coverage taken from the depth channel of an external render | that a ray met something | whether a surface was there | read 100 per cent where alpha was 0.911, min 0.625 |
| 17 | Stage 3's plane branch recorded the object as `np.zeros((1, 3))` | that an object pose existed | the object's shape | every wrist view drew a 5 mm dot; no clip ever contained a moving object |
| 18 | Velocity gate rejected the whole clip on a failed run gate | that some frames were implausible | which frames were implausible | 5 bad frames of 322 cost 148 tracked frames and forced a manual trim |
| 19 | Hand-identity gate rejected the whole clip on any side switch | that two hands appeared | which frames belonged to which hand | 6 Left frames of 325 discarded all 325 |
| 20 | `slerp_fill` held the last measured pose across the unmeasured tail | a pose for every frame | a pose only where the hand was seen | 181 rendered frames with bit-identical camera poses, none of them data |
| 21 | Stage 5 read `hand_valid` in neither branch | that a pose existed for the frame | whether anything measured that pose | drew all 387 frames including 181 frozen ones |
| 22 | `--set` overrides reached no file in the run directory | the config on disk | the config that actually ran | a 57-231 trim was invisible; explaining its effect needed a stage re-run |
| 23 | `StageMeta.to_dict` hand-listed its fields | the fields someone remembered to list | the record's contents | a new `config_overrides` field was dropped in silence the day it was added |
| 24 | `resting_pose` returned the first sustained still run | that the object was still somewhere | whether it was still BEFORE the grasp | real26/bm demo_1 measured the rest position 117 frames after the release and carried the object 7 cm under the desk |
| 25 | `render.wrist_camera.standoff_m` was optional and null | nothing, and warned about it | the camera-to-fingertip distance the rig sets | every take of real06b and real26 shipped with the wrist camera unchecked |
| 26 | Blur threshold taken from the whole clip's median | the average sharpness of a mixed scan | whether a frame is soft for its own pass | appending a sharp pass cost the low pass 18 frames and raised the floor 8.47 to 12.15 cm on identical footage |
| 27 | Grounding DINO's argmax box returned whatever scored highest | the best-scoring box | whether anything was found | a 98 per cent-of-frame box read as "the carton is in every scan frame" when it was not on the mat |
| 28 | Scan height measured from marker-visible frames only | the frames that can see a flat marker | how low the camera went | a frame below 30 cm is 0.15x as likely to show the marker; 45.7 per cent true reads as 11.4 per cent |
| 29 | The marker is treated as always available | that a fiducial was placed | whether anything can see it | id 0 found in 14 of 30 demo first frames; the scan render-band check unmeasurable on two scans running |
| 30 | `estimate.object.prompt` was declared, documented, and read by nothing | the config file | what Stage 3 hands the detector | real27 sent the whole task sentence to Grounding DINO and got a box covering 93 per cent of the frame |
| 31 | The rest-window guard required the still run to END before contact | whether the run overlapped the grasp | whether ENOUGH of it preceded the grasp | refused a valid take whose object sat still from frame 0 to 90 with contact at 75 |
| 32 | The trainer printed a Gaussian count every 1,000 steps and had no ceiling and no checkpoint | that densification was happening | whether the run would fit in the card | real27full reached 8,126,424 Gaussians and 18.9 GiB at step 10,000 with 5,000 refinement steps left, on a 24 GB card holding no checkpoint |
| 33 | The chain watcher grepped the pod transcript for its own completion token | whether the chain had finished | whether the chain had finished, as distinct from whether the watcher had asked | declared success 37 seconds in, matching the token inside its own echoed command |
| 34 | `render.wrist_camera` places the camera by mount offset with no floor | where the camera sits relative to the hand | where the camera sits relative to the desk | 309 of 4,673 exported wrist cameras, 6.6 per cent, sit below the desk plane, the lowest at -20.3 cm |
| 38 | The inlier RATIO rose when retrieval was thinned, while the poses got worse | how many attempted matches survived | whether the poses are right | at `retrieval_top_k=5` demo_22's ratio improved 0.8276 to 0.8687 while it lost 21 of 171 registered frames and a surviving frame moved 36.2 cm and 40.1 degrees |
| 39 | The trainer sized its Gaussians with one dense `cdist` over every seed point | the mean distance to the three nearest neighbours | the same thing without allocating N squared floats | real28scanb seeded 78,961 points and asked for 23.23 GiB on a 23.53 GiB card, dying before step 1; real27full's 36,962 points needed 5.5 GiB and fit, so it had never shown |
| 36 | The config comment claimed the finger tips land 77 per cent down the frame | nothing, it was a comment | where the mount actually puts them, 86.6 per cent | the framing claim went unchecked from the first commit, and narrowing the lens to the real camera's 62.1 degrees would have put the gripper out of frame entirely |
| 37 | WiLoR reports confidence 1.000 on every valid frame of every clip | nothing | how much to trust a detection | two phantom hands in demo_28, on desk clutter with 5 of 21 landmarks in the image, entered the pipeline at full confidence and were reported as a badly shot take |
| 35 | The viewpoint-coverage gate runs in Stage 5, after the GPU render | whether the render extrapolates | whether the render WILL extrapolate, while the GPU is not yet rented | real27full trained 40 minutes and rendered 14,019 frames that fail the project's own coverage limits: nearest scan view median 6.8 cm but p90 24.2 cm against a 15 cm limit, and 22.2 per cent of frames below the scan floor against a 10 per cent limit |
| 40 | `scan_geometry` computes `height_min_m` and gates on `max_baseline_m` and `height_span_m` | whether the scan is VARIED enough | whether the scan went LOW enough | sept02_scan2 reported `height_min_m` 0.1972 m at Stage 1 and passed; that same number became `scan_floor_m` verbatim and failed 56 of 60 clips four hours and one pod later. Across eight reconstructions the gate has never once fired |
| 41 | `close_range_coverage.py` measures the distance from the lens to the MARKER; the gate measures the distance to the nearest SCAN VIEW | how close the scan got to the marker, against 0.45 m | how close the scan got to where the wrist camera will fly, against 0.15 m | two tools named coverage, thresholds 3x apart, and the loose one runs first. It reported 0.290 m and "passes comfortably" on the scan that then failed 57 of 60 clips. Its own docstring names 0.15 m in its first paragraph |
| 42 | The capture SOP fixed the marker at a fifth of the frame width | that the scan got close to the marker | where the lens physically stood | at f 1465 px on a 1920-wide frame, a fifth is 384 px, so a 100 mm marker pins the camera at 1444 x 0.1 / 384 = 0.376 m. The scan's low frames landed at a p90 of 0.38 m against a gate wanting 0.15-0.25 m. A framing rule silently set a position, and the two were never compared |

## Row 7, in detail

The command was:

```
ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate \
  -of csv=p=0 FILE | head -1
```

The ego and scan files carry three streams: video, audio, and a one-frame
timecode data track. The wrist file carries two. ffprobe 9.0.1 emits an empty
record for the extra stream, so its csv output starts with a blank line:

```
ego  : \n a a c , 4 8 0 0 0 \n
wrist:    a a c , 4 8 0 0 0 \n
```

`head -1` returned the blank line. The check reported "no audio" for exactly
the two files that had the extra stream, and worked on the third.

The answer was well formed. The check truncated it.

The pipeline's own `videoio.probe` was never affected. It parses
`-print_format json -show_streams` and filters on `codec_type`.
`videoio.audio_streams` and `videoio.has_audio` now use that same path, so a
text-mode check cannot report this again.

**Cost:** one sync channel was declared impossible. Audio cross-correlation
across both streams in fact works, and gives an offset of +3.779 s with 1 ms
agreement between the two taps. The visual pin that replaced it was wrong by
1.170 s, because a marker-centroid dip fired on the hand entering frame rather
than on the tap.


## Row 9, in detail

`palm_rotations` builds its frame from landmarks 0, 5 and 9. The check asked
whether those points become collinear, which would leave the frame's rotation
poorly determined.

Measured over 322 frames of session real26:

    flagged frames    sin(angle between 5-0 and 9-0)   0.2657
    unflagged frames                                   0.2658

Constant to four decimal places. p10 was 0.2655 against 0.2656.

WiLoR returns a MANO hand. MANO carries a fixed palm shape, set by model
parameters rather than by the image. The angle between those two bones
therefore cannot vary with the observation.

The check measured a property of the model. It could not have measured the
data. A zero-variance result is the signature, and it is why this project now
requires any implausibly low variance to be flagged rather than reported.

**Cost:** none directly, because the check was run as a diagnosis rather than
as a gate. It would have been an expensive gate: it would have passed every
frame forever.

## Row 10, in detail

    seconds = np.diff(indices) / max(fps, 1e-6)      qc.py, before the fix

`indices` counts kept frames. Stage 0 removes blurred frames, so kept frames
are not evenly spaced in time. The manifest records the real times in
`frame_times_s`, and the gate never read them.

Session real26 measured two steps spanning 1.25 s and 1.55 s of real time.
Read as one frame apart at 17 fps, the hand appeared to move at about 4.8 m/s.
The true speeds were 0.23 m/s and 0.12 m/s.

The same error appeared in `s02_localize` for median and maximum camera speed,
which feed the `max_camera_speed_m_s` plausibility check.

**Cost:** the translation figures were wrong. The rejection they were blamed
for was not caused by them: that gate measures palm rotation, and its flagged
steps all sat at normal 0.050 s spacing. Fixing the time base raised the
flagged count from 3 to 5, because the true intervals there are shorter than
the nominal one.

| 43 | `WiLoRHands.process` caught every per-frame exception, logged it at DEBUG, and returned the same empty frame a hand-free image gives | how many frames a hand was DETECTED in | how many frames the detector was ASKED and RAISED in | a torch downgrade made the MPS conv2d path raise on all 291 frames of demo_0; Stage 3 computed `hand detected on 0/291 frames (0%)`, logged it at INFO, and ran on into Stage 4. Ten hours were spent blaming the object dimensions |
| 44 | `summarise` sets `passed = median <= limit` and writes `p90_px` and `max_px` beside it | whether the TYPICAL frame reprojects | whether ANY frame reprojects badly | `max_px` separates the two runs cleanly and the gate never reads it: on the healthy scan2 run 2 of 60 clips exceed the 25 px limit, worst 28.62; on the run with the broken hand path 56 of 60 exceed it, worst 55.17. Both report `"passed": true` on 60 of 60. The median is 0 of 60 over the limit in BOTH runs, so the statistic the gate does read has no discriminating power at all |
| 48 | `RunningQuantileStats.update` computes `np.mean(batch**2)` on the raw uint8 frame | the mean of squares, for a variance | the mean of squares of values that fit in the type | 255 squared wraps to 1 mod 256, so the variance comes out negative and `np.sqrt(np.maximum(0, variance))` clamps it to near zero. Every video-channel `std` this dataset carries is 14 to 20 times too small: ego is [0.0171, 0.0127, 0.0111] against a measured true per-pixel [0.242, 0.215, 0.194]. MEAN_STD normalisation therefore handed the encoder values in [-46, +44] instead of [-2, +2], in Phase 1 as well as Phase 1b |

## Row 48, in detail

This one is upstream, in lerobot, and it is the cleanest example in this file
of the rule at the top of the page: a number was computed, written to disk,
read by the training code, and never once compared to the thing it claimed to
describe.

`lerobot/datasets/compute_stats.py`, in `RunningQuantileStats.update`:

    self._mean = np.mean(batch, axis=0)
    self._mean_of_squares = np.mean(batch**2, axis=0)

`batch` is the raw video frame, dtype uint8. `batch**2` is evaluated IN uint8.
255 squared is 65025, which wraps to 1. 200 squared is 40000, which wraps to
64. So the mean of squares is smaller than the square of the mean, the
variance

    variance = self._mean_of_squares - self._mean**2

comes out negative, and the guard on the next line

    stddev = np.sqrt(np.maximum(0, variance))

turns a negative variance into a std of zero rather than into an error. The
guard is what makes it silent.

Reproduced in isolation on uniform random uint8 frames whose true per-channel
std is 73.78:

| input dtype | std returned |
|---|---|
| uint8, as lerobot passes it | **0.000** |
| the same frames upcast to float32 first | 73.781 |
| true | 73.784 |

The mean is unaffected, because means do not square anything.

On sept02_final the stored ego std is [0.0171, 0.0127, 0.0111] rather than
exactly zero, because the streaming encoder downsamples before it measures and
the partial sums do not cancel perfectly. It is still 14 to 20 times too
small. `NormalizationMode.MEAN_STD` divides by it, so every image the policy
has ever seen arrived scaled by roughly 15x: a [0,1] pixel maps to about
[-46, +44] where ImageNet-normalised input sits in about [-2, +2].

**This applied to Phase 1.** A_prime scored 6.69 mm against NULL's 6.63 and the
conclusion recorded was that the ego camera contributes nothing. That
conclusion now has a second candidate explanation which has nothing to do with
the camera or the encoder: the images were out of range from the first forward
pass. Phase 1b fixes the normalisation, so its arms are the first to see
correctly scaled pixels at all.

The fix used here overrides the image entry in `stats` with ImageNet mean and
std before the processors are built. It is what pretrained weights expect, and
this dataset's TRUE per-pixel std of [0.242, 0.215, 0.194] is close enough to
ImageNet's [0.229, 0.224, 0.225] that the same override is also approximately
right for a scratch encoder.

What would have caught it: any assertion that a normalised batch has
approximately zero mean and unit variance. That check now runs before every
Phase 1b run and refuses to train if it fails.

## Rows 43 and 44, in detail

These two are one incident. Row 43 broke the hand path on 3 September; row 44
is the reason nobody noticed for ten hours.

**Row 43.** A `pip install lerobot` at 15:58 moved torch from 2.13.0 to 2.10.0.
The pipeline's own logs record the moment exactly:

    15:40:49  Stage 5 · render      torch 2.13.0 on device mps
    16:02:13  Stage 3 · estimate    torch 2.10.0 on device mps

Under the changed torch the MPS conv2d path raises on the ViT slice
`x[:, :, :, 32:-32]` for every frame. `process` caught it, logged
`log.debug("WiLoR failed on a frame: %s", exc)`, and returned the same
`HandFrame(detected=False, ...)` that an image with no hand in it produces.

That is the whole defect. **A frame the backend refused to look at and a frame
with nothing in it landed in the same counter.** Stage 3 divided that counter
by the frame count and reported a detection rate of 0 per cent, at INFO, and
continued. The one piece of code that reads `hand_detection_rate < 0.5` is a
fixture escape hatch guarded by `groundtruth_path is not None`, so on real
footage nothing reads it.

The repair was itself a second instance of not reading the record. Torch was
"restored" to 2.5.1 and ultralytics to 8.1.34 from memory. Neither was the
version that worked. `runs/sept02_scan2/wristview.log` had said `torch 2.13.0
on device mps` on every line since the run that produced the good hands, and
2.5.1 fails on the same conv2d in the same place. The environments are now
frozen in `ENV-MAIN.lock.txt` and `ENV-LEROBOT.lock.txt`.

`raised_on_every_frame()` now separates the two cases and Stage 3 stops on it.
It sets no threshold: a backend that raised on 100 per cent of the frames it
was given is not a judgement about footage quality.

**Row 44** is the check that should have caught row 43 within the minute, and
its numbers are the clearest evidence in this file that the family named at the
top of the page is still live.

`summarise` in `reprojection.py` computes three statistics and writes all three
into every `qc.json`:

    "median_px": 8.06, "p90_px": 16.26, "max_px": 27.5, "limit_px": 25.0,
    "passed": true

`passed` is `bool(median <= limit)`. Over the 60 clips of each run:

| run | clips | `passed: true` | median over 25 px | p90 over | **max over** | worst max |
|---|---|---|---|---|---|---|
| sept02_scan2, healthy hands | 60 | 60 | 0 | 0 | **2** | 28.62 |
| broken hand path | 60 | 60 | 0 | 2 | **56** | 55.17 |

Read the two columns the gate ignores. `max_px` puts 2 clips over the limit on
the good run and 56 on the bad one. It is a clean separator, it was correct, it
was written to disk 60 times, and no code compared it to the `limit_px` sitting
in the same dictionary. Meanwhile the statistic the gate does read is 0 of 60
in both runs: **the median cannot see this failure at all**, which is exactly
standing rule 3.

The threshold is not the problem and is not touched here. 25 px is documented
in the file with its derivation. What the gate reads is the problem, and
changing that changes what passes, so it is Sidwyn's call and not a repair to
be slipped in beside an environment fix.
| 45 | `train_arm.py` scored `pred[:, :8]` against `truth[:, :8]` | the policy's prediction against what the operator did | the same thing at the same instants | `generate_actions` returns the horizon sliced `[n_obs_steps-1 : n_obs_steps-1+n_action_steps]`, deltas 0 to 7, and the recorded chunk starts at delta -1. Every arm of real31 was scored against ground truth shifted one frame, 1/15 s, into the past |
| 46 | `analytic_floors.py` scored the floors over all 16 deltas while `train_arm.py` scored the arms over 8 | the floor and the arm | the floor and the arm ON THE SAME WINDOW | the floors carried deltas out to +14, where any predictor is worse, and the arms did not. On sept02_final's holdout the window is worth 6.25 to 6.14 mm to FLOOR-MEAN and 5.31 to 3.72 mm to FLOOR-PERSISTENCE, so the two sides of the gate were never like for like |
| 47 | `DiffusionPolicy(cfg, dataset_stats=stats)` | nothing; lerobot 0.4 takes `**kwargs` and drops it | that the actions are normalised before training | normalisation moved to a processor pipeline in 0.4. The argument is accepted, discarded, and no warning is issued. The policy then trains on raw metres, actions of order 1e-3 against a unit-variance noise schedule, learns nothing, and reads exactly like a true null result. Measured: action sd 0.0148 raw against 0.3850 normalised, a factor of 26 |

## Rows 45 to 47, in detail

All three sit under the Phase 1 gate, none of them raises, and each on its own
would have made the answer meaningless. They were found by reading what the
library does rather than by running it, before a pod was rented.

Rows 45 and 46 are one fault seen twice: **the two sides of a comparison were
measured over different windows.** A diffusion policy configured with
`n_obs_steps=2, horizon=16, n_action_steps=8` has

    action_delta_indices   [-1, 0, 1, 2, ... 14]      the recorded chunk
    generate_actions       [1:9] of that              deltas 0 to 7

so the prediction covers deltas 0 to 7. The arms were scored against
`truth[:, :8]`, deltas -1 to 6, one frame in the past. The floors were scored
against all sixteen, out to +14. Neither matched the other and neither matched
the policy. Both now score deltas 0 to 7, the offset is applied once, and the
score raises if the two lengths ever disagree again.

Row 47 is the most dangerous of the three because its failure is
indistinguishable from the result the experiment is trying to measure.
`DiffusionPolicy.__init__` in lerobot 0.4 reads

    def __init__(self, config: DiffusionConfig, **kwargs):

and the docstring still documents `dataset_stats`, which no longer exists as a
parameter. Normalisation moved to `processor_diffusion.py`. So
`DiffusionPolicy(cfg, dataset_stats=stats)` runs, returns a policy, trains, and
converges to nothing, because the actions are displacements of order 1e-3 m
being fitted against a unit-variance noise schedule. **An arm that scores at
the floor is the finding this experiment exists to produce.** Had this run,
Phase 1 would have reported "53 episodes is not enough to learn this task
offline" and the cause would have been an unread keyword argument.

There is a fourth thing worth recording that is not a defect in the code but a
gap in what it wrote down. real31's nine training runs each recorded twenty
fields of result and not one of them named the version of the library that
produced the numbers. As a result the lerobot that scored A' at 14.73 mm cannot
now be identified: no released version between 0.3.3 and 0.4.4 has a
`predict_action_chunk` that accepts a batch rather than reading an inference
queue, so the code path that produced the published figure cannot be
reconstructed from the repository. Results now carry `lerobot_version`,
`torch_version` and `scored_action_deltas`.

## Rows 11 to 14, in detail

Row 11 is the one that cost a day. Stage 5 measured splat alpha coverage and
reported 100 per cent. That number was true. Gaussians did reach every pixel.
They were the wrong Gaussians, and coverage cannot tell the difference. The
trainer separately rendered eight training views and reported 17.51 dB. That
number was also true, and also read by nobody.

The gate now in `splatqc.py` reads them. It requires 25 dB on the training
views and 200 Gaussians per registered view. The real26 splat fails both, at
17.42 dB and 35 per view. `tools/check_splat.py` applies the same gate to a
splat trained elsewhere, so the GPU result faces it too.

Row 14 is a different shape from the rest and worth separating. The consumer
was written first and the producer never followed. `train_gsplat.py` has looked
for `undistort_export.py`'s two outputs since the real03 run, found neither,
and fell back without comment. Measured on real26 the correction is 0.90 px at
the frame border, so this did not break anything. It was still a fallback that
reported nothing about which branch it took.


## Rows 15 and 16, in detail

Row 15 cost a wrong verdict on a good splat. The real26 GPU splat holds
3,912,607 Gaussians and gsplat renders it at 30.46 dB from a registered scan
pose. The MPS rasteriser scored the same splat, the same pose and the same
photograph at 7.21 dB, and the new gate duly failed it. The gate was not wrong
about its own measurement; it was wrong about what it was measuring.

The rasteriser pads each tile to `max_per_tile` slots and discards the rest. At
the default of 128 it discarded 11,076,570 Gaussian-tile pairs on one view and
returned mean alpha 0.185. It had counted those 11 million and put the number on
the result object. Nothing read it.

Raising the cap to 16,384 leaves 15,224 dropped and lifts the score to 20.63 dB.
A 10 dB gap to gsplat survives and is unexplained. See `MPS-RASTERISER.md`.

Two things changed. Scoring a truncated render now raises instead of returning a
number: `tools/check_splat.py` refuses, and says by how much it was truncated.
And the authoritative gate moved to the GPU, where the renderer is verified:
`tools/cuda_job/measure_splat.py` renders and measures, `check_splat
--measurements` applies the thresholds, so the verdict still has exactly one
implementation.

Row 16 was caught within minutes of writing it, which is the only reason it is a
short entry. The external render returns colour and depth; the first version
took coverage from depth, on the reasoning that a depth of zero means the ray
met nothing. But gsplat's expected-depth channel is alpha-weighted and non-zero
wherever any Gaussian contributes at all. Stage 5 reported 100 per cent coverage
on frames whose mean alpha was 0.911 and whose minimum was 0.625. The fix was to
carry alpha back from the pod and apply the same `alpha > 0.35` rule the local
path always used. Coverage now reads 94 per cent.


## Rows 17 to 23, in detail

These came out of one question about a contact sheet. The numbers had all
passed.

**Row 17 is the worst defect in the project so far.** `s03_estimate.py:424`
recorded the object model as a single point at the origin whenever the plane
solve produced the pose, which is the normal path. Stage 5 drew that point with
`rasterize_points` at a 5 mm radius, five to thirteen pixels wide depending on
range. So every wrist view this project has ever rendered contained a dot where
the object should be. The cube that made the renders look right was the splat's
own copy of it, frozen at its scan position, which does not move when the
operator picks the object up.

A policy trained on those frames cannot learn manipulation. It sees a static
scene and a marker. No metric caught it: the object pose was correct, the
object was drawn, `object_track_rate` was high, and `object_model_points` was
recorded as 1 and never compared to anything.

Stage 3 now writes a box at the measured size with a colour sampled from the
object's own mask pixels, and Stage 5 rasterises it as a mesh so the existing
depth test occludes it correctly. Stage 5 refuses a one-point model rather than
drawing a marker and calling it manipulation data.

**Rows 18 and 19 are the same mistake twice.** Both gates were right that
something was wrong and wrong about the remedy. A gate that cannot say which
frames failed can only reject everything, and rejecting everything makes the
operator reach for a manual trim, which is how real26 lost its tail. Both now
drop the offending frames and report exactly which. Row 19 still rejects when
no majority side exists, because then there is no single-hand trajectory to
extract and picking one would be arbitrary. No threshold moved.

**Rows 20 and 21 compound.** `slerp_fill` clamps leading and trailing gaps to
the nearest valid pose, which is correct for interpolation and wrong at the
ends: it extends a measurement outward into frames where nothing was measured.
Stage 4 did that, and Stage 5 then rendered the result without ever consulting
`hand_valid` in either branch. real26 produced 181 control-rate frames in which
the camera did not move by a single bit, and every downstream number described
them as ordinary frames.

**Rows 22 and 23 are about the record rather than the data.** `--set` does not
reach `config.yaml`, the `stage` subcommand never rewrites that file, and no
`meta.json` held the overrides, so a run directory could not say what produced
it. The 57-231 trim that caused the frozen tail left no trace anywhere and took
a re-run of Stage 4 to identify. Invocations now append to `run_record.jsonl`,
deliberately not by rewriting `config.yaml`: making a one-off override sticky
for every later stage loses the trail a different way.

Row 23 arrived the same hour. `StageMeta.to_dict` builds its payload by naming
each field, so the `config_overrides` field added for row 22 was written to the
dataclass, carried through the code, and dropped on the way to disk. The test
for row 22 is what caught it, one commit after the fix it was testing.


## Rows 24 and 25, in detail

Row 24 is the family exactly: a function computed a plausible answer from the
wrong window, and nothing read whether the window was valid.

`carry.resting_pose` finds where the object sat before a hand lifted it, by
taking the first sustained still run. `carry.solve_carried` then attaches the
object to the hand at the contact frame using that position. The docstring is
explicit about why the order matters: contact is decided while the object still
rests, because the plane solve is reliable at that moment, and that breaks the
circular dependency between contact and the carried pose.

Neither function checks the order it depends on.

On real26/bm demo_1 the object detector found nothing until frame 352.
`resting_pose` returned `rest_run [360, 510]`, which is after the release, and
`solve_carried` used it to attach the object at contact frame 243. The rest
position it measured was a different place on the mat, 117 frames later. The
carried object came out at a median of 7.06 cm BELOW the desk surface, across
109 frames, and nothing in Stage 3 or Stage 4 objected. Stage 4 reported the
clip as healthy: 510 of 510 registered, hand on every frame, zero velocity
outliers, zero dropped frames.

The fix is one comparison: the still run must end before the first contact
frame, or there is no valid resting pose and the carry cannot be solved.

**Fixed 2026-08-27.** `carry.rest_precedes_contact` states the rule and
`solve_carried` raises on it, so every caller is protected rather than just
Stage 3. Stage 3 catches it first, logs why, and leaves the carried frames
unsolved rather than filling them with a wrong answer: an object whose
position is unknown during the carry is honest, and one placed 7 cm under the
desk is not. Checked against the real numbers, it rejects demo_1's
`(360, 510)` against contact at 243 and accepts demo_0's `(0, 100)` against
contact at 100.

Fixing it exposed a second thing. The `lifted_clip` test fixture put the hand
on the object from frame 0, so contact began at frame 0 and no resting window
could exist before it. Real footage does not look like that: real26 saw the
hand arrive at frame 57 of 439. The fixture now models the approach, which is
what lets the guard be tested against a clip that should pass it. A fixture
that cannot represent the correct case cannot test a rule about it.

Row 25 is rule 6 in CLAUDE.md, a fourth time. `standoff_m` sets how far the
wrist camera sits from the finger tips and the rig decides it. It was optional
and shipped null, so Stage 4 warned that the wrist camera could not be checked,
on every take of real06b and real26, and nothing stopped. It now raises when
unset and rejects a value outside 0.05 to 1.0 m. `configs/default.yaml` carries
0.25 m, the UMI placement, with a note that the 0.35 m it once held was a
workaround for an 18 cm scan floor and must not be used to hide a scan again.


## Row 26, in detail

Another wrong population, and this one punishes the capture the SOP asks for.

Stage 0 rejects blurred frames against a threshold of `0.30 x the clip's median
sharpness`. That is right for a clip shot at one distance. A scan shot as
several passes at different heights does not have one sharpness: a close pass
is soft because the lens cannot focus that near, which is a property of the
pass, not a fault in its frames.

Measured on real26, on identical low-pass footage:

| scan | clip median | threshold | low-pass frames kept | reconstructed floor |
|---|---|---|---|---|
| two passes | 132.8 | 39.9 | 225 of 299 | 8.47 cm |
| three passes | 149.6 | 44.9 | 207 of 299 | 12.15 cm |

Appending a third, sharper pass raised the bar for every other pass and cost
the low pass 18 frames it had previously kept. The frames it cost were the
lowest ones, because those are the blurriest, so the floor rose by 3.7 cm
without a single frame of the low pass being reshot.

`ingest.scan_pass_boundaries_s` now names the seams and each pass is judged
against its own median, with the drop cap applied per pass too. The seams are
given rather than detected: whoever concatenated the passes knows where they
are, and a detector would be one more thing to be wrong.

The general shape is worth stating, because it will recur. **A threshold
computed from a population is only valid for a population that is homogeneous
in the thing being thresholded.** Mixing two regimes into one median produces a
threshold that is too strict for one and too loose for the other, and neither
half is visibly wrong.


## Rows 27 and 28, in detail

Row 27 is the `_overflow` shape again. Grounding DINO returns its
highest-scoring box whatever the score means, so when the phrase matches
nothing in particular the winning box is the image. Checking real27's scan for
the tea box returned `[6, 3, 1912, 1075]` on a 1920x1080 frame, 98 per cent of
the area, on 5 of 14 sampled frames. Read as detections those said the carton
appeared in every scan frame. It was not on the mat at all. `detect` now
raises above 60 per cent of frame area, because a caller cannot tell that box
from a real one.

Row 28 is selection bias by marker visibility, for the fourth time, and it is
worth stating precisely because it kills a check that looked obviously
correct.

`tools/check_take.py` was asked to measure how much of a scan sits below
30 cm, from raw video, using the apparent size of marker id 0. The marker lies
flat on the desk. A camera low down and looking ACROSS the mat sees it
obliquely or not at all, and that is exactly the render band the check exists
to police. Measured on real26/d against heights from its own reconstruction:

| height | frames | marker seen | rate |
|---|---|---|---|
| 0-15 cm | 105 | 0 | **0%** |
| 15-20 cm | 127 | 6 | 5% |
| 20-30 cm | 102 | 26 | 25% |
| 30-38 cm | 143 | 66 | 46% |
| 38-60 cm | 254 | 183 | 72% |

A frame below 30 cm is 0.15 times as likely to show the marker as one above.
The true share below 30 cm is 45.7 per cent and the marker-visible subset says
11.4 per cent.

**So a scan shot correctly is the scan whose low frames the marker cannot
see.** Failing on this measurement would reject exactly what it should accept:
real26/d passed the real viewpoint gate at 7.47 cm and this check scores it at
5.8 s below 30 cm, the same as real27 which was genuinely shot high.

The check is therefore a LOWER BOUND and can only pass, never fail. Clearing
the bar on marker-visible frames alone clears it outright, because the unseen
frames are lower still. Not clearing it shows nothing either way.

Making it a real gate needs a capture change, not a code change: a second
marker standing VERTICAL beside the mat stays visible from the low
across-the-mat views and gives the same height and azimuth from one PnP solve.
Until that exists the honest pre-flight is checks 1 to 3, and the render band
is confirmed after Stage 1 with `close_range_coverage`.


## Row 29, in detail

Row 28 said marker visibility biases WHICH frames can be measured. Row 29 is
the blunter problem underneath it: often nothing can be measured at all,
because the marker is not detectable in the frame.

Measured on real27, at full 1920x1080:

- ArUco id 0 was detected in **14 of the 30 demo first frames**. 16 gave
  nothing.
- On the scan, id 0 appeared in 72 of 408 sampled frames, 18 per cent, under
  the 30 per cent floor `check_take` needs. The same was true of the first
  scan at 47 per cent measured but biased. **Two scans in sequence could not
  be checked for render-band coverage**, which is the single question those
  scans were re-shot to answer.

Two causes are visible in the frames themselves, and they are different:

1. **take02: occlusion.** The carton stands between the camera and the marker
   and hides its lower edge. A start spot sits on the sight line.
2. **take13: contrast at the border.** The marker is not hidden and detection
   still fails. Two black clips lie on the left and right edges of the printed
   square. ArUco needs the quiet zone around the pattern to be clean; anything
   crossing the border can break the quad fit even when the pattern itself is
   perfectly visible.

Neither is a code fault, and neither stops this session: scale comes from the
scan, and workspace segment detection uses the scan-match count rather than
marker visibility, which is defect 6's fix still holding.

**A method note that belongs with the row.** The first marker test ran on
480 px wide frames and reported 0 of 30 detections. The same test at
1920x1080 reported 14 of 30. The first number described the test, not the
takes. `check_take` hit the same thing from the other direction: 19 per cent
detection at 960 px, 47 per cent at 1280, 31 per cent at full 1920. ArUco
detection rate is not monotonic in resolution, so any claim about marker
visibility has to state the resolution it was measured at.

Fix before the next session, in the capture and not in the code:

- Nothing may touch the printed square or its white border.
- No start spot may sit between the camera and the marker.
- Stand a second marker VERTICAL beside the mat, on a different id, so the
  low across-the-mat views that the render band is made of have something to
  measure against.


## Rows 30 and 31, in detail

Both are mine, from this session, and both were caught by guards added earlier
in it.

Row 30. `estimate.object.prompt` sat in `configs/default.yaml` with eight lines
of comment explaining why naming the object matters. No code read it. Stage 3
took its detector phrase from the CLI `--instruction`, overridable only by
`estimate.pose.per_clip[clip].prompt`. So real27 handed Grounding DINO "pick up
the tea box and place it on the mat", a sentence describing a task, and the
detector returned a box covering 93 per cent of the frame.

Two things about this are worth keeping.

**A declared key that nothing reads survives the unknown-key check.** Defect 8
made unknown keys a loud error, and this key is not unknown, it is known and
ignored. The check that catches it is a test asserting the source contains the
read, which is now in `test_stage_call_sites.py`.

**It was reported as done.** Earlier in this session I told Sidwyn the prompt
was set and active. I had edited the config and not checked that anything read
it. CLAUDE.md says: do not report a setting as active until you have confirmed
the code reads it. The rule exists because this is easy and the failure is
invisible until a detector is asked to find a sentence.

Row 31 is the opposite error, an over-strict guard. Defect 24's fix required
the still run to end before first contact. That is too strong. A run often
continues past contact for good reason: the hand arrives several frames before
it lifts, and while the object has not moved the plane solve keeps returning
the same point. real27's take01 measured a run of frames 0 to 90 with contact
at 75 and a scatter of 0.04 cm. The object genuinely had not moved, frames 0 to
74 were exactly the evidence needed, and the guard threw the take away.

The rule is now "enough of the run happened before contact": trim at the first
onset and require `min_rest_frames` to remain. Checked against all four cases,
including real26/bm demo_1, which still fails because its run began 117 frames
after the release.

A guard that rejects valid data is a defect, not caution. It costs a re-shoot
just as surely as a guard that passes bad data costs a session.


## Rows 32 to 35, in detail

All four are from the real27 GPU session. Rows 32 and 33 are mine and were
caught in the same session. Rows 34 and 35 are older and were exposed by it.

Row 32. `train_gsplat.py` printed `gaussians N` on every thousandth step. That
line existed because of row 12, where the count sat at the seed value for
30,000 steps and nobody read it. The reader added then asks one question, has
the count grown, and stops asking at step 2,000. Nothing watched it after that.
real26bm finished at 4,998,950 Gaussians and 15.0 GiB, which fits. real27full
passed 5.4 M by step 6,000 and reached 8,126,424 at 18.9 GiB by step 10,000,
with refinement running to step 15,000. There was no checkpoint, so an
out-of-memory crash would have destroyed the whole run.

This is the house defect with a twist. The number was read, once, for one
purpose, and that reading was mistaken for the number having a reader. A
diagnostic needs a reader for each failure it can see. Growth and ceiling are
two failures and the print served one.

The fix adds `--max-gaussians`, default 7,000,000, which stops densification
and says so in the log, and `--checkpoint-every`, default 5,000 steps. The
capped run reached the cap at step 8,001 and finished 30,000 steps at 17.4 GiB
and 26.19 dB.

Row 33. The RunPod proxy shell echoes every command it receives, so the
transcript holds the poll command as well as its output. The watcher grepped
that transcript for `CHAIN_DONE` and found the word inside its own command, on
the first poll, 37 seconds after a 45-minute job started. It reported success
and the harness woke me to a chain that had not started training.

The same family as rows 2 and 9: a measurement that cannot distinguish its
subject from itself. The fix tests process state and an output file rather than
log text, and splits the token so the echoed command cannot contain the string
the test looks for. It was then run against the live chain and made to say
`STATE=running` before being trusted.

Row 34. `mount.py` requires `standoff_m` and rejects values outside 0.05 to
1.0 m, which is a check on the offset. Nothing checks the result. A wrist
camera 20.3 cm below the desk is not a viewpoint, and 309 frames were exported
and rendered from positions like it. The renders from those frames are the
black and blue ones in the contact sheet.

The check to add is on the output, not the parameter: the wrist camera must sit
above the desk plane, and a frame that does not is invalid, not merely odd.

Row 35. This is the expensive one, and the gate was already written and
correct. `coverage.render_viewpoint_coverage` holds `MAX_VIEWPOINT_GAP_M` of
0.15 and `MAX_FRACTION_BELOW_SCAN_FLOOR` of 0.10, and Stage 5 raises on either.
Stage 5 runs on the laptop, after the pod has trained the splat and drawn every
frame, because Stage 5 owns the fisheye remap, the gripper and the object.
`tools/export_wrist_cameras.py` ships the camera set to the pod and applies no
gate at all.

So the sequence was: export 4,673 cameras, rent a 4090, train 40 minutes,
render 14,019 frames, download 1.9 GB, and only then reach the code that says
20.8 per cent of those cameras are further than 15 cm from any scan view and
22.2 per cent sit below the scan floor.

The gate is in the wrong place, not wrong. It belongs in
`export_wrist_cameras.py`, where a failure costs nothing.

**Fixed.** `render_viewpoint_coverage` now runs in `export_wrist_cameras.py`
before anything is written, and the export returns 1 without producing
`wrist_cameras.npz`. The thresholds stay in `wristview.coverage`, so the export
and Stage 5 read one set of numbers. The report is written either way, because
a refusal with no evidence is not reviewable.

Run against real27full, **22 of 30 clips fail**, every one of them on the
scan-floor criterion:

| | clips |
|---|---|
| pass | demo_6, 7, 11, 23, 24, 25, 26, 27 |
| fail, frames below the scan floor | 21 clips, from 12.3 per cent (demo_12) to 60.1 per cent (demo_15) |
| fail, median gap as well | demo_5, at 27.2 cm against the 15 cm limit |

`tests/test_export_gate.py` holds the order check, so a future edit that writes
the cameras first fails a test rather than a session. It was verified by
mutation: removing the `return 1` makes the suite fail.

**A postscript on how nearly this was reported wrongly.** Reading the scan
poses back with pycolmap, I multiplied them by the metric scale factor. Stage 1
had already transformed the reconstruction in place, so the model on disk is
already in metres and I scaled it twice. That put every scan camera at 32.6 to
38.9 cm above the desk and made 90.7 per cent of wrist views fall below the
scan's lowest, which reads as a capture that never went low enough and a
re-shoot. The true scan spans 15.2 to 92.6 cm with a third of its views in the
15 to 25 cm band, and the close pass is present and good.

What caught it was Stage 1's own record disagreeing: it had logged a baseline
of 0.862 m and a path length of 6.468 m where I computed 0.070 m and 0.524 m,
a constant factor of 12.34, which is 1 over the scale factor. Recomputing from
`01_scene/cameras.json`, the poses Stage 1 actually used, reproduced its
numbers exactly.

Standing rule 1 earned its place here: a quantity is not verified until a
source with no access to it agrees. The disagreement was visible in a file
written weeks before and the only reason it surfaced is that I compared against
it before speaking.


## Rows 36 and 37, in detail

Row 36. `configs/default.yaml` carried "These three put the finger tips about
77 percent of the way down the frame" from the first commit. The mount put them
at 86.6 per cent. Nothing compared the two, because a number in a comment has
no reader, which is this project's most repeated defect wearing a different
hat.

It stopped being cosmetic when the lens was matched to the wrist camera the
experiment actually records, 62.1 degrees horizontal against the 90 that was
configured. The finger tips sit 22.38 degrees below the optical axis, fixed by
`mount_back_m` and `mount_up_m`. At 90 degrees the vertical half-angle is 29.4
and they are in frame. At 62.1 it is 18.7 and they land at row 408 of a
360-row image: the gripper leaves the picture.

The claim now lives in `render.wrist_camera.framing_row_fraction`,
`mount.fingertip_row_fraction` computes the delivered value, and
`tests/test_mount_framing.py` fails when they disagree. The same test pins the
sign of the new `pitch_down_deg`, checks that pitch rotates without
translating so the coverage gate cannot be moved by it, and asserts the old
settings would have failed.

Row 37. WiLoR's confidence is 1.000 on every valid frame of all 30 real27
clips. A constant is not a confidence.

**Does anything consume it? No.** `backends/hands.py` line 237 writes
`confidence=1.0` as a literal on the WiLoR path. Stage 3 collects it into
`hand_confidence` and stores it in `hand.npz` as `confidence`. Nothing reads it
back: not Stage 4, not the QC gates, not the reprojection check, not the
retarget velocity gate. Searched across `src` and `tools`, the only other
matches are MediaPipe's own `min_detection_confidence` argument, which is a
different thing.

So the field is inert, and that is the small mercy: no decision was made on a
score that cannot be low. The damage was to a human reading it, which is a real
cost but a recoverable one. **If anything ever starts reading it, it must first
stop being a literal.** The MediaPipe path at line 320 does return a real score,
`scores[best]`, so the field is only constant on the backend actually in use.

The fix shipped is not a better score, it is a different signal:
`handqc.drop_isolated` in Stage 3. Applied retroactively to real27, it removes
exactly 2 frames across all 30 clips, both from demo_28, and turns that clip's
lead-in from 0 into 21 and its 19 gaps into 0. Two frames of demo_28 hold no hand at
all, in the top right corner on desk clutter, with 5 of 21 landmarks inside the
image, and they entered the pipeline at full confidence.

The cost was a wrong diagnosis, not corrupted data: demo_28 was recorded as a
take shot with the hand already in frame, and it was reported that way earlier
in this session. The take is fine. The real hand enters at frame 21, in line
with the other 29 clips at 18 to 52.

What separates a phantom from a legitimate detection at the frame edge is not
the landmark count. Other clips hold valid frames with as few as 2 landmarks in
view and they are real hands entering the shot. It is **isolation**: across all
30 clips, exactly two valid frames have no valid neighbour on either side, and
both are demo_28's phantoms.


## Row 38, in detail

`retrieval_top_k` sets how many scan frames each demo frame is matched against.
Halving it from 10 to 5 halves the matching cost, which is 94 per cent of Stage
2 and about 1.6 hours across 30 clips.

On strong clips it is free. demo_6 and demo_15 registered identically at both
settings and their poses agree to 2.3 mm and 0.21 degrees at worst.

On demo_22, the one clip with a marginal inlier ratio, k=5 dropped 21 of 171
registered frames and moved a surviving frame 36.2 cm and 40.1 degrees.

**And the inlier ratio went up while that happened, 0.8276 to 0.8687.** Fewer
retrieved neighbours means fewer hard pairs are attempted, so the ratio rises
as the evidence thins. The metric moved the right way for the wrong reason,
which is worse than a metric that does not move: a flat number invites a look,
an improving number closes the question.

An adaptive rule cannot save it either. Deciding whether k=5 is safe for a clip
requires the k=10 result to compare against, so the cheap setting can only be
validated by paying for the expensive one. `retrieval_top_k` stays at 10.

The general form, and it is the reason this row exists: **before trusting a
ratio, ask what changed in its denominator.** Standing rule 3 asks whether a
metric can see the failure mode. This one can be actively misled by it.


## Row 39, in detail

`train_gsplat.py` set each Gaussian's initial scale from the mean distance to
its three nearest neighbours, computed as a single `torch.cdist` of every seed
point against every other. That allocates N squared floats.

| run | seed points | allocation | outcome |
|---|---|---|---|
| real27full | 36,962 | 5.5 GiB | fit, unnoticed |
| **real28scanb** | **78,961** | **23.23 GiB** | **out of memory on a 23.53 GiB card** |

The failure mode is the wrong way round, and that is what makes it worth a row:
**the cost is quadratic in a number that grows every time the capture gets
better.** A denser scan is the goal, and this made a denser scan fatal. It
would have gone off on any future session that improved, and it went off on the
first one that did.

It also failed before step 1, so it cost only minutes. Row 32's cap and the
checkpointing added then are what keep a late failure cheap; this one was early
by luck, not design.

Fixed by chunking the rows: peak is now `chunk x N x 4` bytes and independent
of N, sized to a 1 GiB budget. real28scanb runs at 3,399 rows per chunk and
1.00 GiB, and real27full's smaller cloud lands at the same 1.00 GiB rather than
5.5.

## Rows 40 to 42, in detail

Row 40 is the purest case of the family this file opens with, and the most
expensive. `scan_geometry` runs at **Stage 1**. Its own docstring says why:

> Measure whether the scan can constrain geometry. Run this before a render.
> Stage 1 reports this. A reshoot is still cheap at that point.

It computes `height_min_m`. That value is not an approximation of the quantity
that later fails the run. It IS that quantity: `render_viewpoint_coverage` takes
`scan_floor = _heights(scan, ...).min()`, the same minimum over the same
centres. The number was correct, in memory, printed into
`01_scene/meta.json`, and no code compared it to anything.

Its two thresholds ask whether the scan is varied enough, not whether it went
low enough:

| run | `height_min_m` | `h_span` | `baseline` | gate |
|---|---|---|---|---|
| real28c | 0.0708 | 1.0007 | 1.6702 | pass |
| real31full | 0.1205 | 1.0505 | 1.9563 | pass |
| real31scan | 0.1383 | 1.0189 | 1.9475 | pass |
| real31scan0 | 0.1205 | 1.0505 | 1.9563 | pass |
| sept02_05x | 0.2313 | 0.5599 | 1.9729 | pass |
| sept02_1x | 0.3253 | 0.5281 | 1.8432 | pass |
| sept02_scan | 0.2945 | 0.5988 | 3.0654 | pass |
| sept02_scan2 | **0.1972** | 0.6062 | 2.0681 | **pass** |

Eight reconstructions, eight passes. The gate has no discriminating power on
this corpus. `height_min_m`, which it declines to read, separates the two
populations with an empty band between 0.1383 and 0.1972 — every scan that
supported a render sits at or below 0.1383, every scan that was too high sits
at or above 0.1972. A threshold anywhere in that band would have stopped
sept02_scan2 at Stage 1, 45 minutes in, before any pod, instead of four hours
later. The natural value is 0.15, which is already `MAX_VIEWPOINT_GAP_M`; the
scan floor has to sit within about one gap-length of where the wrist camera
flies.

Worse than silent: **the span gate rewards the failure mode.** Height span is
satisfied by going HIGH. The sept02 scans span 0.53 to 0.61 m by starting at
chest height and passing. A tight, correct close pass has a SMALLER span. On
that axis Stage 1 and Stage 5 pull in opposite directions.

Rows 41 and 42 are the same fault one level up, in the tools and the SOP rather
than the pipeline. Two quantities that both sound like "how close was the
scan", with thresholds 3x apart, and the loose one is the one that runs early
and gets quoted. And a capture instruction phrased as framing, which fixes a
distance by arithmetic nobody did, in the opposite direction from the gate.

The three share the shape of every row above them. The information existed. In
row 40 it was computed and unread; in row 41 it was measured against the wrong
constant; in row 42 it was implied by a rule and never derived. **If a number is
worth computing, something must fail on it, or it must not be computed** — and
row 42 adds the corollary: if an instruction sets a number, write the number
down, because otherwise nothing can check it.
