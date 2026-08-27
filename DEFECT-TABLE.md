# Defect table

Each row records a check that reported a wrong answer, and the reason.

The rows share a family. A measurement stood in for the thing it was meant to
test. The substitute agreed with the real quantity most of the time, so nothing
looked wrong until it disagreed.

**One sub-family is the most repeated defect in this project, and it is worth
naming on its own: the code computed the exact number that described the
failure, and then no code read it.** Not an approximation of that number, not a
proxy. The number itself, correct, in memory, discarded.

Rows 7, 9, 10, 11, 12 and 15 are all this. The trainer measured 17.51 dB on its
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
