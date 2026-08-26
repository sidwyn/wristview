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
