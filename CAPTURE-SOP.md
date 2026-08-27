# wristview · Capture procedure

**Who this is for:** anyone shooting a batch, without the pipeline author present.

**The one rule that matters:** the scan must be shot from where the demo camera
sits. Everything else in this document is detail. Two capture sessions failed
on that single point before a third succeeded.

Every rule below states the measurement behind it. Where a rule has no
evidence yet, it says so.

---

## 1. Camera

| Setting | Value | Why |
|---|---|---|
| Lens | **Main wide only.** Never ultra-wide. | The pipeline assumes a pinhole model. Stage 0 rejects footage above 100 degrees horizontal field of view. |
| Stabilization | **Off.** | The most common cause of failed reconstruction. Stabilization warps frames independently, and no camera model fits that. |
| Resolution | 1080p or higher | Sessions to date used 2160x1214. |
| Frame rate | 60 fps | Downsampled later. Scan to 6 fps, demos to 20 fps. |
| Shutter | 1/120 or faster | |
| ISO, white balance, focus | **Locked** | Refocus mid-clip changes the intrinsics, and Stage 1 solves for one camera. |

Use the same camera and the same settings for the scan and every demo in a
session. Stage 2 solves demo poses against the camera Stage 1 calibrated from
the scan.

**Do not move anything on the surface between the scan and the demos.** The
scan is the map. Anything that moves afterwards is not in it.

---

## 2. Workspace fixture

Set this up before shooting anything.

- **Texture under the working area.** A patterned mat, a sheet of newspaper, a
  printed pattern. Anything with non-repeating detail.
- **Three to six small static objects** around the edge of the working area,
  inside the demo's crop. A can, a mouse, a book. They must not move.
- **The task object** in place.
- **The ArUco marker**, see section 4.

**Evidence.** A bare wooden desk fails. Session 2's demo saw almost nothing but
wood grain, whose features are repetitive and cannot match uniquely, and the
pre-flight came out at 0.39 against a 0.40 pass mark. Session 3 added texture
and reached 0.76. Of the features that actually matched in session 3:

```
marker sheet    41.4%
keyboard        22.0%
desk / wood     36.6%
```

**63 percent of usable matches came from added structure.** Wood grain alone
does not carry a capture.

---

## 3. The scan

### How low the main lens can usefully go

**About 20 cm for real detail. About 15 cm for usable. Below that, mostly
blur.**

The wrist camera renders from a median height of 7.3 cm above the desk, so the
scan has to reach down toward it or every wrist frame extrapolates. real26/a
scanned no lower than 18.2 cm and 93 per cent of its render extrapolated
downward from every training view at once.

But coverage and detail pull against each other, and the main lens sets the
limit. Measured on real26/b's low pass, Laplacian variance of registered
frames against camera height above the desk:

| height | median sharpness | share of the sharp value |
|---|---|---|
| 45-100 cm | 308 | 100% |
| 30-45 cm | 244 | 79% |
| 20-30 cm | 150 | 49% |
| 15-20 cm | 91 | 29% |
| 12-15 cm | 52 | 17% |
| 8-10 cm | 47 | 15% |

There is no cliff. Sharpness declines continuously, reaching half at about
25 cm and a third at about 17 cm. And this table flatters the low end: Stage
0's blur filter had already discarded 97 of 299 low-pass samples before these
numbers were taken, so these are the survivors. On the raw clip, the seconds
spent closest to the mat measured a median Laplacian variance of 16.8 against
674 at the sharp start of the same pass.

So a low pass is worth shooting, and it did what it was for: the reconstructed
floor fell from 16.5 cm to 8.5 cm and usable wrist frames rose from 29 to 128.
Just do not expect detail from it. Sweep low for coverage, and keep the bulk of
the take at 20 cm and above where the lens still resolves.

**Getting genuine detail lower needs a macro or ultrawide lens.** That is a
future change and it brings its own intrinsics: a second lens means a second
camera model, and the reconstruction has to be told which frames came from
which. Do not mix lenses in one clip until that is built.

### The passes

One continuous take, about 30 seconds, in two phases.

### Phase 1 · wide orbit, about 12 seconds

Slow orbit of the whole workspace. Vary your height. Large overlap between
viewpoints. No hands, no people.

This phase builds the background the wrist view is rendered against. **It
contributes nothing to demo localization**, see below, so do not spend the
whole take on it.

### Phase 2 · close pass, about 18 seconds and not less

Move in and cover the working area **from the demo camera's own height, tilt,
distance and framing**, with the object in place and hands out of shot. Sweep
slowly across the whole area the demo will cover.

**Evidence, and the reason this phase exists.** Of the 64 scan frames that the
session 3 demo matched best:

```
from the WIDE orbit    0   (0%)
from the CLOSE pass   64   (100%)
median position 76% into the scan timeline
```

Every usable localization match came from the close pass. A scan without one
cannot localize a close-up demo at all: session 1 had a wide orbit at 1.55 m
against demos shot at 0.6 m, and the best possible demo-to-scan match gave 126
features where scan-to-scan neighbours gave over 700. Stage 2 recovered
trajectories of 85 m across a 0.69 m desk from that, and reported 100 percent
of frames registered while doing it.

**Close-pass fraction by session:** 0 percent failed, 37 percent was marginal,
60 percent passed comfortably. Aim for 60 percent.

### Session 6 measured, and it did not obey any of this

The three phases above were written after session 3. Session 6 was shot without
them and nothing in the pipeline noticed, because every existing gate measures
the scan against itself.

```
max baseline between any two scan views     2.96 cm
total camera path, 327 frames               18.5 cm
height above the desk                       41.1 to 43.9 cm   (a 2.7 cm band)
baseline over subject distance              0.069             (an orbit is ~1.0)
view direction spread                       39.7 deg
```

The entire scan was taken from inside a 3 cm ball. It is a rotation, not an
orbit, and a rotation has no parallax, so nothing in it constrains depth. The
reconstruction still reported **327 of 327 registered at 1.3565 px** and the
splat still reached **30.21 dB**, because both of those measure the scan
against its own views, and all 327 of those views are effectively one viewpoint.

Against that, the wrist camera rendered from:

```
wrist camera height above the desk          19.8 to 38.5 cm
distance to the nearest scan view, median   24 to 44 cm depending on the clip
rendered frames below the lowest scan view  100 per cent, on all four clips
```

Every frame of every delivered video is a novel view about 40 cm from anything
the splat was ever shown. That is the floaters, and that is the wash.

**This is now gated.** Stage 1 reports `scan_geometry` and fails loudly on a
baseline under 30 cm or a height band under 15 cm. Stage 5 refuses to render
when the median frame sits more than 15 cm from the nearest scan view, or when
more than 10 per cent of frames fall below the scan floor, unless
`render.allow_extrapolation` is set. Note that the old Stage 5 coverage warning
did not catch this: it reads the splat's alpha, which was 87 to 94 per cent
throughout. Alpha says a Gaussian was drawn there. It does not say a camera was
ever there to constrain it.

### Phase 3 · contact pass, about 6 seconds

Hold the camera **10 to 15 cm from the work surface** and sweep the small area
where the grasp happens. Hands out of shot. This is closer than feels sensible.

**Why.** The virtual wrist camera renders from 0.10 m. Every session so far was
scanned from 0.5 m and further, so every wrist frame is an extrapolation rather
than an interpolation, and the render is soft at exactly the moment that matters,
contact. Training longer does not fix it, and neither does a better splat: the
1.53 M Gaussian splat reaches 35.2 dB on the training views and is still soft at
0.10 m, because no training view was ever there. This is the one defect in the
current renders that only a capture change can fix.

### Do not clear the set until the pipeline has passed end to end

Leave the objects, the marker and the lighting exactly as shot until Stage 5
has produced a render you accept. Not until the scan finishes, and not until
Stage 1 finishes.

**Why.** A scan can only be extended while the set still exists. Session 4's
close passes came out at a median 0.264 m from the surface when they were
intended to be 0.15 m, and only 2 frames of 364 got inside 0.15 m. That is
fixable by appending a 30-second pass at true wrist range, and it is fixable
for about a minute of shooting. The set was cleared before Stage 1 finished,
so the option was gone before the measurement that would have called for it
existed.

Nothing warns you. The scan reconstructs, every clip localizes, the gates
pass, and the defect only appears as softness in the final render, by which
time re-shooting means re-staging the whole session.

This is a process defect, not a code one, which is exactly why it belongs
here.

---

## 4. The ArUco marker

The marker gives metric scale, and gives every demo frame an independent pose
check that does not come from the pipeline's own mathematics.

```
wristview marker --side-cm 10 --out marker.png
```

1. **Print at 100 percent.** No fit-to-page, no scaling.
2. **Measure the printed line** on the sheet with a ruler and confirm it matches
   the number printed beside it. A marker printed at 96 percent biases every
   downstream distance by 4 percent and nothing else in the pipeline catches
   it.
3. **Lie it flat** in the scene, unbent and uncreased.
4. **Place it within 5 to 10 cm of the task object**, so it stays inside the
   demo's tighter crop.
5. Set `scene.scale.aruco_marker_length_m` to the measured side in metres, and
   `scene.scale.aruco_marker_id` to the printed id.

**Evidence for the placement rule.** In session 2 the marker sat outside the
demo crop and was detected in **0 of 143 demo frames**. Moving it beside the
mug took session 3 to **199 of 205, 97 percent**, with the longest blind
stretch being 4 frames.

**Target:** above 90 percent of demo frames, longest gap under 20 frames.
Session 3 measured 90 to 97 percent across five clips.

**Use one marker only.** Extra markers of unknown size corrupt the scale: a
spurious id 17 detected in 8 frames alongside the genuine id 0 in 114 shifted
the scale by 44 percent before the pipeline learned to ignore it. If a second
id appears in the log, either remove it from the scene or pin
`aruco_marker_id`.

---

## 5. The demos

- **10 to 15 seconds** each.
- **Start with the object fully in frame and both hands out of frame**, then
  reach in. Session 1's C007 opened with the object already held, so it had no
  reach to learn from.
- **Keep the object away from every frame edge** for the whole clip. Session
  1's C006 clipped the glass at the left edge, which leaves hand pose intact
  but corrupts object pose.
- **Keep the marker in frame.**
- **Keep the hand in shot until two seconds after the release, then stop
  recording.** Do not let the hand leave the frame while the clip runs on.
  When hand tracking stops, the pipeline holds the last measured pose, so the
  gripper freezes and the wrist camera stares at whatever it last saw. Session
  3's demo_2 ends with 59 frozen frames, 42 per cent of the clip, and demo_3
  with 33. Nothing reports an error, because a held pose is a valid pose.
- **Move your hands slowly. Demo blur is now the limiting defect, not scan
  blur.** Session 6 fixed the scan by slowing the close passes: frames dropped
  for blur fell from 55 to 3. The demos then became the problem, dropping
  **15 per cent** of frames each, capped down from 19 to 22 per cent. The hands
  move fast even when the head is still, and the head mount does nothing about
  that. This is a shooting instruction, not something code can fix: a blurred
  frame has no features to match, whatever is done with it afterwards.
- **Move at a normal working pace.** A wrist cannot turn faster than about 900
  deg/s, and Stage 4 rejects any frame that claims it did. Isolated frames are
  filled from their neighbours, but two in a row, or more than 5 per cent of
  the episode, rejects the whole clip.
- One hand in shot unless the task genuinely needs two. A second hand adjusting
  the scene is picked up by the detector.
- Say the instruction aloud at the start. It becomes the language label.

---

## 6. Pre-flight · go or no-go

**Run this before shooting a full batch.** It takes under a minute and answers
the one question that decides whether the session is usable.

```
wristview preflight --scan scan.mov --demo one_demo.mov
```

Shoot the scan and **one** demo, run this, and only continue if it passes.

```
scan self-match   572 features   the ceiling this footage supports
demo to scan      436 features   best scan frame, median over 6 demo frames
ratio            0.76           pass needs 0.40
PASS
```

It reports a ratio rather than a raw count because the ceiling depends on
texture, resolution and keypoint budget, and comparing a demo against its own
scan's self-match cancels all three.

| Verdict | Ratio | What to do |
|---|---|---|
| PASS | 0.40 and above | Shoot the batch. |
| MARGINAL | 0.25 to 0.40 | Re-shoot the close pass. Do not shoot a batch on this. |
| FAIL | below 0.25 | Re-shoot. The scan does not cover the demo viewpoint. |

**Measured history:**

| Session | Raw matches | Ratio | Verdict | What changed |
|---|---|---|---|---|
| 1 | 126 | — | fail | wide orbit only, bare desk, marker outside demo crop |
| 2 | 212 | 0.39 | marginal | added a 15 s close pass in a 40 s take |
| 3 | **436** | **0.76** | **pass** | texture added, marker moved beside the object, close pass raised to 18 s of 30 |

**Never lower the threshold to make a capture pass.** The threshold is
calibrated against measured passing and failing captures. Moving it does not
change whether the footage localizes; it only removes the warning.

---

## 7. What good looks like, after processing

| Check | Target | Session 3 |
|---|---|---|
| Scan frames registered, Stage 1 | 95 percent or more | 193 of 193, 100 percent |
| Mean reprojection error | under about 2 px | 1.52 px |
| Mean track length | 8 or more | 13.0 |
| Pre-flight ratio | 0.40 or more | 0.64 to 0.77 across five clips |
| Marker in demo frames | 90 percent or more | 90 to 97 percent |
| Stage 2 tracking loss | under 2 percent | see the run summary |
| Camera speed, Stage 2 | under 2 m/s | see the run summary |

Stage 2 rejects an episode whose implied camera motion is not physically
possible for the scene size, so a capture that gets past pre-flight and still
fails there will say why in `02_localize/<clip>/status.json`.

---

## 8. Environment hazard

**Do not keep run directories inside an iCloud-synced folder.** iCloud creates
duplicate files named `frame_00000 2.jpg` beside the originals while a stage is
writing thousands of frames. The pipeline itself is unaffected, because every
stage reads the frame list from `manifest.json` rather than globbing the
directory, but any hand-written analysis that globs will silently count the
duplicates, and one such duplication produced a `KeyError` deep inside a
matcher, sixteen minutes into a run, with nothing pointing at the cause.

**It also locks the COLMAP database.** A fixture run inside `~/Documents` died
at Stage 1 after 842 seconds with `SQLite error: database is locked`, having
completed all 4,826 matches first. The sync daemon had the `.db` open. The same
run outside the synced tree succeeds. Point `--out` somewhere local:

```
python -m wristview.cli run --scan ... --out /tmp/wristview-runs
```

This costs a full matching pass every time it is forgotten, and the error names
SQLite rather than iCloud, so it does not look like a storage problem.
