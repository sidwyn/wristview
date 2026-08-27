# Experiment plan

**27 August 2026.** How to shoot the data, how to train the policies, and how
to read the result.

Read this with `CLAUDE.md` for the standing rules and `CAPTURE-SOP.md` for the
capture details.

---

## 1. The question

We render a robot wrist-camera view from head-mounted human video. The
question is:

> Does a RENDERED wrist view train a policy as well as a REAL one?

Nobody has published this. WARPED renders wrist views but never recorded a
real wrist camera, so it had nothing to compare against. We record both.

---

## 2. What a robot sees, and why it matters

A robot has a camera on its wrist. It has no camera on its head.

So a policy that we can deploy must use the wrist view. The head camera is
how we CAPTURE data. It is not an input the robot will ever have.

This decides the design below. Two of the three groups are single-camera and
deployable. The third is context only.

---

## 3. The task and the object

### 3.1 Today: 1 task only

**LIFT.** Grasp the carton. Raise it about 10 cm. Hold 1 second. Put it
back down. Withdraw.

30 demos plus 30 group-C takes. 60 episodes.

1 task is the right scope. The experiment asks whether a rendered wrist view
helps a policy learn. A task the policy can actually learn from 30 episodes
is what makes that question answerable. A hard task makes every group fail
equally and teaches us nothing about wrist views.

Backup tasks are in Appendix A. Do not shoot them until Lift has run end to
end and produced a result.

### 3.2 The object: a rigid matte carton

| Requirement | Why | Matcha carton |
|---|---|---|
| Rigid | The pose solver assumes a rigid object. The soft toy cube deforms 21% under grip. | yes |
| Matte | Shiny surfaces move their highlights with the camera. This breaks feature matching and splat training. | yes |
| Box-shaped | The renderer draws a box. A cylinder would render as a block. | yes |
| Textured | Print gives the mask and the splat something to hold. | yes |
| Measured | Write the dimensions into `estimate.pose.object_dimensions_m`. | measure it |
| 5 to 8 cm | A real parallel gripper opens 8 to 9 cm. Franka is about 8, UMI about 9. | check it |

**Size rule: 5 to 8 cm on the axis the fingers close along.** Below 5 cm the
hand hides most of the object. Above 8 cm no target gripper can grasp it, so
the demonstration is not executable on the hardware the data is for.

A 60-tissue Kleenex cube is about 11 cm. Too wide. It also squashes under
grip and its top is not flat.

Measure all 3 edges. If it is not a cube, the dimensions are not
interchangeable.

Good household objects at the right size: a tea box, a medicine carton, a
toothpaste box, a small gift box, a deck of cards, a bar of soap in its
carton. Tape a carton shut to stiffen it.

### 3.3 The carton is ABSENT from the scan

Do not put the carton on the mat during the scan.

Its colour is sampled from the demo frames, not the scan, so the scan does
not need it. Leaving it out removes the static ghost that has appeared in
every render so far: the splat's frozen photograph of the object sitting
where it was scanned, while the tracked box moves away from it.

Scan the mat, the marker and any static props. Then place the carton and
shoot.

---

## 4. Filming

### 4.1 Set up once

- Mat taped flat at all 4 corners.
- Marker id 0, 100 mm, where the hand never passes over it.
- Mark 5 or 6 START spots across the mat. The carton sits on one of them
  each take. There are no PLACE spots. Lift does not move the object across
  the mat.
- Object: matcha tea carton, 60 x 60 x 70 mm, cream with black kanji and a
  red stamp. Rigid. Tape it shut or weight it if empty.
- **Stand the carton UPRIGHT, 7 cm tall, on its 6 x 6 footprint. Every
  take.** The config says Z is vertical at 0.070. Lay it on its side once
  and the box renders at the wrong height and the plane solve puts it at the
  wrong distance from the desk.
- Midday light. Write down the shade position.
- Head camera: wide lens, stabilisation off, 1080p60, shutter 1/120, ISO,
  white balance and focus locked.
- Marker id 1 is OFF the wrist camera. It stays off.

Config for this session:

```
estimate.pose.object_dimensions_m: [0.060, 0.060, 0.070]
estimate.pose.object_height_m:     0.070
prompt:                            "cream cardboard tea box"
aruco_marker_length_m:             0.100
```

### 4.2 The scan. 85 s, one continuous take, head camera only

| Time | Height | Purpose |
|---|---|---|
| 25 s | 40 to 60 cm, looking down | Localisation. The demo's viewpoint. |
| 25 s | 20 to 30 cm | **The render band. The virtual camera lives here.** |
| 20 s | 15 to 20 cm | Margin below the render band. Do not go lower. |
| 15 s | 80 to 100 cm | Room context. |

In the two middle passes, look ACROSS the mat as well as down at it. Sweep
in the directions the hand travels. Include the direction the hand enters
from. This is what stops the render from going blank at the start.

Do not go below 15 cm. The lens stops focusing there. Blurry training views
cost sharpness across the whole model, not only where they were taken.

**The carton is OFF the mat for the scan.** See section 3.3. Its colour comes
from the demo frames, not the scan, and leaving it out removes the static
ghost.

### 4.3 Verify before you shoot 30

1. Shoot demo 1.
2. Run Stage 0 and pre-flight on the scan plus demo 1.
3. Run `tools/check_object_mask.py` on demo 1. The carton is cream and the
   mat is pale, so segmentation is the risk. The check compares mask area
   against the area a 60 x 60 x 70 mm box should cover at its measured
   distance, so it does not depend on contrast. A mask that has grabbed the
   mat reads 10 to 20 times too large. Look at the overlay as well.
4. Do NOT shoot the rest until all 3 pass.

This takes 15 minutes. It is the only thing that stops a whole session from
being wasted.

### 4.4 The 30 demos. LIFT. No wrist camera. About 10 s each

Reach in. Pinch grasp. **Raise the carton about 10 cm. Hold 1 second. Lower
it back to the mat. Release.** Withdraw.

The object does not travel across the mat. Lift is grasp and raise. Moving
it to a target is the Place task in Appendix A, and it is not today.

- Hands in frame for the whole clip.
- Pinch, thumb opposed to 2 fingers, on the 6 cm faces. Do not wrap the
  carton. A wrap hides the object and breaks the pose estimate.
- Vary the START spot. Cycle through them.
- Vary approach angle, lift height and speed a little. Never rush the grasp.
- Write down each one: "ep7: start C".
- Return the carton to its spot, upright, between takes. Nothing else moves.

**No taps.** Taps exist to align 2 cameras. There is only 1 camera here.

WARPED shot 30 demonstrations per task in 3 to 5 minutes. Short takes are
correct.

### 4.5 The 30 group-C takes. Wrist camera ON. Taps at both ends

1. Start the head camera first, then the wrist camera.
2. Tap the bare table beside the mat. Pause 1 beat.
3. The same lift, same speed, same grasp.
4. Pause 1 beat. The same tap again.
5. Stop the wrist camera first, then the head camera.

**Vary the START spot here too**, exactly as in 4.4. Both sets need the same
spread, or the comparison is between 2 different distributions.

**Why 30 and not fewer.** 2 reasons.

Equal episode counts. If group C trains on 15 and group B on 30, any difference
between them could be dataset size instead of view type. That is the exact
confound this experiment exists to avoid. At 30 and 30 the comparison is
direct, with no subsampling.

Group C's labels are noisier. Its actions still come from head-camera hand
tracking, and in these takes the wrist rig sits on the knuckles and occludes
the hand. That noise biases group C downward, which makes the render look
CLOSER to real than it is. A bias in our favour is the worst kind to carry
into a result we intend to show people. More episodes average it down.

**Measure what the rig costs. Do not assume it.**

Group C's actions come from head-camera hand tracking, and in those takes the
wrist rig sits on the hand. If the rig degrades tracking, group C's labels are
noisier and the render looks closer to real than it is. The paired design in
5.3 cancels that bias, but only if B' and C really do share the same
degradation, so the size of it has to be known.

Comparing the 30 clean takes against the 30 rigged ones measures it directly.
Report all 4, for both sets:

| Metric | Where it comes from | Why it matters |
|---|---|---|
| hand detection rate | Stage 3, `hand detected on N/M frames` | a rig that hides the hand shows up here first |
| hand reprojection, median and p90 | Stage 3, `hand_position_vs_detected_wrist`, an independent check | the direct measure of tracking accuracy |
| identity switches | Stage 3, `hand_side_switches` | the rig can make one hand look like two |
| velocity flags | Stage 4, frames above 900 deg/s and frames dropped | jitter from partial occlusion appears as implausible rotation |

**The one data point so far says the rig does NOT hurt.** real26/b demo_1,
shot with the rig on: 510/510 hand frames, reprojection 10.78 px median and
13.75 p90, identity stable Right throughout, 0 velocity flags of 510. The
clean demo_0 was 432/432, 13.20 px median, 14.83 p90, stable, 0 flags. The
rigged take tracked BETTER on every measure.

What failed on demo_1 was the OBJECT track, not the hand, and the cause was
defect 24, not occlusion.

One take is not a result. If the 30-versus-30 comparison contradicts it,
B' is compromised and we need to know before trusting it.

If time runs short, stop this set early. 20 is much better than 15.

---

## 5. Policy training

### 5.1 What a policy is

A policy is a function that runs about 30 times a second:

> picture in → next small movement out

Each output is one action: move the gripper a few mm, rotate a little, open
or close the fingers. Hundreds of them chained together make the task.

The task is not an input. We train 1 policy for 1 task, so "move the cube
left" is inside its weights. WARPED trained a separate policy per task.

### 5.2 Inputs and outputs

| | Value |
|---|---|
| Image input | 1 camera view, 640x360, per frame |
| State input | The current end-effector pose (proprioception) |
| Output | Relative pose change + gripper open/close |
| Architecture | Diffusion policy, or ACT. Both are in LeRobot. |

WARPED used a diffusion policy conditioned on wrist-view observations and
proprioception, producing relative pose and gripper action chunks. Copy that.

Our actions come from Stage 4. The retarget already produces an
end-effector pose for every frame. Those poses ARE the actions.

### 5.3 The training runs

There are 2 comparisons and they use 2 different sets of episodes.

**Set 1 — the 30 clean demos.** No wrist camera worn.

| Run | Image input | Deployable |
|---|---|---|
| **NULL** | none, or shuffled | no. Sanity floor. |
| **A** | ego view | no. Context only. |
| **B** | **rendered** wrist view | **yes. This is the product.** |

A and B come from the SAME footage. Train twice, change the input. No extra
filming.

**Set 2 — the 30 group-C takes.** Wrist camera worn, so the same episode
carries both a real wrist video and a renderable head video.

| Run | Image input | Deployable |
|---|---|---|
| **A'** | ego view | no. Paired baseline. |
| **B'** | **rendered** wrist view | yes |
| **C** | **real** wrist view | yes. The upper bound. |

All 3 come from the SAME 30 episodes. Same trajectories, same frames, same
actions. Only the pixels differ.

**Why set 2 exists: B versus C must be PAIRED.**

If B trains on set 1 and C on set 2, they are different episodes with
different trajectories, different grasps and different start spots. The
headline fraction would then mix a view change with a dataset change, and
the denominator would not be the quantity we want. Equal counts control for
size. They do not control for episode identity.

Rendering wrist views from the group-C takes' own head footage gives B' on
exactly the same episodes as C. That is the comparison the question
deserves.

**It also cancels a bias we would otherwise have to argue away.** Group C's
actions come from head-camera hand tracking, and in those takes the wrist
rig sits on the hand. If that degrades tracking, C's labels are noisier,
which flatters the render. In the paired design B' inherits the same
degraded tracking from the same footage, so the bias cancels instead of
needing an excuse.

**Note:** the one group-C take measured so far does NOT show degraded
tracking. real26/b demo_1, shot with the rig on, tracked the hand at
10.78 px median against demo_0's 13.2 px with no rig, 510/510 detected, no
identity switches. What failed on demo_1 was the OBJECT track, and that was
defect 24, not occlusion. Measure it across all 60 takes before assuming
either way.

### 5.4 Rules that decide if the numbers mean anything

**Hold out whole EPISODES, not frames.** Adjacent frames are nearly
identical. A frame-level split leaks the answer into training and every run
scores well.

**Train each run at least 3 times with different random seeds.** With 30
episodes, run-to-run variation can be larger than the effect. Report the
spread, not only the mean. Without this we cannot tell a result from a
coincidence.

**Compare within a set, never across.** The paired design in 5.3 makes this
automatic: A vs B both come from set 1, and A' vs B' vs C all come from set
2. Never compute a fraction that mixes the 2 sets.

**Compare at equal episode count.** If the group-C set ends up shorter than
30, that only affects set 2's internal comparison, which stays valid because
all 3 runs share the same episodes.

---

## 6. How we evaluate. No robot needed

### 6.1 The method

Hold out 3 episodes. Never train on them.

For each frame of a held-out episode:

1. Show the policy the picture ONLY. Hide the recorded action.
2. The policy outputs its predicted action.
3. Compare that prediction to the recorded action.
4. Write down the error.

Repeat for every frame. Average.

It is an exam with an answer key. The held-out episode is the key. We cover
it, ask the question, then mark it.

### 6.2 A worked example

Frame 100. The picture shows the hand approaching the cube.

| | Forward | Down | Error |
|---|---|---|---|
| What I actually did | 3.0 mm | 2.0 mm | — |
| Policy A predicts | 5.0 mm | 4.0 mm | 2.0, 2.0 |
| Policy B predicts | 3.0 mm | 2.5 mm | 0.0, 0.5 |

B is closer on this frame. Do this 300 times per episode, 3 episodes, and
average.

### 6.3 What comes out

Not a video. **A table of numbers.** Something like:

| Run | Set | Action error, mean of 3 seeds | Spread |
|---|---|---|---|
| NULL | 1 | 9.10 mm | ±0.30 |
| A | 1 | 4.20 mm | ±0.40 |
| B | 1 | 2.80 mm | ±0.25 |
| A' | 2 | 4.35 mm | ±0.35 |
| B' | 2 | 2.95 mm | ±0.30 |
| C | 2 | 2.60 mm | ±0.20 |

Nothing moves, because there is no robot. The policy only produces numbers.

**Also make 1 chart:** plot the predicted path against the actual path for
1 held-out episode. Two lines that should sit on top of each other. Where
they separate is where the policy is confused. Much easier to read than the
table.

---

## 7. Reading the result

### 7.1 A vs B — does the render help?

**Set 1, the 30 clean demos.** Compare A's error against B's.

| Outcome | What it means |
|---|---|
| B error < A error | The rendered wrist view adds information the ego view does not carry. This is the product claim. |
| B error = A error | The render adds nothing. It may be wrong, or the task may be too easy to need a wrist view. |
| B error > A error | The render is actively misleading. Look for a defect before believing it. |

Using the example numbers: 4.20 to 2.80 mm. The render helps.

**First check NULL.** If A and B are both close to NULL, nothing learned
anything and the comparison is meaningless.

### 7.2 B' vs C — how close does the render get?

**Set 2, the 30 group-C takes. All 3 runs on the same episodes.**

```
gap that a REAL wrist view buys   = A' error − C error
gap that MY RENDER buys           = A' error − B' error

fraction closed = (A' − B') / (A' − C)
```

With the example numbers:

```
(4.20 − 2.80) / (4.20 − 2.50) = 1.40 / 1.70 = 82%
```

**The render closes 82% of the gap between no wrist view and a real one.**

That is the headline sentence. Every term comes from the same 30 episodes,
so there is no cross-set arithmetic anywhere in it.

WristWorld (arXiv 2510.07313) reports 42.4% for a different method. Anything
near or above that is a real result.

| Outcome | What it means |
|---|---|
| B' close to C | The render carries almost the whole benefit of a real camera. The strongest possible result. |
| B' between A' and C | The render helps but is not equal to real. Report the fraction honestly. |
| B' close to A' | The render adds little. Find out why before shooting more. |

**Sync residual gates this comparison.** Group C's images must align in time
with actions derived from the head camera. A timing error inflates C's error
and makes the render look better than it is, which is the direction we are
already worried about.

**Drop any take whose tap-sync residual exceeds 33 ms, one frame at 30 fps.**

That threshold is derived, not chosen. Measured on real26/bm demo_0, at the
15 Hz control rate:

| | value |
|---|---|
| median action per control step | **1.52 mm** |
| p90 action per control step | 9.38 mm |
| end-effector speed, median | 0.023 m/s |
| end-effector speed, p90 | 0.141 m/s |

A sync error moves the image against the action by speed times the error:

| error | displacement injected | as a share of the median action |
|---|---|---|
| **1 frame, 33 ms** | 0.76 mm median, 4.69 mm p90 | **50%**, 309% at p90 |
| 2 frames, 67 ms | 1.52 mm median, 9.38 mm p90 | 100%, 618% at p90 |
| 3 frames, 100 ms | 2.28 mm median, 14.07 mm p90 | 150%, 927% at p90 |

So one frame is not a small error. It injects half the median action, and
during the fast part of a reach it injects three times it. The concern about
2 frames was right and understated.

**One frame is the drop threshold, not the target.** Audio cross-correlation
on the tap transient pinned both taps to 1 ms agreement on real26. A residual
above about 5 ms therefore means something real went wrong, clock drift or
dropped frames, not measurement noise. Investigate at 5 ms, drop at 33 ms.

**Also check drift, not only offset.** The taps are at both ends for this
reason. If the start and end residuals differ by more than one frame, the two
clocks are drifting and a single offset cannot align the take. Drop it.

The wrist camera duplicates frames in low light, which is a known behaviour
of this rig, so a drifting residual is the expected symptom. Carry the
per-frame ambiguity forward rather than interpolating it away.

### 7.3 The free qualitative check

Play the rendered wrist view beside the real one from the same group-C
episode, aligned by the taps. This works because set 2 holds both for the
same episode.

We cannot compute a number from this. The 2 views come from different
positions and putting them in the same position is the pose-recovery problem
we abandoned. But we can see if they look like the same kind of data, and
it is more persuasive to a person than any table.

---

## 8. What this does NOT prove

**Action error is not task success.** It measures agreement with the
demonstrator, frame by frame. When a policy actually runs, small errors
build up: it drifts, sees a view it never trained on, drifts more.

A real success rate needs a physical arm. WARPED ran 20 trials per task on a
real robot. An SO-101 costs $100 to $360 and is what the field uses. That is
a later step and a stronger claim.

**Known gaps that a better splat will not fix:**

- The object renders as 1 averaged colour. The real cube has 6 bright faces.
- There are 2 cubes in every frame: the tracked box, and the splat's frozen
  photograph of the cube where it sat during the scan.
- The gripper is a proxy, not a real gripper mesh.

State all 3 wherever the result is shown.

---

## 9. The order of work

**Cost of 60 episodes.** About 12 hours. One overnight run.

The 22.4 min/episode figure was measured on 25 to 31 second demos. Our takes
are about 10 s, roughly 200 frames, and stages 2 and 3 scale with frame
count.

| Take length | Per episode | 60 episodes |
|---|---|---|
| 10 s, our plan | 10.7 min | 12.2 h |
| 25 s, what was measured | 23.6 min | 25.0 h |

Ask Claude Code to cut Stage 2 while it runs. It is 69% of the marginal cost
and it named 3 ways: halve `retrieval_top_k` (configs/default.yaml:211, default 10), batch the pairs, or move the stage
to the GPU. Anything it saves applies to every future session.

1. Shoot: scan with NO carton, 1 demo, verify, 30 demos, 30 group-C takes.
2. Process every episode through the pipeline.
3. Convert to LeRobot dataset format.
4. Train NULL, A, B and C. 3 seeds each.
5. Evaluate on the 3 held-out episodes.
6. Report the table, the fraction closed, and the chart.

---

## Appendix A. Backup tasks

Do not shoot these until Lift has produced a result. They are here so the
reasoning is not lost.

### A.1 Place

Grasp the block. Move it to a marked target zone on the mat. Release.

Adds precise placement to Lift. A wrist view should help more here than with
Lift, and that difference is itself informative.

Reuses the Lift scan exactly. No setup change. Just mark the target zone.

### A.2 Stack

Grasp the block. Put it on top of a book or a rigid platform. Release.

The smallest alignment tolerance of the 3, so the hardest case for a policy
with no wrist view, and the best test of whether the render helps.

**How to reuse 1 scan.** Put the book on the mat at setup and never move it.
It is present in the scan and in every demo. Lift and Place ignore it. Stack
uses it as the target.

The book must be 3 to 5 cm tall. Lower and alignment does not matter. Higher
and the block hides behind its edge. Matte, rigid, patterned cover, because
it contributes to the splat.

**Track only the moving block.** The book stays in the splat, where it
renders with its real appearance and real colours instead of the averaged
colour a drawn box gets. Claude Code assessed this as closer to correct than
tracking both, not as a compromise.

2 rules:

1. The block and the book must look different. If they look alike,
   Grounding DINO picks by argmax between near-equal scores, which flips
   mid-clip. That is the same silent damage as the hand-identity switch.
   Name the block's colour in the prompt.
2. Nothing tracks the book's pose. To score stack accuracy in mm later, we
   must first recover the book's position from the reconstruction. Nothing
   does that today.

### A.3 Rotate Box — EXCLUDED

Do not shoot a rotation task with the current renderer.

The renderer draws the object as 1 averaged colour with no texture. A cube
turned 90 degrees about its vertical axis has an identical silhouette. A
rendered wrist view of a rotation shows nothing changing, so a policy
trained on it cannot see orientation at all.

This becomes possible when the object carries real texture, which needs
Sam3D. Until then use tasks where POSITION changes.

### A.4 On cans and cylinders

RoboMimic's Can task uses an aluminium can, and that is standard in the
field. It does not transfer to this pipeline, for 2 reasons that are ours
and not theirs.

RoboMimic runs in simulation, or feeds a real camera straight to the policy.
Nothing reconstructs the can. We photograph the scene, build a 3D model from
it, and re-render.

1. **Specular metal.** Highlights move with the camera, so they are not
   static features. This breaks feature matching and splat training.
2. **A can is a cylinder.** The renderer draws boxes. It would come out as a
   rectangular block.

Both are testable rather than certain. Worth measuring before ruling it out
permanently, and worth revisiting when the renderer handles primitives other
than boxes.

### A.5 Multi-object tracking

Stage 3 assumes exactly 1 object, enforced at 3 independent points:
`objects.py:76` takes argmax over Grounding DINO's boxes, `objects.py:161`
keeps only the largest connected blob, and `clean_mask` rejects a mask above
`max_mask_fraction`. Downstream, `object_pose.npy` is (frames, 4, 4) with no
object index.

Claude Code estimated about 1 day to support 2 objects. Detection and masks
are about 2 hours. The real work is attribution in `carry.py`: when the hand
closes, which object did it pick up? Fingertip proximity solves most of it
and fails exactly where stacking is interesting, with the mover directly
above the base and both within centimetres of the fingertips.

Not needed while the single-object approach in A.2 works.
