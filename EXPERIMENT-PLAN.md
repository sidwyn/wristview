# Experiment plan

**27 August 2026.**

This document has three parts:

- how to record the data — section 4
- how to train the policies — section 5
- how to read the result — sections 6 and 7

Read `CLAUDE.md` for the standing rules. Read `CAPTURE-SOP.md` for the
capture procedure.

---

## 0. Words used in this document

| Word | Meaning |
|---|---|
| ego view | The video from the camera on the head. |
| rendered wrist view | A wrist-camera video that the pipeline makes. It is not recorded. |
| real wrist view | A video from the camera on the wrist. |
| take | One continuous recording. |
| demo | One take that shows the task one time. |
| episode | One demo after the pipeline processes it. |
| set | A collection of episodes. This document has set 1 and set 2. |
| group | One training run, and the cameras that it uses. The groups are A, B and C. |
| policy | The network that the episodes train. |
| action | One small movement of the gripper. The policy gives one action for each frame. |
| splat | The 3D model of the scene. The renderer uses the splat. |
| ghost | A frozen image of the carton in the splat, at the position where you scanned it. |

---

## 1. The question

We make a robot wrist-camera view from head-mounted human video. The
question is:

> Does a RENDERED wrist view train a policy as well as a REAL wrist view?

No person has published an answer. WARPED makes rendered wrist views, but
WARPED did not record a real wrist view. Thus WARPED had nothing to compare
against. We record both views.

---

## 2. What the policy sees, and why this is important

A robot does not wear a head camera. But a robot usually has more than one
camera. DROID uses two exterior cameras and one wrist camera. RoboMimic uses
one agentview camera and one wrist camera. Humanoid robots have head
cameras. A policy that reads two or three cameras is usual.

Thus the design is NESTED. It is not a replacement.

| Group | Cameras that the policy sees |
|---|---|
| A | ego |
| B | ego + **rendered** wrist |
| C | ego + **real** wrist |

Group A and group B are different in one item only: the wrist channel. Thus
a difference between A and B comes from the wrist view, and from nothing
else.

**Why the design is nested.** The customer asks this question: "Must I add
these rendered wrist views to my training data?" The customer has cameras
already. The customer adds one channel. The nested comparison gives the
answer to that decision.

A run with the rendered wrist view alone is a SECONDARY result. It is the
WARPED configuration, and it answers a different question: is a rendered
wrist view sufficient alone? Keep this run. But it is not the primary
result.

The ego view is not a deployment input for a wrist-mounted arm. But the ego
view is not useless. Egocentric data trains visual encoders and gives
actions. In this experiment the ego view is a policy input, because the
nested comparison needs a constant background. We add the wrist channel to
that background.

---

## 3. The task and the object

### 3.1 Today: one task only

**LIFT.** Do these steps:

1. Grasp the carton.
2. Lift it approximately 10 cm.
3. Hold it for 1 second.
4. Put it on the mat.
5. Move the hand away.

Record 30 demos and 30 group-C takes. This gives 60 episodes.

One task is the correct quantity of work. The experiment asks if a rendered
wrist view helps a policy to learn. Thus the policy must be able to learn
the task from 30 episodes. If the task is too difficult, all groups fail
equally, and the result tells us nothing about wrist views.

Appendix A gives the backup tasks. Do not record them until Lift gives a
result.

### 3.2 The object: a rigid matte carton

| Requirement | Why | Matcha carton |
|---|---|---|
| Rigid | The pose solver assumes a rigid object. The soft toy cube changes shape by 21% when you grip it. | yes |
| Matte | A shiny surface moves its highlights when the camera moves. This breaks feature matching and splat training. | yes |
| Box-shaped | The renderer draws a box. It draws a cylinder as a block. | yes |
| Textured | The print gives data to the mask and to the splat. | yes |
| Measured | Write the dimensions into `estimate.pose.object_dimensions_m`. | measure it |
| 5 to 8 cm | A parallel gripper opens 8 to 9 cm. Franka opens approximately 8 cm. UMI opens approximately 9 cm. | yes, 6 cm |

**Size rule: 5 to 8 cm on the axis along which the fingers close.** Below
5 cm the hand hides most of the object. Above 8 cm no gripper can grasp it.
Then the robot cannot do the demonstration, and the data has no use.

A 60-tissue Kleenex cube is approximately 11 cm. It is too wide. It also
changes shape when you grip it, and its top is not flat.

Measure all three edges. If the object is not a cube, the three dimensions
are different. Do not exchange them.

These household objects have the correct size: a tea box, a medicine
carton, a toothpaste box, a small gift box, a deck of cards, a bar of soap
in its carton. Put tape on a carton to make it more rigid.

### 3.3 Keep the carton OFF the mat during the scan

Do not put the carton on the mat during the scan.

The pipeline takes the carton colour from the demo frames. It does not take
the colour from the scan. Thus the scan does not need the carton.

If the carton is in the scan, the splat keeps a ghost. The ghost is a frozen
image of the carton at the scan position. The tracked box then moves away
from the ghost, and both are visible in the render. Every render before this
session had this fault.

Scan the mat, the marker and the static props. Then put the carton on the
mat and record the demos.

---

## 4. How to record

### 4.1 Set up one time

- Put tape on all four corners of the mat, so that the mat is flat.
- Put marker id 0, 100 mm, where the hand does not move above it.
- Mark five or six START spots on the mat. Put the carton on one spot for each
  take. There are no PLACE spots. Lift does not move the object across the
  mat.
- Object: matcha tea carton, 60 x 60 x 70 mm, cream, with black kanji and a
  red stamp. It is rigid. If it is empty, put tape on it or add weight.
- **Put the carton UPRIGHT on its 60 x 60 mm base, 70 mm tall. Do this for
  every take.** The config gives 0.070 m for the vertical axis. If you lay the
  carton on its side, the renderer draws the box at the wrong height. The
  plane solve then puts the box at the wrong distance from the desk.
- Record at midday. Write down the position of the shade.
- Head camera: wide lens, stabilisation off, 1080p60, shutter 1/120. Lock the
  ISO, the white balance and the focus.
- Keep marker id 1 off the wrist camera. It stays off.

Config for this session:

```
estimate.pose.object_dimensions_m: [0.060, 0.060, 0.070]
estimate.pose.object_height_m:     0.070
estimate.object.prompt:            "cream cardboard tea box"
scene.scale.aruco_marker_length_m: 0.100
scene.splat.enabled:               false   # Mac only. The GPU trains the splat.
```

The full key is `scene.scale.aruco_marker_length_m`. The short form
`scene.aruco_marker_length_m` raises an unknown-key error.

**Hold the camera the same way up for every take in the session.** A take
with a different rotation still localises, but each tool must then apply the
rotation metadata in the same way. Do not create that risk.

### 4.2 The scan: 85 s, one continuous take, head camera only

**Measured budget. The camera height is what counts, not the elapsed time.**
real27 failed this section. It put 88% of its frames above 38 cm and only
5.5% in the render band. Count the seconds out loud while you scan.

| Time | Height | Purpose |
|---|---|---|
| 15 s | 80 to 100 cm | Room context. Do this first, then go down. |
| 20 s | 40 to 60 cm, look down | Localisation. This is the viewpoint of the demo. |
| **35 s** | **20 to 30 cm** | **The render band. The virtual camera is here.** |
| 15 s | 15 to 20 cm | Margin below the render band. Do not go lower. |

**A minimum of 45 s of the 85 s must be below 30 cm.** This is 50 s in the
table, which gives you margin.

**Go around the mat. Do not stay on one side.** Divide the mat into 8
sectors, as on a compass. The render band pass must enter all 8 sectors.
real27 occupied 4 of 8, and that alone can fail the viewpoint gate. Move
around the mat one time in the render band, then move around it again in the
opposite direction.

In the two low passes, look ACROSS the mat and also down at it. Move the
camera in the directions in which the hand moves. Include the direction from
which the hand comes. This prevents a blank render at the start of the clip.

Do not go below 15 cm. The lens cannot focus below 15 cm. Blurred training
views make the full model less sharp, not only the part that you recorded
too near.

**Take the carton out of the scanned scene. Off the mat is not sufficient.**
In real27 the carton was off the mat, but it stayed on the desk at the right
side. The splat then holds a ghost of the carton at that position, and the
object detector finds it in the scan frames. Put the carton on a chair
behind the camera. Then no frame of the scan contains it.

### 4.3 Verify before you record 30 takes

There are two checks. Run the fast check first. It costs one minute. The
full check costs 35 minutes, and you run it one time.

**Check 1 — the fast gate. Run this on the raw files, before anything.**

`tools/check_take.py` reads the video file only. It does not reconstruct
anything. It answers five questions:

| Question | Limit |
|---|---|
| Is the demo the correct length? | 8 to 13 s |
| Does the demo start with no hand in the frame? | No hand in the first 30 frames |
| Does the demo have the same rotation tag as the scan? | They must agree |
| How far was the camera from marker id 0 in each scan frame? | 45 s minimum below 30 cm |
| How many of the 8 sectors did the render band cover? | 8 of 8 |

The camera distance comes from the apparent size of marker id 0. The marker
is 100 mm. Thus the distance is the focal length multiplied by 0.100 and
divided by the marker side in pixels. No reconstruction is necessary.

Both faults in real27 are in this table. The full check found them after 35
minutes. This check finds them before you put the camera down.

**Check 2 — the full pre-flight. Run it one time, on the scan and demo 1.**

1. Record demo 1.
2. Run the pre-flight, then stages 0 to 3 with the splat disabled.
3. Run `tools/check_object_mask.py` on demo 1. The carton is cream and the
   mat is pale. Thus segmentation is the risk. The check compares the mask
   area against the area of a 60 x 60 x 70 mm box at its measured distance.
   Thus the check does not depend on contrast. If the mask includes the mat,
   the area is 10 to 20 times too large. Also look at the overlay image.
4. Run `tools/close_range_coverage.py`. This is the true measure of the
   render band. Check 1 is the approximation of it.
5. Do NOT record the other takes until all the checks pass.

**Look at the scan frames yourself.** Extract six frames with ffmpeg and
look at them. The object detector cannot answer the question "is the carton
in the scan", because a whole-frame box is a non-detection that the detector
reports as a detection. Your eyes answer it in 10 seconds.

### 4.4 The 30 demos: LIFT, no wrist camera, approximately 10 s each

**Start every take with the hand OUT of the frame for one full second.**
This is the rule that real27 broke. The hand was on the carton at frame 0.
The pipeline needs a resting window before the grasp, so that it can solve
the carry. With no resting window, the object stays welded to the desk for
every frame, the height spread is 0.000 cm, and the take is not manipulation
data. The guard refused the take instead of giving a wrong trajectory.

The sequence for each take:

1. Start the recording. The hand is out of the frame.
2. Wait one full second. Count it.
3. Move the hand into the frame. Do a pinch grasp.
4. **Lift the carton approximately 10 cm. Hold it for 1 second. Put it on
   the mat. Release it.**
5. Move the hand out of the frame.
6. Wait one full second. Then stop the recording.

One second at each end plus an 8 second lift gives 10 s. Your last take was
17.6 s, and you thought that it was 15 s. Use a timer for the first three
takes, until your count agrees with the clock.

The object does not move across the mat. Lift is a grasp and a lift. To move
the object to a target is the Place task in Appendix A. Do not do it today.

- Keep the hands in the frame for the full clip.
- Do a pinch grasp on the 60 mm faces, with the thumb opposite two fingers.
  Do not put the hand around the carton. A hand around the carton hides the
  object and breaks the pose estimate.
- Change the START spot. Use the spots in sequence.
- Change the approach angle, the lift height and the speed by a small
  quantity. Do not do the grasp quickly.
- Write down each take, for example: "ep7: start C".
- Put the carton back on its spot, upright, between takes. Move nothing
  else.

**Do no taps.** A tap aligns two cameras. There is only one camera in this
set.

WARPED recorded 30 demonstrations for each task in 3 to 5 minutes. Short
takes are correct.

### 4.5 The 30 group-C takes: wrist camera ON, taps at both ends

1. Start the head camera first. Then start the wrist camera.
2. Tap the bare table at the side of the mat. Wait one beat.
3. Do the same lift, at the same speed, with the same grasp.
4. Wait one beat. Do the same tap again.
5. Stop the wrist camera first. Then stop the head camera.

**Change the START spot in this set also**, as in section 4.4. The two sets
must have the same spread of start spots. If they do not, the comparison is
between two different distributions.

**Why 30 takes and not fewer.** There are two reasons.

The first reason is equal episode counts. Group C must not train on 15
episodes while group B trains on 30. Then a difference between them can come
from the quantity of data, and not from the type of view. This experiment
exists to prevent that error. At 30 and 30 the comparison is direct, and we do
not remove episodes.

The second reason is label noise. The actions of group C come from
head-camera hand tracking. In these takes the wrist rig is on the knuckles
and hides part of the hand. This noise makes the group C result worse. Thus
the render looks CLOSER to real than it is. An error in our favour is the
worst type of error to put into a result that we show to other people. More
episodes decrease this noise.

**Measure the cost of the rig. Do not assume it.**

If the rig makes the tracking worse, the labels of group C are noisier. Then
the render looks closer to real than it is. The paired design in section 5.3
cancels this error, but only if B' and C have the same degradation. Thus you
must know the size of the degradation.

A comparison of the 30 clean takes against the 30 takes with the rig
measures it directly. Report all four values, for both sets:

| Metric | Source | Why it is important |
|---|---|---|
| hand detection rate | Stage 3, `hand detected on N/M frames` | If the rig hides the hand, this value decreases first. |
| hand reprojection, median and p90 | Stage 3, `hand_position_vs_detected_wrist`, an independent check | This is the direct measure of tracking accuracy. |
| identity switches | Stage 3, `hand_side_switches` | The rig can make one hand look like two hands. |
| velocity flags | Stage 4, frames above 900 deg/s, and frames dropped | Partial occlusion makes jitter, which shows as impossible rotation. |

**The one measurement that we have says that the rig does NOT make the
tracking worse.** In real26/b demo_1, recorded with the rig on: 510/510 hand
frames, reprojection 10.78 px median and 13.75 px p90, identity stable
Right, 0 velocity flags of 510. The clean demo_0 gave 432/432 frames, 13.20
px median, 14.83 px p90, stable identity, 0 flags. The take with the rig
tracked BETTER on every metric.

**The current clean baseline is real27 demo_0**: 300/300 frames detected,
reprojection 6.03 px median and 9.17 px p90, identity stable Right, 0
velocity flags. This is much better than real26. Compare the rigged takes
against these numbers, not against real26.

The fault in demo_1 was in the OBJECT track, not in the hand track. The
cause was defect 24, not occlusion.

One take is not a result. If the comparison of 30 against 30 disagrees with
it, B' is not usable. You must know this before you use B'.

If you do not have sufficient time, stop this set early. 20 takes are much
better than 15 takes.

---

## 5. Policy training

### 5.1 What a policy is

A policy is a function. It runs approximately 30 times each second:

> image in → next small movement out

Each output is one action: move the gripper a few mm, rotate it a small
quantity, open or close the fingers. Hundreds of actions in sequence do the
task.

The task is not an input. We train one policy for one task. Thus "lift the
carton" is in the weights of that policy. WARPED trained a different policy
for each task.

### 5.2 Inputs and outputs

| Item | Value |
|---|---|
| Image input | One or two camera views, 640 x 360, for each frame. See 5.3. |
| State input | The current end-effector pose (proprioception). |
| Output | Relative pose change, and gripper open or close. |
| Architecture | Diffusion policy. LeRobot also has ACT. |

WARPED used a diffusion policy. The inputs were wrist-view observations and
proprioception. The outputs were chunks of relative pose and gripper
actions. Copy the architecture and the action format.

**These three facts are verified in the LeRobot 0.4.4 source:**

1. The diffusion policy accepts more than one camera. It gets the visual
   features from each camera and stacks them (`modeling_diffusion.py:130`).
   GitHub issue 212 is closed and the correction is in the release. Do not
   change to ACT.
2. **All cameras must have the same image shape**
   (`configuration_diffusion.py:241`). This is a hard limit. The render is
   640 x 360. Thus make the ego frames 640 x 360 also. The ego frames must not
   stay at 1920 x 1080 while the renders are 640 x 360. The training run then
   stops with an error when it reads the config. The failure is not quiet, but
   it occurs after you build the dataset. Do the resize in Stage 6, at export.
3. LeRobotDataset v3.0 holds more than one camera stream. The keys are
   `observation.images.<name>`. Write `observation.images.ego`,
   `observation.images.wrist` and `observation.state` into ONE dataset. Each
   run then selects its cameras with `input_features` at training time. One
   dataset supplies all the runs. This makes the paired comparison exact: run
   A and run B read the same episodes, the same frames and the same actions.
   They are different only in the cameras that they can see.

Our actions come from Stage 4. The retarget makes an end-effector pose for
each frame. These poses ARE the actions.

### 5.3 The training runs

There are two comparisons. They use two different sets of episodes.

**Set 1 — the 30 clean demos.** The operator does not wear the wrist camera.

| Run | Cameras that the policy sees | Role |
|---|---|---|
| **NULL** | none, or shuffled | Sanity floor. |
| **A** | ego | Baseline. |
| **B** | ego + **rendered** wrist | **The product claim.** |
| **B-only** | rendered wrist alone | Secondary. The WARPED configuration. |

A and B come from the SAME video. Train two times and change the input. You
do not record more data. B is different from A in one channel only.

**Set 2 — the 30 group-C takes.** The operator wears the wrist camera. Thus
each episode has a real wrist video and a head video that we can render
from.

| Run | Cameras that the policy sees | Role |
|---|---|---|
| **A'** | ego | Paired baseline. |
| **B'** | ego + **rendered** wrist | The render. |
| **C** | ego + **real** wrist | The upper limit. |

All three runs come from the SAME 30 episodes. The trajectories, the frames
and the actions are the same. Only the pixels are different.

**Why set 2 exists: B against C must be PAIRED.**

B must not train on set 1 while C trains on set 2. Then the two runs use
different episodes. The trajectories, the grasps and the start spots are all
different. The primary fraction would then mix a change of view with a change of
dataset. The denominator would not be the quantity that we want. Equal counts
control the quantity of data. They do not control the identity of the
episodes.

We render wrist views from the head video of the group-C takes. This gives
B' on the same episodes as C. This is the correct comparison.

**The paired design also cancels an error.** The actions of group C come
from head-camera hand tracking, and the wrist rig is on the hand in those
takes. If the rig makes the tracking worse, the labels of C are noisier, and
the render looks better than it is. In the paired design, B' has the same
degraded tracking, because it comes from the same video. Thus the error
cancels.

**Note:** the one group-C take that we measured does NOT show degraded
tracking. See section 4.5 for the values. Measure all 60 takes before you
make a conclusion.

### 5.4 Rules that make the numbers valid

**Hold out full EPISODES. Do not hold out frames.** Two adjacent frames are
almost the same. If you split at frame level, the answer goes into the
training data, and all runs give a good score.

**Train each run a minimum of three times, with different random seeds.**
With 30 episodes, the variation between runs can be larger than the effect.
Report the spread and the mean. Without the spread we cannot know if a
result is real or random.

**Compare inside one set. Never compare across the two sets.** The paired
design in 5.3 makes this automatic: A and B are both in set 1, and A', B'
and C are all in set 2. Never calculate a fraction that mixes the two sets.

**Compare at equal episode count.** If the group-C set has fewer than 30
episodes, this affects only the comparison inside set 2. That comparison
stays valid, because all three runs use the same episodes.

---

## 6. How to evaluate. A robot is not necessary

### 6.1 The method

Hold out three episodes FOR EACH SET. Do not train on them. Set 1 and set 2
each need their own held-out episodes, because we never compare the two sets
against each other.

For each frame of a held-out episode:

1. Show the image to the policy. Hide the recorded action.
2. The policy gives its predicted action.
3. Compare the prediction against the recorded action.
4. Write down the error.

Do this for each frame. Then calculate the mean.

This is a test with an answer sheet. The held-out episode is the answer
sheet. We hide it, ask the question, then mark the answer.

### 6.2 An example

Frame 100. The image shows the hand near the carton.

| | Forward | Down | Error |
|---|---|---|---|
| What the operator did | 3.0 mm | 2.0 mm | — |
| Policy A predicts | 5.0 mm | 4.0 mm | 2.0, 2.0 |
| Policy B predicts | 3.0 mm | 2.5 mm | 0.0, 0.5 |

B is nearer on this frame. Do this for 300 frames in each episode, for three
episodes, and calculate the mean.

### 6.3 The output

The output is not a video. **The output is a table of numbers.** For
example:

| Run | Set | Action error, mean of 3 seeds | Spread |
|---|---|---|---|
| NULL | 1 | 9.10 mm | ±0.30 |
| A | 1 | 4.20 mm | ±0.40 |
| B | 1 | 2.80 mm | ±0.25 |
| B-only | 1 | 3.10 mm | ±0.30 |
| A' | 2 | 4.35 mm | ±0.35 |
| B' | 2 | 2.95 mm | ±0.30 |
| C | 2 | 2.60 mm | ±0.20 |

Nothing moves, because there is no robot. The policy gives numbers only.

**Also make one chart.** Plot the predicted path and the actual path for one
held-out episode. The two lines must be on top of each other. Where they
separate, the policy is not correct. The chart is easier to read than the
table.

---

## 7. How to read the result

### 7.1 A against B: does the render help?

**Set 1, the 30 clean demos.** Compare the error of A against the error of
B.

| Outcome | Meaning |
|---|---|
| B error < A error | The rendered wrist view adds data that the ego view does not have. This is the product claim. |
| B error = A error | The render adds nothing. The render can be incorrect, or the task can be too easy to need a wrist view. |
| B error > A error | The render gives incorrect data to the policy. Look for a defect before you accept this result. |

With the example numbers: 4.20 mm decreases to 2.80 mm. The render helps.

**Look at NULL first.** If A and B are both near NULL, no run learned
anything, and the comparison has no meaning.

### 7.2 B' against C: how near does the render get?

**Set 2, the 30 group-C takes. All three runs use the same episodes.**

```
gap that a REAL wrist view gives   = A' error − C error
gap that MY RENDER gives           = A' error − B' error

fraction closed = (A' − B') / (A' − C)
```

With the example numbers from 6.3, **set 2 only**:

```
(4.35 − 2.95) / (4.35 − 2.60) = 1.40 / 1.75 = 80%
```

**The render closes 80% of the gap between no wrist view and a real wrist
view.**

Use A', B' and C. Do not use A and B from set 1 here. To mix the sets is the
error that section 5.4 prohibits. It is an easy error to make, because the
numbers look almost the same.

That sentence is the primary result. All the terms come from the same 30
episodes. Thus there is no arithmetic across the two sets.

WristWorld (arXiv 2510.07313) reports 42.4% for a different method. A value
near 42.4% or above it is a real result.

| Outcome | Meaning |
|---|---|
| B' near C | The render gives almost all the benefit of a real camera. This is the strongest possible result. |
| B' between A' and C | The render helps, but it is not equal to a real camera. Report the fraction correctly. |
| B' near A' | The render adds little. Find the cause before you record more data. |

**The sync residual controls this comparison.** The images of group C must
align in time with actions that come from the head camera. A timing error
increases the error of C and makes the render look better than it is. This
is the direction of error that we are already concerned about.

**Reject any take with a tap-sync residual above 33 ms.** 33 ms is one frame
at 30 fps.

This threshold is calculated, not selected. The measurement comes from
real26/bm demo_0, at the 15 Hz control rate:

| Item | Value |
|---|---|
| median action for each control step | **1.52 mm** |
| p90 action for each control step | 9.38 mm |
| end-effector speed, median | 0.023 m/s |
| end-effector speed, p90 | 0.141 m/s |

A sync error moves the image against the action. The displacement is the
speed multiplied by the error:

| Error | Displacement added | As a fraction of the median action |
|---|---|---|
| **1 frame, 33 ms** | 0.76 mm median, 4.69 mm p90 | **50%**, 309% at p90 |
| 2 frames, 67 ms | 1.52 mm median, 9.38 mm p90 | 100%, 618% at p90 |
| 3 frames, 100 ms | 2.28 mm median, 14.07 mm p90 | 150%, 927% at p90 |

Thus one frame is not a small error. It adds half of the median action. In
the fast part of a reach it adds three times the median action.

**One frame is the reject threshold. It is not the target.** Audio
cross-correlation on the tap transient put both taps within 1 ms on real26.
Thus a residual above approximately 5 ms shows a real fault: clock drift or
dropped frames, not measurement noise. Investigate at 5 ms. Reject at 33 ms.

**Also check the drift, not only the offset.** This is why there are taps at
both ends of the take. If the residual at the start and the residual at the
end are different by more than one frame, the two clocks drift. Then one
offset cannot align the take. Reject it.

The wrist camera duplicates frames in low light. This is a known behaviour
of this rig. Thus a residual that drifts is the expected symptom. Keep the
ambiguity for each frame. Do not remove it by interpolation.

### 7.3 The qualitative check, which costs nothing

Play the rendered wrist view and the real wrist view from the same group-C
episode, side by side, aligned by the taps. This is possible because set 2
has both views for the same episode.

We cannot calculate a number from this. The two views come from different
positions, and to put them in the same position is the pose-recovery problem
that we stopped. But we can see if they look like the same type of data. A
person finds this more convincing than a table.

---

## 8. What this does NOT prove

**Action error is not task success.** Action error measures agreement with
the operator, frame by frame. When a policy operates a robot, small errors
accumulate: the robot drifts, then it sees a view that it did not train on,
then it drifts more.

A real success rate needs a physical arm. WARPED did 20 trials for each task
on a real robot. An SO-101 arm costs $100 to $360, and the field uses it.
That is a later step, and a stronger claim.

**Known gaps that a better splat does not correct:**

- The renderer draws the object in one average colour. The carton is cream
  with black and red print, so the average is near the true colour. This gap
  is much smaller than it was with the six-colour toy cube. But it is not
  zero.
- The gripper is a proxy box, not a real gripper mesh. A real wrist camera
  always sees its own fingers, and a policy that trains on real data expects
  to see them.
- The renderer draws boxes only. It approximates any object that is not a
  box.

**These faults are corrected. Do not report them as gaps:** the ghost is
gone, because the carton is now off the mat during the scan (3.3). Object
float is corrected. Frozen tails are corrected.

State the remaining gaps wherever you show the result.

---

## 9. The order of work

**Cost of 60 episodes: approximately 13 hours.** This is one overnight run.

The 13 hours are 12.2 h of pipeline, and approximately 1 h for the second
render pass of set 2. The second pass makes B' from the head video of the
group-C takes.

The value of 22.4 min for each episode was measured on demos of 25 to 31
seconds. Our takes are approximately 10 s, which is approximately 200
frames. Stages 2 and 3 increase with the number of frames.

| Take length | For each episode | 60 episodes |
|---|---|---|
| 10 s, our plan | 10.7 min | 12.2 h |
| 25 s, the measured value | 23.6 min | 25.0 h |

Tell Claude Code to decrease the cost of Stage 2 while the pipeline runs.
Stage 2 is 69% of the marginal cost. Claude Code gave three methods: divide
`retrieval_top_k` by two (`configs/default.yaml:211`, default 10), batch the
pairs, or move the stage to the GPU. Each saving applies to every future
session.

1. Record: the scan with NO carton, one demo, the verification, 30 demos,
   30 group-C takes.
2. Process each episode through the pipeline.
3. Render the wrist views for BOTH sets. Set 2 needs its own render pass to
   make B'. This is approximately one hour more than the 12 hours.
4. Convert the data to ONE LeRobot dataset. The dataset holds the ego view,
   the rendered wrist view and the state. Make the ego frames 640 x 360 in
   this step, so that they agree with the renders. See 5.2, item 2.
5. Train NULL, A, B and B-only on set 1. Train A', B' and C on set 2. Use
   three seeds for each run.
6. Evaluate on three held-out episodes FOR EACH SET. Never mix the sets.
7. Report the table, the fraction closed from set 2, and the chart.

---

## Appendix A. Backup tasks

Do not record these tasks until Lift gives a result. They are in this
document so that the reasoning is not lost.

### A.1 Place

Grasp the carton. Move it to a marked target zone on the mat. Release it.

Place adds accurate positioning to Lift. A wrist view must help more with
Place than with Lift, and that difference is itself useful data.

Place uses the Lift scan without a change. Do not change the setup. Only
mark the target zone.

### A.2 Stack

Grasp the carton. Put it on a book or on a rigid platform. Release it.

Stack has the smallest alignment tolerance of the three tasks. Thus it is
the most difficult case for a policy that has no wrist view, and it is the
best test of the render.

**How to use one scan for all three tasks.** Put the book on the mat at
setup, and do not move it. The book is then in the scan and in every demo.
Lift and Place ignore the book. Stack uses it as the target.

The book must be 3 to 5 cm tall. If it is lower, alignment is not important.
If it is higher, the carton goes behind its edge. The book must be matte and
rigid, with a patterned cover, because the book is part of the splat.

**Track the carton only. Do not track the book.** The book stays in the
splat. There it renders with its true appearance and its true colours,
instead of the one average colour that a drawn box gets. Claude Code
assessed this method as more correct than tracking both objects. It is not a
compromise.

There are two rules:

1. The carton and the book must look different. If they look the same,
   Grounding DINO selects by argmax between two almost equal scores, and the
   selection changes in the middle of the clip. This is the same silent
   damage as the hand-identity switch. Name the carton in the prompt.
2. Nothing tracks the pose of the book. To measure stack accuracy in mm, we
   must first recover the position of the book from the reconstruction.
   Nothing does this today.

### A.3 Rotate Box — EXCLUDED

Do not record a rotation task with the current renderer.

The renderer draws the object in one average colour, with no texture. A cube
that turns 90 degrees about its vertical axis has the same silhouette. Thus
a rendered wrist view of a rotation shows no change, and a policy that
trains on it cannot see orientation.

A rotation task becomes possible when the object has real texture. This
needs Sam3D. Until then, use tasks in which the POSITION changes.

### A.4 Cans and cylinders

The RoboMimic Can task uses an aluminium can, and this is usual in the
field. It does not transfer to this pipeline, for two reasons. The reasons
are ours, not theirs.

RoboMimic operates in simulation, or sends a real camera image directly to
the policy. Nothing reconstructs the can. We photograph the scene, build a
3D model, then render a new view.

1. **Specular metal.** The highlights move with the camera. Thus they are
   not static features. This breaks feature matching and splat training.
2. **A can is a cylinder.** The renderer draws boxes. It would draw the can
   as a rectangular block.

Both reasons are testable. They are not certain. Measure them before you
exclude cans permanently. Examine this again when the renderer draws
primitives other than boxes.

### A.5 Tracking of more than one object

Stage 3 assumes exactly one object. Three independent points enforce this:
`objects.py:76` takes the argmax over the boxes from Grounding DINO,
`objects.py:161` keeps only the largest connected blob, and `clean_mask`
rejects a mask above `max_mask_fraction`. After these stages,
`object_pose.npy` has the shape (frames, 4, 4), with no object index.

Claude Code estimated approximately one day of work to support two objects.
Detection and masks are approximately two hours. The difficult work is
attribution in `carry.py`: when the hand closes, which object did it lift?
Fingertip proximity gives the answer in most cases. It fails in the case
that makes stacking interesting: the moving object is directly above the
base object, and both are within centimetres of the fingertips.

This work is not necessary while the single-object method in A.2 operates.
