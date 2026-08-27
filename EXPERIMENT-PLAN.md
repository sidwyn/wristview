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

This decides the design below. Two of the three arms are single-camera and
deployable. The third is context only.

---

## 3. The task and the object

### 3.1 Today: 1 task only

**LIFT.** Grasp the wooden block. Raise it about 10 cm. Hold 1 second. Put it
back down. Withdraw.

30 demos plus 15 arm-C takes. About 45 episodes.

1 task is the right scope. The experiment asks whether a rendered wrist view
helps a policy learn. A task the policy can actually learn from 30 episodes
is what makes that question answerable. A hard task makes every arm fail
equally and teaches us nothing about wrist views.

Backup tasks are in Appendix A. Do not shoot them until Lift has run end to
end and produced a result.

### 3.2 The object: a wooden block

| Requirement | Why | Wooden block |
|---|---|---|
| Rigid | The pose solver assumes a rigid object. The soft toy cube deforms 21% under grip. | yes |
| Matte | Shiny surfaces move their highlights with the camera. This breaks feature matching and splat training. | yes |
| Box-shaped | The renderer draws a box. A cylinder would render as a block. | yes |
| Textured | Grain gives the mask and the splat something to hold. | yes |
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

### 3.3 The block is ABSENT from the scan

Do not put the block on the mat during the scan.

Its colour is sampled from the demo frames, not the scan, so the scan does
not need it. Leaving it out removes the static ghost that has appeared in
every render so far: the splat's frozen photograph of the object sitting
where it was scanned, while the tracked box moves away from it.

Scan the mat, the marker and any static props. Then place the block and
shoot.

---

## 4. Filming

### 4.1 Set up once

- Mat taped flat at all 4 corners.
- Marker id 0, 100 mm, where the hand never passes over it.
- Mark 5 or 6 START spots on the right half of the mat. Mark 3 or 4 PLACE
  spots on the left, clear of the marker.
- Cube, 70 mm, at start spot A.
- Midday light. Write down the shade position.
- Head camera: wide lens, stabilisation off, 1080p60, shutter 1/120, ISO,
  white balance and focus locked.
- Marker id 1 is OFF the wrist camera. It stays off.

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

### 4.3 Verify before you shoot 30

1. Shoot demo 1.
2. Run Stage 0 and pre-flight on the scan plus demo 1.
3. Do NOT shoot the rest until it passes.

This takes 15 minutes. It is the only thing that stops a whole session from
being wasted.

### 4.4 The 30 demos. No wrist camera. About 10 s each

Reach in. Pinch grasp. Move the cube right to left. Place. Hold 1 second.
Withdraw.

- Hands in frame for the whole clip.
- Pinch, thumb opposed to 2 fingers. Do not wrap the cube.
- Vary the START and PLACE spots. Cycle through them.
- Vary approach angle and speed a little. Never rush the grasp.
- Write down each one: "ep7: start C, place 2".
- Reset the cube between takes. Nothing else moves.

**No taps.** Taps exist to align 2 cameras. There is only 1 camera here.

WARPED shot 30 demonstrations per task in 3 to 5 minutes. Short takes are
correct.

### 4.5 The 15 arm-C takes. Wrist camera ON. Taps at both ends

1. Start the head camera first, then the wrist camera.
2. Tap the bare table beside the mat. Pause 1 beat.
3. The same task, same speed, same grasp.
4. Pause 1 beat. The same tap again.
5. Stop the wrist camera first, then the head camera.

15, not 5. If arm C trains on 5 episodes and arm B on 30, any difference
between them could be dataset size instead of view type. That is the exact
confound this experiment exists to avoid.

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

### 5.3 The 4 training runs

All 4 use the same episodes and the same architecture. Only the image input
changes.

| Run | Image input | From | Deployable |
|---|---|---|---|
| **NULL** | none, or shuffled | — | no. Sanity floor. |
| **A** | ego view | the 30 | no. Context only. |
| **B** | **rendered** wrist view | the 30 | **yes. This is the product.** |
| **C** | **real** wrist view | the 15 | yes. The upper bound. |

A and B come from the SAME footage. We train twice and change the input.
No extra filming.

### 5.4 Rules that decide if the numbers mean anything

**Hold out whole EPISODES, not frames.** Adjacent frames are nearly
identical. A frame-level split leaks the answer into training and every run
scores well.

**Train each run at least 3 times with different random seeds.** With 30
episodes, run-to-run variation can be larger than the effect. Report the
spread, not only the mean. Without this we cannot tell a result from a
coincidence.

**Compare at equal episode count.** For the headline B versus C number,
subsample B to 15 episodes. Report B at 30 separately as a bonus.

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

| Run | Action error, mean of 3 seeds | Spread |
|---|---|---|
| NULL | 9.10 mm | ±0.30 |
| A | 4.20 mm | ±0.40 |
| B | 2.80 mm | ±0.25 |
| C | 2.50 mm | ±0.20 |

Nothing moves, because there is no robot. The policy only produces numbers.

**Also make 1 chart:** plot the predicted path against the actual path for
1 held-out episode. Two lines that should sit on top of each other. Where
they separate is where the policy is confused. Much easier to read than the
table.

---

## 7. Reading the result

### 7.1 A vs B — does the render help?

Compare A's error against B's.

| Outcome | What it means |
|---|---|
| B error < A error | The rendered wrist view adds information the ego view does not carry. This is the product claim. |
| B error = A error | The render adds nothing. It may be wrong, or the task may be too easy to need a wrist view. |
| B error > A error | The render is actively misleading. Look for a defect before believing it. |

Using the example numbers: 4.20 → 2.80 mm. The render helps.

**First check NULL.** If A and B are both close to NULL, nothing learned
anything and the comparison is meaningless.

### 7.2 B vs C — how close does the render get?

Compare B's error against C's, as a fraction of the whole gap.

```
gap that a REAL wrist view buys   = A error − C error
gap that MY RENDER buys           = A error − B error

fraction closed = (A − B) / (A − C)
```

With the example numbers:

```
(4.20 − 2.80) / (4.20 − 2.50) = 1.40 / 1.70 = 82%
```

**The render closes 82% of the gap between no wrist view and a real one.**

That is the headline sentence. WristWorld (arXiv 2510.07313) reports 42.4%
for a different method. Anything near or above that is a real result.

| Outcome | What it means |
|---|---|
| B close to C | The render carries almost the whole benefit of a real camera. The strongest possible result. |
| B between A and C | The render helps but is not equal to real. Report the fraction honestly. |
| B close to A | The render adds little. Find out why before shooting more. |

### 7.3 The free qualitative check

Play the rendered wrist view beside the real one from the same arm-C
episode, aligned by the taps.

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

**Cost of 45 episodes.** 85 minutes of fixed work plus 22.4 minutes per
episode is about 18 hours of compute. That is 2 overnight runs.

Ask Claude Code to cut Stage 2 while it runs. It is 69% of the marginal cost
and it named 3 ways: halve `val_top_k`, batch the pairs, or move the stage
to the GPU. Anything it saves applies to every future session.

1. Shoot: scan with NO block, 1 demo, verify, 30 demos, 15 arm-C takes.
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
