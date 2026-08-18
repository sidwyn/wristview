# Session 6 wrist views

Four clips, shot 17 August. Egocentric source on the left, a robot wrist camera
on the right, rendered from a 2,734,543-Gaussian splat of the same desk.

Rendered after the hand-lift fix of 17 August. The versions in
`deliverables/superseded/` came from a hand pose with a 25x lateral error and
must not be shown.

## Read this before watching

**demo_0, demo_1 and demo_2 are single-arm views of two-handed tasks.** The
operator used both hands. Only the right hand is tracked, retargeted and
rendered. The left hand is doing work the wrist view does not show and the
trajectory does not contain. Nothing on screen marks its absence.

Only **demo_3** is genuinely single-handed, and it is the only clip where the
wrist view accounts for the whole task.

**No object is composited.** The manipulated object appears only where the scan
left it, inside the static scene splat. Each view is correct up to the moment
the operator picks the object up and wrong for the carry after that. The
caption reads `object CARRIED by gripper` during a grasp, which describes the
grasp state, not a moved object.

## The clips

| clip | source | task | object | coverage | contact frames |
|---|---|---|---|---|---|
| demo_0 | C032 | parallel pick and place, one object per hand | tape measure | 92.4% | 14, from frame 70 |
| demo_1 | C033 | left hand puts the cube in the bin, right hand steadies the bin | the bin | 88.5% | 159, from frame 22 |
| demo_2 | C034 | cube handed from left hand to right, then placed | cube | 86.8% | none, see below |
| demo_3 | C035 | single-handed pick and place | tape measure | 93.9% | 43, from frame 14 |

demo_1's right hand never manipulates anything. It holds the bin still while
the left hand does the task, so its 159 contact frames are a hold, not a grasp.

demo_2 reports no contact and that is the method, not a miss. Contact onset is
decided by the fingers reaching the object's **resting** position, which is the
one position known to be right at that moment. In a handoff the cube arrives in
mid-air and is never collected from rest, so the test cannot fire. Its grasp
state falls back to the older fingertip detector, which is weaker evidence.

## What is measured, and how well

Every 3D quantity is projected back into the frame it came from. Per clip, in
`03_estimate/<clip>/qc.json` and `04_retarget/<clip>/qc.json`:

| clip | hand position | effector pullback | wrist standoff |
|---|---|---|---|
| demo_0 | 12.48 px (p90 17.1) | 0.0203 m | 0.2500 m |
| demo_1 | 10.79 px (p90 12.47) | 0.0203 m | 0.2500 m |
| demo_2 | 12.93 px (p90 15.05) | 0.0202 m | 0.2500 m |
| demo_3 | 8.27 px (p90 16.46) | 0.0208 m | 0.2500 m |

The bound on hand position is 25 px. The same measurement on the superseded
renders would read 614 px.

**The gripper width is a weaker claim than the hand position, and the captions
do not distinguish them.** WiLoR infers the hand for a camera at about 12 m
with a focal length near 37500 px, which is very nearly orthographic; its own
3D and 2D outputs agree there to 0.0 px. Our camera has that hand at about
0.48 m, where it spans 19 per cent of its own depth. Projecting each joint at
its own depth gives 15 to 98 px depending on how the hand is angled, while
projecting them all at the root depth, the assumption WiLoR actually made,
gives 6.5 px on every clip. So the hand's **position** is sound and its
internal **shape** carries a weak-perspective approximation. Gripper width is a
within-hand measurement and inherits it. Treat `gripper proxy N cm` as
indicative and `hand tracked` as measured.

## Quality, plainly

demo_3 remains the cleanest at 93.9 per cent coverage and the only whole task.
All four carry cloud-like floaters, Gaussians that only resolve from the angles
the scan actually flew, and all four go dark where the wrist camera looks
somewhere the scan never covered.

Coverage held up despite the corrected trajectories sweeping a much larger
volume: the hand's path length grew by 29 to 88 per cent once the lateral
collapse was undone, and demo_1 and demo_2 gained coverage rather than losing
it, at 81 to 89 and 85 to 87 per cent.

Frames where the right hand was not detected, or where two detections both
claimed it, are captioned `hand LOST, pose held`. At most 7 per clip.

## Files

- `demo_*.mp4` — captioned. Read numbers off these.
- `clean/demo_*.mp4` — no overlay, for showing the result.
- `hero/HERO.mp4` — demo_0, frames 0 to 98, 4.9 s, cut about a second past the
  point the object starts moving, past which the static object is visibly wrong.

All four hero windows came in at 1.8 to 4.9 s against a 6 to 10 s target,
because the object starts moving early in every clip.

## Provenance

Stages 1 to 4 ran locally. The splat was trained on a rented RTX 4090 and the
wrist views were rendered there, then pulled back and hash-verified before use.
Composition is local. Coverage figures come from the renderer's own
`report.json`, pulled from the pod, not reconstructed from a log.
