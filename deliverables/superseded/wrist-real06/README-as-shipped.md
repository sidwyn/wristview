# Session 6 wrist views

Four clips, shot 17 August. Egocentric source on the left, a robot wrist camera
on the right, rendered from a 2,734,543-Gaussian splat of the same desk.

## Read this before watching

**demo_0, demo_1 and demo_2 are single-arm views of two-handed tasks.** The
operator used both hands. Only the right hand is tracked, retargeted and
rendered. The left hand is doing work that the wrist view does not show and the
trajectory does not contain. Nothing in the frame marks its absence, so it is
stated here instead.

Only **demo_3** is a genuinely single-handed clip, and it is the only one where
the wrist view accounts for the whole task.

**No object is composited in any clip.** The manipulated object appears only
where the scan left it, inside the static scene splat, which the caption states
as `object STATIC at scan position`. Each view is correct up to the moment the
operator picks the object up and wrong for the whole of the carry after that.

## The clips

| clip | source | task | tracked hand | tracked object | scene coverage |
|---|---|---|---|---|---|
| demo_0 | C032 | parallel pick and place, one object per hand | right | tape measure | 93% |
| demo_1 | C033 | left hand puts the cube in the bin, right hand steadies the bin | right | the bin | 81% |
| demo_2 | C034 | cube handed from left hand to right, then placed | right | cube | 85% |
| demo_3 | C035 | single-handed pick and place | right | tape measure | 95% |

demo_1 deserves a second note. The right hand never manipulates anything; it
holds the bin still while the left hand does the task. Its trajectory is a
valid recording of a hand holding a bin, and it is not a manipulation.

## Quality, plainly

demo_3 is the one that holds up. It has the highest scene coverage at 95%, it
is the only whole task, and the surfaces read as surfaces.

demo_0, demo_1 and demo_2 are visibly rougher. They carry cloud-like floaters,
Gaussians that only resolve correctly from the angles the scan actually flew,
and demo_0 opens on a near-white wash where the wrist camera sits close to a
surface the scan only ever saw from a distance. They are watchable. They are
not the frames to lead with.

Frames where the right hand was not detected, or where two detections both
claimed to be the right hand, are captioned `hand LOST, pose held`. The pose is
held at its last value across those frames rather than interpolated. There are
at most 7 such frames in any clip.

## Files

- `demo_*.mp4` — captioned. Every claim the render makes is on screen, including
  what is not measured. Read numbers off these.
- `clean/demo_*.mp4` — the same pairs with no overlay, for showing the result.
- `hero/` — short windows cut to end about a second after the object starts
  moving, past which the static object becomes visibly wrong.

`hero/HERO_demo_3_reach_to_grasp.mp4` is the one to use. The window picker
scored demo_1 highest, but it ranks on object displacement, and demo_1's
"object" is the bin the right hand merely steadies, so its displacement is a
nudge rather than a grasp. `hero/HERO_toolpick_demo_1.mp4` is kept so the
disagreement is visible rather than hidden.

All hero windows came out at 1.8 to 2.15 s, short of the 6 to 10 s the picker
aims for, because in every clip the object starts moving within the first few
seconds and the window has to close shortly after.

## How these were made

Stages 1 to 4 ran on this machine. The splat was trained on a rented RTX 4090
and the wrist views were rendered there, then pulled back and hash-verified
before use. The side-by-side composition is local.

Hand selection is by handedness label rather than by detection size. Selecting
the largest box crosses between hands the moment both are in shot, which is
what rejected demo_0, demo_1 and demo_2 outright on the previous run, at 27, 7
and 12 crossings. Note that the label-switch guard reads zero under label
selection **by construction** and proves nothing there; the measure that does
carry information is the largest single-frame wrist displacement, against the
40 to 60 cm the operator's two hands are apart:

| clip | largest single-frame wrist move | p99 |
|---|---|---|
| demo_0 | 3.5 cm | 2.67 cm |
| demo_1 | 3.15 cm | 2.62 cm |
| demo_2 | 18.05 cm | 2.92 cm |
| demo_3 | 14.12 cm | 9.23 cm |

demo_2's 18.05 cm is a single frame at the hand's entry, and it is 18 cm along
the viewing axis with 7 mm of lateral motion, so it is the depth estimate
settling rather than a crossing between hands. No clip shows a crossing.
