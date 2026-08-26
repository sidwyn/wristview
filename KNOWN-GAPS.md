# Known gaps

Not defects. Nothing here is broken, mis-measured, or unread. These are
differences between what the pipeline produces and what the world contains,
recorded so they are never a surprise in a result.

A defect gets fixed. A gap gets stated, and closed when the thing that closes
it arrives.

---

## The rendered object carries one averaged colour

**What the render shows.** A box at the measured size, filled with a single
colour: the median of the object's own mask pixels. On real26 that is
`(0.357, 0.518, 0.447)`, a greenish grey, over 52,054 sampled pixels. Shading
is one headlight term, so faces differ by angle to the camera and by nothing
else.

**What the world contains.** The cube has six distinct bright faces. It is a
different object to look at.

**Where it matters.** This is a domain difference between the arms:

- **Arm B** trains on rendered wrist views and sees a flat greenish box.
- **Arm C** trains on real wrist-camera footage and sees six coloured faces.

Any comparison between B and C carries this difference. If B underperforms C on
anything that could depend on object appearance, colour identification, face or
pose discrimination, or grasp selection that keys on which face is up, this gap
is a candidate explanation and has to be ruled out before a conclusion is
drawn. It cannot be ruled out by looking at the trajectories, because the
trajectories are the same.

The median is the right summary of one colour. The problem is not the median,
it is that one colour is the whole model.

**What closes it.** A Sam3D reconstruction of the cube, which carries per-face
appearance. Stage 5 asks for vertices, faces and a colour, and does not care
where they came from, so this drops in without touching the render path. That
is Harry's item 2.

**Until then**, say so wherever a rendered wrist view is shown. It is in the
caveats file beside every render.

**First recorded** 2026-08-26, real26.
