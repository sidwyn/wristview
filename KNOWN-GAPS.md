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


---

## Splat quality: the lever is Gaussians, not views

**Observed** on real26/bm, 2026-08-26.

| | real26/a | real26/bm |
|---|---|---|
| registered training views | 298 | 586 |
| Gaussians | 3,912,607 | 4,998,950 |
| **Gaussians per view** | **13,130** | **8,531** |
| gate PSNR, gsplat | 30.57 dB (12 views) | 27.82 dB (586 views) |

The merged scan scores 2.75 dB lower. Two explanations fit and this session
cannot separate them:

1. The low pass is blurry, and soft training views smear Gaussians that sharp
   views also depend on.
2. The same Gaussian budget is spread across twice the views, so each view gets
   fewer Gaussians to reproduce it.

Separating them needs a control: the same scan retrained without the low-pass
frames. That was priced at about $0.50 and 40 minutes and deliberately not
run, because the 22 views below 12 cm are the only ones that see the desk from
where the wrist camera actually is, and cutting them costs 30 per cent of
usable render frames to chase a couple of dB in bands already at 28 dB and
already passing the gate.

What the numbers do support: **the per-view budget fell by 35 per cent while
the view count doubled.** gsplat's `DefaultStrategy` grows the cloud from
gradient signal and prunes on opacity; it has no per-view target. So a longer
scan does not automatically buy a proportionally larger splat.

**If splat quality ever has to go up, raise the Gaussian count, do not cut
views.** Cutting views trades coverage, which is the thing the render is short
of, for detail, which is already adequate. Raising the count costs GPU memory
and nothing else. Peak use here was 15.0 GiB of 23.5 at 5.0 M Gaussians, so
there is room for roughly 8 M on the same card before the cache-free trainer
runs out.

Levers, in order of preference: a longer `refine_stop_iter` so densification
runs further, a lower `grow_grad2d` so more Gaussians qualify to split, or a
larger card. Not fewer views.
