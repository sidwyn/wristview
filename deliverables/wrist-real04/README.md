# Wrist-view renders, session real04

Rendered from the **point cloud**, not a Gaussian splat. The splat did not run:
three RunPod pods failed SSH authentication, and the one that worked could not
be restarted because its host had no free GPUs. Nothing about the pipeline
blocked it.

```
HERO_demo_0_reach_to_grasp.mp4   the website asset, 6.05 s, no captions
clean/                           full-length, no captions, all three clips
diagnostic/                      same renders with captions and coverage
```

## Which clips are here, and why only three

Five clips were shot. Two were rejected by gates, not by choice:

| clip | outcome |
|---|---|
| demo_0 (C020) | accepted |
| demo_1 (C021) | accepted |
| demo_2 (C022) | **rejected**, hand velocity: 2 consecutive frames above 900 deg/s against a limit of 1 |
| demo_3 (C023) | accepted |
| demo_4 (C024) | **rejected**, Stage 2 tracking loss 12.8 per cent against a 2 per cent limit |

Neither threshold was relaxed to recover a clip.

## The hero clip

`HERO_demo_0_reach_to_grasp.mp4`, demo_0 frames 101 to 221, 6.05 seconds:
the reach, the descent onto the cube, and one second past the grasp.

Chosen on measurement, over the same window in each clip:

| clip | scene coverage | sharpness | near-black frames |
|---|---|---|---|
| **demo_0** | **70%** | **7202** | **0** |
| demo_1 | 60% | 6296 | 0 |
| demo_3 | 49% | 5248 | 29 |

demo_0 wins on all three. It also has the best marker-versus-pose agreement of
any clip in the session, 2.20 cm and 1.73 degrees, so its camera path is the
most trustworthy as well as the best looking.

The window ends one second after the grasp deliberately. After that the hand
carries the cube away while the rendered cube stays where the scan left it, and
the illusion breaks.

## What is measured

- **Camera pose per frame**, registered against the scan, 98.6 to 100 per cent
  of frames across the accepted clips.
- **Metric scale**, verified twice and independently. The ArUco marker gives
  0.10095 m per reconstruction unit. The cube, measured with a ruler at
  76.2 mm, reconstructs at 81.5, 80.3 and 72.7 mm, a **+2.6 per cent**
  disagreement. The marker is the reference; they are not averaged.
- **Gripper roll**, resolved against world up from the marker plane. Every clip
  is now clean: 3 to 0, 2 to 0 and 0 to 0 flips, **zero inverted frames**, and
  the camera up axis sits 11 to 22 degrees from world up.
- **Hand pose**, checked against a 900 deg/s human limit measured on the palm.
  Isolated failures are filled from neighbours and recorded per frame.

## What is approximated, and what is wrong

**The cube renders wherever the scan put it.** It is not composited at its
tracked pose. So it is correct until the moment it is picked up, and wrong for
the whole of the transport, sitting on the desk while the real one is in the
operator's hand. This is deliberate for this render: the tracked object pose is
not trustworthy yet, and a wrong object placed confidently is worse than an
object that visibly stays put. See `OBJECT-POSE-FIX.md`.

**Scene coverage is 50 to 51 per cent over the full clips**, 70 per cent in the
hero window. The black regions are space the scan never modelled, mostly above
the desk. They are missing data, not black geometry.

**Everything is soft, and the standoff is why it is not softer.** The camera
renders at **0.25 m**, chosen to match the scan's measured close-range coverage
rather than for appearance. The scan's closest pass sat at a median 0.264 m
from the surface, with only 2 frames of 364 inside 0.15 m. Rendering at 0.15 m
would be extrapolating past anything the scan saw, and it would come back
blurred however good the reconstruction.

**The point cloud is 366,330 points**, built from 47 scan frames. real03's was
2.8 M. This one was built under heavy memory pressure on the laptop and is
correspondingly thinner; a splat run would raise coverage substantially, as it
did on real03 where it went from 44 to 66 per cent up to 94 to 99 per cent.

**The gripper proxy is drawn, not detected.** It shows where a parallel jaw
would sit given the measured grasp.
