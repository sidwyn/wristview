# The object pose fix, specified for real05

Written the night of 16 August so it can be run tomorrow without re-deriving
it. Nothing here has been implemented yet, deliberately.

## What is wrong

Stage 3 tracks the object well in 2D and places it wrongly in 3D.

Measured on real04, three clips:

| Check | Measured | Expected |
|---|---|---|
| Cube centre above the desk, at rest | 120 to 127 mm | **38.1 mm** (half of 76.2) |
| Fingertip to object centre, while the object is demonstrably moving | 113 to 145 mm laterally | roughly 40 to 70 mm |
| Standard deviation of that offset | 19 to 60 mm | small, if it were a fixed bias |
| ICP residual | 1.1 to 1.7 mm | fine |
| 2D mask coverage | 100 per cent of frames | fine |

The object floats about 85 mm above the desk and sits about 120 mm to one side
of the hand holding it. The offset is not constant, so it is not a single
mis-registration that could be subtracted.

Every reported number is true and none of them constrain the world position.
The ICP residual measures how well the model fits **its own back-projected
points**, not whether those points are in the right place. That is why this
survived: it looks like a healthy tracker.

**Consequence.** Contact-based grasp detection cannot work. Two things that
disagree about where they are by 120 mm never touch. demo_3 reported zero
contact frames across an obviously clean pick and place.

## Why it happens

The object's 3D pose comes from SAM 2 masks back-projected through **monocular
depth**, anchored per frame. Monocular depth is up to an unknown affine
transform per frame, and the anchoring fits that transform against sparse
COLMAP observations. Where the object is small in frame and the observations
near it are few, the fit is dominated by background geometry and the object
lands at the wrong depth.

The hand does not share the error because it is lifted differently.

## The fix

Anchor the object to the **desk plane**, which is the one surface in this
scene that is large, flat, textured and reconstructed reliably.

1. **Fit the desk plane once, in Stage 1**, from the sparse reconstruction.
   Take the dominant horizontal plane by RANSAC, with `world_up` from the
   marker as the expected normal. Store it in `01_scene/scene.json` as a point
   and a normal, in metres. It never moves, so it is fitted once per session,
   not per frame.

2. **Constrain the object's resting pose.** Before the grasp the object sits on
   the desk, so its lowest points lie in that plane. Solve the object pose with
   that as a constraint rather than taking whatever depth returns: the model's
   support face is coplanar with the desk. This fixes the 85 mm float directly
   and gives a correct object scale as a by-product.

3. **Re-anchor depth against the object's own observations.** When fitting the
   per-frame affine depth transform, weight the sparse observations that fall
   **inside the object mask** far more heavily than the background ones. The
   current fit is global, so a small object contributes almost nothing to it.

4. **During transport, prefer the hand.** Once contact is established the
   object moves with the hand by definition. Rather than trusting per-frame
   monocular depth through the occluded phase, carry the object rigidly from
   its pose at contact and use the tracked pose only to detect release.

## How to verify it, before trusting any render

Three checks, all cheap, all with a number known in advance:

| Check | Passes if |
|---|---|
| Object centre above the desk, at rest | within 5 mm of half the measured object height |
| Fingertip to object centre, while the object is moving | under 70 mm, with a standard deviation under 20 mm |
| Reconstructed object edge against the ruler | within 5 per cent, as the marker cross-check already achieves |

The first is the one that would have caught this. It costs one line and it has
a right answer known from a ruler.

## What to add to the QC gates

`OBJECT_REST_HEIGHT_TOLERANCE_M = 0.005`, checked in Stage 3 whenever an object
is tracked and a desk plane exists. Fail loudly. This belongs with the other
gates in `qc.py`, and it is the gate whose absence cost real04 its grasp
detection.
