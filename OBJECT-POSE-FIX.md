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

---

## Session 6: the plane solve pins the object to the desk

Measured on real06b clip 5, 17 August:

| quantity | value |
|---|---|
| tracked object height above desk | 2.000 cm on **all 185** valid frames |
| spread of that height | **0.0000 mm** |
| operator wrist height above desk | 4.9 cm min, 10.5 cm median, **24.7 cm max** |
| fingertip-to-object gap | 13.8 cm median, 3 frames under 3 cm |
| grasp detection | **0 of 186 frames** |

`object_pose_on_plane` sets `centre_offset = offset + height_m / 2` and
intersects the silhouette ray with that plane. The height above the desk is
therefore a constant of the method, not a measurement. The object cannot rise,
so once the hand lifts it the tracked pose stays behind on the table and the
fingertip gap grows to the lift height. The 13.8 cm median gap **is** the median
lift height.

This is correct before contact, which is why the pre-contact rest check reads
3.27 cm, and wrong for every frame of the carry, which is the part a
manipulation dataset exists to record.

### The proposed gate would not have caught it

The verification table above lists "object centre above the desk, at rest:
within 5 mm of half the measured object height". That gate passes on **every
frame of this defect**, at zero error, because the method computes the quantity
the gate checks. A gate that restates its subject's own construction measures
nothing. The gate that works asks the opposite question: does the tracked height
ever *change*? Here it did not change at all, to 4 decimal places in
millimetres, and that is the signature.

### The fix

`src/wristview/carry.py`, tested in `tests/test_carry.py`.

The silhouette ray is correct throughout: it points at the object whether the
object rests or is carried. Only the distance along it is wrong once the object
lifts. On the desk, the plane fixes that distance. In the hand, the hand fixes
it. Neither branch estimates depth.

- **Resting pose** from frames where the hand is more than 15 cm clear.
- **Contact onset** where the fingertips reach that resting position. Decided
  against the resting pose, which is the one pose known to be right, and that is
  what breaks the circularity between contact and carry.
- **Carry** by the rigid hand-to-object transform recorded at onset, with the
  centre placed at the point on the silhouette ray nearest the hand's prediction.
- **Release** by disagreement, not distance. While the object is held, the ray
  and the hand agree. When it is set down and the hand withdraws, they diverge.

A first draft decided release the same way as onset, by distance to the resting
position. That reports release two frames into every pick-up, because a carried
object is by definition no longer where it was resting. The test suite caught it
rather than the pipeline.
