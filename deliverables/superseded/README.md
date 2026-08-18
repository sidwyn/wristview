# Superseded. Do not show these.

Everything under this directory was produced from a hand pose that is wrong,
and it is kept only so the corrected version can be compared against it.

## What is wrong with them

The 3D hand position was computed by rescaling WiLoR's translation onto our
camera and scaling all three components by `fx / scaled_focal_length`. Uniform
scaling leaves `X/Z` unchanged while the focal length changes, and the
projected offset from the optical axis is `f * X/Z`, so the hand collapsed
toward the centre of the frame by that same factor.

Measured on real06 demo_0, 17 August:

| quantity | value |
|---|---|
| wrist motion across the image, from the 2D detector | 2221 px |
| wrist motion implied by the 3D position | 94 px |
| ratio | **about 25x** |
| median reprojection error of the 3D wrist | **614 px** |
| bounding box of the wrist in world coordinates | 2.6 x 1.8 x 16.9 cm |

The last row is the one to read. The operator reaches across a desk and picks
up a tape measure. The recorded hand barely moves sideways at all.

The wrist camera pose is derived from the hand pose, so every frame in these
videos was rendered from a camera in the wrong place. They look plausible
because the splat, the scene reconstruction and the camera localisation are all
sound, and because the 2D overlays were always correct. Only the 3D lift was
broken, and nothing in the pipeline compared the two.

## What is still true in them

The scene reconstruction, the camera localisation, the 2,734,543-Gaussian splat
and the object poses solved on the desk plane are unaffected. Those come from
masks and from structure from motion, not from the hand.

## What replaced them

`deliverables/wrist-real06/`, rendered after the fix in
`src/wristview/backends/hands.py`, which preserves the pixel the hand sits on
and rescales only the depth. Reprojection on the same frames went from a median
of 614 px to 13 to 16 px.

Regression test: `tests/test_hands_wilor.py::
test_the_3d_hand_reprojects_onto_its_own_2d_detection`.
