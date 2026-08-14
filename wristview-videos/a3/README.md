# A3 · HaMeR frames

Ten stills from the three cup-to-coaster demo clips, for the HaMeR demo on Hugging Face.
Milestone A3 of `wristview/RUNBOOK.md`.

Source clips are in `atlas/wristview-videos/`.
All frames are full resolution, 2160x1214, rotation applied.
Timestamps are in the filename.

## What to look for

HaMeR tracks an open hand well.
It struggles when the fingers wrap an object and occlude themselves.
So the pre-grasp frames are the control, and the contact frames are the test.

Judge each output on one question: does the mesh keep the fingers on the glass, or does it flatten the hand open when the fingers disappear behind the glass?

## Upload order

Go clip by clip, in this order. The contrast within a clip is the signal.

| # | File | Moment | Role |
|---|---|---|---|
| 1 | `C005_1-pregrasp_t1.00s.jpg` | Hand open, approaching, not touching | Control |
| 2 | `C005_2-contact_t1.50s.jpg` | Fingers closing on the glass | Test |
| 3 | `C005_2b-contact-alt-tightwrap_t2.20s.jpg` | Wrap complete, fingers occluded | Test, harder |
| 4 | `C005_3-transport_t2.75s.jpg` | Glass mid-travel, hand wrapped | Test |
| 5 | `C006_1-pregrasp_t0.80s.jpg` | Hand open, spread, away from glass | Control |
| 6 | `C006_2-contact_t1.20s.jpg` | Wrap complete, fingers behind glass | Test |
| 7 | `C006_3-transport_t2.45s.jpg` | Glass mid-travel, hand wrapped | Test |
| 8 | `C007_2-contact_t4.50s.jpg` | Wrapped grasp, static, sharp | Test, cleanest |
| 9 | `C007_3-transport_t2.50s.jpg` | Glass mid-travel, hand wrapped | Test |
| 10 | `C007_4-release-openhand_t5.60s.jpg` | Hand open, just off the glass | Control |

Screenshot the outputs.
Frame 8 is the cleanest wrapped grasp in the set, so it is the best single screenshot if you only keep one.
Frames 1 and 8 side by side are the best pair, because they are the control and the test.

## Two problems in the source clips

**C007 has no pre-grasp.**
The clip starts with the hand already gripping the glass, and the glass is clipped by the left frame edge.
So C007 gets a release frame at 5.60s instead, where the hand opens just off the glass.
That is an open-hand control, but it is not a reach.

**C006 clips the glass at the left edge** during the pre-grasp and early contact frames.
The hand is fully visible, so hand pose is unaffected.
Object pose in Stage 3 would be affected.

Both are recording framing issues, not processing issues.
Start the demo with the object fully inside the frame and the hand out of frame, then reach in.

## Event timeline per clip

Derived from contact sheets at 2 fps, then refined at 10 fps around each event.

| Clip | Length | Reach starts | Contact | Transport | Release |
|---|---|---|---|---|---|
| C005 | 7.85 s | 0.9 s | 1.4 to 2.4 s | 2.4 to 3.1 s | ~6.0 s |
| C006 | 9.57 s | 0.5 s | 1.0 to 2.0 s | 2.0 to 2.9 s | ~6.0 s |
| C007 | 6.75 s | before 0.0 s | already held | 1.9 to 3.0 s | 5.6 s |

C005 has a second hand in frame from about 1.2 to 2.0 s, adjusting the coaster.
Expect HaMeR to detect two hands in those frames.
