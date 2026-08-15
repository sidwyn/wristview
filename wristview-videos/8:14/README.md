# wristview-videos

Source footage. The clips are **not committed**: they are hundreds of
megabytes and git has no business holding them without LFS. `.gitignore`
excludes `*.mov` and `*.mp4` here.

## What belongs here

| File | What it is |
|---|---|
| `overview.mov` | Room scan. Slow orbit of the workspace. No hands, no people. |
| `A001_*_C005.mov` | Demo 1. Glass to coaster. |
| `A001_*_C006.mov` | Demo 2. |
| `A001_*_C007.mov` | Demo 3. |
| `a3/` | Ten stills pulled from the demos, used to check hand pose. Committed, they are small. |

## The current set

Shot on one camera in one session. 2160x1214 after rotation, 60 fps, HEVC.
The files store 1214x2160 portrait with a -90 degree rotation flag, and
ffmpeg applies it on decode, so extracted frames come out landscape.

**No EXIF focal length**, so Stage 1 self-calibrates a SIMPLE_RADIAL camera
rather than freezing an unmeasured guess.

**No ARKit trajectory and no ArUco marker**, so there is no source of metric
scale. Stage 1 refuses to continue by default, because a wrong scale fails
silently and every downstream distance inherits it. See the manual scale
section in `configs/default.yaml`.

## Known problems in this set

Both are recording-framing issues, not processing issues, and both are worth
fixing on the next shoot.

**C007 has no pre-grasp.** The clip opens with the hand already gripping the
glass, and the glass is clipped by the left frame edge. There is no reach to
learn from.

**C006 clips the glass at the left edge** during the pre-grasp and early
contact. Hand pose is unaffected because the hand stays fully visible, but
object pose in Stage 3 is.

**C005 has a second hand in frame** from about 1.2 to 2.0 s, adjusting the
coaster. Stage 3 takes the largest detected hand, which is the manipulating
one.

## What to do differently next time

**The scan must cover the demo's viewpoint.** This is what currently blocks the
pipeline, and it blocks it completely. The demos are tight top-down close-ups
covering about 30 cm of desk. The scan is a wide oblique orbit of the whole
desk. They overlap so little that the best demo-to-scan feature match gives
126 matches where scan-to-scan neighbours give over 700, and Stage 2 cannot
recover a usable camera pose from that.

So: after the wide orbit, **move in and scan the working area slowly from
directly above**, at the same height and framing the demo camera uses, with
the object in place. Twenty extra seconds. Without it nothing downstream of
Stage 2 can run.

Start with the object fully inside the frame and the hand out of frame, then
reach in. Keep the object away from every edge for the whole clip.

Put a printed ArUco marker of known size flat in the scene during the scan, or
capture an ARKit trajectory. Either one gives metric scale for free and
removes the one manual step in the pipeline.

## Event timeline

Derived from contact sheets at 2 fps, refined at 10 fps around each event.

| Clip | Length | Reach starts | Contact | Transport | Release |
|---|---|---|---|---|---|
| C005 | 7.85 s | 0.9 s | 1.4 to 2.4 s | 2.4 to 3.1 s | ~6.0 s |
| C006 | 9.57 s | 0.5 s | 1.0 to 2.0 s | 2.0 to 2.9 s | ~6.0 s |
| C007 | 6.75 s | before 0.0 s | already held | 1.9 to 3.0 s | 5.6 s |
