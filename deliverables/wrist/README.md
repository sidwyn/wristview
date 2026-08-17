# Wrist-view renders · session 3, 15 August

Five videos, one per demo clip. Each frame is **source egocentric camera on the
left, synthesised virtual wrist camera on the right**.

| File | Frames | Scene coverage | Hand tracked |
|---|---|---|---|
| `demo_0.mp4` | 205 | 60% | 58% |
| `demo_1.mp4` | 173 | 52% | 91% |
| `demo_2.mp4` | 139 | 46% | 51% |
| `demo_3.mp4` | 180 | 44% | 78% |
| `demo_4.mp4` | 180 | 66% | 58% |

---

## What is measured

Everything below is recovered from the footage, in metres, and independently
checked.

- **Camera pose per demo frame.** Registered against the scan reconstruction,
  205/205, 173/173, 139/139, 180/180 and 180/180 frames, 0% tracking loss.
- **Metric scale.** 0.0587 m per reconstruction unit, from a 100 mm ArUco
  marker triangulated across 114 scan frames.
- **Independent pose check.** An ArUco marker solved from four coplanar
  corners agrees with the reconstruction's pose to **1.2 to 2.7 cm and 1.2 to
  2.5 degrees**, median over 131 to 199 frames per clip. Different
  mathematics, same answer.
- **Camera trajectories.** 0.12 to 0.42 m of travel over a 0.69 m desk, at a
  median 0.02 m/s. Physically consistent with a hand-held camera.
- **Hand pose.** WiLoR, 21 MANO joints, lifted to metric world coordinates.
  Every frame is checked against a human angular limit of 900 deg/s, measured
  on the palm rather than the fingers. 2 to 4 frames per clip fail and are
  filled from their neighbours. `ee_trajectory.npz` carries `hand_measured`
  and `hand_filled` per frame, so a filled frame is never presented as a
  measured one.
- **Gripper roll.** Resolved against world up, taken from the ArUco marker
  plane. Before this, four of five clips rendered mostly upside down, at a
  median 73 to 161 degrees from world up. All five now sit at 16 to 35
  degrees.
- **Scene geometry.** 2.8 M points, built by anchoring monocular depth to the
  COLMAP points each scan frame observes. Median depth correlation 0.91.

## What is approximated

- **The scene is a point cloud, not a Gaussian splat.** The splat needs a long
  CUDA training run. The point cloud is metrically correct but visibly
  speckled, and it leaves holes where the scan never saw a surface. Every
  black region in the right panel is missing data, not black geometry.
- **The gripper is a parametric parallel jaw**, not a specific robot. Its
  opening comes from the measured thumb-to-index distance, clamped to a URDF.
- **The wrist camera offset is a guess**: a fixed mount 10 cm behind and 10 cm
  above the fingertips, aimed at a point 14 cm ahead of them, 90 degree field
  of view. It frames the fingertips 77% down the image, which is where DROID
  and Open X wrist views put them. The geometry is exact; the mount itself has
  not been matched to any real robot.
- **The held object is carried, not tracked.** The mug is segmented once from
  the scan and moved rigidly with the gripper after the grasp. Before the
  grasp it sits where the scan found it. The caption names which of the two is
  showing on every frame, `object STATIC at scan position` or `object CARRIED`.

## What is wrong, plainly

**1. The hand leaves frame before two clips end.** demo_2 holds its last
measured pose for 59 frames, 42% of the clip, and demo_3 for 33. The gripper
freezes and the camera stares. This is a capture fault, not a code one, and
`CAPTURE-SOP.md` now requires the hand to stay in shot until two seconds after
the release.

**2. Two clips keep one roll flip each.** demo_0 and demo_1. The wrist is
steady through them while the fingers reconfigure, so the thumb-index axis
that sets the gripper roll swings on its own. Removing these means taking the
roll from the palm instead, which is a change to the pose mapping rather than
a bug fix. Every other clip is at zero.

**3. Coverage is 44% to 66%.** The rest is holes in the scan. Coverage is
worst when the camera is close to the surface or looking at a grazing angle.
Every black region is missing data, not black geometry.

**4. Hand tracking drops out.** 51% to 91% by clip. Separately, 2 to 4 frames
per clip are rejected by the plausibility gate and filled from their
neighbours. Both are recorded per frame, never silently.

**5. The gripper rarely closes.** Grasp detection prefers to confirm contact
against a tracked object position. Object tracking returned 0% on this run, so
the open/closed signal falls back to finger distance alone, which fires on any
pinch in mid-air. Recorded in the status as `signal: fingers_only`.

---

## How this was produced

```bash
python -m tools.render_wrist_videos --run runs/real03 --out deliverables/wrist --splat-px 2
```

Scene cloud: `runs/real03/01_scene/dense.ply`, built by
`src/wristview/backends/dense_cloud.py`.

Per-frame numbers: `render_report.json` beside these videos.

## What is still open

The carried object is rigid, so it cannot show the mug rotating in the grasp.
Real object tracking needs the splat for metric depth, which is the CUDA job in
`tools/cuda_job/`. The wrist mount is a plausible guess and needs a decision
about where the camera actually sits on the target robot, which is a
specification question rather than a code one.
