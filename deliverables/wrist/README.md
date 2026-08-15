# Wrist-view renders · session 3, 15 August

Five videos, one per demo clip. Each frame is **source egocentric camera on the
left, synthesised virtual wrist camera on the right**.

| File | Frames | Scene coverage | Hand tracked |
|---|---|---|---|
| `demo_0.mp4` | 205 | 72% | 60% |
| `demo_1.mp4` | 173 | 57% | 92% |
| `demo_2.mp4` | 139 | 59% | 53% |
| `demo_3.mp4` | 180 | 51% | 80% |
| `demo_4.mp4` | 180 | 68% | 60% |

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
- **Scene geometry.** 2.8 M points, built by anchoring monocular depth to the
  COLMAP points each scan frame observes. Median depth correlation 0.91.

## What is approximated

- **The scene is a point cloud, not a Gaussian splat.** The splat needs a long
  CUDA training run. The point cloud is metrically correct but visibly
  speckled, and it leaves holes where the scan never saw a surface. Every
  black region in the right panel is missing data, not black geometry.
- **The gripper is a parametric parallel jaw**, not a specific robot. Its
  opening comes from the measured thumb-to-index distance, clamped to a URDF.
- **The wrist camera offset is a guess**: 4 cm below and 8 cm behind the grasp
  point, pitched down 25 degrees, 90 degree field of view. It has not been
  matched to any real robot.

## What is wrong, plainly

**1. The held object is missing from the render.** The point cloud comes from
the scan, where the mug sat on the desk. In the demos the mug is picked up and
moved, so the wrist view shows the desk *without* the object the gripper is
holding. This is the most visible defect and it is not a rendering bug: Stage 3
tracks object pose to place it, and object tracking returned 0% on this run
because the splat was disabled, so there is nothing to composite.

**2. The wrist camera looks across the desk, not down at the grasp.** The
end-effector frame takes its approach direction from the palm through the
grasp point. Wrapping a hand around a mug makes that direction roughly
horizontal, so a camera looking along it sees the keyboard and monitor rather
than the work surface. This is geometrically faithful to the hand pose, and it
is probably not what a robot wrist camera would be mounted to see. Expect to
revisit the offset against a real gripper.

**3. Coverage is 51% to 72%.** The rest is holes in the scan. Coverage is
worst when the camera is close to the surface or looking at a grazing angle.

**4. Hand tracking drops out.** 53% to 92% by clip. When the hand is lost the
end-effector pose is held at its last value, which reads as a frozen camera.
The caption says `hand LOST, pose held` on those frames.

**5. The gripper rarely closes.** Grasp detection needs the object position to
confirm contact, and object tracking returned 0%, so the open/closed signal
falls back to finger distance alone.

---

## How this was produced

```bash
python -m tools.render_wrist_videos --run runs/real03 --out deliverables/wrist --splat-px 2
```

Scene cloud: `runs/real03/01_scene/dense.ply`, built by
`src/wristview/backends/dense_cloud.py`.

Per-frame numbers: `render_report.json` beside these videos.

## What would fix the top two defects

The missing object needs Stage 3 object tracking, which needs the splat for
metric depth. That is the CUDA job in `tools/cuda_job/`. The camera direction
needs a decision about where the wrist camera actually sits on the target
robot, which is a specification question rather than a code one.
