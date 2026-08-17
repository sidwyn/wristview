# Wrist-view renders from the Gaussian splat · session 3

Five videos, one per demo clip. Each frame is **source egocentric camera on the
left, synthesised virtual wrist camera on the right**.

Rendered from a 1,530,082 Gaussian splat trained on an RTX 4090, replacing the
2.8 M point cloud used in `../wrist/`. Both sets use the same trajectories, the
same wrist mount and the same captions, so they can be compared frame for frame.

| File | Frames | Splat coverage | Object carried | Near-black frames |
|---|---|---|---|---|
| `demo_0.mp4` | 205 | 98% | 115 | 9 |
| `demo_1.mp4` | 173 | 98% | 85 | 2 |
| `demo_2.mp4` | 139 | 95% | 97 | 11 |
| `demo_3.mp4` | 180 | 94% | 108 | 43 |
| `demo_4.mp4` | 180 | 99% | 117 | 0 |

---

## What is measured

- **Camera pose per demo frame.** Registered against the scan reconstruction,
  every frame of every clip, 0% tracking loss.
- **Metric scale.** 0.0587 m per reconstruction unit, from a 100 mm ArUco
  marker triangulated across 114 scan frames.
- **Independent pose check.** A marker solved from four coplanar corners agrees
  with the reconstruction to 1.2 to 2.7 cm and 1.2 to 2.5 degrees. Different
  mathematics, same answer.
- **Hand pose.** WiLoR, 21 MANO joints, lifted to metric world coordinates.
  Every frame is checked against a 900 deg/s human wrist limit, measured on the
  palm rather than the fingers. 2 to 4 frames per clip fail and are filled from
  their neighbours, and the caption says so on those frames.
- **Gripper roll.** Resolved against world up, taken from the ArUco marker
  plane rather than a convention. Before this fix four of five clips rendered
  mostly upside down.
- **Scene.** 1.53 M Gaussians, mean PSNR **35.21 dB** over all 193 training
  views, measured by re-rendering every view rather than reading the training
  log.

## What is approximated

- **The wrist mount is a guess.** Fixed, 10 cm behind and 10 cm above the
  fingertips, aimed 14 cm ahead, 90 degree field of view. It frames the
  fingertips 77% down the image, matching DROID and Open X. The geometry is
  exact; the mount has not been matched to a real robot.
- **The gripper is a parametric parallel jaw**, not a specific robot. Its
  opening is the measured thumb-to-index distance clamped to a URDF.
- **The held object is carried, not tracked.** The mug's own Gaussians are
  selected from the splat and moved rigidly with the gripper from the grasp
  onward. That is exact only while the grasp is firm and the mug does not
  rotate in the hand. Every frame says which of the two applies.
- **Rendered with view-dependent colour switched off** (SH degree 0 instead of
  the trained 3). The wrist camera sits 10 cm from surfaces that were all
  scanned from 0.5 m or further, so the view-dependent terms are evaluated far
  outside the angles they were fitted on and produce iridescent smears. Degree
  0 trades specular highlights for legible geometry. Side by side, this was the
  single largest visual improvement.

## What is wrong, plainly

**0. Read the coverage column with care.** It counts pixels where the splat
returned a finite depth, which a very dim Gaussian tail satisfies. demo_3 reads
94% coverage on frames that are visibly almost black. Coverage says the camera
is inside the modelled volume; it does not say the frame is useful. The
near-black count in the same table is the honest companion number.

**1. demo_3 goes near-black for its last 42 frames, and demo_2 for 11.** The
hand leaves the frame before the clip ends, the trajectory holds its last
measured pose, and the camera ends up staring into space the scan never
covered. 79% of demo_3's dark frames and 73% of demo_2's fall on frames where
the hand was never detected. This is a capture fault. `CAPTURE-SOP.md` now
requires the hand to stay in shot until two seconds after the release.

**2. Everything is soft at close range.** The scan was shot from 0.5 m and up;
the wrist camera renders from 0.10 m. That is extrapolation, not
interpolation, and no amount of training fixes it. The fix is a capture change:
add a very close pass, or accept softness at contact.

**3. demo_0 and demo_1 keep one roll flip each.** The wrist is steady while the
fingers reconfigure, so the thumb-index axis that sets gripper roll swings on
its own. Correcting it means taking roll from the palm, which is a change to
the pose mapping rather than a bug fix.

**4. The gripper proxy is drawn, not tracked.** It shows where a parallel jaw
would be given the measured grasp. It is not a detection of anything.

## Compared with the point-cloud renders in `../wrist/`

| | point cloud | Gaussian splat |
|---|---|---|
| Scene coverage | 44 to 66% | **94 to 99%** |
| Holes in the frame | constant, large | rare |
| Held object | speckled point overlay | continuous surface, from the splat |
| Near-black frames | 0 | 65 of 877, 43 in demo_3 |
| Laplacian variance | 7192 | 1071 |

**Coverage is the real gain.** The point-cloud renders were between a third and
a half empty; the splat fills nearly the whole frame, so the desk, keyboard and
monitor read as surfaces instead of as scattered dots.

**Do not read the Laplacian numbers as the point cloud being sharper.** That
metric responds to high-frequency contrast, and an unfilled point cloud is
nothing but high-frequency contrast: isolated bright points against a black
background score far above a smooth surface. It measures speckle here, not
detail. Judged by eye, the splat resolves the keyboard keys and the monitor
bezel, which the point cloud never does.

**The splat is worse in one respect.** It can go near-black where the point
cloud merely goes sparse, because a Gaussian field falls off to background
outside its fitted volume while stray points persist. Every one of those frames
is in a stretch where the trajectory was already holding a stale pose.

## How this was produced

```bash
# on the laptop
python -m tools.export_for_gsplat --run runs/real03 --out exports/real03

# on the CUDA box
python undistort_export.py /root/job/exports/real03
python train_gsplat.py --data /root/job/exports/real03 --out /workspace/result2 \
    --iterations 30000
python render_splat_wrist.py --sh-render 0 --out-dir /root/wrist_splat

# back on the laptop
python -m tools.compose_splat_videos --run runs/real03 \
    --wrist <pulled videos> --out deliverables/wrist-gsplat
```

Per-clip numbers: `render_report.json` beside these videos.
