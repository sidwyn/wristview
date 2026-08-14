# wristview · Runbook

**Read this first.** Two paths. They are not exclusive. Run A tonight, and set B going overnight.

| Path | Time | Machine | Produces |
|---|---|---|---|
| **A · Evidence** | 2 hours, hands on | Your Mac | Two numbers and a screenshot. Enough for Actuate. |
| **B · Build** | Overnight, unattended | Your Mac first, GPU only if it blocks | Rendered wrist views, if it gets there |

**A is the one that must finish before the conference.** B may or may not reach a wrist view on the first night. Set it running and let it try. You lose nothing but sleep.

---

# PATH A · Two hours · do this before Actuate

The goal is not a product. The goal is to be the one person at that conference who ran an experiment.

## A1 · Record · 30 minutes

Install **Blackmagic Camera** (free, iOS). The stock app will not let you disable stabilization, and stabilization breaks reconstruction.

| Setting | Value |
|---|---|
| Lens | **Main wide only.** Not ultra-wide. |
| Stabilization | **Off.** The most common failure. |
| Resolution / rate | 1080p, 60 fps |
| Shutter | 1/120 or faster |
| ISO / WB / Focus | **Locked** |

Head strap or cap clip, rear camera forward and slightly down. Check one test clip: **both hands must be visible when you reach for the table.**

**One scan clip.** Slow orbit of the workspace, 45 to 60 seconds. Vary your height. Large overlap between viewpoints. **No hands, no people. It is a scan of the static room.**

**Three demo clips.** Same settings, same session, furniture unmoved. One rigid object on a table — a can, a mug, a block. Say the instruction out loud, do the task at natural speed, 10 to 20 seconds, hands in frame, reset visibly at the end.

Transfer at original quality.

## A2 · Gate 0 · 1 hour, mostly waiting

Runs on your Mac. COLMAP's sparse reconstruction is CPU-only, so no GPU needed.

```bash
brew install colmap ffmpeg

mkdir -p ~/wristview/gate0/frames && cd ~/wristview/gate0
ffmpeg -i /path/to/scan.mov -vf fps=4 -q:v 2 frames/%05d.jpg
ls frames | wc -l          # want 150 to 400

colmap automatic_reconstructor \
  --workspace_path . \
  --image_path frames \
  --data_type video \
  --single_camera 1 \
  --dense 0 \
  --quality medium

ls sparse/                 # want exactly one folder: 0
colmap model_analyzer --path sparse/0
```

**Write down the registration rate.** That is the number you carry into the conference.

| Check | Pass |
|---|---|
| Folders in `sparse/` | Exactly one |
| Registered images | 80 percent or more |
| Mean reprojection error | Under about 1.0 px |

If it fails: several models means the scan broke into disconnected chunks, so move slower with more overlap. Low registration means blur or stabilization. High error means lens distortion, so use the main wide.

Optional: `colmap gui`, then File > Import model > sparse/0. The picture tells you more than the numbers.

## A3 · Hand pose · 30 minutes

No install, no GPU. Run a handful of demo frames through the **HaMeR demo on Hugging Face**. Screenshot the output.

**What you want to see:** the hand mesh tracking *through* the grasp, not only when the hand is open and clearly visible.

## A4 · What you now say at Actuate

> "I recorded egocentric demos on an iPhone. COLMAP registered *[your number]* percent of the scan. HaMeR tracks my hands through the grasp. I am trying to work out whether pose from a hundred-dollar rig is good enough to train on."

Two facts and an open question. That gets Cheng Chi and the Data Wars panel to engage in a way a pitch never will.

**Then the conference is your research.** Ask Cheng Chi where UMI breaks at volume. Ask the panel what pose error stops mattering. Ask whether anyone has reproduced WARPED. Come back knowing whether to build at all.

---

# PATH B · Build · start it tonight, unattended

## B1 · Where it runs

**Try the Mac first.** Do not pre-emptively rent a GPU. Most stages run on Apple Silicon, and you only pay for the ones that do not.

Two things will snag it. Name them in the prompt so it does not spend an hour discovering them.

**Gaussian splatting.** `gsplat` is CUDA-only, so Stages 1 and 5 would die on Apple Silicon. Mac-native alternatives exist:

- **[Brush](https://github.com/ArthurBrussee/brush)** — wgpu, cross-platform, trains and renders. The safest bet.
- **[gsplat-mlx](https://a1091150.github.io/gsplat-mlx/)** — gsplat's API on MLX. Closest to a drop-in.
- **[splat-apple](https://github.com/ghif/splat-apple)** — native Apple Silicon 3DGS.

**HaMeR.** The likelier wall. It pulls in detectron2 and ViTPose, both painful on Apple Silicon. If it blocks, rent a box for that one stage.

### If you do need a GPU

One rented Linux box, everything on it over SSH. Not a second development machine — the only machine, for that stage. RunPod, Lambda or Vast.ai, a 4090 or A10, roughly $0.40 to $0.80 an hour. Shut it down after.

## B2 · One prompt, then go to bed

Put `wristview-build-plan.md` in an empty repo, and paste this:

> Read `wristview-build-plan.md`. Build Stages 0 through 5, so I can run one command on a scan video plus demo videos and get rendered robot wrist-camera views out.
>
> Target platform is macOS on Apple Silicon. Python 3.11 with `uv`. Use Metal and MPS where possible. **For Gaussian splatting use Brush or gsplat-mlx, not gsplat, since gsplat is CUDA-only.**
>
> **Run all stages through to Stage 5 without waiting for my approval.** Write each stage's artifact to disk and log what you produced and where. If a dependency cannot be installed on Apple Silicon, say so clearly in the log, skip that stage if the pipeline can continue without it, and keep going.
>
> Do not build Stage 6 or 7. Do not scaffold future stages.

Then leave it. Read the log in the morning.

## B3 · What to look at, per stage

The log tells you the file each time. This is what a correct one looks like.

| Stage | Verify by |
|---|---|
| 0 · Ingest | Registration rate on the scan |
| 1 · Scene | A splat you can fly through, plus a scale factor in metres that matches a real measured object |
| 2 · Localize | The demo trajectory drawn inside the scan, following the path you actually walked |
| 3 · Estimate | Hand mesh and object mask overlaid on video, tracking **through** the grasp |
| 4 · Retarget | Gripper trajectory in 3D, closing and opening at the right moments |
| 5 · Render | **The first wrist view.** |

## B4 · The honest expectation

**Your constraint is not the machine.** Stages 3 and 4 — object pose and hand-to-gripper retargeting — are hard *problems*, not hard *compute*. A GPU makes them faster to iterate on, not easier to solve. WARPED does a joint hand-object optimization we are approximating.

So: do not expect a wrist view on the first night. Expect to get somewhere between Stage 2 and Stage 4, and to find out exactly which dependency wall you hit. That itself is worth the night.

**Rules to enforce when you pick it back up:**

- Every stage writes files and reads files. **No stage calls the next.** If it starts chaining them, stop it.
- When something fails, paste the traceback and `meta.json`. Do not describe the failure in prose.
- If Stage 3 or Stage 4 stalls for more than a day, that is expected, not a sign you did it wrong.

---

# THE GATE THAT MATTERS · at the first wrist view

**Stop building. Do not start Stage 6.**

Send three rendered frames to Georgios at Acumino. It doubles as the data sample you already owe him.

> "I built a pipeline that turns head-mounted human video into robot wrist-camera views with end-effector trajectories. Three frames attached. Two questions. Would you train on this? And what is wrong with it?"

| His answer | What you do |
|---|---|
| "Yes, we would train on this" | Build Stages 6 and 7. Then scale capture to 100 episodes. |
| "Close, but X is wrong" | X is your roadmap. Fix it before scaling anything. |
| "No, and here is why" | You learned it in a week instead of a quarter. That is why you stopped here. |

---

# Cost

| Item | Cost |
|---|---|
| Blackmagic Camera | Free |
| Head mount | About $15 |
| COLMAP, ffmpeg, HaMeR demo | Free |
| Cloud GPU, only if the Mac blocks | $10 to $100 |

Path A costs about $15 and two hours. Path B costs nothing until a dependency forces a GPU, and then well under $100.
