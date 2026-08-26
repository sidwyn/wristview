# CLAUDE.md

Read this file first. It holds the rules that do not change between sessions.

## What this project does

Head-mounted video in. Rendered robot wrist-camera video out. We train a
policy on the result. We measure if the rendered wrist views increase task
success.

The real wrist camera is NOT part of the product. It records arm C of the
experiment, and nothing else. Do not try to recover its pose. That work is
retracted. See `synthesis/struggles&decisions.md` for the 4 failed attempts.

## WHERE EACH STAGE RUNS

This is the most important operational rule in this file.

| Stage | Name | Runs on | Why |
|---|---|---|---|
| 0 | Ingest | Mac | CPU only. |
| 1 SfM | Reconstruct | Mac | CPU only. COLMAP. |
| 1 splat | Train splat | **RunPod GPU** | Needs CUDA. See below. |
| 2 | Localise | Mac | CPU only. |
| 3 | Estimate hand | Mac | WiLoR. 162 s local. GPU is faster but not required. |
| 4 | Retarget | **RunPod GPU** | Does not need CUDA. It runs there to avoid a second transfer. |
| 5 | Render | **RunPod GPU** | Needs CUDA. See below. |
| 6 | Export | Mac | CPU only. |

**Never train a splat on the Mac.** MPS gives about 17 dB. CUDA gives about
31 dB on the same data. The Mac result is unusable.

**Never render a real splat on the Mac.** The MPS rasteriser is broken. See
`MPS-RASTERISER.md`. It drops Gaussians silently and scores a good splat at
7 dB. It is a preview tool only. It is not authoritative for any measurement.

**Splat quality gates must run on the GPU**, with gsplat. A gate that judges
a splat through an unverified renderer cannot tell a bad splat from a
renderer that cannot draw it. That error cost a full day.

Use `GPU-COMMANDS-real26.md` as the transfer template. RTX 4090, 24 GB, is
enough: peak use is 14.2 GiB at 1920 with 3.9 M Gaussians. Do not use an
A100 or an H100. They cost 2 to 4 times more and give no speed increase,
because splat training uses memory bandwidth, not tensor cores.

Stop the pod as soon as the results are downloaded. It bills while it runs,
not while it works.

## Standing rules

1. A quantity is not verified until a source with no access to it agrees.
2. A per-clip metric with too little variance across independent clips is a
   systematic fault, not a result.
3. Before you trust a metric, ask if it can see the failure mode.
4. A script written to investigate a problem needs the same doubt as the
   code it investigates.
5. If 4 failures in a row are physical, not analytical, the instrument is
   the wrong instrument. Stop the build.
6. Defaults for physical quantities must not exist. Make them required
   parameters. A forgotten override must be a loud error.

## Gate rules

**Two categories. Encode the category. Do not decide it per run.**

- A gate that measures if something is WRONG is a hard stop.
- A gate that measures if something could be CHECKED is reported, never a
  stop. But low coverage must COST something: widen the reported
  uncertainty. "We could not check" must never render as "it is fine".

Never relax a threshold to pass a clip. Fix the cause, or trim by a
CONTENT-defined boundary and say plainly what the trim excluded.

Never use `--keep-going`. Stop at the first failed gate.

## The most repeated defect

**This codebase computes the number that describes a failure, and then does
not read it.** It has happened at least 6 times. See `DEFECT-TABLE.md`.

Examples: `_overflow` counted 11,076,570 dropped Gaussians and nothing read
it. A quality threshold sat in the config and no code read it. Densification
counts were logged and not checked.

When you add a metric, make something read it and act on it.

## Config traps

- `aruco_marker_length_m` was 0.15 from the first commit. Every marker this
  project printed is 0.100 m. Pass it explicitly. Never use the default.
- Unknown config keys must raise. `estimate.hand.stride` was passed, did not
  exist, and was ignored in silence.
- `init_points` was a ceiling, not a target. It is fixed. Check any other
  parameter that reads like a target.
- `object_height_m` was 0.0762 from the first commit, which is exactly 3
  inches. The cube is 70 mm. A stale imperial constant survives because a cube
  of the wrong size still sits on the desk and no check compares it to the
  object. Measure the object for every session.
- Watch for any default that is a round number in the wrong unit. 0.0762 m is
  3 in, 0.15 m is not a marker this project owns. A tidy imperial value in a
  metric field is almost always inherited, not measured.

## Marker ids

- **id 0** — table marker, 100 mm. Scale AND workspace-segment detection key
  on this id ONLY.
- **id 1** — on the wrist camera mount. It moves. Ignore it completely.
  Never let it satisfy segment detection.
- **id 17** — a stray. Ignore it.

## Segment detection

Use scan-match count per frame. Do NOT use marker visibility. The hand
covers the marker for 4.25 s during a manipulation, which is longer than a
sync segment, so no threshold on marker visibility can separate the two
cases.

## Sync

Two taps on the bare table beside the mat, one at each end of the take, made
with the right hand. Three channels: audio transient, visual contact frame,
and wrist-camera image motion going to zero. Report each separately.

The wrist camera duplicates frames in low light. Carry the per-frame
ambiguity forward. Do not interpolate it away.

## Reference documents

| File | Holds |
|---|---|
| `synthesis/human-priors-aug-25.md` | The current plan. What to do now. |
| `synthesis/struggles&decisions.md` | Why each rule exists. The history. |
| `DEFECT-TABLE.md` | Every defect found, and its family. |
| `MPS-RASTERISER.md` | The broken local renderer. Deferred, not abandoned. |
| `GPU-COMMANDS-real26.md` | The RunPod transfer and training template. |
| `CAPTURE-SOP.md` | How to shoot a session. |

## Reporting style

Numbers only. Broken things first. Do not report a setting as active until
you have confirmed the code reads it.
