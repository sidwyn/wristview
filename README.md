# wristview

Rebuild of the [WARPED](https://arxiv.org/html/2604.10809v1) pipeline in our own code.

**In:** head-mounted egocentric video of a person doing a task, plus a scan of the room.
**Out:** robot wrist-camera views with end-effector trajectories, ready to train on.

WARPED has released no code. This is a reimplementation.

## Files

| File | What it is |
|---|---|
| `RUNBOOK.md` | **Start here.** What to do, in order. Two paths: two hours of evidence, and an overnight build. |
| `BUILD-PLAN.md` | The spec. Hand this to Claude Code. Seven stages, data contracts, risks. |

## The short version

1. Record one room scan and three demo clips on an iPhone. Stabilization off.
2. Run COLMAP on the scan. If it does not register, stop and re-shoot.
3. Point Claude Code at `BUILD-PLAN.md` and let it run overnight.
4. At the first rendered wrist view, stop and send three frames to a buyer.

## Why this matters

Raw egocentric video is worth $2 to $5 an hour with a 90 percent reject rate. Data carrying pose labels sells for roughly 100 times that. This pipeline is the difference between the two, using a camera that costs nothing because you already own it.

## Status

Not started. Gate 0 is unrun.
