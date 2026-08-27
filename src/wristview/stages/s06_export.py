"""Stage 6. Write the run's episodes as one LeRobotDataset.

One dataset, every training run. Runs A, B, A', B' and C select their cameras
through the policy's `input_features` and read the same rows, so the paired
comparison cannot drift: A' and B' are the same frames.

Verified end to end on a 3-episode dataset. See `tools/verify_lerobot.py`.
"""

from __future__ import annotations

import numpy as np

from ..lerobot_export import build_dataset
from ..logging_setup import get
from ..runctx import RunContext, StageRecorder, read_json

STAGE = 6
NAME = "export"

log = get(__name__)


def _episode(ctx: RunContext, clip_id: str) -> dict | None:
    """Gather one clip's images, poses and gripper widths.

    Returns None when the clip cannot contribute, with the reason logged. A
    clip that is silently skipped is a clip that quietly shrinks the dataset.
    """
    render_dir = ctx.episode_dir(5, clip_id, create=False) / "wrist"
    status = read_json(ctx.episode_dir(5, clip_id, create=False) / "status.json")
    if not render_dir.is_dir() or not status:
        log.warning("%s: no Stage 5 render, so it has no wrist view", clip_id)
        return None

    wrist = sorted(render_dir.glob("*.png"))
    # Stage 5 renders only the frames that carry a hand measurement, and
    # records which source frames those were.
    source_index = status.get("source_frame_index")
    if source_index is None:
        log.warning("%s: Stage 5 did not record source_frame_index, so its "
                    "frames cannot be matched to the ego frames", clip_id)
        return None

    manifest = read_json(ctx.root / "00_ingest" / "manifest.json")
    clip = (manifest.get("clips") or {}).get(clip_id)
    if not clip:
        log.warning("%s: not in the ingest manifest", clip_id)
        return None
    frames_dir = ctx.root / clip["frames_dir"]
    names = clip["frame_names"]

    trajectory = np.load(ctx.episode_dir(4, clip_id, create=False) / "ee_trajectory.npz")
    poses = trajectory["poses"]
    widths = trajectory["width_m"]
    fps = float(trajectory["source_fps"])
    times = trajectory["timestamps_s"]

    # Map each rendered frame back to the ego frame it came from.
    demo_index = np.clip(np.round(times * fps).astype(int), 0, len(names) - 1)
    ego = [frames_dir / names[demo_index[i]] for i in source_index]
    if len(ego) != len(wrist):
        log.warning("%s: %d ego frames against %d wrist frames", clip_id, len(ego), len(wrist))
        return None

    return {
        "ego": ego,
        "wrist": wrist,
        "poses": poses[source_index],
        "gripper": widths[source_index],
        "fps": float(status.get("fps") or 15),
    }


def run(ctx: RunContext) -> dict:
    rec = StageRecorder(ctx, STAGE, NAME)
    try:
        cfg = ctx.config.get("export", {}) or {}
        summary = read_json(ctx.stage_dir(5, create=False) / "summary.json") or {}
        clips = sorted(summary)

        episodes, used, skipped = [], [], []
        for clip_id in clips:
            episode = _episode(ctx, clip_id)
            if episode is None:
                skipped.append(clip_id)
                continue
            episodes.append(episode)
            used.append(clip_id)

        if not episodes:
            raise ValueError(
                "no clip produced a complete episode, so there is nothing to "
                "export. Stage 5 must have run and recorded source_frame_index."
            )

        out = ctx.stage_dir(STAGE) / "lerobot"
        repo_id = str(cfg.get("repo_id") or f"wristview/{ctx.run_id}")
        instruction = (read_json(ctx.root / "sources.json") or {}).get(
            "instruction") or "manipulate the object"

        build_dataset(episodes, out, repo_id=repo_id,
                      fps=int(round(episodes[0]["fps"])), task=instruction)

        rec.output("lerobot_dataset", out)
        rec.metric("episodes_written", len(episodes))
        rec.metric("episodes_skipped", skipped)
        rec.metric("frames", int(sum(len(e["wrist"]) - 1 for e in episodes)))
        rec.metric("repo_id", repo_id)
        if skipped:
            log.warning("%d clip(s) skipped: %s", len(skipped), skipped)
        rec.write("ok")
        return {"episodes": len(episodes), "skipped": skipped}
    except Exception as exc:
        rec.note(f"{type(exc).__name__}: {exc}")
        rec.write("failed")
        raise
