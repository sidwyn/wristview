"""Stage 6. Write the run's episodes as one LeRobotDataset.

One dataset, every training run. Runs A, B, A', B' and C select their cameras
through the policy's `input_features` and read the same rows, so the paired
comparison cannot drift: A' and B' are the same frames.

Verified end to end on a 3-episode dataset. See `tools/verify_lerobot.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..lerobot_export import build_dataset
from ..logging_setup import get
from ..runctx import RunContext, StageRecorder, read_json

STAGE = 6
NAME = "export"

log = get(__name__)


def _episode(ctx: RunContext, clip_id: str, cfg_real_root=None) -> dict | None:
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
    times = trajectory["timestamps_s"]

    # Map each rendered frame back to the ego frame it came from, by TRUE
    # frame time, not by an averaged rate.
    #
    # This was `round(times * fps)` with fps = `source_fps`, which is Stage 0's
    # `effective_fps`: kept frames divided by duration. On real31's demo_1 that
    # is 268 / 13.95 = 19.2115, and it is a fiction. Stage 0 extracts at a
    # clean 20 fps and then drops blurred and duplicate frames, so the
    # SURVIVING frames sit at irregular indices while `frame_times_s` records
    # what each one's true time is. Indexing by an averaged rate assumes the
    # drops were spread evenly. They are not.
    #
    # Measured on demo_1, 279 extracted and 268 kept: the old expression picked
    # the wrong frame on 134 of 134 rows, drifting from 5 frames early to 6
    # frames late, median time error 183 ms and worst 317 ms. At 15 Hz that is
    # up to five rows out. Every dataset this project has exported carries it.
    #
    # `frame_times_s` has been in the manifest since Stage 0 was written and
    # nothing read it. That is this codebase's most repeated defect, and the
    # fix is to read the number that was already there.
    frame_times = np.asarray(clip.get("frame_times_s") or [], dtype=np.float64)
    if len(frame_times) != len(names):
        raise ValueError(
            f"{clip_id}: the manifest holds {len(names)} frame names and "
            f"{len(frame_times)} frame times. Without a true time per frame "
            f"the row-to-image mapping falls back to an averaged rate, which "
            f"is the fault this check exists to stop."
        )
    demo_index = np.abs(frame_times[None, :] - np.asarray(times)[:, None]).argmin(axis=1)
    ego = [frames_dir / names[demo_index[i]] for i in source_index]
    if len(ego) != len(wrist):
        log.warning("%s: %d ego frames against %d wrist frames", clip_id, len(ego), len(wrist))
        return None

    # Arm C's pixels. Sampled by tools/extract_wrist_real.py at the instants
    # these very rows use, so C reads the same rows as A' and B' and the
    # paired comparison cannot drift. A clip without them cannot be in the
    # dataset at all: giving A' and B' an episode C cannot see would break the
    # error cancellation the fraction-closed number depends on.
    real_dir = Path(str(cfg_real_root)) / clip_id if cfg_real_root else None
    if real_dir is None or not real_dir.is_dir():
        log.warning("%s: no real wrist frames at %s, so arm C cannot read it",
                    clip_id, real_dir)
        return None
    wrist_real = sorted(real_dir.glob("*.png"))
    if len(wrist_real) != len(wrist):
        log.warning("%s: %d real wrist frames against %d rendered; the two "
                    "cameras must sample the same instants", clip_id,
                    len(wrist_real), len(wrist))
        return None

    return {
        "ego": ego,
        "wrist": wrist,
        "wrist_real": wrist_real,
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
        # `export.clips`, when set, restricts the dataset to a named list.
        # The experiment needs this: real31 rendered 28 clips of which 11 have
        # a hand-identity crossing that Stage 3 calls INVALID, and those must
        # not enter a training set even though they render. Naming the clips
        # explicitly beats filtering on a metric here, because the reason a
        # clip is excluded belongs in the run record, not in a threshold.
        wanted = cfg.get("clips")
        if wanted:
            wanted = [str(c) for c in wanted]
            missing = [c for c in wanted if c not in summary]
            if missing:
                raise ValueError(
                    f"export.clips names {missing}, which Stage 5 did not "
                    f"render. It rendered {sorted(summary)}."
                )
            clips = [c for c in clips if c in set(wanted)]
            log.info("export.clips restricts this dataset to %d of %d rendered "
                     "clips", len(clips), len(summary))

        episodes, used, skipped = [], [], []
        for clip_id in clips:
            episode = _episode(ctx, clip_id, cfg.get("wrist_real_root"))
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
