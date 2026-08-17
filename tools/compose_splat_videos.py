"""Put the splat-rendered wrist views beside the source footage.

The wrist views come back from the CUDA box as one video per clip. The source
frames, the trajectory and the captions are all local, so the pairing happens
here rather than shipping 180 MB of demo frames to a rented machine.

    python -m tools.compose_splat_videos --run runs/real03 \
        --wrist /path/to/wrist_mp4 --out deliverables/wrist-gsplat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wristview.logging_setup import get, setup  # noqa: E402
from wristview.runctx import read_json  # noqa: E402

log = get(__name__)
PANEL_W, PANEL_H = 640, 480


def label(image: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    for row, (text, colour) in enumerate(lines):
        y = 18 + row * 17
        cv2.rectangle(image, (4, y - 13), (8 + 7 * len(text), y + 4), (0, 0, 0), -1)
        cv2.putText(image, text, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--wrist", required=True, help="folder of <clip>.mp4 from the box")
    parser.add_argument("--out", default="deliverables/wrist-gsplat")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    setup(None, verbose=False)
    root = Path(args.run).resolve()
    wrist_dir = Path(args.wrist).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(root / "00_ingest" / "manifest.json")["clips"]
    box_report = json.loads((wrist_dir / "report.json").read_text())
    report = {}

    for clip_id, meta in sorted(manifest.items()):
        video = wrist_dir / f"{clip_id}.mp4"
        if not video.exists():
            continue
        traj = np.load(root / "04_retarget" / clip_id / "ee_trajectory.npz")
        widths = traj["width_video_rate"]
        closed = traj["closed_video_rate"].astype(bool)
        detected = traj["hand_valid"].astype(bool)
        filled = traj["hand_filled"].astype(bool) if "hand_filled" in traj else np.zeros_like(detected)
        onsets = np.nonzero(closed)[0]
        grasp_onset = int(onsets[0]) if len(onsets) else None

        frames_dir = root / meta["frames_dir"]
        names = meta["frame_names"]
        coverage = box_report.get(clip_id, {}).get("coverage_mean", 0.0)

        capture = cv2.VideoCapture(str(video))
        writer = None
        target = out_dir / f"{clip_id}.mp4"
        index = 0
        while True:
            ok, right = capture.read()
            if not ok or index >= len(names):
                break
            right = np.ascontiguousarray(cv2.resize(right, (PANEL_W, PANEL_H)))
            source = cv2.imread(str(frames_dir / names[index]))
            if source is None:
                break
            left = np.ascontiguousarray(cv2.resize(source, (PANEL_W, PANEL_H)))

            carried = grasp_onset is not None and index >= grasp_onset and closed[index]
            if carried:
                object_caption = ("object CARRIED by gripper (rigid, from grasp)", (140, 220, 255))
            else:
                object_caption = ("object STATIC at scan position", (180, 180, 180))

            if filled[index]:
                hand_line = ("hand frame FILLED (failed velocity gate)", (120, 200, 255))
            elif detected[index]:
                hand_line = ("hand tracked", (180, 220, 180))
            else:
                hand_line = ("hand LOST, pose held", (120, 160, 255))

            label(left, [("SOURCE  egocentric camera", (255, 255, 255))])
            label(right, [
                ("RENDERED  wrist camera, 1.53 M Gaussian splat", (255, 255, 255)),
                (f"scene coverage {coverage * 100:.0f}%", (180, 220, 180)),
                hand_line,
                (f"gripper proxy {'CLOSED' if closed[index] else 'open'} "
                 f"{widths[index] * 100:.1f} cm", (200, 200, 255)),
                object_caption,
            ])

            pair = np.hstack([left, np.full((PANEL_H, 4, 3), 40, np.uint8), right])
            if writer is None:
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (pair.shape[1], pair.shape[0]))
            writer.write(pair)
            index += 1
        capture.release()
        if writer is not None:
            writer.release()
        report[clip_id] = {"frames": index, "splat_coverage": coverage,
                           "carried_frames": int(sum(closed[:index] & (
                               np.arange(index) >= (grasp_onset if grasp_onset is not None else index))))}
        log.info("%s: %d frames, splat coverage %.0f%%", clip_id, index, coverage * 100)

    (out_dir / "render_report.json").write_text(json.dumps(report, indent=2))
    log.info("wrote %d videos to %s", len(report), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
