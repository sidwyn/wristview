"""Is the object mask on the object, or on the mat and the hand?

Segmentation fails quietly. A mask that has drifted onto the mat still has an
area, still has a centroid, and `plane.object_pose_on_plane` still returns a
pose from it. Every downstream number then describes the mat.

real27's object is a cream carton on a white printed mat, so contrast is low
and this is where it will fail if it fails. Run this on the first demo before
shooting the rest.

What it reports:

- area per frame, and how stable that area is. A real object's silhouette
  changes smoothly as the hand turns it. A mask that jumps between the object
  and the mat changes area in steps.
- how much of the mask overlaps the hand, using the hand landmarks Stage 3
  already produced. A mask that has grabbed the hand is the common failure and
  it is invisible in the area alone.
- an overlay strip, so the numbers can be checked against the picture.

    python -m tools.check_object_mask --run runs/real27 --clip demo_0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# A mask whose area jumps by more than this between neighbouring frames is not
# tracking a rigid object smoothly. real26/bm demo_0 tracked correctly with a
# p95 of 0.104.
AREA_JUMP_P95_LIMIT = 0.25

# How far the measured silhouette may sit from the area the object's real size
# predicts, as a ratio. A box of side s at distance z covers about
# s^2 * fx * fy / z^2 pixels, and a real silhouette runs between one and two
# faces of that depending on the viewing angle, so the honest band is wide.
# Outside it the mask is not the object.
AREA_RATIO_MIN, AREA_RATIO_MAX = 0.25, 4.0

# Overlap with the hand is NOT used as a test.
#
# It was, and it was wrong. A held object legitimately sits inside the hand's
# convex hull, and a hand passing over the object overlaps it without touching
# it. Measured on real26/bm demo_0, which tracked correctly throughout: held
# frames had a median overlap of 0.255, and frames with no contact at all
# reached 0.341 at p95. The metric cannot separate "the mask grabbed the hand"
# from "the hand is where the object is", which is most of a manipulation clip.
# It is still reported, as an observation, and nothing fails on it.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--clip", default="demo_0")
    parser.add_argument("--stills", default=None)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    run = Path(args.run)
    est = run / "03_estimate" / args.clip
    masks_dir = est / "object_masks"
    frames_dir = run / "00_ingest" / args.clip / "frames"
    if not masks_dir.is_dir():
        raise SystemExit(f"no masks at {masks_dir}; run Stage 3 first")

    hand = np.load(est / "hand.npz")
    hand_px = hand["landmarks_px"]
    hand_valid = hand["valid"].astype(bool)

    # Predict the silhouette from the object's real size and how far away it
    # actually is. This is the test that separates the object from the mat:
    # a mask on the mat is far larger than the object could ever project to.
    import yaml

    cfg = yaml.safe_load((run / "config.yaml").read_text())
    pose_cfg = cfg["estimate"]["pose"]
    per_clip = (pose_cfg.get("per_clip") or {}).get(args.clip, {})
    merged = {**pose_cfg, **per_clip}
    dims = merged.get("object_dimensions_m")
    if not dims:
        h = float(merged.get("object_height_m") or 0)
        dims = [h, h, h]
    dims = np.asarray(dims, dtype=float)
    # Two of the three faces are visible at a general viewing angle. Use the
    # geometric mean of the face areas as the central expectation.
    faces = np.array([dims[0] * dims[1], dims[0] * dims[2], dims[1] * dims[2]])
    face_area_m2 = float(np.exp(np.mean(np.log(faces))))

    intr = json.loads((run / "01_scene" / "cameras.json").read_text())["intrinsics"]
    obj_pose = np.load(est / "object_pose.npy")
    obj_valid = np.load(est / "object_valid.npy").astype(bool)
    cam_poses = np.load(run / "02_localize" / args.clip / "camera_poses.npy")

    names = sorted(p.name for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    rows = []
    for path in sorted(masks_dir.iterdir()):
        if path.suffix.lower() != ".png":
            continue
        index = int(path.stem)
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        flag = mask > 127
        area = int(flag.sum())
        if area == 0:
            continue
        ys, xs = np.nonzero(flag)

        # How much of the mask sits on the hand. The landmarks are in full-frame
        # pixels and the mask is at the work resolution, so scale them.
        overlap = np.nan
        if index < len(hand_valid) and hand_valid[index]:
            sx = mask.shape[1] / 1920.0
            sy = mask.shape[0] / 1080.0
            hull = cv2.convexHull(
                np.stack([hand_px[index][:, 0] * sx, hand_px[index][:, 1] * sy], axis=1)
                .astype(np.int32)
            )
            hand_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.fillConvexPoly(hand_mask, hull, 1)
            overlap = float((flag & (hand_mask > 0)).sum()) / area

        # What area should this object cover, from where the camera actually is?
        expected = np.nan
        if index < len(obj_valid) and obj_valid[index] and index < len(cam_poses):
            centre = obj_pose[index][:3, 3]
            distance = float(np.linalg.norm(centre - cam_poses[index][:3, 3]))
            if distance > 1e-3:
                sx = mask.shape[1] / float(intr["width"])
                sy = mask.shape[0] / float(intr["height"])
                fx = float(intr["fx"]) * sx
                fy = float(intr["fy"]) * sy
                expected = face_area_m2 * fx * fy / (distance ** 2)

        rows.append({
            "frame": index, "area": area,
            "area_fraction": area / flag.size,
            "expected_area": expected,
            "area_ratio": area / expected if np.isfinite(expected) and expected > 0 else np.nan,
            "cx": float(xs.mean()), "cy": float(ys.mean()),
            "hand_overlap": overlap,
        })

    if not rows:
        raise SystemExit("every mask was empty")

    area = np.array([r["area"] for r in rows], dtype=float)
    frames = np.array([r["frame"] for r in rows])
    overlap = np.array([r["hand_overlap"] for r in rows], dtype=float)

    # Compare neighbours in the source clip, not neighbours in the kept list:
    # a gap in the masks is not a jump in the object.
    adjacent = np.diff(frames) == 1
    jump = np.abs(np.diff(area)) / np.maximum(area[:-1], 1)
    jump = jump[adjacent] if adjacent.any() else np.array([0.0])

    report = {
        "run": str(run), "clip": args.clip,
        "frames_with_a_mask": len(rows),
        "frames_in_clip": len(names),
        "coverage_pct": round(100.0 * len(rows) / max(len(names), 1), 1),
        "area_px_median": int(np.median(area)),
        "area_px_p5": int(np.percentile(area, 5)),
        "area_px_p95": int(np.percentile(area, 95)),
        "area_fraction_median": round(float(np.median([r["area_fraction"] for r in rows])), 5),
        "area_jump_median": round(float(np.median(jump)), 4),
        "area_jump_p95": round(float(np.percentile(jump, 95)), 4),
        "area_jump_limit": AREA_JUMP_P95_LIMIT,
        "hand_overlap_median": round(float(np.nanmedian(overlap)), 4) if np.isfinite(overlap).any() else None,
        "hand_overlap_p95": round(float(np.nanpercentile(overlap, 95)), 4) if np.isfinite(overlap).any() else None,
        "hand_overlap_note": (
            "observation only, nothing fails on it: a held object sits inside "
            "the hand's hull, so this cannot separate contamination from a grasp"
        ),
    }

    ratio = np.array([r["area_ratio"] for r in rows], dtype=float)
    good = np.isfinite(ratio)
    if good.any():
        report["area_ratio_median"] = round(float(np.median(ratio[good])), 3)
        report["area_ratio_p5"] = round(float(np.percentile(ratio[good], 5)), 3)
        report["area_ratio_p95"] = round(float(np.percentile(ratio[good], 95)), 3)
        report["object_face_area_cm2"] = round(face_area_m2 * 1e4, 2)
        report["area_ratio_band"] = [AREA_RATIO_MIN, AREA_RATIO_MAX]

    failures = []
    if good.any() and not (AREA_RATIO_MIN <= report["area_ratio_median"] <= AREA_RATIO_MAX):
        direction = "larger" if report["area_ratio_median"] > 1 else "smaller"
        failures.append(
            f"the mask is {report['area_ratio_median']:.1f}x the area this "
            f"object projects to at its measured distance, which is {direction} "
            f"than any viewing angle explains. The mask is not on the object: "
            f"too large means the mat or the background, too small means a "
            f"fragment or a specular patch."
        )
    if report["area_jump_p95"] > AREA_JUMP_P95_LIMIT:
        failures.append(
            f"the mask area jumps by {report['area_jump_p95'] * 100:.0f} per cent "
            f"between adjacent frames at p95, over {AREA_JUMP_P95_LIMIT * 100:.0f}. "
            f"A rigid object's silhouette does not do that; the mask is moving "
            f"between things."
        )
    if report["coverage_pct"] < 50:
        failures.append(
            f"only {report['coverage_pct']:.0f} per cent of frames have a mask. "
            f"real26/bm demo_1 had 31 per cent and its carried object solved "
            f"7 cm below the desk."
        )
    report["passed"] = not failures
    report["failures"] = failures

    print(json.dumps(report, indent=2))

    if args.stills:
        out = Path(args.stills)
        out.mkdir(parents=True, exist_ok=True)
        picks = np.linspace(0, len(rows) - 1, min(args.samples, len(rows))).astype(int)
        tiles = []
        for slot in sorted(set(int(i) for i in picks)):
            row = rows[slot]
            frame = cv2.imread(str(frames_dir / names[row["frame"]]))
            mask = cv2.imread(str(masks_dir / f"{row['frame']:05d}.png"), cv2.IMREAD_GRAYSCALE)
            frame = cv2.resize(frame, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
            tinted = frame.copy()
            tinted[mask > 127] = (0, 0, 255)
            blended = cv2.addWeighted(frame, 0.5, tinted, 0.5, 0)
            pixels = frame[mask > 127]
            label = (f"f{row['frame']} {row['area']}px "
                     f"BGR {np.median(pixels, axis=0).astype(int)} "
                     f"hand {row['hand_overlap'] * 100:.0f}%"
                     if len(pixels) else f"f{row['frame']} empty")
            cv2.putText(blended, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            tiles.append(blended)
        half = max(1, len(tiles) // 2)
        strip = np.vstack([np.hstack(tiles[:half]), np.hstack(tiles[half:half * 2])]) \
            if len(tiles) >= 2 * half else np.hstack(tiles)
        cv2.imwrite(str(out / f"object_mask_{args.clip}.png"), strip)
        print(f"wrote {out / f'object_mask_{args.clip}.png'}")

    if not report["passed"]:
        print("\nOBJECT MASK SUSPECT. Do not shoot the rest of the session on this setup.")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nobject mask looks sound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
