"""Consecutive-frame palm rotation, and a landmarks_px diff against a reference.

The palm frame is built from three landmarks that do not move relative to each
other while the fingers do: the wrist (0), the index MCP (5) and the pinky MCP
(17). Gram-Schmidt gives an orthonormal basis, and the angle between two
consecutive bases is the rotation the palm made between those frames.

A hand tracker that has lost the hand emits a basis that jumps. That is what
this measures. It is reported over CONSECUTIVE VALID PAIRS only, because a
gap of invalid frames is a gap, not a rotation.
"""
import sys
import numpy as np

WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17


def palm_bases(lm: np.ndarray) -> np.ndarray:
    """(N,21,3) landmarks -> (N,3,3) orthonormal palm rotation matrices."""
    origin = lm[:, WRIST]
    x = lm[:, INDEX_MCP] - origin
    y = lm[:, PINKY_MCP] - origin
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=2)


def step_angles_deg(R: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pair = valid[:-1] & valid[1:]
    rel = np.einsum("nij,nkj->nik", R[1:], R[:-1])          # R_{t+1} R_t^T
    trace = np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(trace))[pair]


def report(tag: str, path: str, key: str) -> np.ndarray:
    z = np.load(path)
    lm, valid = z[key], z["valid"]
    ang = step_angles_deg(palm_bases(lm), valid)
    over90 = int((ang > 90).sum())
    print(f"{tag:22s} {key:17s} n={len(ang):4d}  "
          f"median {np.median(ang):6.2f} deg  mean {ang.mean():6.2f}  "
          f"max {ang.max():7.2f}  >90deg {over90:3d}  "
          f"valid {int(valid.sum())}/{len(valid)}")
    return ang


RUNS = [
    ("scan2 REFERENCE", "runs/sept02_scan2/03_estimate/demo_0/hand.npz"),
    ("broken evidence", "runs/sept02_final/_evidence_broken_hands/"
                        "demo_0_broken_reference/hand.npz"),
    ("scratch NEW", "runs/_scratch_s3_health/03_estimate/demo_0/hand.npz"),
]

for key in ("landmarks_world", "landmarks_cam"):
    print(f"--- palm rotation from {key} ---")
    for tag, path in RUNS:
        try:
            report(tag, path, key)
        except FileNotFoundError:
            print(f"{tag:22s} {key:17s} MISSING {path}")
    print()

print("--- landmarks_px against scan2 ---")
ref = np.load(RUNS[0][1])
for tag, path in RUNS[1:]:
    try:
        z = np.load(path)
    except FileNotFoundError:
        print(f"{tag:22s} MISSING")
        continue
    a, b = ref["landmarks_px"], z["landmarks_px"]
    if a.shape != b.shape:
        print(f"{tag:22s} SHAPE {a.shape} vs {b.shape}")
        continue
    both = ref["valid"] & z["valid"]
    d = np.linalg.norm(a - b, axis=2)
    print(f"{tag:22s} maxdiff {np.abs(a-b).max():.8f}  "
          f"per-landmark px: median {np.median(d[both]):.4f} "
          f"max {d[both].max():.4f}  "
          f"valid-agree {int((ref['valid']==z['valid']).sum())}/{len(both)}")
