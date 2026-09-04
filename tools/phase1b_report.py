"""Read the Phase 1b encoder probe and place it beside Phase 1.

The comparison that matters is NULL at 6.63 mm, not the floor. Phase 1
established that a scratch encoder earns nothing; this asks whether a
pretrained one earns anything.

The reported figure is the MEAN OF THE LAST 6 VALIDATION CHECKS.
`still_falling_at_end` is not read: it looks at four endpoints and has already
produced one false conclusion on a 0.004 mm margin.
"""
import json
import sys
from pathlib import Path

import numpy as np

BASELINES = [
    ("FLOOR-PERSISTENCE", 3.72, 2.15, 8.16, None, "reference, not a gate"),
    ("FLOOR-MEAN", 6.14, 4.41, 12.01, None, "the Phase 1 bar"),
    ("NULL (no camera)", 6.63, 5.18, 12.48, 7.07, "THE COMPARISON"),
    ("A_prime scratch", 6.69, 5.07, 12.53, 6.97, "Phase 1"),
]
RUNS = [
    ("KEEP_imagenet_frozen.json", "ImageNet FROZEN"),
    ("KEEP_imagenet_finetuned.json", "ImageNet fine-tuned"),
    ("KEEP_r3m_frozen.json", "R3M FROZEN"),
    ("KEEP_r3m_finetuned.json", "R3M fine-tuned"),
]

root = Path(sys.argv[1])
print(f"{'':24s} {'mean':>7s} {'median':>7s} {'p90':>7s} {'last6':>7s}   note")
for name, m, med, p90, l6, note in BASELINES:
    l6s = f"{l6:7.2f}" if l6 is not None else f"{'-':>7s}"
    print(f"{name:24s} {m:7.2f} {med:7.2f} {p90:7.2f} {l6s}   {note}")
print()

rows, hw = [], {}
for fname, label in RUNS:
    p = root / fname
    if not p.exists():
        print(f"{label:24s} NO RESULT FILE ({fname})")
        continue
    d = json.load(open(p))
    curve = [c["val_mm_mean"] for c in d["val_curve"]]
    last6 = float(np.mean(curve[-6:]))
    rows.append((label, d, last6))
    print(f"{label:24s} {d['action_error_mm_mean']:7.2f} "
          f"{d['action_error_mm_median']:7.2f} {d['action_error_mm_p90']:7.2f} "
          f"{last6:7.2f}   pod {d.get('pod_id')}")
    hw[label] = d.get("hardware", {})

print()
for label, d, _ in rows:
    curve = " ".join(f"{c['val_mm_mean']:.2f}" for c in d["val_curve"])
    print(f"{label}")
    print(f"  val curve  {curve}")
    print(f"  loss {d['val_curve'][0]['train_loss']:.4f} -> "
          f"{d['val_curve'][-1]['train_loss']:.4f}   {d['train_minutes']:.1f} min   "
          f"{d['peak_vram_gb']:.1f} GB   BN frozen {d.get('batchnorm_modules_frozen')}   "
          f"group_norm {d.get('use_group_norm')}")
    nc = d.get("normalised_batch_check")
    if nc:
        print(f"  batch reaching the backbone: mean {nc['per_channel_mean']} "
              f"std {nc['per_channel_std']}")
    r = d.get("r3m_load")
    if r:
        print(f"  R3M: {r['convnet_keys']} keys, missing {r['missing_keys']}, "
              f"unexpected {r['unexpected_keys']}, "
              f"differs from fresh init {r['differs_from_fresh_init']}")

print()
keys = ["gpu_name", "driver_version", "cuda_version", "cudnn_version",
        "torch_version", "torchvision_version"]
distinct = {k: {str(v.get(k)) for v in hw.values()} for k in keys}
mismatch = {k: v for k, v in distinct.items() if len(v) > 1}
if mismatch:
    print("HARDWARE DIFFERS BETWEEN PODS:", json.dumps({k: sorted(v) for k, v in mismatch.items()}, indent=1))
else:
    print("hardware identical across both pods:",
          {k: next(iter(v)) for k, v in distinct.items()})

print()
if rows:
    best_label, best_d, best = min(rows, key=lambda r: r[2])
    print(f"BEST: {best_label} at {best:.2f} mm (last-6 mean) against NULL 6.63 mm")
    if best < 6.63:
        print("  -> a pretrained encoder BEATS the no-camera policy. Pixels "
              "contribute. Phase 2 is Sidwyn's call.")
    else:
        print("  -> NO run beats NULL. On this dataset and this metric the "
              "encoder was never the bottleneck. Do not extend the step "
              "budget; the answer is that offline action error cannot see what "
              "a wrist view does.")
