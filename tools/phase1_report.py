"""Read the Phase 1 results and apply the gate exactly as PHASE1-PROTOCOL says.

The reported figure is the MEAN OF THE LAST 6 VALIDATION CHECKS, not the final
one. `still_falling_at_end` is not read: it looks at four endpoints and on
real31 it fired on a 0.004 mm margin.
"""
import json
import sys
from pathlib import Path

import numpy as np

results = Path(sys.argv[1])
floors = json.load(open(sys.argv[2]))
BAR = floors["FLOOR_MEAN"]["action_error_mm_mean"]
PERSIST = floors["FLOOR_PERSISTENCE"]["action_error_mm_mean"]

print(f"holdout {floors['holdout']}")
print(f"scored deltas {floors['scored_action_deltas']}, "
      f"{floors['test_rows']} held-out rows\n")
print(f"{'':16s} {'mean':>8s} {'median':>8s} {'p90':>8s}   {'last6 mean':>10s}")
print(f"{'FLOOR-MEAN':16s} {BAR:8.2f} "
      f"{floors['FLOOR_MEAN']['action_error_mm_median']:8.2f} "
      f"{floors['FLOOR_MEAN']['action_error_mm_p90']:8.2f}   {'-':>10s}   <- THE BAR")
print(f"{'FLOOR-PERSIST':16s} {PERSIST:8.2f} "
      f"{floors['FLOOR_PERSISTENCE']['action_error_mm_median']:8.2f} "
      f"{floors['FLOOR_PERSISTENCE']['action_error_mm_p90']:8.2f}   {'-':>10s}   "
      f"reference, not a gate")

verdicts = {}
for arm in ("NULL", "A_prime"):
    path = results / f"{arm}_seed1.json"
    if not path.exists():
        print(f"{arm:16s} NO RESULT FILE")
        verdicts[arm] = None
        continue
    d = json.load(open(path))
    curve = [c["val_mm_mean"] for c in d["val_curve"]]
    last6 = float(np.mean(curve[-6:]))
    verdicts[arm] = last6
    print(f"{arm:16s} {d['action_error_mm_mean']:8.2f} "
          f"{d['action_error_mm_median']:8.2f} {d['action_error_mm_p90']:8.2f}   "
          f"{last6:10.2f}")

print()
for arm in ("NULL", "A_prime"):
    path = results / f"{arm}_seed1.json"
    if not path.exists():
        continue
    d = json.load(open(path))
    curve = [f"{c['val_mm_mean']:.2f}" for c in d["val_curve"]]
    print(f"{arm} val curve (every 250 steps): {' '.join(curve)}")
    print(f"  train loss {d['val_curve'][0]['train_loss']:.4f} -> "
          f"{d['val_curve'][-1]['train_loss']:.4f}   "
          f"{d['train_minutes']:.1f} min, {d['peak_vram_gb']:.1f} GB, "
          f"lerobot {d.get('lerobot_version')}, torch {d.get('torch_version')}")
    print(f"  scored deltas {d.get('scored_action_deltas')}")

print()
a = verdicts.get("A_prime")
if a is None:
    print("GATE: CANNOT BE READ. A_prime produced no result.")
else:
    beat = a < BAR
    print(f"GATE: A_prime last-6 mean {a:.2f} mm against FLOOR-MEAN {BAR:.2f} mm "
          f"-> {'PASS' if beat else 'FAIL'}")
    if beat and a > PERSIST:
        print(f"      but it loses to FLOOR-PERSISTENCE at {PERSIST:.2f} mm, "
              f"which on this holdout is the better trivial predictor on every "
              f"statistic. Reported, not acted on.")
    if not beat:
        print("      STOP. No B_prime, no C, no further seeds, and no change to "
              "steps, config or holdout.")
