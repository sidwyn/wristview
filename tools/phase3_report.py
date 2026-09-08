"""Phase 3: the seed table, the paired table, and the decision words.

The rule is CC-PHASE-3-SEEDS.md's, committed before any number existed, and it
is implemented here rather than applied by eye:

    a difference is ESTABLISHED when |mean| clears 2 x SE
    AND its sign is the same in at least 4 of 5 seeds.
    Otherwise it is NOT ESTABLISHED. There is no third word.

The score of a run is the mean of its last 6 validation checks. Never the
minimum: the 10 held-out episodes are the only test set there is, so picking
the best checkpoint by held-out error would be fitting to it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ARMS = ["A_prime", "B_prime", "C"]
FLOOR_MEAN, FLOOR_PERSISTENCE = 6.14, 3.72


def load(root: str) -> dict:
    """{arm: {seed: record}}. A run is scored only if it reached 10k steps."""
    out: dict[str, dict[int, dict]] = {a: {} for a in ARMS}
    for f in sorted(glob.glob(os.path.join(root, "seed*_*.json"))):
        d = json.load(open(f))
        arm, seed = d["arm"], int(d["seed"])
        curve = [x["val_mm_mean"] for x in d["val_curve"]]
        steps = [x["step"] for x in d["val_curve"]]
        complete = bool(steps) and steps[-1] >= d["steps"]
        half = curve[len(curve) // 2:]
        out[arm][seed] = {
            "score": float(np.mean(curve[-6:])) if complete else None,
            "final": curve[-1], "min": min(curve),
            "min_step": steps[int(np.argmin(curve))],
            "slope": float(np.polyfit(np.arange(len(half)), half, 1)[0]),
            "settle": next((s for s, v in zip(steps, curve)
                            if abs(v - curve[-1]) <= 0.1), None),
            "curve": curve, "steps": steps, "complete": complete,
            "last_step": steps[-1] if steps else 0,
            "minutes": d.get("train_minutes"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="runs/sept02_final/07_train/results_p3")
    args = ap.parse_args()
    R = load(args.results)

    seeds = sorted({s for a in ARMS for s in R[a]})
    print("SEED TABLE. score = mean of the last 6 validation checks.\n")
    print(f"{'arm':9s}" + "".join(f"{'s%d' % s:>8s}" for s in seeds) + f"{'mean':>8s}{'sd':>7s}")
    stats = {}
    for a in ARMS:
        cells = []
        vals = []
        for s in seeds:
            r = R[a].get(s)
            if r is None:
                cells.append(f"{'-':>8s}")
            elif not r["complete"]:
                cells.append(f"{'INCOMP':>8s}")
            else:
                cells.append(f"{r['score']:8.2f}"); vals.append(r["score"])
        v = np.array(vals)
        stats[a] = v
        m = f"{v.mean():8.2f}" if len(v) else f"{'-':>8s}"
        sd = f"{v.std(ddof=1):7.3f}" if len(v) > 1 else f"{'-':>7s}"
        print(f"{a:9s}" + "".join(cells) + m + sd)

    for a in ARMS:
        n = len(stats[a])
        if n < 3:
            print(f"\n{a}: only {n} complete seeds. Its paired table is VOID.")

    print("\n\nPAIRED DIFFERENCES, per seed. These cancel the shared dice.\n")
    print(f"{'pair':10s}" + "".join(f"{'s%d' % s:>8s}" for s in seeds)
          + f"{'mean':>8s}{'sd':>7s}{'SE':>7s}{'2xSE':>7s}{'signs':>7s}  verdict")
    for x, y in (("A_prime", "C"), ("A_prime", "B_prime"), ("C", "B_prime")):
        diffs, cells = [], []
        for s in seeds:
            rx, ry = R[x].get(s), R[y].get(s)
            if rx and ry and rx["complete"] and ry["complete"]:
                d = rx["score"] - ry["score"]; diffs.append(d); cells.append(f"{d:+8.3f}")
            else:
                cells.append(f"{'-':>8s}")
        if len(diffs) < 3:
            print(f"{x[:1]}'-{y[:1]:8s}" + "".join(cells) + "   VOID, fewer than 3 seeds")
            continue
        v = np.array(diffs); sd = v.std(ddof=1); se = sd / np.sqrt(len(v))
        pos = int((v > 0).sum()); neg = int((v < 0).sum())
        agree = max(pos, neg)
        established = abs(v.mean()) > 2 * se and agree >= 4
        short = {"A_prime": "A'", "B_prime": "B'", "C": "C"}
        label = short[x] + " - " + short[y]
        print(f"{label:10s}" + "".join(cells)
              + f"{v.mean():+8.3f}{sd:7.3f}{se:7.3f}{2*se:7.3f}{agree:5d}/{len(v)}"
              + ("  ESTABLISHED" if established else "  NOT ESTABLISHED"))

    print(f"\n\nFLOOR-PERSISTENCE {FLOOR_PERSISTENCE}   FLOOR-MEAN {FLOOR_MEAN}\n")
    for a in ARMS:
        if not len(stats[a]):
            continue
        v = stats[a]
        below_p = int((v < FLOOR_PERSISTENCE).sum())
        print(f"  {a:9s} mean {v.mean():.2f}  "
              f"vs persistence {v.mean()-FLOOR_PERSISTENCE:+.2f} ({below_p}/{len(v)} seeds below)  "
              f"vs floor-mean {v.mean()-FLOOR_MEAN:+.2f}")

    print("\n\nDIAGNOSTICS. min is NOT the score, it shows how far the curve moved after settling.\n")
    print(f"{'arm':9s}{'seed':>5s}{'score':>8s}{'final':>8s}{'min':>8s}{'@step':>7s}{'slope':>8s}{'settle':>8s}")
    for a in ARMS:
        for s in seeds:
            r = R[a].get(s)
            if not r:
                continue
            sc = f"{r['score']:8.2f}" if r["complete"] else f"{'INCOMP':>8s}"
            print(f"{a:9s}{s:5d}{sc}{r['final']:8.2f}{r['min']:8.2f}"
                  f"{r['min_step']:7d}{r['slope']:+8.3f}{str(r['settle']):>8s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
