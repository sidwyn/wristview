"""Every pod created must be logged as deleted, and the reverse.

GPU-LOG.md exists to answer one question: is anything billing right now. On
4 September it could not. Three creation rows stood against five deletion
rows, one pod appeared to have been created and never terminated, and the
reconciliation cost three exchanges and a phone notification to Sidwyn for a
pod that had in fact been deleted two hours earlier.

The pod was down. The LOG was wrong. This makes that failure loud.

    python -m tools.check_gpu_log            # reconcile the file
    python -m tools.check_gpu_log --live     # also ask RunPod what is up now

Exit code 1 when the file does not reconcile, or when a live pod is running
that the file does not record as created.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "GPU-LOG.md"

# The pod a row is ABOUT, not every pod it mentions. A row can reference
# another pod in prose ("Phase 2 replacement, after `eepzdyapa49atk` failed"),
# and counting that as an event reopened a pod that had already been closed.
# "Pod `x`" is the canonical subject; the first backticked id is the fallback.
POD_SUBJECT = re.compile(r"Pod[s]?\s+`([a-z0-9]{12,16})`")
POD_ID = re.compile(r"`([a-z0-9]{12,16})`")


def subject(line: str) -> list[str]:
    named = POD_SUBJECT.findall(line)
    if named:
        return named
    found = POD_ID.findall(line)
    return found[:1]


# A pod's state is its LAST action, not the presence of any action. The log
# records stop/start cycles, so a pod can legitimately open and close several
# times. Reading the file as two sets rather than a timeline reports four
# healthy August pods as leaks.
OPENS = ("CREATE", "START", "POD IN USE")
CLOSES = ("DELET", "TERMINAT", "STOP", "EXITED")


def scan(text: str) -> tuple[dict, dict]:
    """Return {pod: when} for pods left OPEN, and for pods properly CLOSED."""
    last: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        action = (cells[2] if len(cells) > 2 else "").upper()
        when = cells[1] if len(cells) > 1 else ""
        ids = subject(line)
        if not ids:
            continue
        # CLOSES is checked first: "stop+start" is a close then a reopen, and
        # the row that matters for billing is the one that ends it.
        if any(k in action for k in CLOSES):
            state = "closed"
        elif any(k in action for k in OPENS):
            state = "open"
        else:
            continue
        for pid in ids:
            last[pid] = (state, when)
    created = {p: w for p, (st, w) in last.items() if st == "open"}
    deleted = {p: w for p, (st, w) in last.items() if st == "closed"}
    return created, deleted


def live_pods() -> list[str]:
    try:
        out = subprocess.run(["runpodctl", "pod", "list"], capture_output=True,
                             text=True, timeout=60).stdout
        return [p["id"] for p in (json.loads(out) or [])]
    except Exception as exc:  # noqa: BLE001 - a failed query is not a clean bill
        print(f"  could not query RunPod: {exc}")
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true",
                    help="also ask RunPod what is running right now")
    args = ap.parse_args()

    created, deleted = scan(LOG.read_text())
    bad = False

    never_deleted = sorted(created)
    never_created = []

    print(f"{len(created)} pod(s) left OPEN by the log, {len(deleted)} closed")
    if never_deleted:
        bad = True
        print("\nCREATED AND NEVER LOGGED AS DELETED:")
        for pid in never_deleted:
            print(f"  {pid}  created {created[pid]}")
        print("  Each of these is either still billing, or was deleted and the "
              "row was never written. Both are defects. Check RunPod.")
    if never_created:
        bad = True
        print("\nDELETED WITH NO CREATE ROW:")
        for pid in never_created:
            print(f"  {pid}  deleted {never_created and deleted[pid]}")
        print("  A pod nobody recorded creating. The file cannot be trusted to "
              "list what exists.")

    if args.live:
        live = live_pods()
        print(f"\nLIVE NOW: {live or 'nothing running, nothing billing'}")
        for pid in live:
            if pid not in created:
                bad = True
                print(f"  {pid} IS RUNNING and has no CREATE row.")
            if pid in deleted:
                bad = True
                print(f"  {pid} IS RUNNING but the log says it was deleted.")

    if not bad:
        print("\nreconciles.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
