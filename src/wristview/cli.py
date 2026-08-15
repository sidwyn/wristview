"""Command line entry point.

    wristview run --scan scan.mov --demos d0.mov d1.mov --out runs/

One command over a scan video plus demo videos, producing rendered robot
wrist-camera views.

This module is the only place that sequences stages. Each stage reads files
and writes files, and no stage calls another. That rule is what lets any one
stage be re-run, swapped, or replaced without touching the others:

    wristview stage 5 --run runs/2026-08-14_room01     # re-render only
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import wristview  # noqa: F401 - sets KMP_DUPLICATE_LIB_OK before torch loads

from .config import Config
from .logging_setup import get, setup
from .runctx import RunContext, read_json, write_json

log = get(__name__)

STAGE_MODULES = {
    0: "s00_ingest",
    1: "s01_scene",
    2: "s02_localize",
    3: "s03_estimate",
    4: "s04_retarget",
    5: "s05_render",
}
STAGE_NAMES = {
    0: "ingest",
    1: "scene",
    2: "localize",
    3: "estimate",
    4: "retarget",
    5: "render",
}
# Stages 6 and 7 are deliberately absent. The build plan stops at the first
# rendered wrist view, and says to get buyer judgment before building export.
LAST_STAGE = 5


def _load_stage(index: int):
    from importlib import import_module

    return import_module(f".stages.{STAGE_MODULES[index]}", package="wristview")


def _resolve_arkit(scan: Path, demos: list[Path], explicit: list[str] | None) -> dict[str, str]:
    """Find ARKit trajectory JSON files.

    Explicit paths win. Otherwise look for `arkit_<clip>.json` beside the
    videos, which is what the fixture generator writes.
    """
    found: dict[str, str] = {}
    if explicit:
        for entry in explicit:
            if "=" in entry:
                clip, path = entry.split("=", 1)
                found[clip] = str(Path(path).resolve())
            else:
                found["scan"] = str(Path(entry).resolve())
        return found

    candidates = {"scan": scan}
    for index, demo in enumerate(demos):
        candidates[f"demo_{index}"] = demo
    for clip_id, video in candidates.items():
        for candidate in (
            video.parent / f"arkit_{clip_id}.json",
            video.with_suffix(".arkit.json"),
        ):
            if candidate.exists():
                found[clip_id] = str(candidate.resolve())
                break
    return found


def _run_stages(ctx: RunContext, stages: list[int], keep_going: bool) -> dict:
    """Drive the stages in order. The only place chaining happens."""
    results: dict[int, dict] = {}
    for index in stages:
        banner = f"===== Stage {index} · {STAGE_NAMES[index]} ====="
        log.info("")
        log.info(banner)
        started = time.perf_counter()
        module = _load_stage(index)
        try:
            results[index] = {
                "status": "ok",
                "result": module.run(ctx),
                "seconds": round(time.perf_counter() - started, 1),
            }
            log.info(
                "Stage %d finished in %.1fs, artifacts in %s",
                index, results[index]["seconds"], ctx.rel(ctx.stage_dir(index)),
            )
        except Exception as exc:  # noqa: BLE001 - the driver reports, it does not hide
            elapsed = round(time.perf_counter() - started, 1)
            results[index] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": elapsed,
            }
            log.error("Stage %d FAILED after %.1fs: %s", index, elapsed, exc)
            log.debug("%s", traceback.format_exc())
            if not keep_going:
                log.error(
                    "stopping. Stage %d wrote meta.json with status failed at %s",
                    index, ctx.rel(ctx.stage_dir(index) / "meta.json"),
                )
                break
            log.warning("--keep-going is set; continuing to the next stage")
    return results


def _summarize(ctx: RunContext, results: dict) -> None:
    """Print what was produced and where. This is the log read in the morning."""
    log.info("")
    log.info("===== Run summary =====")
    log.info("run directory: %s", ctx.root)

    for index in sorted(results):
        entry = results[index]
        marker = "ok    " if entry["status"] == "ok" else "FAILED"
        log.info("  stage %d %-9s %s  %6.1fs", index, STAGE_NAMES[index], marker, entry["seconds"])
        if entry["status"] == "failed":
            log.info("           %s", entry.get("error", ""))

    log.info("")
    log.info("artifacts:")
    for index in sorted(results):
        meta_path = ctx.stage_dir(index, create=False) / "meta.json"
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        for key, value in meta.get("outputs", {}).items():
            log.info("  %-28s %s", f"{STAGE_NAMES[index]}.{key}", value)

    payload = {
        "run_id": ctx.run_id,
        "root": str(ctx.root),
        "git_sha": ctx.git_sha,
        "config_hash": ctx.config.hash,
        "platform": ctx.platform,
        "stages": {
            str(k): {"status": v["status"], "seconds": v["seconds"], "error": v.get("error")}
            for k, v in results.items()
        },
    }
    write_json(ctx.root / "run_summary.json", payload)
    log.info("")
    log.info("summary written to %s", ctx.root / "run_summary.json")


def command_run(args: argparse.Namespace) -> int:
    scan = Path(args.scan).resolve()
    demos = [Path(p).resolve() for p in args.demos]
    if not scan.exists():
        print(f"scan video not found: {scan}", file=sys.stderr)
        return 2
    missing = [str(p) for p in demos if not p.exists()]
    if missing:
        print(f"demo videos not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    overrides: dict = {}
    if args.set:
        overrides = _parse_overrides(args.set)
    config = Config.load(args.config, overrides)

    run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M%S}_{scan.stem}"
    ctx = RunContext.create(args.out, run_id, config)
    setup(ctx.root / "wristview.log", verbose=args.verbose)

    log.info("wristview run %s", run_id)
    log.info("  scan:   %s", scan)
    for index, demo in enumerate(demos):
        log.info("  demo_%d: %s", index, demo)
    log.info("  config hash %s, git %s", config.hash, ctx.git_sha)
    log.info("  platform %s", ctx.platform)

    arkit = _resolve_arkit(scan, demos, args.arkit)
    if arkit:
        for clip_id, path in sorted(arkit.items()):
            log.info("  arkit %s: %s", clip_id, path)
    else:
        log.warning("  no ARKit trajectories found")

    write_json(
        ctx.root / "sources.json",
        {
            "scan": str(scan),
            "demos": [str(p) for p in demos],
            "arkit": arkit,
            "instruction": args.instruction,
        },
    )

    stages = _parse_stage_range(args.stages)
    results = _run_stages(ctx, stages, keep_going=args.keep_going)
    _summarize(ctx, results)

    failed = [k for k, v in results.items() if v["status"] == "failed"]
    return 1 if failed else 0


def command_stage(args: argparse.Namespace) -> int:
    """Re-run one stage, or a range, over an existing run directory."""
    root = Path(args.run).resolve()
    if not root.exists():
        print(f"run directory not found: {root}", file=sys.stderr)
        return 2

    config_path = root / "config.yaml"
    overrides = _parse_overrides(args.set) if args.set else {}
    config = Config.load(config_path if config_path.exists() else None, overrides)
    ctx = RunContext(root=root, config=config, run_id=root.name)
    setup(root / "wristview.log", verbose=args.verbose)

    stages = _parse_stage_range(args.stages)
    log.info("re-running stages %s in %s", stages, root)
    results = _run_stages(ctx, stages, keep_going=args.keep_going)
    _summarize(ctx, results)
    return 1 if any(v["status"] == "failed" for v in results.values()) else 0


def _parse_stage_range(spec: str) -> list[int]:
    """Accept `0-5`, `3`, or `1,3,5`."""
    spec = spec.strip()
    if "," in spec:
        stages = sorted({int(part) for part in spec.split(",") if part.strip()})
    elif "-" in spec:
        low, high = spec.split("-", 1)
        stages = list(range(int(low), int(high) + 1))
    else:
        stages = [int(spec)]

    for index in stages:
        if index not in STAGE_MODULES:
            raise SystemExit(
                f"stage {index} does not exist. This build stops at stage {LAST_STAGE}: "
                f"export and QC are deliberately not built."
            )
    return stages


def _parse_overrides(pairs: list[str]) -> dict:
    """Turn `--set render.width=800` into a nested override dict."""
    import yaml

    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set needs key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def command_preflight(args: argparse.Namespace) -> int:
    """Answer one question: will these two clips localize?"""
    import logging

    from .device import resolve as resolve_device
    from .preflight import PASS_RATIO, run_preflight

    scan = Path(args.scan).resolve()
    demo = Path(args.demo).resolve()
    for path in (scan, demo):
        if not path.exists():
            print(f"not found: {path}", file=sys.stderr)
            return 2

    # The whole point is a number and a verdict, so silence everything else
    # unless the caller asks for the working.
    setup(None, verbose=False)
    if not args.verbose:
        logging.getLogger().setLevel(logging.ERROR)

    device = resolve_device("auto", True)
    try:
        result = run_preflight(
            scan, demo, device,
            scan_frames=args.scan_frames,
            demo_frames=args.demo_frames,
            max_keypoints=args.max_keypoints,
        )
    except Exception as exc:  # noqa: BLE001 - a one-line answer, even on failure
        print(f"FAIL  could not measure: {exc}")
        return 1

    print(f"scan self-match   {result.reference_matches:6.0f} features   (the ceiling this footage supports)")
    print(f"demo to scan      {result.demo_matches:6.0f} features   (best scan frame, median over {result.demo_frames} demo frames)")
    print(f"ratio             {result.ratio:6.2f}              (pass needs {PASS_RATIO:.2f})")
    print()
    print(f"{result.verdict}")
    if result.verdict != "PASS":
        print()
        print("The scan does not cover the demo's viewpoint. After the wide orbit,")
        print("scan the working area again from the demo camera's height and framing.")
    return 0 if result.passed else 1


def command_marker(args: argparse.Namespace) -> int:
    """Write a printable ArUco marker at an exact physical size."""
    from .marker import generate, verify

    setup(None, verbose=False)
    path = generate(
        Path(args.out), side_m=args.side_cm / 100.0, marker_id=args.id,
        dictionary_name=args.dictionary,
    )
    found = verify(path, args.dictionary)
    if args.id not in found:
        print(f"generated {path} but detection failed, found {found}", file=sys.stderr)
        return 1

    print(f"wrote {path}")
    print(f"  marker {args.id} of {args.dictionary}, black square {args.side_cm:.1f} cm")
    print()
    print("  1. print at 100 percent, no fit-to-page")
    print("  2. measure the printed line and confirm it is "
          f"{args.side_cm:.1f} cm")
    print("  3. lie it flat in the scene, in view for much of the scan")
    print(f"  4. set scene.scale.aruco_marker_length_m to {args.side_cm / 100:.4f}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    """Score a run against synthetic ground truth."""
    from .evaluate import evaluate_run, format_reports

    setup(None, verbose=args.verbose)
    run_root = Path(args.run).resolve()
    truth = Path(args.groundtruth).resolve()
    for path in (run_root, truth):
        if not path.exists():
            print(f"not found: {path}", file=sys.stderr)
            return 2

    reports = evaluate_run(run_root, truth)
    if not reports:
        print("no episodes to evaluate", file=sys.stderr)
        return 1
    print(format_reports(reports))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wristview",
        description="Egocentric head-mounted video in, robot wrist-camera views out.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the pipeline end to end")
    run_parser.add_argument("--scan", required=True, help="room scan video")
    run_parser.add_argument("--demos", required=True, nargs="+", help="demo videos")
    run_parser.add_argument("--out", default="runs", help="directory to hold run folders")
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--config", default=None, help="YAML config layered over the default")
    run_parser.add_argument("--stages", default=f"0-{LAST_STAGE}", help="for example 0-5 or 2,3")
    run_parser.add_argument(
        "--arkit", nargs="*", default=None,
        help="ARKit trajectory JSON, as PATH or clip=PATH. Auto-discovered when omitted.",
    )
    run_parser.add_argument(
        "--instruction", default="pick up the object and place it down",
        help="language instruction for the task, used to prompt object detection",
    )
    run_parser.add_argument("--keep-going", action="store_true",
                            help="continue to later stages after a failure")
    run_parser.add_argument("--set", nargs="*", default=None, help="config override, key=value")
    run_parser.add_argument("-v", "--verbose", action="store_true")
    run_parser.set_defaults(func=command_run)

    stage_parser = subparsers.add_parser("stage", help="re-run stages over an existing run")
    stage_parser.add_argument("stages", help="for example 5, or 3-5")
    stage_parser.add_argument("--run", required=True, help="existing run directory")
    stage_parser.add_argument("--keep-going", action="store_true")
    stage_parser.add_argument("--set", nargs="*", default=None)
    stage_parser.add_argument("-v", "--verbose", action="store_true")
    stage_parser.set_defaults(func=command_stage)

    pre_parser = subparsers.add_parser(
        "preflight", help="will a scan and a demo localize? answers in under a minute"
    )
    pre_parser.add_argument("--scan", required=True, help="room or workspace scan video")
    pre_parser.add_argument("--demo", required=True, help="one demo clip")
    pre_parser.add_argument("--scan-frames", type=int, default=30)
    pre_parser.add_argument("--demo-frames", type=int, default=6)
    pre_parser.add_argument("--max-keypoints", type=int, default=1024)
    pre_parser.add_argument("-v", "--verbose", action="store_true")
    pre_parser.set_defaults(func=command_preflight)

    marker_parser = subparsers.add_parser(
        "marker", help="write a printable ArUco marker for metric scale"
    )
    marker_parser.add_argument("--out", default="aruco_marker.png")
    marker_parser.add_argument("--side-cm", type=float, default=15.0,
                               help="printed side of the black square, in cm")
    marker_parser.add_argument("--id", type=int, default=0)
    marker_parser.add_argument("--dictionary", default="DICT_4X4_50")
    marker_parser.set_defaults(func=command_marker)

    eval_parser = subparsers.add_parser(
        "evaluate", help="score a run against synthetic ground truth"
    )
    eval_parser.add_argument("--run", required=True)
    eval_parser.add_argument("--groundtruth", required=True)
    eval_parser.add_argument("-v", "--verbose", action="store_true")
    eval_parser.set_defaults(func=command_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "stage":
        args.stages = args.stages
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
