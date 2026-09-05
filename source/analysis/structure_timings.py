"""Summarize Protoss structure timings from a handful of replays.

For each structure this reports four related times:

* order: exact build-command time from the replay event stream;
* IL grid: the action time after replay_parser serializes actions onto its 4s grid;
* complete: exact completion time from the structure object's lifetime;
* RL sees: first 4s decision boundary at or after exact completion.

Exact completion time is the best basis for an RL deadline. The other columns
make timing quantization and parser queue shifts visible rather than silently
mixing those semantics together.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sc2reader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from obs_spec import ACTION_NAMES, DECISION_INTERVAL_SECONDS, OBS_SIZE  # noqa: E402
from replay_parser import (  # noqa: E402
    BUILD_COMMAND_TO_STRUCTURE,
    ReplayParser,
    STRUCTURE_NAME_MAP,
    MIN_REPLAY_BUILD,
    calibrate_fps,
    is_command_event,
)


DEFAULT_REPLAY_DIR = Path(r"C:\dev\BetaStar\replays\raw")
BUILDABLE_STRUCTURES = tuple(dict.fromkeys(
    BUILD_COMMAND_TO_STRUCTURE.values()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show exact and 4-second-grid Protoss structure timings for a "
            "small replay sample."))
    parser.add_argument(
        "replays", nargs="*", type=Path,
        help="Specific .SC2Replay files. If omitted, sample --replay-dir.")
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--count", type=int, default=8,
                        help="Number of valid replays to report.")
    parser.add_argument("--seed", type=int, default=54,
                        help="Seed used when sampling the replay directory.")
    parser.add_argument(
        "--match", default=None,
        help="Optional case-insensitive substring required in the filename.")
    parser.add_argument(
        "--structures", default=None,
        help=("Optional comma-separated structure subset. The default is all "
              "buildable Protoss structures."))
    parser.add_argument(
        "--until", type=float, default=None,
        help="End the displayed 4s action log at this game time.")
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Hide per-window action logs and show only structure timings.")
    parser.add_argument(
        "--include-non-pvz", action="store_true",
        help="Include Protoss replays without a Zerg opponent.")
    parser.add_argument(
        "--result", choices=("all", "win", "loss"), default="all",
        help="Filter by the Protoss player's result.")
    parser.add_argument(
        "--json", type=Path, default=None,
        help="Optionally save the complete report as JSON.")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.until is not None and args.until < 0:
        parser.error("--until cannot be negative")
    return args


def _selected_structures(args: argparse.Namespace) -> tuple[str, ...]:
    if args.structures is None:
        return BUILDABLE_STRUCTURES
    selected = tuple(
        value.strip().upper()
        for value in args.structures.split(",")
        if value.strip()
    )
    unknown = sorted(set(selected) - set(BUILDABLE_STRUCTURES))
    if unknown:
        valid = ", ".join(BUILDABLE_STRUCTURES)
        raise ValueError(
            f"Unknown structure(s): {', '.join(unknown)}. Valid: {valid}")
    return selected


def _candidate_paths(args: argparse.Namespace) -> list[Path]:
    if args.replays:
        paths = args.replays
    else:
        paths = sorted(args.replay_dir.glob("*.SC2Replay"))
        random.Random(args.seed).shuffle(paths)
    if args.match:
        needle = args.match.casefold()
        paths = [path for path in paths if needle in path.name.casefold()]
    return paths


def _type_name_at(unit, frame: int) -> str | None:
    """Resolve an object's type at a frame, before later morphs rename it."""
    history = getattr(unit, "type_history", None)
    if not history:
        return getattr(unit, "name", None)
    resolved = None
    items = sorted(history.items())
    for history_frame, unit_type in items:
        if history_frame > frame:
            break
        resolved = getattr(unit_type, "name", None)
    return resolved or getattr(items[0][1], "name", None)


def _first_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def _next_policy_grid(second: float | None) -> float | None:
    if second is None:
        return None
    grid = DECISION_INTERVAL_SECONDS
    return float(math.ceil(second / grid) * grid)


def analyze_replay(
    path: Path,
    structures: tuple[str, ...],
    include_non_pvz: bool,
    until_seconds: float | None,
) -> dict | None:
    replay = sc2reader.load_replay(str(path), load_level=4)
    protoss = [p for p in replay.players if p.play_race == "Protoss"]
    has_zerg = any(p.play_race == "Zerg" for p in replay.players)
    if (getattr(replay, "build", 0) < MIN_REPLAY_BUILD
            or not all(p.is_human for p in replay.players)
            or not protoss
            or (not include_non_pvz and not has_zerg)):
        return None

    player = protoss[0]
    pid = player.pid
    fps = calibrate_fps(replay)

    orders: dict[str, list[float]] = defaultdict(list)
    for event in replay.events:
        if not is_command_event(event):
            continue
        owner = getattr(event, "player", None)
        if owner is None or owner.pid != pid:
            continue
        structure = BUILD_COMMAND_TO_STRUCTURE.get(event.ability_name)
        if structure in structures:
            frame = getattr(event, "frame", None)
            exact_second = (
                float(frame) / fps if frame is not None
                else float(event.second))
            orders[structure].append(exact_second)

    completions: dict[str, list[float]] = defaultdict(list)
    objects = getattr(replay, "objects", None) or {}
    for unit in objects.values():
        owner = getattr(unit, "owner", None)
        if owner is None or getattr(owner, "pid", None) != pid:
            continue
        started = getattr(unit, "started_at", None)
        finished = getattr(unit, "finished_at", None)
        # Exclude starting structures and cancelled/incomplete constructions.
        if started is None or started <= 0 or finished is None:
            continue
        raw_name = _type_name_at(unit, int(started))
        structure = STRUCTURE_NAME_MAP.get(raw_name)
        if structure in structures:
            completions[structure].append(float(finished) / fps)

    # Parse through the real training path so this log includes collision
    # queueing, legality demotions, and do_nothing exactly as dataset.npz does.
    replay_parser = ReplayParser(debug=False)
    sequence = replay_parser.parse_replay(replay)
    if sequence is None:
        return None
    action_ids = sequence[:, OBS_SIZE].astype(int)
    full_action_log = [
        {
            "time_seconds": window * DECISION_INTERVAL_SECONDS,
            "action_id": int(action_id),
            "action": ACTION_NAMES[int(action_id)],
        }
        for window, action_id in enumerate(action_ids)
    ]
    if until_seconds is not None:
        final_window = int(until_seconds / DECISION_INTERVAL_SECONDS)
        action_log = full_action_log[:final_window + 1]
    else:
        action_log = full_action_log

    action_to_structure = {
        replay_parser.EVENT_TO_ACTION[ability]: structure
        for ability, structure in BUILD_COMMAND_TO_STRUCTURE.items()
        if ability in replay_parser.EVENT_TO_ACTION
    }
    grid_orders: dict[str, list[float]] = defaultdict(list)
    for entry in full_action_log:
        structure = action_to_structure.get(entry["action_id"])
        if structure in structures:
            grid_orders[structure].append(
                float(entry["time_seconds"]))

    timings = {}
    for structure in structures:
        order = _first_or_none(orders[structure])
        grid_order = _first_or_none(grid_orders[structure])
        complete = _first_or_none(completions[structure])
        timings[structure] = {
            "order_seconds": order,
            "il_grid_seconds": grid_order,
            "complete_seconds": complete,
            "rl_observable_seconds": _next_policy_grid(complete),
        }

    opponents = [
        f"{p.name} ({p.play_race})" for p in replay.players if p.pid != pid
    ]
    return {
        "file": path.name,
        "path": str(path.resolve()),
        "map": replay.map_name,
        "protoss_player": player.name,
        "protoss_result": str(getattr(player, "result", "Unknown")),
        "opponents": opponents,
        "action_log": action_log,
        "timings": timings,
    }


def _format_time(value: float | None) -> str:
    if value is None:
        return "-"
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes}:{seconds:04.1f}"


def _print_replay(report: dict, index: int, summary_only: bool) -> None:
    opponent = ", ".join(report["opponents"])
    print(f"\n[{index}] {report['file']}")
    print(f"    {report['protoss_player']} vs {opponent} | "
          f"{report['map']} | {report['protoss_result']}")
    if not summary_only:
        print("\n    4-SECOND ACTION LOG (exact IL training labels)")
        for entry in report["action_log"]:
            print(f"    {entry['time_seconds']:>4}s: {entry['action']}")

    print("\n    FIRST STRUCTURE TIMINGS")
    print(f"    {'Structure':<22}{'Order':>9}{'IL grid':>10}"
          f"{'Complete':>11}{'RL sees':>10}")
    print(f"    {'-' * 62}")
    for structure, timing in report["timings"].items():
        print(
            f"    {structure:<22}"
            f"{_format_time(timing['order_seconds']):>9}"
            f"{_format_time(timing['il_grid_seconds']):>10}"
            f"{_format_time(timing['complete_seconds']):>11}"
            f"{_format_time(timing['rl_observable_seconds']):>10}")


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    p25, p50, p75, p90 = np.percentile(values, [25, 50, 75, 90])
    return {
        "p25": round(float(p25), 2),
        "p50": round(float(p50), 2),
        "p75": round(float(p75), 2),
        "p90": round(float(p90), 2),
    }


def build_aggregate(reports: list[dict], structures: tuple[str, ...]) -> dict:
    aggregate = {}
    for structure in structures:
        order_values = []
        completion_values = []
        for report in reports:
            timing = report["timings"][structure]
            if timing["order_seconds"] is not None:
                order_values.append(timing["order_seconds"])
            if timing["complete_seconds"] is not None:
                completion_values.append(timing["complete_seconds"])
        completion_stats = _percentiles(completion_values)
        aggregate[structure] = {
            "replays_with_order": len(order_values),
            "replays_with_completion": len(completion_values),
            "order_seconds": _percentiles(order_values),
            "completion_seconds": completion_stats,
            "p75_completion_grid_seconds": (
                _next_policy_grid(completion_stats["p75"])
                if completion_stats else None),
            "p90_completion_grid_seconds": (
                _next_policy_grid(completion_stats["p90"])
                if completion_stats else None),
        }
    return aggregate


def _print_aggregate(aggregate: dict, replay_count: int) -> None:
    print(f"\n{'=' * 88}")
    print(f"CONSENSUS ACROSS {replay_count} REPLAYS (exact replay seconds)")
    print(f"{'=' * 88}")
    print(f"{'Structure':<22}{'N':>4}  {'Order p50':>10} {'Order p90':>10}  "
          f"{'Done p50':>10} {'Done p75':>10} {'Done p90':>10}")
    print("-" * 88)
    for structure, stats in aggregate.items():
        order = stats["order_seconds"] or {}
        done = stats["completion_seconds"] or {}
        print(
            f"{structure:<22}{stats['replays_with_completion']:>4}  "
            f"{_format_time(order.get('p50')):>10}"
            f" {_format_time(order.get('p90')):>10}  "
            f"{_format_time(done.get('p50')):>10}"
            f" {_format_time(done.get('p75')):>10}"
            f" {_format_time(done.get('p90')):>10}")
    print("\nFor an RL completion deadline, use the completion columns. P75 means"
          " 75% of this sample completed by then; P90 is a looser target.")
    print(f"Round a deadline to the {DECISION_INTERVAL_SECONDS}s policy grid; "
          "the JSON report includes both rounded P75 and P90 values.")


def main() -> None:
    # Some replay filenames contain characters unavailable in Windows' legacy
    # console encoding. Replacing only unprintable glyphs keeps analysis alive.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    args = parse_args()
    try:
        structures = _selected_structures(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    candidates = _candidate_paths(args)
    if not candidates:
        raise SystemExit("No replay files matched the requested input.")

    reports = []
    skipped = []
    for path in candidates:
        if len(reports) >= args.count:
            break
        try:
            report = analyze_replay(
                path, structures,
                include_non_pvz=args.include_non_pvz,
                until_seconds=args.until)
        except Exception as exc:  # report a bad replay without losing the sample
            skipped.append({"file": path.name, "reason": str(exc)})
            continue
        if report is None:
            skipped.append({
                "file": path.name,
                "reason": "outside supported human PvZ corpus",
            })
            continue
        if (args.result != "all"
                and report["protoss_result"].casefold() != args.result):
            skipped.append({
                "file": path.name,
                "reason": f"Protoss result is {report['protoss_result']}",
            })
            continue
        reports.append(report)
        _print_replay(report, len(reports), args.summary_only)

    if not reports:
        raise SystemExit("No valid Protoss-vs-Zerg replays were found.")

    aggregate = build_aggregate(reports, structures)
    _print_aggregate(aggregate, len(reports))
    if skipped:
        print(f"\nSkipped {len(skipped)} candidate replay(s).")

    if args.json is not None:
        payload = {
            "grid_interval_seconds": DECISION_INTERVAL_SECONDS,
            "structures": structures,
            "replays": reports,
            "aggregate": aggregate,
            "skipped": skipped,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON report: {args.json.resolve()}")


if __name__ == "__main__":
    main()
