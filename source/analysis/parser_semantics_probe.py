"""
parser_semantics_probe.py — Confirm two sc2reader semantics before rewriting the parser
=======================================================================================
The divergence audit proved the parser's state reconstruction is wrong. Before
rewriting it, two mechanisms need confirming on YOUR sc2reader version:

Q1. MORPH NAMING. UnitDoneEvent for a Gateway that later becomes a Warpgate
    appears to report unit.name == "WarpGate" (the FINAL type), which is why the
    audit found GATEWAY dones=14 vs WARPGATE dones=343. If sc2reader exposes a
    per-frame type history, we can recover the type as of the event frame and fix
    the gateway/warpgate features exactly. This prints whichever of
    type_history / _type_class / type is available, plus a concrete Gateway
    timeline.

Q2. STARTING UNITS. The parser used to hardcode counts["NEXUS"]=1 and
    counts["PROBE"]=12 while sc2reader ALSO emits UnitBornEvent for those same
    starting units at t=0, so every game began at 2 Nexuses and 24 Probes. State
    now comes from object lifetimes; this verifies the counts are correct.

Also reports, for one replay: how many units are born per production command
(the label-undercount ratio), broken down by unit type.

Usage:
    python parser_semantics_probe.py                      # first replay found
    python parser_semantics_probe.py <path-to-replay>
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import sc2reader
from sc2reader.events import (
    UnitBornEvent, UnitDoneEvent, UnitDiedEvent,
    BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent,
)

try:
    from sc2reader.events import UnitInitEvent
    HAVE_UNIT_INIT = True
except ImportError:  # pragma: no cover
    UnitInitEvent = ()
    HAVE_UNIT_INIT = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from replay_parser import (  # noqa: E402
    WindowedState, STRUCTURE_NAME_MAP, UNIT_NAME_MAP, TRAIN_COMMAND_TO_UNIT,
    is_command_event,
)

DEFAULT_REPLAY_DIR = r"C:\dev\BetaStar\replays\raw"


def section(title: str):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def find_protoss_pid(replay):
    for p in replay.players:
        if p.play_race == "Protoss":
            return p.pid
    return None


def owned_by(event, pid):
    unit = getattr(event, "unit", None)
    owner = getattr(unit, "owner", None) if unit is not None else None
    return owner is not None and owner.pid == pid


def q1_morph_naming(replay, pid):
    section("Q1. MORPH NAMING — can we recover unit type at event time?")

    # What attributes does a Unit object actually expose?
    sample_unit = None
    for event in replay.events:
        if isinstance(event, (UnitBornEvent, UnitDoneEvent)) and owned_by(event, pid):
            sample_unit = event.unit
            break

    if sample_unit is None:
        print("  No owned unit events found.")
        return

    print(f"  Sample unit: {sample_unit.name}  (type {type(sample_unit).__name__})")
    for attr in ("type_history", "_type_class", "type", "unit_type_name",
                 "is_building", "finished_at", "started_at", "died_at"):
        if hasattr(sample_unit, attr):
            val = getattr(sample_unit, attr)
            shown = repr(val)
            if len(shown) > 90:
                shown = shown[:90] + "..."
            print(f"    has .{attr:<16} = {shown}")
        else:
            print(f"    (no .{attr})")

    has_hist = hasattr(sample_unit, "type_history")
    print(f"\n  --> type_history available: {has_hist}")
    if has_hist:
        print("      If populated with {frame: type}, we can resolve the type as of")
        print("      any event frame and fix GATEWAY/WARPGATE exactly.")

    # Concrete timeline for gateway-type buildings.
    print(f"\n  Gateway/Warpgate lifecycle (first 8 such buildings):")
    seen = {}
    order = []
    for event in replay.events:
        if not owned_by(event, pid):
            continue
        if not isinstance(event, (UnitInitEvent, UnitDoneEvent, UnitDiedEvent)
                          if HAVE_UNIT_INIT else (UnitDoneEvent, UnitDiedEvent)):
            continue
        unit = event.unit
        name = unit.name
        if name not in ("Gateway", "WarpGate"):
            continue
        tag = getattr(unit, "id", id(unit))
        if tag not in seen:
            if len(order) >= 8:
                continue
            seen[tag] = []
            order.append(tag)
        seen[tag].append(
            (event.second, type(event).__name__, name))

    if not order:
        print("    none found")
    for tag in order:
        trail = "  ".join(f"{t:.0f}s:{ev}={nm}" for t, ev, nm in seen[tag])
        hist = ""
        print(f"    unit {tag}: {trail}{hist}")

    print("\n  Interpretation: if a unit shows UnitDone=WarpGate at ~46s, sc2reader")
    print("  is reporting the FINAL type, confirming the audit's GATEWAY dones=14.")


def q2_starting_units(replay, pid):
    section("Q2. STARTING UNITS — are they double counted?")

    print("  Born/Done events in the first 5 seconds (owned by Protoss player):")
    n = 0
    counts = defaultdict(int)
    for event in replay.events:
        if event.second > 5:
            break
        if isinstance(event, (UnitBornEvent, UnitDoneEvent)) and owned_by(event, pid):
            counts[event.unit.name] += 1
            n += 1
            if n <= 20:
                print(f"    [{event.second:5.1f}s] {type(event).__name__:<14} "
                      f"{event.unit.name}")
    if n > 20:
        print(f"    ... and {n - 20} more")
    if n == 0:
        print("    NONE")
    print(f"\n  Totals in first 5s: {dict(counts)}")

    # The parser no longer seeds counts in __init__; state comes from object
    # lifetimes, so starting units are counted exactly once.
    state = WindowedState(replay, pid)
    if not len(state):
        print("\n  No windows reconstructed.")
        return
    first = state.windows[0]
    nexus = first["structures_done"].get("NEXUS", 0)
    probes = first["units_done"].get("PROBE", 0)

    print(f"\n  Parser state at window 0 (t=0):")
    print(f"    NEXUS={nexus}  PROBE={probes}")
    print(f"  A real game starts with 1 Nexus / 12 Probes.")

    if nexus == 1 and probes == 12:
        print(f"\n  [OK] Counted exactly once.")
    else:
        print(f"\n  [WARN] Unexpected starting counts — investigate.")


def q3_label_undercount(replay, pid):
    section("Q3. LABEL UNDERCOUNT — units born per production command")

    cmds = defaultdict(int)
    borns = defaultdict(int)
    # command timestamps per unit type, to show clustering
    cmd_times = defaultdict(list)
    born_times = defaultdict(list)

    for event in replay.events:
        if is_command_event(event) and event.player.pid == pid:
            key = TRAIN_COMMAND_TO_UNIT.get(event.ability_name)
            if key:
                cmds[key] += 1
                cmd_times[key].append(event.second)
        elif isinstance(event, UnitBornEvent) and owned_by(event, pid):
            key = UNIT_NAME_MAP.get(event.unit.name)
            if key:
                borns[key] += 1
                born_times[key].append(event.second)

    print(f"  {'Unit':<16}  {'commands':>9}  {'born':>7}  {'born/cmd':>9}")
    print(f"  {'-' * 16}  {'-' * 9}  {'-' * 7}  {'-' * 9}")
    for key in sorted(set(list(cmds) + list(borns)), key=lambda k: -borns.get(k, 0)):
        c = cmds.get(key, 0)
        b = borns.get(key, 0)
        ratio = (b / c) if c else float("inf")
        flag = "  <-- multi-unit commands" if c and ratio > 1.5 else ""
        rs = f"{ratio:>9.2f}" if c else f"{'n/a':>9}"
        print(f"  {key:<16}  {c:>9,}  {b:>7,}  {rs}{flag}")

    print("\n  born/cmd > 1 means one command event produced several units")
    print("  (multiple production buildings selected at once). Every extra unit is")
    print("  production the single-action-per-window label cannot represent.")

    # Show one concrete cluster for the worst offender.
    worst = None
    for key in borns:
        c = cmds.get(key, 0)
        if c and borns[key] / c > 1.5:
            if worst is None or borns[key] > borns[worst]:
                worst = key
    if worst:
        print(f"\n  Example — {worst}: first 3 commands and the births that follow")
        for ct in cmd_times[worst][:3]:
            following = [t for t in born_times[worst] if ct <= t <= ct + 60]
            print(f"    command at {ct:6.1f}s -> {len(following)} born within 60s: "
                  f"{[round(t, 1) for t in following[:8]]}")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        if not Path(DEFAULT_REPLAY_DIR).is_dir():
            print(f"ERROR: {DEFAULT_REPLAY_DIR} not found. Pass a replay path.")
            return
        files = sorted(f for f in os.listdir(DEFAULT_REPLAY_DIR)
                       if f.endswith(".SC2Replay"))
        if not files:
            print(f"ERROR: no replays in {DEFAULT_REPLAY_DIR}")
            return
        # skip the odd "..SC2Replay" entry if present
        pick = next((f for f in files if len(f) > 12), files[0])
        path = os.path.join(DEFAULT_REPLAY_DIR, pick)

    print(f"Probing replay: {path}")
    print(f"sc2reader version: {getattr(sc2reader, '__version__', 'unknown')}")
    print(f"UnitInitEvent available: {HAVE_UNIT_INIT}")

    replay = sc2reader.load_replay(path, load_level=4)
    pid = find_protoss_pid(replay)
    if pid is None:
        print("No Protoss player in this replay.")
        return
    print(f"Protoss pid: {pid}  |  build: {getattr(replay, 'build', '?')}")

    q1_morph_naming(replay, pid)
    q2_starting_units(replay, pid)
    q3_label_undercount(replay, pid)

    section("DONE")


if __name__ == "__main__":
    main()
