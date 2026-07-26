"""
obs_divergence_audit.py — Quantify training/inference observation skew
======================================================================
Cross-checks the parser's reconstructed state against an INDEPENDENT
reconstruction built from the tracker event stream.

The parser (replay_parser.WindowedState) derives state from sc2reader per-object
lifetimes (started_at / finished_at / died_at / type_history). This audit derives
"structures under construction" from the UnitInit -> UnitDone event stream, which
is exactly what the live API reports as `structures(X).not_ready.amount` at
inference. Two independent paths to the same quantity, so a disagreement is a
real defect in one of them.

History: the original command-counting parser drifted from a true ~1-4 buildings
under construction to a reported ~32 by minute 28. Section B is the regression
test for that -- error should now stay flat near zero at every game minute.

Reports
-------
  A. Command/completion conservation per type
  B. Parser pending-structure error vs UnitInit ground truth, by game minute
  C. Ignored cancel events
  D. Resource staleness from PlayerStatsEvent cadence
  E. Feature-range sanity (time normalization)

Usage
-----
    python obs_divergence_audit.py
    python obs_divergence_audit.py --replays C:/dev/BetaStar/replays/raw --limit 40
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sc2reader
from sc2reader.events import (
    PlayerStatsEvent, UnitBornEvent, UnitDiedEvent, UnitDoneEvent,
    BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent,
)

# UnitInitEvent is what makes the exact structure reconstruction possible.
try:
    from sc2reader.events import UnitInitEvent
    HAVE_UNIT_INIT = True
except ImportError:  # pragma: no cover - depends on sc2reader version
    UnitInitEvent = ()
    HAVE_UNIT_INIT = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from replay_parser import (  # noqa: E402
    WindowedState, GRID_INTERVAL_SECONDS, STRUCTURES,
    STRUCTURE_NAME_MAP, UNIT_NAME_MAP,
    BUILD_COMMAND_TO_STRUCTURE, TRAIN_COMMAND_TO_UNIT,
)
from obs_spec import TIME_NORM  # noqa: E402

DEFAULT_REPLAY_DIR = r"C:\dev\BetaStar\replays\raw"

# Cancel-type abilities. The parser maps none of these, so every occurrence
# leaves a pending counter permanently incremented.
CANCEL_ABILITIES = {
    "CancelBuilding", "CancelLast", "CancelSlot", "Cancel",
    "HaltBuilding", "CancelMorphArchon", "CancelGravitonBeam",
    "AdeptPhaseShiftCancel", "AdeptShadePhaseShiftCancel",
}

COMMAND_EVENTS = (BasicCommandEvent, TargetPointCommandEvent,
                  TargetUnitCommandEvent)


def section(title: str):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def find_protoss_pid(replay):
    for player in replay.players:
        if player.play_race == "Protoss":
            return player.pid
    return None


def owned_by(event, pid: int) -> bool:
    unit = getattr(event, "unit", None)
    owner = getattr(unit, "owner", None) if unit is not None else None
    return owner is not None and owner.pid == pid


class ReplayAudit:
    """Per-replay measurements, aggregated across the corpus by AuditTotals."""

    def __init__(self):
        # A. conservation
        self.build_commands = defaultdict(int)
        self.train_commands = defaultdict(int)
        self.struct_inits = defaultdict(int)
        self.struct_dones = defaultdict(int)
        self.unit_borns = defaultdict(int)
        self.cancels = defaultdict(int)

        # B. pending error, per game-minute bucket
        # minute -> list of (parser_total_pending, truth_total_pending)
        self.pending_by_minute = defaultdict(list)
        # structure key -> list of signed errors (parser - truth)
        self.pending_err_by_struct = defaultdict(list)
        # end-of-game leftover pending in the parser
        self.final_parser_pending = {}

        # C. staleness
        self.stats_intervals = []
        self.staleness_samples = []

        # D. range sanity
        self.max_time = 0.0
        self.windows = 0
        self.windows_over_720 = 0


def audit_replay(replay, pid: int) -> ReplayAudit:
    """
    Walk the replay once, maintaining three things in parallel:
      1. the parser's reconstructed pending counts (object lifetimes)
      2. ground-truth in-construction counts (UnitInit -> UnitDone events)
      3. staleness / conservation bookkeeping
    Snapshots are taken at the same 4s grid boundaries the parser uses.
    """
    a = ReplayAudit()
    G = GRID_INTERVAL_SECONDS

    # The parser's own reconstruction, which we are cross-checking.
    parser_state = WindowedState(replay, pid)

    # Ground truth: count of structures currently between Init and Done.
    truth_pending = defaultdict(int)

    current_grid = 0
    last_stats_time = 0.0
    have_stats = False

    def parser_pending(grid_idx: int) -> dict:
        if not len(parser_state):
            return {}
        idx = min(grid_idx, len(parser_state) - 1)
        return parser_state.windows[idx]["structures_pending"]

    def snapshot(grid_idx: int):
        t = grid_idx * G
        pend = parser_pending(grid_idx)
        parser_total = sum(pend.values())
        truth_total = sum(truth_pending.values())
        minute = int(t // 60)
        a.pending_by_minute[minute].append((parser_total, truth_total))
        for s in STRUCTURES:
            err = pend.get(s, 0) - truth_pending.get(s, 0)
            if err != 0:
                a.pending_err_by_struct[s].append(err)
        if have_stats:
            a.staleness_samples.append(max(0.0, t - last_stats_time))
        a.windows += 1
        if t > TIME_NORM:
            a.windows_over_720 += 1
        a.max_time = max(a.max_time, float(t))

    snapshot(0)

    for event in replay.events:
        t = event.second
        new_grid = int(t / G)
        while current_grid < new_grid:
            current_grid += 1
            snapshot(current_grid)

        if isinstance(event, PlayerStatsEvent):
            if event.player.pid == pid:
                if have_stats:
                    a.stats_intervals.append(event.second - last_stats_time)
                last_stats_time = event.second
                have_stats = True

        elif HAVE_UNIT_INIT and isinstance(event, UnitInitEvent):
            if owned_by(event, pid):
                key = STRUCTURE_NAME_MAP.get(event.unit.name)
                if key:
                    truth_pending[key] += 1
                    a.struct_inits[key] += 1

        elif isinstance(event, (UnitBornEvent, UnitDoneEvent)):
            if owned_by(event, pid):
                name = event.unit.name
                skey = STRUCTURE_NAME_MAP.get(name)
                ukey = UNIT_NAME_MAP.get(name)
                if isinstance(event, UnitDoneEvent) and skey:
                    truth_pending[skey] = max(0, truth_pending[skey] - 1)
                    a.struct_dones[skey] += 1
                if ukey:
                    a.unit_borns[ukey] += 1

        elif isinstance(event, UnitDiedEvent):
            if owned_by(event, pid):
                skey = STRUCTURE_NAME_MAP.get(event.unit.name)
                # A building that dies while under construction (cancel or
                # killed) must stop counting as pending.
                if skey and truth_pending.get(skey, 0) > 0:
                    truth_pending[skey] -= 1

        elif isinstance(event, COMMAND_EVENTS):
            if event.player.pid == pid:
                ability = event.ability_name
                if ability in BUILD_COMMAND_TO_STRUCTURE:
                    a.build_commands[BUILD_COMMAND_TO_STRUCTURE[ability]] += 1
                if ability in TRAIN_COMMAND_TO_UNIT:
                    a.train_commands[TRAIN_COMMAND_TO_UNIT[ability]] += 1
                if ability in CANCEL_ABILITIES:
                    a.cancels[ability] += 1

    last = (parser_state.windows[-1] if len(parser_state) else
            {"structures_pending": {}, "units_pending": {}})
    a.final_parser_pending = {
        k: v for k, v in last["structures_pending"].items() if v != 0}
    a.final_parser_pending_units = {
        k: v for k, v in last["units_pending"].items() if v != 0}
    return a


class AuditTotals:
    def __init__(self):
        self.build_commands = defaultdict(int)
        self.train_commands = defaultdict(int)
        self.struct_inits = defaultdict(int)
        self.struct_dones = defaultdict(int)
        self.unit_borns = defaultdict(int)
        self.cancels = defaultdict(int)
        self.pending_by_minute = defaultdict(list)
        self.pending_err_by_struct = defaultdict(list)
        self.stats_intervals = []
        self.staleness_samples = []
        self.final_pending_struct = []
        self.final_pending_units = []
        self.windows = 0
        self.windows_over_720 = 0
        self.max_time = 0.0
        self.n_replays = 0

    def add(self, a: ReplayAudit):
        self.n_replays += 1
        for src, dst in (
            (a.build_commands, self.build_commands),
            (a.train_commands, self.train_commands),
            (a.struct_inits, self.struct_inits),
            (a.struct_dones, self.struct_dones),
            (a.unit_borns, self.unit_borns),
            (a.cancels, self.cancels),
        ):
            for k, v in src.items():
                dst[k] += v
        for m, vals in a.pending_by_minute.items():
            self.pending_by_minute[m].extend(vals)
        for s, errs in a.pending_err_by_struct.items():
            self.pending_err_by_struct[s].extend(errs)
        self.stats_intervals.extend(a.stats_intervals)
        self.staleness_samples.extend(a.staleness_samples)
        self.final_pending_struct.append(sum(a.final_parser_pending.values()))
        self.final_pending_units.append(
            sum(getattr(a, "final_parser_pending_units", {}).values()))
        self.windows += a.windows
        self.windows_over_720 += a.windows_over_720
        self.max_time = max(self.max_time, a.max_time)


def report_conservation(T: AuditTotals):
    section("A. COMMAND vs COMPLETION CONSERVATION (drift magnitude)")
    print("  The parser increments pending on a command and decrements on")
    print("  completion. Commands that never complete (cancels, killed builders,")
    print("  destroyed production) leave the counter permanently inflated.\n")

    print(f"  {'Structure':<20}  {'cmds':>7}  {'inits':>7}  {'dones':>7}  "
          f"{'cmd-done':>9}  {'leak %':>7}")
    print(f"  {'-' * 20}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 9}  {'-' * 7}")
    for s in STRUCTURES:
        cmds = T.build_commands.get(s, 0)
        if cmds == 0 and T.struct_dones.get(s, 0) == 0:
            continue
        inits = T.struct_inits.get(s, 0)
        dones = T.struct_dones.get(s, 0)
        leak = cmds - dones
        leak_pct = (100.0 * leak / cmds) if cmds else 0.0
        flag = "  <-- LEAK" if leak_pct > 10 else ""
        print(f"  {s:<20}  {cmds:>7,}  {inits:>7,}  {dones:>7,}  "
              f"{leak:>9,}  {leak_pct:>6.1f}%{flag}")

    print(f"\n  {'Unit':<20}  {'cmds':>7}  {'borns':>7}  {'cmd-born':>9}  {'leak %':>7}")
    print(f"  {'-' * 20}  {'-' * 7}  {'-' * 7}  {'-' * 9}  {'-' * 7}")
    for u, cmds in sorted(T.train_commands.items(), key=lambda kv: -kv[1]):
        borns = T.unit_borns.get(u, 0)
        leak = cmds - borns
        leak_pct = (100.0 * leak / cmds) if cmds else 0.0
        flag = "  <-- LEAK" if leak_pct > 10 else ""
        print(f"  {u:<20}  {cmds:>7,}  {borns:>7,}  {leak:>9,}  "
              f"{leak_pct:>6.1f}%{flag}")

    fs = np.array(T.final_pending_struct, dtype=float)
    fu = np.array(T.final_pending_units, dtype=float)
    print(f"\n  End-of-game leftover pending counts (should be near 0):")
    print(f"    structures: mean={fs.mean():.1f}  median={np.median(fs):.1f}  "
          f"max={fs.max():.0f}")
    print(f"    units:      mean={fu.mean():.1f}  median={np.median(fu):.1f}  "
          f"max={fu.max():.0f}")
    if fu.mean() > 3:
        print(f"    [WARN] Large leftover unit pending confirms counter drift.")


def report_pending_error(T: AuditTotals):
    section("B. PARSER PENDING-STRUCTURE ERROR vs GROUND TRUTH (by game minute)")
    if not HAVE_UNIT_INIT:
        print("  UnitInitEvent unavailable in this sc2reader version — skipping.")
        print("  (Section A still measures conservation without it.)")
        return

    print("  Parser = object lifetimes. Ground truth = UnitInit..UnitDone events,")
    print("  which is what the live API reports as not_ready.amount at inference.")
    print("  Error should stay flat near zero at every minute.\n")
    print(f"  {'Minute':>7}  {'N':>8}  {'parser mean':>12}  {'truth mean':>11}  "
          f"{'mean err':>9}  {'MAE':>7}")
    print(f"  {'-' * 7}  {'-' * 8}  {'-' * 12}  {'-' * 11}  {'-' * 9}  {'-' * 7}")

    worst_mae = 0.0
    for minute in sorted(T.pending_by_minute):
        vals = T.pending_by_minute[minute]
        if len(vals) < 20:
            continue
        arr = np.array(vals, dtype=float)
        parser_mean = arr[:, 0].mean()
        truth_mean = arr[:, 1].mean()
        err = arr[:, 0] - arr[:, 1]
        mae = float(np.abs(err).mean())
        worst_mae = max(worst_mae, mae)
        print(f"  {minute:>7}  {len(vals):>8,}  {parser_mean:>12.2f}  "
              f"{truth_mean:>11.2f}  {err.mean():>+9.2f}  {mae:>7.2f}")

    print(f"\n  Worst per-minute MAE: {worst_mae:.2f}")
    if worst_mae < 1.0:
        print("  [OK] Parser tracks ground truth. The drift regression is fixed.")
    elif worst_mae < 3.0:
        print("  [NOTE] Small residual error — likely grid quantization at window")
        print("         boundaries. Acceptable, but check it does not grow with time.")
    else:
        print("  [WARN] Large error remains. If it GROWS with game minute the")
        print("         reconstruction is still leaking somewhere.")

    print(f"\n  Per-structure signed error (parser - truth), nonzero samples only:")
    print(f"  {'Structure':<20}  {'N':>8}  {'mean err':>9}  {'p95 err':>8}")
    print(f"  {'-' * 20}  {'-' * 8}  {'-' * 9}  {'-' * 8}")
    for s in STRUCTURES:
        errs = T.pending_err_by_struct.get(s)
        if not errs:
            continue
        arr = np.array(errs, dtype=float)
        print(f"  {s:<20}  {len(arr):>8,}  {arr.mean():>+9.2f}  "
              f"{np.percentile(arr, 95):>8.1f}")


def report_cancels(T: AuditTotals):
    section("C. IGNORED CANCEL EVENTS (drift mechanism)")
    total = sum(T.cancels.values())
    if total == 0:
        print("  No cancel abilities observed.")
        return
    print("  None of these are in the parser's command maps, so each one leaves")
    print("  a pending counter incremented forever.\n")
    for ability, n in sorted(T.cancels.items(), key=lambda kv: -kv[1]):
        print(f"    {ability:<34}  {n:>7,}")
    print(f"\n    {'TOTAL':<34}  {total:>7,}")


def report_staleness(T: AuditTotals):
    section("D. RESOURCE STALENESS (PlayerStatsEvent cadence)")
    if T.stats_intervals:
        iv = np.array(T.stats_intervals, dtype=float)
        print(f"  PlayerStatsEvent interval: mean={iv.mean():.2f}s  "
              f"median={np.median(iv):.2f}s  max={iv.max():.2f}s")
        print(f"  Grid interval:             {GRID_INTERVAL_SECONDS}s")
        print(f"  -> minerals/gas are refreshed every ~{np.median(iv):.0f}s but "
              f"sampled every {GRID_INTERVAL_SECONDS}s.")
    if T.staleness_samples:
        st = np.array(T.staleness_samples, dtype=float)
        print(f"\n  Age of resource reading at each grid snapshot:")
        print(f"    mean={st.mean():.2f}s  median={np.median(st):.2f}s  "
              f"p95={np.percentile(st, 95):.2f}s  max={st.max():.2f}s")
        print(f"    windows with >4s stale resources: "
              f"{100.0 * (st > 4).mean():.1f}%")
        print(f"\n  At inference bot.minerals/bot.vespene are frame-accurate, so the")
        print(f"  mineral/gas one-hot bins are systematically fresher than training.")


def report_ranges(T: AuditTotals):
    section("E. FEATURE RANGE SANITY")
    print(f"  Grid windows audited: {T.windows:,}")
    print(f"  Longest game time:    {T.max_time:.0f}s "
          f"({T.max_time / 60:.1f} min)")
    over = 100.0 * T.windows_over_720 / max(T.windows, 1)
    print(f"  TIME_NORM (obs_spec): {TIME_NORM:.0f}s")
    print(f"  Windows beyond TIME_NORM: {T.windows_over_720:,}  ({over:.1f}%)")
    if over > 1:
        print(f"  [NOTE] {over:.1f}% of rows hit the time clip and are")
        print(f"         indistinguishable in the time feature.")
    else:
        print(f"  [OK] Almost all rows fall inside the time normalization range.")


def report_conservation_note():
    section("NOTE ON SECTION A")
    print("  Section A compares build COMMANDS to completions. Large gaps are")
    print("  expected and no longer harmful: state is reconstructed from object")
    print("  lifetimes, so commands that never produce a building are simply")
    print("  never counted. The numbers remain useful for understanding how")
    print("  unreliable command events are as a signal (they still drive LABELS).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replays", default=DEFAULT_REPLAY_DIR)
    ap.add_argument("--limit", type=int, default=40,
                    help="max replays to audit (default 40; use 0 for all)")
    args = ap.parse_args()

    folder = args.replays
    if not Path(folder).is_dir():
        print(f"ERROR: replay folder not found: {folder}")
        return

    files = sorted(f for f in os.listdir(folder) if f.endswith(".SC2Replay"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"ERROR: no .SC2Replay files in {folder}")
        return

    print(f"Auditing {len(files)} replay(s) from {folder}")
    print(f"UnitInitEvent available: {HAVE_UNIT_INIT}")

    T = AuditTotals()
    failed = skipped = 0
    for i, fname in enumerate(files, 1):
        try:
            replay = sc2reader.load_replay(
                os.path.join(folder, fname), load_level=4)
            if getattr(replay, "build", 0) < 73286:
                skipped += 1
                continue
            pid = find_protoss_pid(replay)
            if pid is None:
                skipped += 1
                continue
            T.add(audit_replay(replay, pid))
            print(f"  [{i}/{len(files)}] {fname}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAILED {fname}: {exc}")
            failed += 1

    if T.n_replays == 0:
        print("\nNo replays audited.")
        return

    print(f"\nAudited {T.n_replays} replay(s)  "
          f"(skipped {skipped}, failed {failed})")

    report_conservation(T)
    report_pending_error(T)
    report_cancels(T)
    report_staleness(T)
    report_ranges(T)
    report_conservation_note()

    section("DONE")
    print("  Section B is the regression test: error should be flat and near")
    print("  zero at every game minute (it used to climb to +31).\n")


if __name__ == "__main__":
    main()
