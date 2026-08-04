"""
parse_log.py — structured logging for replay parsing
====================================================
The parser only ever printed to stdout, which meant capturing it via a shell
redirect (producing UTF-16 files) and grepping half a megabyte of inline
conflict spam to answer questions like "which actions are losing labels?".

This writes the same information as data, following the same convention as
decision_log.py:

    <log_dir>/parse_<timestamp>.jsonl        one JSON object per replay
    <log_dir>/parse_<timestamp>.summary.txt  human-readable aggregate

The jsonl is for slicing (find every replay with >50% do_nothing, or every
conflict for action 31). The summary is for eyeballing a run and diffing it
against the previous one.

Everything is best-effort: if the log cannot be opened, parsing continues.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path


def _pct(part: float, whole: float) -> float:
    return (100.0 * part / whole) if whole else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return float(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0)


class ParseLogger:
    """Accumulates per-replay parse results and writes a run summary."""

    def __init__(self, log_dir: str, meta: dict | None = None,
                 echo: bool = True):
        self.echo = echo
        self.enabled = True
        self._fh = None
        self.path = None
        self.summary_path = None

        self.records: list[dict] = []
        self.action_counts: Counter = Counter()      # action_id -> count
        self.conflict_counts: Counter = Counter()    # action_id -> count
        self.conflict_reasons: Counter = Counter()   # "id:reason" -> count
        self.skip_reasons: Counter = Counter()
        self.lengths: list[int] = []
        self.n_parsed = 0
        self.n_skipped = 0
        self.n_failed = 0
        self._t0 = time.time()

        stamp = time.strftime("%Y%m%d-%H%M%S")
        try:
            directory = Path(log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f"parse_{stamp}.jsonl"
            self.summary_path = directory / f"parse_{stamp}.summary.txt"
            self._fh = open(self.path, "w", encoding="utf-8")
            record = {"_meta": dict(meta or {})}
            record["_meta"].setdefault("python", sys.version.split()[0])
            record["_meta"].setdefault("started", stamp)
            self._write(record)
            if echo:
                print(f"[parse-log] writing to {self.path}")
        except Exception as exc:  # noqa: BLE001 - never block parsing
            print(f"[parse-log] disabled ({exc})")
            self.enabled = False

    # -- internals ---------------------------------------------------------

    def _write(self, record: dict):
        if not self._fh:
            return
        try:
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass

    # -- per-replay events -------------------------------------------------

    def replay_parsed(self, fname: str, *, build: int, windows: int,
                      action_counts: Counter, conflicts: Counter,
                      max_lag: int, seconds: float):
        """
        conflicts: Counter keyed by (action_id, action_name, reason).
        action_counts: Counter keyed by action_id (0 included).
        """
        self.n_parsed += 1
        self.lengths.append(windows)
        self.action_counts.update(action_counts)

        detail: dict[str, int] = {}
        for (action_id, action_name, reason), n in conflicts.items():
            self.conflict_counts[action_id] += n
            self.conflict_reasons[f"{action_id}:{action_name}:{reason}"] += n
            detail[f"{action_id}:{action_name}:{reason}"] = n

        idle = int(action_counts.get(0, 0))
        record = {
            "file": fname,
            "status": "parsed",
            "build": build,
            "windows": windows,
            "game_seconds": windows * self._grid(),
            "do_nothing": idle,
            "do_nothing_pct": round(_pct(idle, windows), 1),
            "conflicts": int(sum(conflicts.values())),
            "max_lag_windows": max_lag,
            "parse_seconds": round(seconds, 2),
            "actions": {str(k): int(v) for k, v in sorted(action_counts.items())
                        if k != 0},
        }
        if detail:
            record["conflict_detail"] = detail
        self.records.append(record)
        self._write(record)

    def replay_skipped(self, fname: str, reason: str, **extra):
        self.n_skipped += 1
        self.skip_reasons[reason] += 1
        record = {"file": fname, "status": "skipped", "reason": reason}
        record.update(extra)
        self.records.append(record)
        self._write(record)

    def replay_failed(self, fname: str, error: str):
        self.n_failed += 1
        record = {"file": fname, "status": "failed", "error": error}
        self.records.append(record)
        self._write(record)

    def _grid(self) -> int:
        return 4  # overridden via meta; only used for a convenience field

    # -- summary -----------------------------------------------------------

    def finish(self, *, action_names: list[str], unmapped: dict,
               dataset_path: str | None = None, grid_seconds: int = 4,
               extra: dict | None = None):
        if not self.enabled:
            return

        lines: list[str] = []

        def out(text: str = ""):
            lines.append(text)

        total_windows = sum(self.lengths)
        idle = int(self.action_counts.get(0, 0))
        non_idle = total_windows - idle
        elapsed = time.time() - self._t0

        out("=" * 72)
        out("  REPLAY PARSE SUMMARY")
        out("=" * 72)
        out(f"  Finished:        {time.strftime('%Y-%m-%d %H:%M:%S')}")
        out(f"  Wall time:       {elapsed / 60:.1f} min")
        out(f"  Grid interval:   {grid_seconds}s")
        if dataset_path:
            out(f"  Dataset:         {dataset_path}")
        if self.path:
            out(f"  Per-replay log:  {self.path.name}")
        for key, value in (extra or {}).items():
            out(f"  {key + ':':16} {value}")

        out()
        out("-" * 72)
        out("  CORPUS")
        out("-" * 72)
        out(f"  Parsed:          {self.n_parsed}")
        out(f"  Skipped:         {self.n_skipped}")
        for reason, n in self.skip_reasons.most_common():
            out(f"      {reason:<40} {n:>6}")
        out(f"  Failed:          {self.n_failed}")

        if self.lengths:
            out()
            out(f"  Sequence lengths (windows):")
            out(f"      min={min(self.lengths)}  max={max(self.lengths)}  "
                f"mean={total_windows / len(self.lengths):.0f}  "
                f"median={_median(self.lengths):.0f}")
            out(f"  Total windows:   {total_windows:,}")
            out(f"  Longest game:    {max(self.lengths) * grid_seconds}s "
                f"({max(self.lengths) * grid_seconds / 60:.1f} min)")

            buckets = [(0, 100), (100, 200), (200, 300), (300, 400),
                       (400, 600), (600, 10 ** 9)]
            labels = ["<100", "100-200", "200-300", "300-400", "400-600",
                      "600+"]
            out()
            out(f"  Length histogram:")
            for (lo, hi), label in zip(buckets, labels):
                n = sum(1 for x in self.lengths if lo <= x < hi)
                out(f"      {label:>8}  {'#' * min(n, 50):<50} {n}")

        out()
        out("-" * 72)
        out("  LABEL DISTRIBUTION")
        out("-" * 72)
        out(f"  do_nothing:      {idle:,}  ({_pct(idle, total_windows):.1f}% of all windows)")
        out(f"  real actions:    {non_idle:,}  ({_pct(non_idle, total_windows):.1f}%)")
        out()
        out(f"  {'ID':>4}  {'Action':<28}  {'Count':>8}  {'% all':>7}  "
            f"{'% real':>7}")
        out(f"  {'-' * 4}  {'-' * 28}  {'-' * 8}  {'-' * 7}  {'-' * 7}")
        order = sorted(range(len(action_names)),
                       key=lambda a: (a != 0, -self.action_counts.get(a, 0)))
        for a in order:
            n = int(self.action_counts.get(a, 0))
            flag = "  <-- ZERO SAMPLES" if n == 0 and a != 0 else ""
            out(f"  {a:>4}  {action_names[a]:<28}  {n:>8,}  "
                f"{_pct(n, total_windows):>6.2f}%  "
                f"{_pct(n, non_idle) if a != 0 else 0:>6.2f}%{flag}")

        out()
        out("-" * 72)
        out("  CONFLICT DEMOTIONS (label judged illegal -> became do_nothing)")
        out("-" * 72)
        total_conflicts = int(sum(self.conflict_counts.values()))
        out(f"  Total:           {total_conflicts:,}")
        if total_conflicts:
            out(f"  As % of real actions: "
                f"{_pct(total_conflicts, non_idle + total_conflicts):.2f}%")
            out()
            out(f"  {'ID':>4}  {'Action':<28}  {'Dropped':>8}  {'Kept':>8}  "
                f"{'Loss %':>7}")
            out(f"  {'-' * 4}  {'-' * 28}  {'-' * 8}  {'-' * 8}  {'-' * 7}")
            for a, n in self.conflict_counts.most_common():
                name = action_names[a] if a < len(action_names) else f"?{a}"
                kept = int(self.action_counts.get(a, 0))
                out(f"  {a:>4}  {name:<28}  {n:>8,}  {kept:>8,}  "
                    f"{_pct(n, n + kept):>6.1f}%")
            out()
            out(f"  Top reasons:")
            for key, n in self.conflict_reasons.most_common(15):
                out(f"      {n:>7,}  {key}")
        else:
            out("  None.")

        if unmapped:
            out()
            out("-" * 72)
            out("  UNMAPPED ABILITIES (ignored, produce no label)")
            out("-" * 72)
            for ability, n in sorted(unmapped.items(), key=lambda kv: -kv[1])[:30]:
                shown = ability if ability.strip() else "<empty ability name>"
                out(f"      {n:>8,}  {shown}")
            remaining = len(unmapped) - 30
            if remaining > 0:
                out(f"      ... and {remaining} more")

        # Outliers worth a human glance.
        parsed = [r for r in self.records if r.get("status") == "parsed"]
        if parsed:
            out()
            out("-" * 72)
            out("  OUTLIERS")
            out("-" * 72)
            worst_idle = sorted(parsed, key=lambda r: -r["do_nothing_pct"])[:5]
            out("  Highest do_nothing %:")
            for r in worst_idle:
                out(f"      {r['do_nothing_pct']:>5.1f}%  "
                    f"({r['windows']:>4} windows)  {r['file']}")
            worst_conf = [r for r in parsed if r.get("conflicts")]
            worst_conf.sort(key=lambda r: -r["conflicts"])
            if worst_conf:
                out("  Most conflicts:")
                for r in worst_conf[:5]:
                    out(f"      {r['conflicts']:>5}  {r['file']}")
            slowest = sorted(parsed, key=lambda r: -r["parse_seconds"])[:3]
            out("  Slowest to parse:")
            for r in slowest:
                out(f"      {r['parse_seconds']:>6.2f}s  {r['file']}")

        out()
        out("=" * 72)

        text = "\n".join(lines)
        try:
            self.summary_path.write_text(text, encoding="utf-8")
            if self.echo:
                print(text)
                print(f"[parse-log] summary written to {self.summary_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[parse-log] could not write summary ({exc})")

        if self._fh:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
