"""
decision_log.py — runtime introspection for the policy
======================================================
Answers "why didn't the bot do X?" with a lookup instead of detective work.

For every decision it records what the model wanted, what the mask allowed, what
was sampled, and -- crucially -- whether the action actually had any effect on
the game. That last field is what was missing when the bot picked
build_assimilator correctly for a whole game while the execution layer silently
dropped every one of them.

Outcome detection is deferred by one decision. burnysc2 batches actions and
sends them at the end of on_step, so nothing is observable immediately after
execute_action(). Instead we snapshot a progress metric for the action's target
at decision time and re-read it at the next decision (~4s later), by which point
an accepted order shows up in already_pending(). Each record is therefore written
one decision behind.

Outcomes
--------
    executed        progress increased -- the order landed
    noop            progress unchanged -- silently dropped (a bug, or unaffordable)
    suppressed      blocked on purpose by the MAX_CONCURRENT_BUILDS guard
    n/a             no measurable target (do_nothing, attack_enemy_base)

Output
------
    <log_dir>/decisions_<timestamp>.jsonl   one JSON object per decision
    <log_dir>/decisions_<timestamp>.summary.txt
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

import obs_spec
from actions import ACTIONS
from helpers import MAX_CONCURRENT_BUILDS, DEFAULT_MAX_CONCURRENT_BUILDS

# ---------------------------------------------------------------------------
# What each action is trying to produce, so its effect can be measured.
#   ("structure", UnitTypeId)  ("unit", UnitTypeId)
#   ("upgrade", (UpgradeId, ...))   ("none", None)
# ---------------------------------------------------------------------------

ACTION_TARGET: dict[str, tuple[str, object]] = {
    "do_nothing":              ("none", None),
    "train_probe":             ("unit", UnitTypeId.PROBE),
    "build_pylon":             ("structure", UnitTypeId.PYLON),
    "build_gateway":           ("structure", UnitTypeId.GATEWAY),
    "build_cyberneticscore":   ("structure", UnitTypeId.CYBERNETICSCORE),
    "build_assimilator":       ("structure", UnitTypeId.ASSIMILATOR),
    "build_nexus":             ("structure", UnitTypeId.NEXUS),
    "build_forge":             ("structure", UnitTypeId.FORGE),
    "build_stargate":          ("structure", UnitTypeId.STARGATE),
    "build_robotics_facility": ("structure", UnitTypeId.ROBOTICSFACILITY),
    "build_twilight_council":  ("structure", UnitTypeId.TWILIGHTCOUNCIL),
    "build_photon_cannon":     ("structure", UnitTypeId.PHOTONCANNON),
    "build_fleet_beacon":      ("structure", UnitTypeId.FLEETBEACON),
    "build_templar_archive":   ("structure", UnitTypeId.TEMPLARARCHIVE),
    "build_robotics_bay":      ("structure", UnitTypeId.ROBOTICSBAY),
    "build_shield_battery":    ("structure", UnitTypeId.SHIELDBATTERY),
    "train_zealot":            ("unit", UnitTypeId.ZEALOT),
    "train_stalker":           ("unit", UnitTypeId.STALKER),
    "train_immortal":          ("unit", UnitTypeId.IMMORTAL),
    "train_voidray":           ("unit", UnitTypeId.VOIDRAY),
    "train_carrier":           ("unit", UnitTypeId.CARRIER),
    "train_high_templar":      ("unit", UnitTypeId.HIGHTEMPLAR),
    "warp_in_zealot":          ("unit", UnitTypeId.ZEALOT),
    "warp_in_stalker":         ("unit", UnitTypeId.STALKER),
    "warp_in_high_templar":    ("unit", UnitTypeId.HIGHTEMPLAR),
    "research_charge":         ("upgrade", (UpgradeId.CHARGE,)),
    "research_warp_gate":      ("upgrade", (UpgradeId.WARPGATERESEARCH,)),
    "upgrade_ground_weapons":  ("upgrade", (
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3)),
    "upgrade_air_weapons":     ("upgrade", (
        UpgradeId.PROTOSSAIRWEAPONSLEVEL1,
        UpgradeId.PROTOSSAIRWEAPONSLEVEL2,
        UpgradeId.PROTOSSAIRWEAPONSLEVEL3)),
    "upgrade_shields":         ("upgrade", (
        UpgradeId.PROTOSSSHIELDSLEVEL1,
        UpgradeId.PROTOSSSHIELDSLEVEL2,
        UpgradeId.PROTOSSSHIELDSLEVEL3)),
    "attack_enemy_base":       ("none", None),
    "train_adept":             ("unit", UnitTypeId.ADEPT),
    "train_phoenix":           ("unit", UnitTypeId.PHOENIX),
    "train_colossus":          ("unit", UnitTypeId.COLOSSUS),
    "warp_in_adept":           ("unit", UnitTypeId.ADEPT),
}


def _progress(bot: BotAI, kind: str, target) -> float | None:
    """
    A scalar that must strictly increase if the action took effect.

    Counts pending AND completed so a fast build that finishes between two
    decisions still registers as executed.
    """
    try:
        if kind == "structure":
            return (float(bot.already_pending(target))
                    + bot.structures(target).ready.amount)
        if kind == "unit":
            return (float(bot.already_pending(target))
                    + bot.units(target).amount)
        if kind == "upgrade":
            total = 0.0
            for uid in target:
                if uid in bot.state.upgrades:
                    total += 1.0
                else:
                    total += float(bot.already_pending_upgrade(uid))
            return total
    except Exception:      # never let instrumentation break the game
        return None
    return None


class DecisionLogger:
    """Records one JSON object per policy decision, plus an end-of-game summary."""

    def __init__(self, log_dir: str, top_k: int = 5,
                 include_obs: bool = True, echo: bool = True):
        self.top_k = top_k
        self.include_obs = include_obs
        self.echo = echo
        self.enabled = True

        self._pending: dict | None = None      # record awaiting outcome
        self._n = 0
        self._blocked = 0
        self._chosen = {}                      # action name -> count
        self._noop = {}                        # action name -> count
        self._suppressed = {}                  # action name -> count
        self._feature_names = obs_spec.feature_names()

        stamp = time.strftime("%Y%m%d-%H%M%S")
        try:
            directory = Path(log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f"decisions_{stamp}.jsonl"
            self.summary_path = directory / f"decisions_{stamp}.summary.txt"
            self._fh = open(self.path, "w", encoding="utf-8")
            # First line describes the schema so the obs arrays below are
            # interpretable without having to guess the feature layout.
            self._write({
                "_meta": {
                    "feature_names": self._feature_names,
                    "actions": list(ACTIONS),
                    "decision_interval_seconds":
                        obs_spec.DECISION_INTERVAL_SECONDS,
                    "obs_size": obs_spec.OBS_SIZE,
                    "outcomes": {
                        "executed": "order landed",
                        "noop": "chosen but had no effect",
                        "suppressed": "blocked by MAX_CONCURRENT_BUILDS",
                        "n/a": "no measurable target",
                    },
                }
            })
            print(f"[decision-log] writing to {self.path}")
        except Exception as exc:               # keep playing without logging
            print(f"[decision-log] disabled ({exc})")
            self.enabled = False
            self._fh = None

    # -- internals ---------------------------------------------------------

    def _write(self, record: dict):
        if not self.enabled or self._fh is None:
            return
        try:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()
        except Exception:
            pass

    def _settle(self, record: dict, bot: BotAI | None):
        """Attach an outcome to a buffered record and write it out."""
        name = record["action"]
        kind = record.pop("_kind")
        target = record.pop("_target")
        before = record.pop("_progress_before")

        if record.get("suppressed"):
            outcome = "suppressed"
            self._suppressed[name] = self._suppressed.get(name, 0) + 1
        elif kind == "none" or before is None or bot is None:
            outcome = "n/a"
        else:
            after = _progress(bot, kind, target)
            if after is None:
                outcome = "n/a"
            elif after > before + 1e-9:
                outcome = "executed"
            else:
                outcome = "noop"
                self._noop[name] = self._noop.get(name, 0) + 1
            record["progress_after"] = after

        record["outcome"] = outcome
        self._write(record)

        if self.echo and outcome in ("noop", "suppressed"):
            tag = "SUPPRESSED" if outcome == "suppressed" else "NO-OP"
            print(f"[decision-log] {tag}: {name} at {record['t']:.1f}s "
                  f"had no effect")

    # -- public API --------------------------------------------------------

    def resolve_previous(self, bot: BotAI):
        """Call at the start of each decision, before choosing a new action."""
        if not self.enabled or self._pending is None:
            return
        record, self._pending = self._pending, None
        try:
            self._settle(record, bot)
        except Exception:
            pass

    def log_decision(self, bot: BotAI, iteration: int, obs: list[float],
                     action_id: int, diag: dict):
        """Buffer a decision; its outcome is resolved at the next decision."""
        if not self.enabled:
            return
        try:
            name = ACTIONS[action_id] if action_id < len(ACTIONS) else str(action_id)
            kind, target = ACTION_TARGET.get(name, ("none", None))
            before = _progress(bot, kind, target)

            # Was the execution layer going to refuse this on purpose?
            suppressed = False
            if kind == "structure":
                cap = MAX_CONCURRENT_BUILDS.get(
                    target, DEFAULT_MAX_CONCURRENT_BUILDS)
                try:
                    suppressed = bot.already_pending(target) >= cap
                except Exception:
                    suppressed = False

            record = {
                "t": round(float(bot.time), 2),
                "iteration": int(iteration),
                "action_id": int(action_id),
                "action": name,
                "suppressed": bool(suppressed),
                "minerals": int(bot.minerals),
                "vespene": int(bot.vespene),
                "supply_used": int(bot.supply_used),
                "supply_cap": int(bot.supply_cap),
                "n_legal": diag.get("n_legal"),
                "chosen_prob": diag.get("chosen_prob"),
                "greedy_prob": diag.get("greedy_prob"),
                "model_top1": diag.get("raw_top1_name"),
                "masked_top1": diag.get("masked_top1_name"),
                "mask_blocked_top1": diag.get("blocked_top1"),
                "top_actions": diag.get("top_named"),
                "_kind": kind,
                "_target": target,
                "_progress_before": before,
            }
            if self.include_obs:
                record["obs"] = [round(float(v), 4) for v in obs]

            self._n += 1
            self._chosen[name] = self._chosen.get(name, 0) + 1
            if diag.get("blocked_top1"):
                self._blocked += 1

            self._pending = record
        except Exception:
            self._pending = None

    def finish(self, bot: BotAI | None = None, game_result=None):
        """Resolve the last decision, write the summary, close the file."""
        if not self.enabled:
            return
        self.resolve_previous(bot)

        lines = []
        def out(text=""):
            lines.append(text)
            print(text)

        out()
        out("=" * 68)
        out("  DECISION LOG SUMMARY")
        out("=" * 68)
        if game_result is not None:
            out(f"  Result: {game_result}")
        out(f"  Decisions: {self._n}")
        if self._n:
            idle = self._chosen.get("do_nothing", 0)
            out(f"  do_nothing: {idle} ({100.0 * idle / self._n:.1f}%)")
            out(f"  mask blocked the model's top choice: {self._blocked} "
                f"({100.0 * self._blocked / self._n:.1f}%)")

        out()
        out(f"  {'action':<26}{'chosen':>8}{'no-op':>8}{'suppr':>7}"
            f"{'no-op %':>9}")
        out(f"  {'-' * 26}{'-' * 8}{'-' * 8}{'-' * 7}{'-' * 9}")
        for name in sorted(self._chosen, key=lambda k: -self._chosen[k]):
            chosen = self._chosen[name]
            noop = self._noop.get(name, 0)
            suppr = self._suppressed.get(name, 0)
            measurable = chosen - suppr
            pct = (100.0 * noop / measurable) if measurable else 0.0
            flag = ""
            if measurable >= 3 and pct >= 90:
                flag = "  <-- NEVER TAKES EFFECT"
            elif measurable >= 5 and pct >= 50:
                flag = "  <-- often dropped"
            out(f"  {name:<26}{chosen:>8}{noop:>8}{suppr:>7}{pct:>8.0f}%{flag}")

        out()
        out("  no-op = chosen by the model but had no effect on the game.")
        out("  A high rate means the execution layer is dropping it (a bug),")
        out("  or it was unaffordable. 'suppr' is the deliberate")
        out("  MAX_CONCURRENT_BUILDS guard and is expected.")
        out("=" * 68)

        try:
            self.summary_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"[decision-log] summary written to {self.summary_path}")
        except Exception:
            pass
        try:
            if self._fh:
                self._fh.close()
        except Exception:
            pass
