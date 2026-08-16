"""
replay_parser.py — Fixed-Grid Sequence Dataset Builder
=======================================================
Builds (observation, action) sequences from human replays on a fixed time grid.

State reconstruction (rewritten)
--------------------------------
State used to be reconstructed by incrementing/decrementing counters as command
and completion events streamed by. That leaked badly: a divergence audit over 40
replays found the parser reporting ~32 structures under construction by minute 28
when the true figure was ~1, error growing monotonically all game. Causes were
build commands that never produce a building (spam clicks, replaced orders),
missing cancel handling, and structures absent from the command maps entirely.

State now comes from sc2reader's per-object lifetimes, which sc2reader has
already resolved:

    under construction : started_at  <= t < finished_at
    completed & alive  : finished_at <= t < died_at
    type as of time t  : latest type_history entry <= t

This is exact, cannot drift, needs no build-time constants, and handles cancels
for free (a cancelled building simply never gets a finished_at). Using
type_history also fixes Gateway/Warpgate: sc2reader's unit.name returns the
unit's FINAL type, so a Gateway that later morphs reported as "WarpGate" for its
whole life, leaving the completed-Gateway feature near zero across the corpus.

Pending UNITS cannot come from lifetimes — a queued unit has no object until it
pops. They are instead derived by FIFO-matching each production command to the
next matching birth, so unmatched commands (cancelled, or the producing building
died) are discarded rather than leaking upward.

Observations are emitted through obs_spec.build_obs_vector(), the same function
the live bot uses at inference.
"""

from collections import defaultdict, Counter
import os
import time

import numpy as np
import sc2reader
from sc2reader.events import (
    PlayerStatsEvent, UnitBornEvent, UnitDiedEvent, UnitDoneEvent,
    BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent,
)

try:
    from sc2reader.events import UnitInitEvent
    _HAVE_UNIT_INIT = True
except ImportError:  # pragma: no cover - depends on sc2reader version
    UnitInitEvent = ()
    _HAVE_UNIT_INIT = False

import obs_spec
from obs_spec import (
    STRUCTURES, UNITS, PENDING_STRUCTURES, UPGRADE_KEYS, OBS_SIZE,
    STRUCT_IDX, UNIT_IDX, PEND_STRUCT_IDX, PEND_UNIT_IDX,
    IDX_GROUND_WEAPONS_LVL, IDX_SHIELDS_LVL, IDX_AIR_WEAPONS_LVL,
    IDX_IDLE_GW_WG, IDX_IDLE_SG, IDX_IDLE_ROBO, IDX_IDLE_WG,
    IDX_SUPPLY_USED, IDX_SUPPLY_CAP, SUPPLY_NORM,
    ACTION_SUPPLY_COST, SUPPLY_EPS, TRAINING_SUPPLY_SLACK,
    DECISION_INTERVAL_SECONDS, ACTION_NAMES, build_obs_vector,
)
from parse_log import ParseLogger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Shared with the live bot via obs_spec so the two cadences cannot drift apart.
GRID_INTERVAL_SECONDS = DECISION_INTERVAL_SECONDS

# A production command is matched to a birth at most this far in the future.
# Beyond it we assume the order never delivered.
MAX_PRODUCTION_LAG_SECONDS = 180.0

MIN_REPLAY_BUILD = 73286   # 4.0.0 — older replays are unreadable here

# Where parse logs are written (matches LOG_DIR in protoss_bot.py).
LOG_DIR = r"C:\dev\BetaStar\logs"

# ---------------------------------------------------------------------------
# sc2reader unit name -> canonical obs_spec name
# ---------------------------------------------------------------------------

STRUCTURE_NAME_MAP = {
    "Nexus":            "NEXUS",
    "Pylon":            "PYLON",
    "Gateway":          "GATEWAY",
    "WarpGate":         "WARPGATE",
    "Forge":            "FORGE",
    "TwilightCouncil":  "TWILIGHTCOUNCIL",
    "PhotonCannon":     "PHOTONCANNON",
    "ShieldBattery":    "SHIELDBATTERY",
    "TemplarArchive":   "TEMPLARARCHIVE",
    "RoboticsBay":      "ROBOTICSBAY",
    "RoboticsFacility": "ROBOTICSFACILITY",
    "Assimilator":      "ASSIMILATOR",
    "CyberneticsCore":  "CYBERNETICSCORE",
    "Stargate":         "STARGATE",
    "FleetBeacon":      "FLEETBEACON",
}

UNIT_NAME_MAP = {
    "Probe":       "PROBE",
    "Zealot":      "ZEALOT",
    "Stalker":     "STALKER",
    "HighTemplar": "HIGHTEMPLAR",
    "Archon":      "ARCHON",
    "Immortal":    "IMMORTAL",
    "Carrier":     "CARRIER",
    "VoidRay":     "VOIDRAY",
    "Adept":       "ADEPT",
    "Phoenix":     "PHOENIX",
    "Colossus":    "COLOSSUS",
}

# Supply cost of every Protoss unit, keyed by sc2reader name.
#
# This deliberately covers units that are NOT in UNIT_NAME_MAP. The feature
# vocabulary only decides which unit COUNTS become observation features; supply
# is consumed by everything. Pros build Observers, Sentries, Warp Prisms and
# Oracles constantly, so restricting the sum to the 11 tracked types would
# undercount supply badly and make the derived value useless.
#
# Structures cost no supply in Protoss, so anything absent from this table
# contributes 0 and needs no entry.
PROTOSS_SUPPLY_COST = {
    "Probe":            1,
    "Zealot":           2,
    "Stalker":          2,
    "Sentry":           2,
    "Adept":            2,
    "HighTemplar":      2,
    "DarkTemplar":      2,
    "Archon":           4,
    "Observer":         1,
    "ObserverSiegeMode": 1,
    "WarpPrism":        2,
    "WarpPrismPhasing": 2,
    "Immortal":         4,
    "Colossus":         6,
    "Disruptor":        3,
    "Phoenix":          2,
    "VoidRay":          4,
    "Oracle":           3,
    "Tempest":          5,
    "Carrier":          6,
    "Mothership":       8,
    "MothershipCore":   2,      # pre-LotV replays
    # Interceptors and AdeptPhaseShift are free; listed for the reader.
    "Interceptor":      0,
    "AdeptPhaseShift":  0,
}

# Protoss supply providers, and the hard game cap.
NEXUS_SUPPLY = 15
PYLON_SUPPLY = 8
SUPPLY_CAP_MAX = 200

TRAIN_COMMAND_TO_UNIT = {
    "TrainProbe":        "PROBE",
    "TrainZealot":       "ZEALOT",
    "TrainStalker":      "STALKER",
    "TrainImmortal":     "IMMORTAL",
    "TrainVoidRay":      "VOIDRAY",
    "TrainCarrier":      "CARRIER",
    "TrainHighTemplar":  "HIGHTEMPLAR",
    "TrainAdept":        "ADEPT",
    "TrainPhoenix":      "PHOENIX",
    "TrainColossus":     "COLOSSUS",
    "WarpInZealot":      "ZEALOT",
    "WarpInStalker":     "STALKER",
    "WarpInHighTemplar": "HIGHTEMPLAR",
    "WarpInAdept":       "ADEPT",
}

# Kept for reference/back-compat. State no longer depends on these: structure
# tracking comes from object lifetimes, which is why SHIELDBATTERY and
# ROBOTICSBAY (absent below, and absent from the old map too) are now handled.
BUILD_COMMAND_TO_STRUCTURE = {
    "BuildNexus":            "NEXUS",
    "BuildPylon":            "PYLON",
    "BuildGateway":          "GATEWAY",
    "BuildForge":            "FORGE",
    "BuildTwilightCouncil":  "TWILIGHTCOUNCIL",
    "BuildPhotonCannon":     "PHOTONCANNON",
    "BuildShieldBattery":    "SHIELDBATTERY",
    "BuildTemplarArchive":   "TEMPLARARCHIVE",
    "BuildRoboticsBay":      "ROBOTICSBAY",
    "BuildRoboticsFacility": "ROBOTICSFACILITY",
    "BuildAssimilator":      "ASSIMILATOR",
    "BuildCyberneticsCore":  "CYBERNETICSCORE",
    "BuildStargate":         "STARGATE",
    "BuildFleetBeacon":      "FLEETBEACON",
}

UPGRADE_COMMAND_TO_LEVEL = {
    "UpgradeGroundWeapons1": ("GROUND_WEAPONS", 1),
    "UpgradeGroundWeapons2": ("GROUND_WEAPONS", 2),
    "UpgradeGroundWeapons3": ("GROUND_WEAPONS", 3),
    "UpgradeShields1":       ("SHIELDS",        1),
    "UpgradeShields2":       ("SHIELDS",        2),
    "UpgradesShields3":      ("SHIELDS",        3),  # sc2reader typo variant
    "UpgradeShields3":       ("SHIELDS",        3),
    "UpgradeAirWeapons1":    ("AIR_WEAPONS",    1),
    "UpgradeAirWeapons2":    ("AIR_WEAPONS",    2),
    "UpgradeAirWeapons3":    ("AIR_WEAPONS",    3),
}

COMMAND_EVENTS = (BasicCommandEvent, TargetPointCommandEvent,
                  TargetUnitCommandEvent)

_EPS = 0.01

# ---------------------------------------------------------------------------
# Action legality (parser-side mirror of action_mask.build_training_mask)
# ---------------------------------------------------------------------------
#
# Indices are pulled from obs_spec rather than computed by hand. The previous
# version derived them arithmetically and was off by one for cybernetics core,
# stargate, robotics facility, twilight council and templar archive, so it
# checked the wrong building's pending count when validating a label.
#
# Prerequisites use PENDING-OR-COMPLETE: within one 4s window a player can
# command a building and the thing it unlocks, and the snapshot is taken at the
# window start. Requiring completion would reject valid human labels.

_IDX = STRUCT_IDX
_PIDX = PEND_STRUCT_IDX


def _action_legal_numpy(obs: list[float], action_id: int) -> tuple[bool, str]:
    """Return (is_legal, reason) for a label at this observation."""
    if action_id == 0:
        return True, ""

    def done(name: str) -> bool:
        return obs[_IDX[name]] > _EPS

    def pend(name: str) -> bool:
        return obs[_PIDX[name]] > _EPS

    def poc(name: str) -> bool:
        return done(name) or pend(name)

    has_nexus = done("NEXUS")
    has_forge = done("FORGE")
    has_fleet = done("FLEETBEACON")
    has_robobay = done("ROBOTICSBAY")
    has_twilight = done("TWILIGHTCOUNCIL")
    has_temparch = done("TEMPLARARCHIVE")

    poc_pylon = poc("PYLON")
    poc_cybcore = poc("CYBERNETICSCORE")
    poc_stargate = poc("STARGATE")
    poc_robo = poc("ROBOTICSFACILITY")
    poc_twilight = poc("TWILIGHTCOUNCIL")
    poc_temparch = poc("TEMPLARARCHIVE")
    # Warpgate has no pending slot — it is a morph, never "under construction".
    poc_warpgate = done("WARPGATE")

    # Gateway-type production: a Warpgate IS a morphed Gateway and still gates
    # the same units and the same tech (cybernetics core). Once Warp Gate research
    # finishes, pros morph every Gateway, so done("GATEWAY") legitimately drops to
    # zero while they keep producing. Requiring GATEWAY alone demoted thousands of
    # valid TrainAdept/TrainStalker/TrainZealot labels to do_nothing.
    #
    # This was previously hidden by the old command-counter leak: pending GATEWAY
    # was permanently inflated (340 commands, 14 recorded completions), so
    # pend("GATEWAY") was almost always non-zero and kept these labels alive by
    # accident. Fixing the leak removed the crutch and exposed the real
    # inconsistency -- the strict inference mask already combines GATEWAY+WARPGATE
    # via the idle_gw_wg feature, so training and inference disagreed.
    poc_gateway_type = poc("GATEWAY") or done("WARPGATE")

    under_cybcore_cap = obs[_IDX["CYBERNETICSCORE"]] < (1.5 / 10.0)

    has_army = any(
        obs[UNIT_IDX[u]] > _EPS
        for u in UNITS if u != "PROBE"
    )

    rules = {
        1:  (has_nexus, "needs nexus"),
        2:  (True, ""),
        3:  (poc_pylon, "needs poc_pylon"),
        4:  (poc_gateway_type and under_cybcore_cap,
             "needs gateway/warpgate and under_cybcore_cap"),
        5:  (has_nexus, "needs nexus"),
        6:  (True, ""),
        7:  (poc_pylon, "needs poc_pylon"),
        8:  (poc_cybcore, "needs poc_cybcore"),
        9:  (poc_cybcore, "needs poc_cybcore"),
        10: (poc_cybcore and not has_twilight,
             "needs poc_cybcore and no_twilight"),
        11: (has_forge, "needs forge"),
        12: (poc_stargate and not has_fleet, "needs poc_stargate and no_fleet"),
        13: (poc_twilight and not has_temparch,
             "needs poc_twilight and no_temparch"),
        14: (poc_robo and not has_robobay, "needs poc_robo and no_robobay"),
        15: (poc_cybcore, "needs poc_cybcore"),
        16: (poc_gateway_type, "needs gateway or warpgate"),
        17: (poc_gateway_type and poc_cybcore, "needs gateway/warpgate and poc_cybcore"),
        18: (poc_robo, "needs poc_robo"),
        19: (poc_stargate, "needs poc_stargate"),
        20: (poc_stargate and has_fleet, "needs poc_stargate and has_fleet"),
        21: (poc_gateway_type and poc_temparch,
             "needs gateway/warpgate and poc_temparch"),
        22: (poc_warpgate, "needs warpgate"),
        23: (poc_warpgate and poc_cybcore, "needs warpgate and poc_cybcore"),
        24: (poc_warpgate and poc_temparch, "needs warpgate and poc_temparch"),
        25: (poc_twilight, "needs poc_twilight"),
        26: (poc_cybcore, "needs poc_cybcore"),
        27: (has_forge, "needs forge"),
        28: (poc_cybcore, "needs poc_cybcore"),
        29: (has_forge, "needs forge"),
        30: (has_army, "needs army"),
        31: (poc_gateway_type and poc_cybcore, "needs gateway/warpgate and poc_cybcore"),
        32: (poc_stargate, "needs poc_stargate"),
        33: (poc_robo and has_robobay, "needs poc_robo and has_robobay"),
        34: (poc_warpgate and poc_cybcore, "needs warpgate and poc_cybcore"),
    }
    ok, reason = rules.get(action_id, (False, "unknown action"))
    if not ok:
        return ok, reason

    # Supply headroom, mirroring action_mask._apply_supply_gate with the same
    # training slack. These two must stay in lockstep: if this is stricter than
    # build_training_mask we keep labels the loss then masks out, and if it is
    # more lenient we demote labels the model was allowed to predict.
    cost = ACTION_SUPPLY_COST.get(action_id)
    if cost is not None:
        remaining = (obs[IDX_SUPPLY_CAP] - obs[IDX_SUPPLY_USED]) * SUPPLY_NORM
        if remaining < cost - TRAINING_SUPPLY_SLACK - SUPPLY_EPS:
            return False, f"needs {cost} supply headroom"
    return True, ""


# ---------------------------------------------------------------------------
# Frame/second calibration
# ---------------------------------------------------------------------------

def calibrate_fps(replay) -> float:
    """
    Object lifetimes (started_at / finished_at / died_at) are in FRAMES while
    events expose .second. Derive the conversion from the replay itself rather
    than hardcoding it, and sanity-check the result.
    """
    ratios = []
    for event in replay.events:
        second = getattr(event, "second", 0)
        frame = getattr(event, "frame", None)
        if frame and second and second > 0:
            ratios.append(frame / second)
            if len(ratios) >= 400:
                break
    if not ratios:
        return 16.0
    fps = float(np.median(ratios))
    if not (1.0 < fps < 100.0):
        return 16.0
    return fps


# ---------------------------------------------------------------------------
# Windowed state reconstruction
# ---------------------------------------------------------------------------

class WindowedState:
    """
    Per-window game state for one player, reconstructed from object lifetimes.

    Exposes counts at each grid window:
        structures_done / units_done       (completed and alive)
        structures_pending                 (under construction)
        units_pending                      (ordered, not yet delivered)
        minerals / vespene / supply        (from PlayerStatsEvent, held forward)
        upgrade_lvls                       (highest level commanded)
    """

    def __init__(self, replay, pid: int, grid: int = GRID_INTERVAL_SECONDS):
        self.grid = grid
        self.fps = calibrate_fps(replay)
        self.pid = pid

        self._last_window = 0

        # window -> list of (bucket, key, delta)
        self._deltas: dict[int, list[tuple[str, str, int]]] = defaultdict(list)

        # window -> change in supply consumed. Derived from object lifetimes for
        # the same reason the counts are: PlayerStatsEvent only fires about every
        # 10s, so the stats-event supply is up to 10s stale against a 4s grid,
        # while the live bot reads bot.supply_used exactly. Deriving it here makes
        # the training value frame-accurate and kills that train/inference skew.
        self._supply_deltas: dict[int, float] = defaultdict(float)

        self._resource_samples: list[tuple[float, float, float, float, float]] = []
        self._upgrade_events: list[tuple[float, str, int]] = []
        self._production_commands: list[tuple[float, str]] = []
        self._births: dict[str, list[float]] = defaultdict(list)

        self._scan_events(replay)
        self._scan_objects(replay)
        self._match_production()
        self._accumulate()

    # -- frame/time helpers ------------------------------------------------

    def _sec(self, frame) -> float:
        return float(frame) / self.fps

    def _win(self, frame) -> int:
        return int(self._sec(frame) / self.grid)

    def _win_from_sec(self, second: float) -> int:
        return int(second / self.grid)

    def _note_window(self, w: int):
        if w > self._last_window:
            self._last_window = w

    # -- pass 1: events ----------------------------------------------------

    def _scan_events(self, replay):
        for event in replay.events:
            second = getattr(event, "second", 0)
            self._note_window(self._win_from_sec(second))

            if isinstance(event, PlayerStatsEvent):
                if event.player.pid != self.pid:
                    continue
                minerals = getattr(event, "minerals_current",
                                   getattr(event, "minerals", 0)) or 0
                vespene = getattr(event, "vespene_current",
                                  getattr(event, "vespene", 0)) or 0
                supply_used = getattr(event, "supply_used",
                                      getattr(event, "food_used", 0)) or 0
                supply_cap = getattr(event, "supply_made",
                                     getattr(event, "food_made", 0)) or 0
                self._resource_samples.append(
                    (float(second), float(minerals), float(vespene),
                     float(supply_used), float(supply_cap)))

            elif isinstance(event, COMMAND_EVENTS):
                if event.player.pid != self.pid:
                    continue
                ability = event.ability_name
                unit_key = TRAIN_COMMAND_TO_UNIT.get(ability)
                if unit_key:
                    self._production_commands.append((float(second), unit_key))
                upgrade = UPGRADE_COMMAND_TO_LEVEL.get(ability)
                if upgrade:
                    key, level = upgrade
                    self._upgrade_events.append((float(second), key, level))

        self._resource_samples.sort(key=lambda s: s[0])
        self._upgrade_events.sort(key=lambda s: s[0])
        self._production_commands.sort(key=lambda s: s[0])

    # -- pass 2: object lifetimes -----------------------------------------

    def _type_segments(self, unit) -> list[tuple[int, str | None]]:
        """[(frame, canonical_name_or_None)] sorted, from sc2reader type_history."""
        history = getattr(unit, "type_history", None)
        segments: list[tuple[int, str | None]] = []
        if history:
            for frame, unit_type in sorted(history.items()):
                name = getattr(unit_type, "name", None)
                segments.append((int(frame), name))
        if not segments:
            segments = [(0, getattr(unit, "name", None))]
        return segments

    @staticmethod
    def _canonical(name: str | None) -> tuple[str, str] | None:
        """Return (bucket, canonical_name) or None if not tracked."""
        if not name:
            return None
        if name in STRUCTURE_NAME_MAP:
            return ("structure", STRUCTURE_NAME_MAP[name])
        if name in UNIT_NAME_MAP:
            return ("unit", UNIT_NAME_MAP[name])
        return None

    def _emit_supply(self, cost: float, w_start: int, w_end: int | None):
        """Hold `cost` supply over [w_start, w_end), None meaning until the end."""
        if w_end is not None and w_end <= w_start:
            return
        self._supply_deltas[w_start] += cost
        if w_end is not None:
            self._supply_deltas[w_end] -= cost
            self._note_window(w_end)
        self._note_window(w_start)

    def _emit(self, bucket: str, key: str, w_start: int, w_end: int | None):
        """Add +1 over [w_start, w_end), where None means 'until the end'."""
        if w_end is not None and w_end <= w_start:
            return
        self._deltas[w_start].append((bucket, key, +1))
        if w_end is not None:
            self._deltas[w_end].append((bucket, key, -1))
            self._note_window(w_end)
        self._note_window(w_start)

    def _scan_objects(self, replay):
        objects = getattr(replay, "objects", None) or {}
        for unit in objects.values():
            owner = getattr(unit, "owner", None)
            if owner is None or getattr(owner, "pid", None) != self.pid:
                continue

            started = getattr(unit, "started_at", None)
            finished = getattr(unit, "finished_at", None)
            died = getattr(unit, "died_at", None)

            segments = self._type_segments(unit)
            first_name = segments[0][1] if segments else None
            first = self._canonical(first_name)

            # --- supply consumed: [started_at, died_at) -----------------------
            # SC2 charges supply the moment production begins and refunds it when
            # the unit dies (or when production is cancelled, which also sets
            # died_at). Supply cost is invariant across a unit's morphs -- two
            # 2-supply templar merging into a 4-supply archon conserves it, and
            # the merge shows up as two deaths plus one birth -- so keying on the
            # first type name is correct.
            cost = PROTOSS_SUPPLY_COST.get(first_name, 0)
            if cost:
                # Starting probes exist from frame 0 with no started_at.
                supply_from = started if started is not None else (finished or 0)
                self._emit_supply(cost, self._win(supply_from),
                                  self._win(died) if died is not None else None)

            # --- under construction: [started_at, finished_at) ---
            if first is not None and first[0] == "structure" and started is not None:
                if finished is not None and finished > started:
                    self._emit("pending_struct", first[1],
                               self._win(started), self._win(finished))
                elif finished is None and died is not None and died > started:
                    # Cancelled or destroyed mid-construction.
                    self._emit("pending_struct", first[1],
                               self._win(started), self._win(died))

            # --- completed and alive: [finished_at, died_at) ---
            if finished is None:
                continue
            end_win = self._win(died) if died is not None else None
            start_win = self._win(finished)

            # Split the alive interval by type, so a Gateway counts as a
            # Gateway until the frame it morphs into a Warpgate.
            for i, (seg_frame, seg_name) in enumerate(segments):
                canon = self._canonical(seg_name)
                if canon is None:
                    continue
                bucket = ("done_struct" if canon[0] == "structure"
                          else "done_unit")

                seg_start = max(self._win(seg_frame), start_win)
                if i + 1 < len(segments):
                    seg_end = self._win(segments[i + 1][0])
                    if end_win is not None:
                        seg_end = min(seg_end, end_win)
                else:
                    seg_end = end_win

                self._emit(bucket, canon[1], seg_start, seg_end)

                if bucket == "done_unit" and i == 0:
                    self._births[canon[1]].append(self._sec(finished))

        for key in self._births:
            self._births[key].sort()

    # -- pass 3: pending units via FIFO command->birth matching ------------

    def _match_production(self):
        """
        Pair each production command with the next unmatched birth of that unit
        type. Pending units at time t = pairs with cmd_time <= t < birth_time.
        Commands with no matching birth inside MAX_PRODUCTION_LAG_SECONDS are
        dropped, so failed orders cannot accumulate.
        """
        cursor: dict[str, int] = defaultdict(int)
        for cmd_time, key in self._production_commands:
            births = self._births.get(key)
            if not births:
                continue
            i = cursor[key]
            while i < len(births) and births[i] < cmd_time:
                i += 1
            if i >= len(births):
                cursor[key] = i
                continue
            birth = births[i]
            if birth - cmd_time > MAX_PRODUCTION_LAG_SECONDS:
                cursor[key] = i
                continue
            cursor[key] = i + 1
            self._emit("pending_unit", key,
                       self._win_from_sec(cmd_time),
                       self._win_from_sec(birth))

    # -- pass 4: prefix-sum the deltas into per-window snapshots ------------

    def _accumulate(self):
        n = self._last_window + 1
        counts = {
            "done_struct": {k: 0 for k in STRUCTURES},
            "done_unit": {k: 0 for k in UNITS},
            "pending_struct": {k: 0 for k in PENDING_STRUCTURES},
            "pending_unit": {k: 0 for k in UNITS},
        }

        self.windows: list[dict] = []

        res_i = 0
        minerals, vespene = 50.0, 0.0
        # Kept only as a fallback if a replay yields no usable object lifetimes.
        stats_supply_used, stats_supply_cap = 12.0, 15.0
        supply_used = 0.0

        upg_i = 0
        upgrade_lvls = {k: 0 for k in UPGRADE_KEYS}

        for w in range(n):
            for bucket, key, delta in self._deltas.get(w, ()):
                bucket_counts = counts.get(bucket)
                if bucket_counts is None or key not in bucket_counts:
                    continue
                bucket_counts[key] = max(0, bucket_counts[key] + delta)

            t = w * self.grid

            while (res_i < len(self._resource_samples)
                   and self._resource_samples[res_i][0] <= t):
                _, minerals, vespene, stats_supply_used, stats_supply_cap = \
                    self._resource_samples[res_i]
                res_i += 1

            # Supply, derived rather than sampled. Both terms come from
            # frame-accurate object lifetimes, so unlike minerals/vespene these
            # cannot lag the 4s grid. Cap is whatever our COMPLETED pylons and
            # nexuses provide; the game hard-caps it at 200.
            supply_used += self._supply_deltas.get(w, 0.0)
            done_struct = counts["done_struct"]
            derived_cap = min(
                float(SUPPLY_CAP_MAX),
                NEXUS_SUPPLY * done_struct.get("NEXUS", 0)
                + PYLON_SUPPLY * done_struct.get("PYLON", 0))
            if derived_cap > 0.0:
                w_supply_used, w_supply_cap = supply_used, derived_cap
            else:
                # No lifetimes recovered (very old or truncated replay): fall back
                # to the stats event. Never write back into `supply_used`, which
                # is a running accumulator for the remaining windows.
                w_supply_used, w_supply_cap = stats_supply_used, stats_supply_cap

            while (upg_i < len(self._upgrade_events)
                   and self._upgrade_events[upg_i][0] <= t):
                _, key, level = self._upgrade_events[upg_i]
                upgrade_lvls[key] = max(upgrade_lvls[key], level)
                upg_i += 1

            self.windows.append({
                "time": float(t),
                "minerals": minerals,
                "vespene": vespene,
                "supply_used": w_supply_used,
                "supply_cap": w_supply_cap,
                "structures_done": dict(counts["done_struct"]),
                "units_done": dict(counts["done_unit"]),
                "structures_pending": dict(counts["pending_struct"]),
                "units_pending": dict(counts["pending_unit"]),
                "upgrade_lvls": dict(upgrade_lvls),
            })

    # -- public ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.windows)

    def obs_at(self, window: int) -> list[float]:
        w = self.windows[min(window, len(self.windows) - 1)]
        return build_obs_vector(
            time_s=w["time"],
            minerals=w["minerals"],
            vespene=w["vespene"],
            supply_used=w["supply_used"],
            supply_cap=w["supply_cap"],
            structures_done=w["structures_done"],
            units_done=w["units_done"],
            structures_pending=w["structures_pending"],
            units_pending=w["units_pending"],
            upgrade_lvls=w["upgrade_lvls"],
        )


# ---------------------------------------------------------------------------
# ReplayParser
# ---------------------------------------------------------------------------

class ReplayParser:
    def __init__(
        self,
        replay_folder=r"C:\dev\BetaStar\replays\raw",
        output_file=r"C:\dev\BetaStar\replays\parsed\dataset.npz",
        debug=True,
        log_dir=LOG_DIR,
    ):
        self.replay_folder = replay_folder
        self.output_file = output_file
        self.debug = debug
        self.log_dir = log_dir

        self.unmapped_abilities = defaultdict(int)
        self.mapped_actions = defaultdict(int)
        self.conflicts_dropped = 0
        self.max_queue_lag_seen = 0

        # Per-replay accumulators, reset at the start of each parse_replay.
        # Conflicts are keyed by (action_id, action_name, reason) so the log can
        # answer "which actions are losing labels, and why" -- previously only a
        # single global total existed, and the detail was buried in stdout spam.
        self._replay_conflicts: Counter = Counter()
        self._replay_actions: Counter = Counter()
        self._replay_max_lag = 0

        self.EVENT_TO_ACTION = {
            "TrainProbe":             1,
            "BuildPylon":             2,
            "BuildGateway":           3,
            "BuildCyberneticsCore":   4,
            "BuildAssimilator":       5,
            "BuildNexus":             6,
            "BuildForge":             7,
            "BuildStargate":          8,
            "BuildRoboticsFacility":  9,
            "BuildTwilightCouncil":  10,
            "BuildPhotonCannon":     11,
            "BuildFleetBeacon":      12,
            "BuildTemplarArchive":   13,
            "BuildRoboticsBay":      14,
            "BuildShieldBattery":    15,
            "TrainZealot":           16,
            "TrainStalker":          17,
            "TrainImmortal":         18,
            "TrainVoidRay":          19,
            "TrainCarrier":          20,
            "TrainHighTemplar":      21,
            "WarpInZealot":          22,
            "WarpInStalker":         23,
            "WarpInHighTemplar":     24,
            "ResearchCharge":        25,
            "ResearchWarpGate":      26,
            "UpgradeGroundWeapons1": 27,
            "UpgradeGroundWeapons2": 27,
            "UpgradeGroundWeapons3": 27,
            "UpgradeAirWeapons1":    28,
            "UpgradeAirWeapons2":    28,
            "UpgradeAirWeapons3":    28,
            "UpgradeShields1":       29,
            "UpgradeShields2":       29,
            "UpgradesShields3":      29,   # sc2reader typo variant
            "UpgradeShields3":       29,
            "TrainAdept":            31,
            "TrainPhoenix":          32,
            "TrainColossus":         33,
            "WarpInAdept":           34,
        }
        self._action_names = {v: k for k, v in self.EVENT_TO_ACTION.items()}

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def find_protoss_pid(replay) -> int | None:
        for player in replay.players:
            if player.play_race == "Protoss":
                return player.pid
        return None

    def _collect_actions(self, replay, pid: int) -> tuple[dict[int, int], int]:
        """
        Map command events onto grid slots. One action per window; collisions
        are pushed to the next free slot.
        """
        grid_actions: dict[int, int] = {}
        last_window = 0
        G = GRID_INTERVAL_SECONDS

        for event in replay.events:
            if not isinstance(event, COMMAND_EVENTS):
                continue
            if event.player.pid != pid:
                continue

            ability = event.ability_name
            action_id = self.EVENT_TO_ACTION.get(ability)
            if action_id is None:
                self.unmapped_abilities[ability] += 1
                if self.debug and self.unmapped_abilities[ability] == 1:
                    print(f"    [UNMAPPED] {ability}")
                continue

            cmd_window = int(event.second / G)
            slot = cmd_window
            while slot in grid_actions:
                slot += 1

            lag = slot - cmd_window
            if lag > self.max_queue_lag_seen:
                self.max_queue_lag_seen = lag
            if lag > self._replay_max_lag:
                self._replay_max_lag = lag

            grid_actions[slot] = action_id
            last_window = max(last_window, slot)
            self.mapped_actions[ability] += 1

        return grid_actions, last_window

    # -- main entry point --------------------------------------------------

    def parse_replay(self, replay, min_length: int = 10) -> np.ndarray | None:
        self._replay_conflicts = Counter()
        self._replay_actions = Counter()
        self._replay_max_lag = 0

        if getattr(replay, "build", 0) < MIN_REPLAY_BUILD:
            if self.debug:
                print(f"    [SKIP] build {getattr(replay, 'build', '?')} "
                      f"older than {MIN_REPLAY_BUILD}")
            return None

        pid = self.find_protoss_pid(replay)
        if pid is None:
            return None

        state = WindowedState(replay, pid)
        grid_actions, action_last_window = self._collect_actions(replay, pid)
        last_window = max(len(state) - 1, action_last_window)

        rows = []
        for window in range(last_window + 1):
            obs = state.obs_at(window)
            action_id = grid_actions.get(window, 0)

            if action_id != 0:
                is_legal, reason = _action_legal_numpy(obs, action_id)
                if not is_legal:
                    self.conflicts_dropped += 1
                    name = self._action_names.get(action_id, "unknown")
                    self._replay_conflicts[(action_id, name, reason)] += 1
                    if self.debug:
                        self._print_conflict(window, action_id, reason, obs)
                    action_id = 0

            self._replay_actions[action_id] += 1
            rows.append(obs + [float(action_id)])

        if len(rows) < min_length:
            return None

        return np.array(rows, dtype=np.float32)

    def _print_conflict(self, window: int, action_id: int, reason: str,
                        obs: list[float]):
        name = self._action_names.get(action_id, "unknown")
        parts = []
        for s in STRUCTURES:
            done = obs[STRUCT_IDX[s]] * obs_spec.STRUCT_NORM
            pend = (obs[PEND_STRUCT_IDX[s]] * obs_spec.STRUCT_NORM
                    if s in PEND_STRUCT_IDX else 0.0)
            if done > 0 or pend > 0:
                parts.append(f"{s}(h={done:.0f},p={pend:.0f})")
        for u in UNITS:
            done = obs[UNIT_IDX[u]] * obs_spec.UNIT_NORM
            pend = obs[PEND_UNIT_IDX[u]] * obs_spec.UNIT_NORM
            if done > 0 or pend > 0:
                parts.append(f"{u}(h={done:.0f},p={pend:.0f})")
        state_str = ", ".join(parts) if parts else "No structures/units"
        print(f"    [CONFLICT] window={window} action={action_id} ({name}) "
              f"- Failed: {reason}")
        print(f"               State: {state_str}")

    # -- statistics --------------------------------------------------------

    def print_statistics(self):
        print("\n" + "=" * 60)
        print("PARSING STATISTICS")
        print("=" * 60)
        print(f"\nGrid interval:          {GRID_INTERVAL_SECONDS}s")
        print(f"Max queue lag observed: {self.max_queue_lag_seen} window(s) "
              f"({self.max_queue_lag_seen * GRID_INTERVAL_SECONDS}s)")

        print("\nMapped Actions (queued into dataset):")
        for ability, count in sorted(self.mapped_actions.items(),
                                     key=lambda x: -x[1]):
            action_id = self.EVENT_TO_ACTION.get(ability, 0)
            print(f"  [{action_id:2d}] {ability:30s}: {count:5d} samples")

        total_mapped = sum(self.mapped_actions.values())
        print(f"\nTotal mapped samples:   {total_mapped}")
        print(f"Conflict demotions:     {self.conflicts_dropped} "
              f"(label illegal at snapshot time, replaced with do_nothing)")

        if self.unmapped_abilities:
            print("\nUnmapped Abilities (ignored):")
            for ability, count in sorted(self.unmapped_abilities.items(),
                                         key=lambda x: -x[1]):
                print(f"  {ability:30s}: {count:5d} occurrences")
        else:
            print("\nNo unmapped abilities found.")

    def parse_replay_folder(self):
        sequences = []
        skipped = failed = 0
        bot_replays = []

        replay_files = [f for f in os.listdir(self.replay_folder)
                        if f.endswith(".SC2Replay")]
        print(f"Found {len(replay_files)} replay(s) to process.")
        print(f"Grid interval: {GRID_INTERVAL_SECONDS}s | "
              f"state from object lifetimes | UnitInit available: "
              f"{_HAVE_UNIT_INIT}\n")

        log = ParseLogger(self.log_dir, meta={
            "grid_interval_seconds": GRID_INTERVAL_SECONDS,
            "obs_size": OBS_SIZE,
            "num_actions": len(ACTION_NAMES),
            "replay_folder": self.replay_folder,
            "output_file": self.output_file,
            "sc2reader": getattr(sc2reader, "__version__", "unknown"),
            "unit_init_available": _HAVE_UNIT_INIT,
            "min_replay_build": MIN_REPLAY_BUILD,
            "time_norm": obs_spec.TIME_NORM,
            "actions": ACTION_NAMES,
            "feature_names": obs_spec.feature_names(),
        })

        for fname in replay_files:
            path = os.path.join(self.replay_folder, fname)
            t_start = time.time()
            try:
                replay = sc2reader.load_replay(path, load_level=4)

                races = {p.play_race for p in replay.players}
                if "Protoss" not in races:
                    skipped += 1
                    log.replay_skipped(fname, "no protoss player")
                    continue

                if not all(p.is_human for p in replay.players):
                    skipped += 1
                    bot_replays.append(fname)
                    log.replay_skipped(fname, "contains a computer player")
                    continue

                seq = self.parse_replay(replay)
                build = getattr(replay, "build", 0)
                if seq is None:
                    skipped += 1
                    if 0 < build < MIN_REPLAY_BUILD:
                        print(f"  {fname}: skipped (old patch {build})")
                        log.replay_skipped(fname, "replay build too old",
                                           build=build)
                    else:
                        print(f"  {fname}: too short, skipped")
                        log.replay_skipped(fname, "too short", build=build)
                    continue

                actions = seq[:, OBS_SIZE].astype(int)
                n_idle = int((actions == 0).sum())
                pct_idle = 100.0 * n_idle / len(actions)
                sequences.append(seq)
                log.replay_parsed(
                    fname,
                    build=build,
                    windows=len(seq),
                    action_counts=self._replay_actions,
                    conflicts=self._replay_conflicts,
                    max_lag=self._replay_max_lag,
                    seconds=time.time() - t_start,
                )
                print(f"  {fname}: {len(seq)} windows  "
                      f"(do_nothing: {n_idle}/{len(seq)} = {pct_idle:.0f}%)")

            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  FAILED {fname}: {exc}")
                log.replay_failed(fname, f"{type(exc).__name__}: {exc}")
                failed += 1

        if not sequences:
            print("No training data collected.")
            log.finish(action_names=ACTION_NAMES,
                       unmapped=dict(self.unmapped_abilities),
                       grid_seconds=GRID_INTERVAL_SECONDS)
            return

        seq_array = np.empty(len(sequences), dtype=object)
        for i, s in enumerate(sequences):
            seq_array[i] = s

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        np.savez(self.output_file, sequences=seq_array)

        total_steps = sum(len(s) for s in sequences)
        lengths = [len(s) for s in sequences]
        all_actions = np.concatenate(
            [s[:, OBS_SIZE].astype(int) for s in sequences])
        n_idle = int((all_actions == 0).sum())
        pct_idle = 100.0 * n_idle / len(all_actions)

        print(f"\nDone. {len(sequences)} sequences | {total_steps} total windows")
        print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.0f}")
        print(f"do_nothing: {n_idle}/{len(all_actions)} = {pct_idle:.1f}% of rows")
        print(f"Skipped: {skipped}  |  Failed: {failed}")
        if bot_replays:
            print(f"Bot replays skipped: {bot_replays}")
        print(f"\nSaved to: {self.output_file}")
        self.print_statistics()

        log.finish(
            action_names=ACTION_NAMES,
            unmapped=dict(self.unmapped_abilities),
            dataset_path=self.output_file,
            grid_seconds=GRID_INTERVAL_SECONDS,
            extra={
                "Max queue lag": f"{self.max_queue_lag_seen} window(s) "
                                 f"({self.max_queue_lag_seen * GRID_INTERVAL_SECONDS}s)",
                "Bot replays": len(bot_replays),
            },
        )


if __name__ == "__main__":
    ReplayParser().parse_replay_folder()
