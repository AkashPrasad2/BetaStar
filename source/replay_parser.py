"""
replay_parser.py — Queued-Action Fixed-Grid Sequence Dataset Builder
=====================================================================

Core design
-----------
The bot takes one macro action every GRID_INTERVAL_SECONDS (4s, matching
action_cooldown=22 at ~5.6 on_step/s).  The parser mirrors this exactly:

  • Time is divided into 4-second windows: [0,4), [4,8), [8,12), ...
  • The observation snapshot for window W is taken at t = W * 4s, using
    the game state BEFORE any events inside that window fire.  This is
    exactly what the live bot observes when predict_action is called.
  • Each mapped command from the replay is pushed into the next FREE window
    slot, simulating the bot's single-action-per-cooldown cadence.
  • No lag cap — every command is queued regardless of how far it gets pushed.
  • Windows with no queued command receive label 0 (do_nothing).  This teaches
    the model when NOT to act, which the previous parser never did.

OBS_SIZE = 57  (matches observation_wrapper.py exactly)
"""

from collections import defaultdict
import os
import sc2reader
import numpy as np
from sc2reader.events import (
    PlayerStatsEvent, UnitBornEvent, UnitDiedEvent,
    UnitDoneEvent, BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_INTERVAL_SECONDS = 4   # must match bot action_cooldown cadence

STRUCTURE_NAME_MAP = {
    "Nexus":             "NEXUS",
    "Pylon":             "PYLON",
    "Gateway":           "GATEWAY",
    "WarpGate":          "WARPGATE",
    "Forge":             "FORGE",
    "TwilightCouncil":   "TWILIGHTCOUNCIL",
    "PhotonCannon":      "PHOTONCANNON",
    "ShieldBattery":     "SHIELDBATTERY",
    "TemplarArchive":    "TEMPLARARCHIVE",
    "RoboticsBay":       "ROBOTICSBAY",
    "RoboticsFacility":  "ROBOTICSFACILITY",
    "Assimilator":       "ASSIMILATOR",
    "CyberneticsCore":   "CYBERNETICSCORE",
    "Stargate":          "STARGATE",
    "FleetBeacon":       "FLEETBEACON",
}

UNIT_NAME_MAP = {
    "Probe":        "PROBE",
    "Zealot":       "ZEALOT",
    "Stalker":      "STALKER",
    "HighTemplar":  "HIGHTEMPLAR",
    "Archon":       "ARCHON",
    "Immortal":     "IMMORTAL",
    "Carrier":      "CARRIER",
    "VoidRay":      "VOIDRAY",
}

STRUCTURES = [
    "NEXUS", "PYLON", "GATEWAY", "WARPGATE", "FORGE", "TWILIGHTCOUNCIL",
    "PHOTONCANNON", "SHIELDBATTERY", "TEMPLARARCHIVE", "ROBOTICSBAY",
    "ROBOTICSFACILITY", "ASSIMILATOR", "CYBERNETICSCORE", "STARGATE", "FLEETBEACON",
]
UNITS = [
    "PROBE", "ZEALOT", "STALKER", "HIGHTEMPLAR", "ARCHON", "IMMORTAL", "CARRIER", "VOIDRAY",
]

OBS_SIZE = 57  # 6 base + 15 structs + 8 units + 15 pending structs + 8 pending units + 1 opp + 4 idle

BUILD_COMMAND_TO_STRUCTURE = {
    "BuildNexus":             "NEXUS",
    "BuildPylon":             "PYLON",
    "BuildGateway":           "GATEWAY",
    "BuildForge":             "FORGE",
    "BuildTwilightCouncil":   "TWILIGHTCOUNCIL",
    "BuildPhotonCannon":      "PHOTONCANNON",
    "BuildTemplarArchive":    "TEMPLARARCHIVE",
    "BuildRoboticsFacility":  "ROBOTICSFACILITY",
    "BuildAssimilator":       "ASSIMILATOR",
    "BuildCyberneticsCore":   "CYBERNETICSCORE",
    "BuildStargate":          "STARGATE",
    "BuildFleetBeacon":       "FLEETBEACON",
}

TRAIN_COMMAND_TO_UNIT = {
    "TrainProbe":         "PROBE",
    "TrainZealot":        "ZEALOT",
    "TrainStalker":       "STALKER",
    "TrainImmortal":      "IMMORTAL",
    "TrainVoidRay":       "VOIDRAY",
    "TrainCarrier":       "CARRIER",
    "TrainHighTemplar":   "HIGHTEMPLAR",
    "WarpInZealot":       "ZEALOT",
    "WarpInStalker":      "STALKER",
    "WarpInHighTemplar":  "HIGHTEMPLAR",
}

MORPH_MAP = {
    "WarpGate": "GATEWAY",
}

# Obs feature indices (must match observation_wrapper.py)
_IDX_NEXUS = 6
_IDX_PYLON = 7
_IDX_GATEWAY = 8
_IDX_WARPGATE = 9
_IDX_FORGE = 10
_IDX_TWILIGHTCOUNCIL = 11
_IDX_PHOTONCANNON = 12
_IDX_TEMPLARARCHIVE = 14
_IDX_ROBOTICSFACILITY = 16
_IDX_CYBERNETICSCORE = 18
_IDX_STARGATE = 19
_IDX_FLEETBEACON = 20
_IDX_HIGHTEMPLAR = 24

_EPS = 0.01


def _action_legal_numpy(obs: list[float], action_id: int) -> bool:
    """
    Pure-numpy mirror of action_mask.build_legal_mask for a single obs vector.
    Must stay in sync with action_mask.py.
    do_nothing (0) is always legal.
    """
    if action_id == 0:
        return True

    has_nexus = obs[_IDX_NEXUS] > _EPS
    has_pylon = obs[_IDX_PYLON] > _EPS
    has_gateway = obs[_IDX_GATEWAY] > _EPS
    has_forge = obs[_IDX_FORGE] > _EPS
    has_twilight = obs[_IDX_TWILIGHTCOUNCIL] > _EPS
    has_temparch = obs[_IDX_TEMPLARARCHIVE] > _EPS
    has_cybcore = obs[_IDX_CYBERNETICSCORE] > _EPS
    has_stargate = obs[_IDX_STARGATE] > _EPS
    has_fleet = obs[_IDX_FLEETBEACON] > _EPS
    has_2ht = obs[_IDX_HIGHTEMPLAR] > (1.5 / 30.0)

    pending_probes = obs[44] * 30.0
    nexus_count = obs[_IDX_NEXUS] * 10.0
    probe_queue_ok = pending_probes < (2.0 * nexus_count)

    _IDLE_EPS = 0.5 / 5.0
    has_idle_gw_wg = obs[53] > _IDLE_EPS
    has_idle_sg = obs[54] > _IDLE_EPS
    has_idle_robo = obs[55] > _IDLE_EPS
    has_idle_wg = obs[56] > _IDLE_EPS

    has_army = any(obs[i] > _EPS for i in range(22, 29))

    rules = {
        1:  has_nexus and probe_queue_ok,
        2:  True,
        3:  has_pylon,
        4:  has_gateway,
        5:  has_nexus,
        6:  True,
        7:  has_pylon,
        8:  has_cybcore,
        9:  has_cybcore,
        10: has_cybcore,
        11: has_forge,
        12: has_stargate,
        13: has_twilight,
        14: has_idle_gw_wg,
        15: has_idle_gw_wg and has_cybcore,
        16: has_idle_robo,
        17: has_idle_sg,
        18: has_idle_sg and has_fleet,
        19: has_idle_gw_wg and has_temparch,
        20: has_idle_wg,
        21: has_idle_wg and has_cybcore,
        22: has_idle_wg and has_temparch,
        23: has_2ht,
        24: has_twilight,
        25: has_cybcore,
        26: has_forge,
        27: has_cybcore,
        28: has_forge,
        29: has_army,
    }
    return rules.get(action_id, False)


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

class GameState:
    """Tracks full Protoss game state including in-progress counts."""

    def __init__(self):
        self.time = 0.0
        self.minerals = 50.0
        self.vespene = 0.0
        self.supply_used = 12.0
        self.supply_cap = 15.0

        self.counts = {k: 0 for k in STRUCTURES + UNITS}
        self.pending_structures = {k: 0 for k in STRUCTURES}
        self.pending_units = {k: 0 for k in UNITS}

        self.counts["NEXUS"] = 1
        self.counts["PROBE"] = 12

        self.opp_supply_used = 0.0

    def update_from_stats(self, event: PlayerStatsEvent):
        self.time = event.second
        self.minerals = getattr(event, "minerals_current",
                                getattr(event, "minerals",  0))
        self.vespene = getattr(event, "vespene_current",
                               getattr(event, "vespene",   0))
        self.supply_used = getattr(
            event, "supply_used",      getattr(event, "food_used", 0))
        self.supply_cap = getattr(
            event, "supply_made",      getattr(event, "food_made", 0))

    def update_opp_from_stats(self, event: PlayerStatsEvent):
        self.opp_supply_used = getattr(
            event, "supply_used", getattr(event, "food_used", 0))

    def on_build_command(self, ability_name: str):
        key = BUILD_COMMAND_TO_STRUCTURE.get(ability_name)
        if key:
            self.pending_structures[key] += 1

    def on_train_command(self, ability_name: str):
        key = TRAIN_COMMAND_TO_UNIT.get(ability_name)
        if key:
            self.pending_units[key] += 1

    def unit_born_or_done(self, unit_type_name: str):
        unit_key = UNIT_NAME_MAP.get(unit_type_name)
        structure_key = STRUCTURE_NAME_MAP.get(unit_type_name)

        if unit_key:
            self.counts[unit_key] += 1
            self.pending_units[unit_key] = max(
                0, self.pending_units[unit_key] - 1)

        if structure_key:
            self.counts[structure_key] += 1
            self.pending_structures[structure_key] = max(
                0, self.pending_structures[structure_key] - 1)

        predecessor = MORPH_MAP.get(unit_type_name)
        if predecessor:
            self.counts[predecessor] = max(0, self.counts[predecessor] - 1)

    def unit_died(self, unit_type_name: str):
        key = UNIT_NAME_MAP.get(
            unit_type_name) or STRUCTURE_NAME_MAP.get(unit_type_name)
        if key:
            self.counts[key] = max(0, self.counts[key] - 1)

    def to_obs(self, override_time: float | None = None) -> list[float]:
        t = override_time if override_time is not None else self.time
        ideal_workers = max(self.counts["NEXUS"], 1) * 22
        worker_saturation = self.counts["PROBE"] / ideal_workers

        obs = [
            t / 720.0,
            self.minerals / 1800.0,
            self.vespene / 700.0,
            self.supply_used / 200.0,
            self.supply_cap / 200.0,
            worker_saturation,
        ]
        for s in STRUCTURES:
            obs.append(self.counts[s] / 10.0)
        for u in UNITS:
            obs.append(self.counts[u] / 30.0)
        for s in STRUCTURES:
            obs.append(self.pending_structures[s] / 10.0)
        for u in UNITS:
            obs.append(self.pending_units[u] / 30.0)
        obs.append(self.opp_supply_used / 200.0)

        # Idle production building features (indices 53-56)
        gw_wg_total = self.counts["GATEWAY"] + self.counts["WARPGATE"]
        gw_wg_busy = (self.pending_units["ZEALOT"]
                      + self.pending_units["STALKER"]
                      + self.pending_units["HIGHTEMPLAR"])
        idle_gw_wg = max(0, gw_wg_total - gw_wg_busy)

        sg_busy = self.pending_units["VOIDRAY"] + self.pending_units["CARRIER"]
        idle_sg = max(0, self.counts["STARGATE"] - sg_busy)

        robo_busy = self.pending_units["IMMORTAL"]
        idle_robo = max(0, self.counts["ROBOTICSFACILITY"] - robo_busy)

        wg_count = self.counts["WARPGATE"]
        idle_wg = max(
            0, wg_count - max(0, gw_wg_busy - self.counts["GATEWAY"]))

        obs.append(idle_gw_wg / 5.0)   # index 53
        obs.append(idle_sg / 5.0)   # index 54
        obs.append(idle_robo / 5.0)   # index 55
        obs.append(idle_wg / 5.0)   # index 56

        assert len(
            obs) == OBS_SIZE, f"Obs size mismatch: {len(obs)} vs {OBS_SIZE}"
        return obs


# ---------------------------------------------------------------------------
# ReplayParser
# ---------------------------------------------------------------------------

class ReplayParser:
    """
    Parses SC2 replays into fixed-grid sequence arrays for LSTM training.

    Queue mechanic
    --------------
    As mapped commands are encountered, each is assigned to the next FREE
    grid slot at or after the command's actual game time.  There is no lag
    cap — every command is queued regardless of how far it gets pushed.

    Grid slots without a queued command receive label 0 (do_nothing).

    Each row = [obs_at_window_start (OBS_SIZE floats), action_id (1 float)]
    where obs_at_window_start is snapshotted BEFORE any events in that window.
    """

    def __init__(
        self,
        replay_folder=r"C:\dev\BetaStar\replays\raw",
        output_file=r"C:\dev\BetaStar\replays\parsed\dataset.npz",
        debug=True,
    ):
        self.replay_folder = replay_folder
        self.output_file = output_file
        self.debug = debug

        self.unmapped_abilities = defaultdict(int)
        self.mapped_actions = defaultdict(int)
        self.conflicts_dropped = 0
        self.max_queue_lag_seen = 0   # diagnostic: worst-case lag observed

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
            "TrainZealot":           14,
            "TrainStalker":          15,
            "TrainImmortal":         16,
            "TrainVoidRay":          17,
            "TrainCarrier":          18,
            "TrainHighTemplar":      19,
            "WarpInZealot":          20,
            "WarpInStalker":         21,
            "WarpInHighTemplar":     22,
            "ArchonWarp":            23,
            "MorphToArchon":         23,
            "ResearchCharge":        24,
            "ResearchWarpGate":      25,
        }

    # ------------------------------------------------------------------
    # Core parse logic
    # ------------------------------------------------------------------

    def parse_replay(self, replay, min_length: int = 10) -> np.ndarray | None:
        """
        Walk replay events and build a queued-action grid.

        Algorithm
        ---------
        1. Walk all events in order, maintaining GameState and snapshotting
           obs at each new grid window boundary (BEFORE events in that window).
        2. For each mapped BasicCommandEvent at game time t:
           a. cmd_window = floor(t / G)
           b. Find the first slot >= cmd_window with no action assigned yet.
           c. Assign this action to that slot (no lag cap).
        3. After the full event walk, emit one row per window from 0 to
           last_window.  Slots with no assignment get action 0 (do_nothing).
        4. Non-zero labels that contradict the legal mask are demoted to
           do_nothing so sequence length stays aligned with real game time.
        """
        protoss_player = None
        zerg_player = None
        for player in replay.players:
            if player.play_race == "Protoss":
                protoss_player = player
            elif player.play_race == "Zerg":
                zerg_player = player

        if protoss_player is None or zerg_player is None:
            return None

        pid = protoss_player.pid
        opp_pid = zerg_player.pid

        state = GameState()
        G = GRID_INTERVAL_SECONDS

        grid_obs = {}
        grid_actions = {}   # slot -> action_id (non-zero only)

        current_grid = 0
        grid_obs[0] = state.to_obs(override_time=0.0)
        last_window = 0

        for event in replay.events:
            t = event.second

            # Snapshot the start of every new grid window BEFORE this event.
            new_grid = int(t / G)
            while current_grid < new_grid:
                current_grid += 1
                grid_obs[current_grid] = state.to_obs(
                    override_time=float(current_grid * G))
            last_window = max(last_window, new_grid)

            # --- Update game state ---
            if isinstance(event, PlayerStatsEvent):
                if event.player.pid == pid:
                    state.update_from_stats(event)
                elif event.player.pid == opp_pid:
                    state.update_opp_from_stats(event)

            elif isinstance(event, (UnitBornEvent, UnitDoneEvent)):
                unit = event.unit
                owner = getattr(unit, "owner", None)
                if owner is None or owner.pid != pid:
                    continue
                state.unit_born_or_done(unit.name)

            elif isinstance(event, UnitDiedEvent):
                unit = event.unit
                owner = getattr(unit, "owner", None)
                if owner is None or owner.pid != pid:
                    continue
                state.unit_died(unit.name)

            elif isinstance(event, (BasicCommandEvent, TargetPointCommandEvent, TargetUnitCommandEvent)):
                if event.player.pid != pid:
                    continue

                ability_name = event.ability_name

                # Always update pending counts for state accuracy
                state.on_build_command(ability_name)
                state.on_train_command(ability_name)

                action_id = self.EVENT_TO_ACTION.get(ability_name)
                if action_id is None:
                    self.unmapped_abilities[ability_name] += 1
                    if self.debug and self.unmapped_abilities[ability_name] == 1:
                        print(f"    [UNMAPPED] {ability_name}")
                    continue

                # Find next free slot — no lag cap, queue as far as needed
                cmd_window = int(t / G)
                slot = cmd_window
                while slot in grid_actions:
                    slot += 1

                lag = slot - cmd_window
                if lag > self.max_queue_lag_seen:
                    self.max_queue_lag_seen = lag
                    if self.debug:
                        print(f"    [NEW MAX LAG] {ability_name} at t={t:.1f}s "
                              f"pushed {lag} window(s) → slot {slot} "
                              f"(t={slot * G:.0f}s)")

                grid_actions[slot] = action_id
                last_window = max(last_window, slot)
                self.mapped_actions[ability_name] += 1

        # Ensure obs snapshots exist for all slots up to last_window
        while current_grid < last_window:
            current_grid += 1
            grid_obs[current_grid] = state.to_obs(
                override_time=float(current_grid * G))

        # --- Build rows ---
        rows = []
        for window in range(last_window + 1):
            obs = grid_obs.get(window)
            if obs is None:
                continue

            action_id = grid_actions.get(window, 0)   # default: do_nothing

            if action_id != 0 and not _action_legal_numpy(obs, action_id):
                self.conflicts_dropped += 1
                if self.debug:
                    print(f"    [CONFLICT] window={window} t={window*G:.0f}s "
                          f"action={action_id} illegal at snapshot — demoted to do_nothing")
                action_id = 0

            rows.append(obs + [float(action_id)])

        if len(rows) < min_length:
            return None

        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def print_statistics(self):
        print("\n" + "=" * 60)
        print("PARSING STATISTICS")
        print("=" * 60)
        print(f"\nGrid interval:          {GRID_INTERVAL_SECONDS}s")
        print(f"Max queue lag observed: {self.max_queue_lag_seen} window(s) "
              f"({self.max_queue_lag_seen * GRID_INTERVAL_SECONDS}s)")

        print("\nMapped Actions (queued into dataset):")
        for ability, count in sorted(self.mapped_actions.items(), key=lambda x: -x[1]):
            action_id = self.EVENT_TO_ACTION.get(ability, 0)
            print(f"  [{action_id:2d}] {ability:30s}: {count:5d} samples")

        total_mapped = sum(self.mapped_actions.values())
        print(f"\nTotal mapped samples:   {total_mapped}")
        print(f"Conflict demotions:     {self.conflicts_dropped} "
              f"(label illegal at snapshot time, replaced with do_nothing)")

        if self.unmapped_abilities:
            print("\nUnmapped Abilities (ignored):")
            for ability, count in sorted(self.unmapped_abilities.items(), key=lambda x: -x[1]):
                print(f"  {ability:30s}: {count:5d} occurrences")
        else:
            print("\nNo unmapped abilities found.")

    # ------------------------------------------------------------------
    # Folder processing
    # ------------------------------------------------------------------

    def parse_replay_folder(self):
        sequences = []
        skipped = 0
        failed = 0
        bot_replays = []

        replay_files = [
            f for f in os.listdir(self.replay_folder) if f.endswith(".SC2Replay")
        ]
        print(f"Found {len(replay_files)} replay(s) to process.")
        print(f"Grid interval: {GRID_INTERVAL_SECONDS}s  "
              f"(no queue lag cap — all commands preserved)\n")

        for fname in replay_files:
            path = os.path.join(self.replay_folder, fname)
            try:
                replay = sc2reader.load_replay(path, load_level=4)

                races = {p.play_race for p in replay.players}
                if races != {"Protoss", "Zerg"}:
                    skipped += 1
                    continue

                if not all(p.is_human for p in replay.players):
                    skipped += 1
                    bot_replays.append(fname)
                    continue

                seq = self.parse_replay(replay)
                if seq is None:
                    skipped += 1
                    print(f"  {fname}: too short, skipped")
                    continue

                actions = seq[:, OBS_SIZE].astype(int)
                n_do_nothing = (actions == 0).sum()
                pct_idle = 100 * n_do_nothing / len(actions)
                sequences.append(seq)
                print(f"  {fname}: {len(seq)} windows  "
                      f"(do_nothing: {n_do_nothing}/{len(seq)} = {pct_idle:.0f}%)")

            except Exception as e:
                print(f"  FAILED {fname}: {e}")
                failed += 1

        if not sequences:
            print("No training data collected.")
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
        n_do_nothing = (all_actions == 0).sum()
        pct_idle = 100 * n_do_nothing / len(all_actions)

        print(
            f"\nDone. {len(sequences)} sequences | {total_steps} total windows")
        print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.0f}")
        print(
            f"do_nothing: {n_do_nothing}/{len(all_actions)} = {pct_idle:.1f}% of all rows")
        print(f"Skipped: {skipped}  |  Failed: {failed}")
        if bot_replays:
            print(f"Bot replays skipped: {bot_replays}")
        print(f"\nSaved to: {self.output_file}")
        self.print_statistics()


if __name__ == "__main__":
    parser = ReplayParser()
    parser.parse_replay_folder()
