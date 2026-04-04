"""
replay_parser.py — Queued-Action Fixed-Grid Sequence Dataset Builder
=====================================================================
[unchanged header — see original for full docstring]

Changes in this version
-----------------------
- Removed probe_queue_ok from _action_legal_numpy. The queue-cap check is
  correct at inference time but wrong for parsing: the parser's pending probe
  count drifts upward over long games because cancelled/lost probes don't
  always fire UnitBornEvent to decrement the counter. This was silently
  demoting dozens of valid probe-train labels per replay.

- Added 1-of-building caps for CYBERNETICSCORE, TWILIGHTCOUNCIL, FLEETBEACON,
  TEMPLARARCHIVE. Pros never build a second of these. Capping the mask here
  prevents the model from ever learning to build duplicates, and also avoids
  the (rare) human duplicate build from polluting training labels.

- _IDX_SHIELDBATTERY added (was missing, causing index offset comment confusion).
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

GRID_INTERVAL_SECONDS = 4

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

OBS_SIZE = 57

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

# Obs feature indices — completed structures (indices 6-20, matching observation_wrapper.py)
_IDX_NEXUS = 6
_IDX_PYLON = 7
_IDX_GATEWAY = 8
_IDX_WARPGATE = 9
_IDX_FORGE = 10
_IDX_TWILIGHTCOUNCIL = 11
_IDX_PHOTONCANNON = 12
_IDX_SHIELDBATTERY = 13   # present in obs, not used by mask but listed for clarity
_IDX_TEMPLARARCHIVE = 14
_IDX_ROBOTICSBAY = 15
_IDX_ROBOTICSFACILITY = 16
_IDX_ASSIMILATOR = 17
_IDX_CYBERNETICSCORE = 18
_IDX_STARGATE = 19
_IDX_FLEETBEACON = 20

# Completed units (indices 21-28)
_IDX_HIGHTEMPLAR = 24

_EPS = 0.01


def _action_legal_numpy(obs: list[float], action_id: int) -> bool:
    """
    Pure-numpy mirror of action_mask.build_legal_mask for a single obs vector,
    with PARSER-SPECIFIC relaxations to avoid false conflict demotions.

    Key differences from the inference mask (action_mask.py):
    ----------------------------------------------------------------
    1. probe_queue_ok is NOT checked. The parser's pending probe count drifts
       over long games causing false-illegal calls on valid probe trains.

    2. Prerequisite structure checks use PENDING-OR-COMPLETE rather than
       COMPLETE-ONLY. Pro players queue the next building the moment the
       prerequisite is placed (not when it finishes, ~60s later). The window
       snapshot is taken at the START of the window, so a gateway placed at
       t=128s won't appear as completed until t~190s. Using pending-or-complete
       means "the player has committed to building this" which is the right
       semantic for parsing intent.

       This affects: build_cyberneticscore (needs gateway), train_adept/stalker
       (needs cybcore), warp_in_stalker (needs warpgate), train_immortal (needs
       robo), train_voidray (needs stargate), and the entire tech tree chain.

    3. 1-of building caps are applied (same as inference mask): cybercore,
       twilight council, fleet beacon, templar archive. Pros never build
       duplicates — any such label in a replay is noise.

    4. Idle building checks are REMOVED for unit training actions. The parser's
       idle counts (indices 53-56) are derived from pending_units which drift
       for the same reason as pending probes. A pro training a stalker is valid
       if a gateway is pending-or-complete + cybcore is pending-or-complete,
       regardless of what the idle count says at snapshot time.

    The inference mask keeps all strict checks because the bot must actually
    be able to execute the action right now.
    """
    if action_id == 0:
        return True

    # Completed structure counts (indices 6-20, normalised /10)
    has_nexus = obs[_IDX_NEXUS] > _EPS
    has_pylon = obs[_IDX_PYLON] > _EPS
    has_gateway = obs[_IDX_GATEWAY] > _EPS
    has_warpgate = obs[_IDX_WARPGATE] > _EPS
    has_forge = obs[_IDX_FORGE] > _EPS
    has_twilight = obs[_IDX_TWILIGHTCOUNCIL] > _EPS
    has_temparch = obs[_IDX_TEMPLARARCHIVE] > _EPS
    has_cybcore = obs[_IDX_CYBERNETICSCORE] > _EPS
    has_stargate = obs[_IDX_STARGATE] > _EPS
    has_fleet = obs[_IDX_FLEETBEACON] > _EPS
    has_robobay = obs[_IDX_ROBOTICSBAY] > _EPS
    has_robo = obs[_IDX_ROBOTICSFACILITY] > _EPS
    has_2ht = obs[_IDX_HIGHTEMPLAR] > (1.5 / 30.0)
    has_army = any(obs[i] > _EPS for i in range(22, 29))

    # Pending structure counts (indices 29-43, same order as completed, /10)
    # Index mapping: pending_NEXUS=29, PYLON=30, GATEWAY=31, WARPGATE=32,
    # FORGE=33, TWILIGHTCOUNCIL=34, PHOTONCANNON=35, SHIELDBATTERY=36,
    # TEMPLARARCHIVE=37, ROBOTICSBAY=38, ROBOTICSFACILITY=39, ASSIMILATOR=40,
    # CYBERNETICSCORE=41, STARGATE=42, FLEETBEACON=43
    _P = 29  # pending block starts at index 29
    pend_gateway = obs[_P + 2] > _EPS   # GATEWAY is 3rd in STRUCTURES list
    pend_cybcore = obs[_P + 12] > _EPS   # CYBERNETICSCORE is 13th
    pend_stargate = obs[_P + 13] > _EPS   # STARGATE is 14th
    pend_robo = obs[_P + 10] > _EPS   # ROBOTICSFACILITY is 11th
    pend_twilight = obs[_P + 5] > _EPS   # TWILIGHTCOUNCIL is 6th
    pend_warpgate = obs[_P + 3] > _EPS   # WARPGATE is 4th
    pend_temparch = obs[_P + 8] > _EPS   # TEMPLARARCHIVE is 9th

    # Pending-or-complete: "player has committed to building this"
    poc_gateway = has_gateway or pend_gateway
    poc_cybcore = has_cybcore or pend_cybcore
    poc_stargate = has_stargate or pend_stargate
    poc_robo = has_robo or pend_robo
    poc_twilight = has_twilight or pend_twilight
    poc_warpgate = has_warpgate or pend_warpgate
    poc_temparch = has_temparch or pend_temparch

    # 1-of building caps
    no_cybcore = not has_cybcore
    no_twilight = not has_twilight
    no_fleet = not has_fleet
    no_temparch = not has_temparch

    rules = {
        # train_probe: needs nexus (no queue cap in parser)
        1:  has_nexus,

        # build_pylon: always legal
        2:  True,

        # build_gateway: needs pylon
        3:  has_pylon,

        # build_cyberneticscore: gateway pending-or-complete, and no existing cybcore
        4:  poc_gateway and no_cybcore,

        # build_assimilator: needs nexus
        5:  has_nexus,

        # build_nexus: always legal
        6:  True,

        # build_forge: needs pylon
        7:  has_pylon,

        # build_stargate: cybcore pending-or-complete
        8:  poc_cybcore,

        # build_robotics_facility: cybcore pending-or-complete
        9:  poc_cybcore,

        # build_twilight_council: cybcore pending-or-complete, no existing twilight
        10: poc_cybcore and no_twilight,

        # build_photon_cannon: needs completed forge
        11: has_forge,

        # build_fleet_beacon: stargate pending-or-complete, no existing fleet beacon
        12: poc_stargate and no_fleet,

        # build_templar_archive: twilight pending-or-complete, no existing templar archive
        13: poc_twilight and no_temparch,

        # train_zealot: gateway pending-or-complete (don't require idle count — drifts)
        14: poc_gateway,

        # train_stalker: gateway + cybcore both pending-or-complete
        15: poc_gateway and poc_cybcore,

        # train_immortal: robo pending-or-complete
        16: poc_robo,

        # train_voidray: stargate pending-or-complete
        17: poc_stargate,

        # train_carrier: stargate pending-or-complete + fleet beacon complete
        18: poc_stargate and has_fleet,

        # train_high_templar: gateway pending-or-complete + templar archive pending-or-complete
        19: poc_gateway and poc_temparch,

        # warp_in_zealot: warpgate pending-or-complete
        20: poc_warpgate,

        # warp_in_stalker: warpgate + cybcore both pending-or-complete
        21: poc_warpgate and poc_cybcore,

        # warp_in_high_templar: warpgate + templar archive both pending-or-complete
        22: poc_warpgate and poc_temparch,

        # archon_warp: needs 2 completed high templars
        23: has_2ht,

        # research_charge: twilight pending-or-complete
        24: poc_twilight,

        # research_warp_gate: cybcore pending-or-complete
        25: poc_cybcore,

        # upgrade_ground_weapons: needs completed forge
        26: has_forge,

        # upgrade_air_weapons: cybcore pending-or-complete
        27: poc_cybcore,

        # upgrade_shields: needs completed forge
        28: has_forge,

        # attack_enemy_base: needs army
        29: has_army,

        # train_adept: gateway + cybcore pending-or-complete
        30: poc_gateway and poc_cybcore,

        # train_phoenix: stargate pending-or-complete
        31: poc_stargate,

        # train_colossus: robo pending-or-complete + robobay complete
        32: poc_robo and has_robobay,

        # warp_in_adept: warpgate + cybcore pending-or-complete
        33: poc_warpgate and poc_cybcore,
    }
    return rules.get(action_id, False)


# ---------------------------------------------------------------------------
# GameState  (unchanged from original)
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
            event, "supply_used",  getattr(event, "food_used", 0))
        self.supply_cap = getattr(
            event, "supply_made",  getattr(event, "food_made", 0))

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
# ReplayParser  (unchanged from original except conflict log is more specific)
# ---------------------------------------------------------------------------

class ReplayParser:
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
        self.max_queue_lag_seen = 0

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
            "ArchonWarpSelection":   23,
            "MorphToArchon":         23,
            "ResearchCharge":        24,
            "ResearchWarpGate":      25,
            "TrainAdept":            30,
            "TrainPhoenix":          31,
            "TrainColossus":         32,
            "WarpInAdept":           33,
        }

    def parse_replay(self, replay, min_length: int = 10) -> np.ndarray | None:
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

            new_grid = int(t / G)
            while current_grid < new_grid:
                current_grid += 1
                grid_obs[current_grid] = state.to_obs(
                    override_time=float(current_grid * G))
            last_window = max(last_window, new_grid)

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
