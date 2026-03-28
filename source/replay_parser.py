"""
SC2 Replay Parser — Fixed-Grid Sequence Dataset Builder
========================================================
Key change from event-driven version:

  Observations are now sampled on a fixed temporal grid that matches the
  live bot's cooldown cadence (GRID_INTERVAL_SECONDS). A row is added only
  when a mapped macro action fires within a grid window — idle windows are
  skipped entirely, so there is no do_nothing noise in the dataset.

  This closes the training/inference distribution gap: the model now learns
  "given the state at the start of an 8-second window, which macro action
  (if any) should I take?" — exactly the question predict_action asks.

Grid mechanics:
  - Windows: [0, 8s), [8s, 16s), [16s, 24s), ...
  - Obs snapshot: taken at the START of the window, before any events in it.
    This is what the live bot observes when predict_action is called.
  - Action label: the first mapped command that fires inside the window.
    If multiple mapped commands fire, only the first is the label; all still
    update pending-count state so subsequent snapshots stay accurate.
  - Empty windows (no mapped action): skipped, not added as do_nothing.

GRID_INTERVAL_SECONDS = 8 matches cooldown=44 at ~5.6 on_step/s ≈ 7.8s.
Adjust if you change the bot cooldown.

OBS_SIZE remains 53. Layout identical to observation_wrapper.py.
"""

from collections import defaultdict
import os
import sc2reader
import numpy as np
from sc2reader.events import (
    PlayerStatsEvent, UnitBornEvent, UnitDiedEvent,
    UnitDoneEvent, BasicCommandEvent,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_INTERVAL_SECONDS = 8  # change this if you change the bot cooldown

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

OBS_SIZE = 53  # 6 base + 15 structures + 8 units + 15 pending structures + 8 pending units + 1 opp

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

# Obs feature indices for completed structures (must match observation_wrapper.py)
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
_IDX_HIGHTEMPLAR = 24  # completed unit

_EPS = 0.01  # > 0 but < 0.1 (= 1 structure normalised /10)


def _action_legal_numpy(obs: list[float], action_id: int) -> bool:
    """
    Pure-numpy mirror of action_mask.build_legal_mask for a single obs vector.
    Used during parsing to discard rows where the label contradicts the mask,
    which would cause log(0) = -inf loss during training.

    Checks only completed structure/unit counts (indices 6-28), not pending,
    because a structure under construction doesn't yet unlock tech.
    """
    has_nexus = obs[_IDX_NEXUS] > _EPS
    has_pylon = obs[_IDX_PYLON] > _EPS
    has_gateway = obs[_IDX_GATEWAY] > _EPS
    has_warpgate = obs[_IDX_WARPGATE] > _EPS
    has_forge = obs[_IDX_FORGE] > _EPS
    has_twilight = obs[_IDX_TWILIGHTCOUNCIL] > _EPS
    has_temparch = obs[_IDX_TEMPLARARCHIVE] > _EPS
    has_robofac = obs[_IDX_ROBOTICSFACILITY] > _EPS
    has_cybcore = obs[_IDX_CYBERNETICSCORE] > _EPS
    has_stargate = obs[_IDX_STARGATE] > _EPS
    has_fleet = obs[_IDX_FLEETBEACON] > _EPS
    has_2ht = obs[_IDX_HIGHTEMPLAR] > (1.5 / 30.0)

    # Any combat unit present (needed for attack action)
    # indices 22-28 are zealot, stalker, hightemplar, archon, immortal, carrier, voidray
    has_army = any(obs[i] > _EPS for i in range(22, 29))

    rules = {
        0:  True,                             # do_nothing
        1:  has_nexus,                        # train_probe
        2:  True,                             # build_pylon
        3:  has_pylon,                        # build_gateway
        4:  has_gateway,                      # build_cyberneticscore
        5:  has_nexus,                        # build_assimilator
        6:  True,                             # build_nexus
        7:  has_pylon,                        # build_forge
        8:  has_cybcore,                      # build_stargate
        9:  has_cybcore,                      # build_robotics_facility
        10: has_cybcore,                      # build_twilight_council
        11: has_forge,                        # build_photon_cannon
        12: has_stargate,                     # build_fleet_beacon
        13: has_twilight,                     # build_templar_archive
        14: has_gateway,                      # train_zealot
        15: has_gateway and has_cybcore,      # train_stalker
        16: has_robofac,                      # train_immortal
        17: has_stargate,                     # train_voidray
        18: has_stargate and has_fleet,       # train_carrier
        19: has_gateway and has_temparch,     # train_high_templar
        20: has_warpgate,                     # warp_in_zealot
        21: has_warpgate and has_cybcore,     # warp_in_stalker
        22: has_warpgate and has_temparch,    # warp_in_high_templar
        23: has_2ht,                          # archon_warp
        24: has_twilight,                     # research_charge
        25: has_cybcore,                      # research_warp_gate
        26: has_forge,                        # upgrade_ground_weapons
        27: has_cybcore,                      # upgrade_air_weapons
        28: has_forge,                        # upgrade_shields
        29: has_army,                         # attack_enemy_base
    }
    return rules.get(action_id, False)


# ---------------------------------------------------------------------------
# GameState — unchanged from original
# ---------------------------------------------------------------------------

class GameState:
    """
    Tracks full Protoss game state including in-progress counts.
    Identical to original; kept here so the parser is self-contained.
    """

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
                                getattr(event, "minerals", 0))
        self.vespene = getattr(event, "vespene_current",
                               getattr(event, "vespene",  0))
        self.supply_used = getattr(event, "supply_used",
                                   getattr(event, "food_used", 0))
        self.supply_cap = getattr(event, "supply_made",
                                  getattr(event, "food_made", 0))

    def update_opp_from_stats(self, event: PlayerStatsEvent):
        self.opp_supply_used = getattr(event, "supply_used",
                                       getattr(event, "food_used", 0))

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
        """
        Serialize to flat vector matching ObservationWrapper.get_observation().
        override_time lets the caller pass the exact grid-boundary time rather
        than relying on the last PlayerStatsEvent time.
        """
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

        assert len(
            obs) == OBS_SIZE, f"Obs size mismatch: {len(obs)} vs {OBS_SIZE}"
        return obs


# ---------------------------------------------------------------------------
# ReplayParser
# ---------------------------------------------------------------------------

class ReplayParser:
    """
    Parses SC2 replays into fixed-grid sequence arrays for LSTM training.

    Each replay becomes one (T, OBS_SIZE+1) array where:
      - T   = number of grid windows that contained at least one macro action.
      - row = [obs_at_window_start (53 floats), action_id (1 float)]

    The obs snapshot uses the game time at the window START (override_time),
    so the time feature is always exactly aligned with the grid — not
    dependent on when the last PlayerStatsEvent happened to fire.
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
        self.conflicts_dropped = 0   # rows where label contradicts legal mask

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
        Walk replay events on a fixed GRID_INTERVAL_SECONDS grid.

        Algorithm:
          1. Walk all events chronologically, maintaining GameState.
          2. Whenever an event's timestamp crosses into a new grid window,
             snapshot the current state at that window's START time.
          3. For BasicCommandEvents: always update pending counts.
             If the command maps to a known action AND this window doesn't
             already have a label, record it as this window's action.
          4. After the loop, emit one row per window that has a label.

        Returns float32 array (T, OBS_SIZE+1), or None if T < min_length.
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

        # grid_obs[i]    = obs snapshot at the START of window i (time = i*G)
        # grid_actions[i] = action_id of the first mapped command in window i
        grid_obs = {}
        grid_actions = {}

        current_grid = 0
        # Snapshot window 0 from the initial (pre-event) state
        grid_obs[0] = state.to_obs(override_time=0.0)

        for event in replay.events:
            t = event.second

            # --- Advance grid snapshots ---
            # Before processing this event, snapshot the start of every new
            # grid window we've entered.  The snapshot captures state AFTER
            # all events from prior windows but BEFORE events in this window.
            new_grid = int(t / G)
            while current_grid < new_grid:
                current_grid += 1
                grid_obs[current_grid] = state.to_obs(
                    override_time=float(current_grid * G))

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

            elif isinstance(event, BasicCommandEvent):
                if event.player.pid != pid:
                    continue

                ability_name = event.ability_name

                # Always update pending counts for state accuracy
                state.on_build_command(ability_name)
                state.on_train_command(ability_name)

                action_id = self.EVENT_TO_ACTION.get(ability_name)
                if action_id is not None:
                    # First mapped action in this window wins; later ones
                    # are dropped as labels (but state was already updated)
                    window = int(t / G)
                    if window not in grid_actions:
                        grid_actions[window] = action_id
                        self.mapped_actions[ability_name] += 1
                    # else: second action in same window — state updated, label ignored
                else:
                    self.unmapped_abilities[ability_name] += 1
                    if self.debug and self.unmapped_abilities[ability_name] == 1:
                        print(f"    [UNMAPPED] {ability_name}")

        # --- Build rows ---
        # Emit one (obs, action) row per window that had a mapped action,
        # in chronological order.  Skip any row where the action contradicts
        # the legal mask — those cause log(0) = -inf loss during training.
        rows = []
        for window in sorted(grid_actions.keys()):
            obs = grid_obs.get(window)
            if obs is None:
                continue
            action_id = grid_actions[window]
            if not _action_legal_numpy(obs, action_id):
                self.conflicts_dropped += 1
                continue
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
        print("\nMapped Actions (included in dataset):")
        for ability, count in sorted(self.mapped_actions.items(), key=lambda x: -x[1]):
            action_id = self.EVENT_TO_ACTION.get(ability, 0)
            print(f"  [{action_id:2d}] {ability:30s}: {count:5d} samples")
        total = sum(self.mapped_actions.values())
        print(f"\nTotal mapped samples: {total}")
        if self.unmapped_abilities:
            print("\nUnmapped Abilities (omitted):")
            for ability, count in sorted(self.unmapped_abilities.items(), key=lambda x: -x[1]):
                print(f"  {ability:30s}: {count:5d} occurrences")
        else:
            print("\nNo unmapped abilities found!")

    # ------------------------------------------------------------------
    # Folder processing
    # ------------------------------------------------------------------

    def parse_replay_folder(self):
        sequences = []
        bot_replays = []
        skipped = 0
        failed = 0

        replay_files = [
            f for f in os.listdir(self.replay_folder) if f.endswith(".SC2Replay")
        ]
        print(f"Found {len(replay_files)} replay(s) to process.")
        print(f"Grid interval: {GRID_INTERVAL_SECONDS}s "
              f"(change GRID_INTERVAL_SECONDS to match bot cooldown)\n")

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

                sequences.append(seq)
                print(f"  {fname}: {len(seq)} action windows")

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

        print(
            f"\nDone. {len(sequences)} sequences | {total_steps} total action windows")
        print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.0f}")

        # Sanity check: average seconds per action window
        # (total game time across replays / total action windows)
        # Should be close to GRID_INTERVAL_SECONDS * some coverage factor
        print(f"\nApprox. actions per game minute (mean seq length × "
              f"{GRID_INTERVAL_SECONDS}s / 60): "
              f"{np.mean(lengths) * GRID_INTERVAL_SECONDS / 60:.1f}")

        print(f"Skipped: {skipped}  |  Failed: {failed}")
        if bot_replays:
            print(f"Bot replays skipped: {bot_replays}")
        print(f"\nSaved to: {self.output_file}")
        self.print_statistics()


if __name__ == "__main__":
    parser = ReplayParser()
    parser.parse_replay_folder()
