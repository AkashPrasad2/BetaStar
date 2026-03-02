from collections import defaultdict
import os
import sc2reader
import numpy as np
from sc2reader.events import PlayerStatsEvent, UnitBornEvent, UnitDiedEvent, UnitDoneEvent, BasicCommandEvent

# Maps sc2reader unit type names -> state dict keys
# Must match the order in observation_wrapper.py's PROTOSS_STRUCTURES / PROTOSS_UNITS
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

# ordered lists matching observation_wrapper.py exactly
STRUCTURES = [
    "NEXUS", "PYLON", "GATEWAY", "WARPGATE", "FORGE", "TWILIGHTCOUNCIL",
    "PHOTONCANNON", "SHIELDBATTERY", "TEMPLARARCHIVE", "ROBOTICSBAY",
    "ROBOTICSFACILITY", "ASSIMILATOR", "CYBERNETICSCORE", "STARGATE", "FLEETBEACON",
]
UNITS = [
    "PROBE", "ZEALOT", "STALKER", "HIGHTEMPLAR", "ARCHON", "IMMORTAL", "CARRIER", "VOIDRAY",
]

# Units that transition into something else on completion (Gateway -> WarpGate)
# We handle these as special cases so counts stay accurate
MORPH_MAP = {
    "WarpGate": "GATEWAY",  # when a gateway morphs, lose 1 gateway, gain 1 warpgate
}


class GameState:
    """
    Tracks the game state of the Protoss player as events are processed

    The sc2reader library only provides snapshots of the gamestate every 7 seconds.
    Because of this, we must update unit counts manually in the interim
    """

    def __init__(self):
        # intitialize resources with the game's start values
        self.time = 0.0
        self.minerals = 50.0
        self.vespene = 0.0
        self.supply_used = 12.0
        self.supply_cap = 15.0

        # unit/structure counts which are updated with events
        self.counts = {k: 0 for k in STRUCTURES + UNITS}
        self.counts["NEXUS"] = 1
        self.counts["PROBE"] = 12

        # opponent supply (from their PlayerStatsEvent)
        self.opp_supply_used = 0.0

    def update_from_stats(self, event: PlayerStatsEvent):
        self.time = event.second
        # handle compatibility with older replay versions as well
        self.minerals = getattr(event, 'minerals_current',
                                getattr(event, 'minerals', 0))
        self.vespene = getattr(event, 'vespene_current',
                               getattr(event, 'vespene', 0))
        self.supply_used = getattr(
            event, 'supply_used', getattr(event, 'food_used', 0))
        self.supply_cap = getattr(
            event, 'supply_made', getattr(event, 'food_made', 0))

    def update_opp_from_stats(self, event: PlayerStatsEvent):
        # handle compatibility with older replay versions as well
        self.opp_supply_used = getattr(
            event, 'supply_used', getattr(event, 'food_used', 0))

    def unit_born_or_done(self, unit_type_name: str):
        """Call when a unit/structure finishes construction or is born."""
        key = UNIT_NAME_MAP.get(
            unit_type_name) or STRUCTURE_NAME_MAP.get(unit_type_name)
        if key:
            self.counts[key] += 1

        # Gateway morphing into WarpGate: remove the gateway
        predecessor = MORPH_MAP.get(unit_type_name)
        if predecessor:
            self.counts[predecessor] = max(0, self.counts[predecessor] - 1)

    def unit_died(self, unit_type_name: str):
        """Call when a unit/structure is destroyed."""
        key = UNIT_NAME_MAP.get(
            unit_type_name) or STRUCTURE_NAME_MAP.get(unit_type_name)
        if key:
            self.counts[key] = max(0, self.counts[key] - 1)

    def to_obs(self) -> list[float]:
        """
        Serialize to the same flat vector as ObservationWrapper.get_observation().
        Order: time, minerals, vespene, supply_used, supply_cap, supply_left,
               worker_saturation, [15 structures], [8 units], opp_supply_used
        Total: 8 + 15 + 8 = 31 floats
        """
        supply_left = max(0.0, self.supply_cap - self.supply_used)
        ideal_workers = max(self.counts["NEXUS"], 1) * 22
        worker_saturation = self.counts["PROBE"] / ideal_workers

        # normalize as ObersvationWrapper does
        obs = [
            self.time / 720.0,
            self.minerals / 1800.0,
            self.vespene / 700.0,
            self.supply_used / 200.0,
            self.supply_cap / 200.0,
            supply_left / 200.0,
            worker_saturation,
        ]
        for s in STRUCTURES:
            obs.append(self.counts[s] / 10.0)
        for u in UNITS:
            obs.append(self.counts[u] / 30.0)
        obs.append(self.opp_supply_used / 200.0)

        return obs


class ReplayParser:
    """
    Parses a folder of SC2 replays (PvZ, AbyssalReefLE only) into a training dataset.
    Each sample is a (observation_vector, action_id) pair captured at the moment
    the Protoss player issues a macro command.
    """

    def __init__(
        self,
        replay_folder=r"C:\Users\akash\Projects\sc2_bot\replays\raw",
        output_file=r"C:\Users\akash\Projects\sc2_bot\replays\parsed\dataset.npz",
        debug=True,
    ):
        self.replay_folder = replay_folder
        self.output_file = output_file
        self.debug = debug

        # Track unmapped abilities for debugging
        self.unmapped_abilities = defaultdict(int)
        self.mapped_actions = defaultdict(int)

        self.EVENT_TO_ACTION = {
            # Training units
            "TrainProbe":           1,
            "TrainZealot":          14,
            "TrainStalker":         15,
            "TrainImmortal":        16,
            "TrainVoidRay":         17,
            "TrainCarrier":         18,
            "TrainHighTemplar":     19,
            # Building structures
            "BuildPylon":           2,
            "BuildGateway":         3,
            "BuildCyberneticsCore": 4,
            "BuildAssimilator":     5,
            "BuildNexus":           6,
            "BuildForge":           7,
            "BuildStargate":        8,
            "BuildRoboticsFacility": 9,
            "BuildTwilightCouncil": 10,
            "BuildPhotonCannon":    11,
            "BuildFleetBeacon":     12,
            "BuildTemplarArchive":  13,
            # Warp-ins
            "WarpInZealot":         20,
            "WarpInStalker":        21,
            "WarpInHighTemplar":    22,
            # Research
            "ResearchCharge":       24,
            "ResearchWarpGate":     25,
            # Archon
            "ArchonWarp":           23,
            "MorphToArchon":        23,
        }

    def parse_replay(self, replay) -> list[tuple[list[float], int]]:
        """
        Walks through all events in a single replay chronologically
        Returns a list of (obs_vector, action_id) training pairs.
        """
        # identify players
        protoss_player = None
        zerg_player = None
        for player in replay.players:
            if player.play_race == "Protoss":
                protoss_player = player
            elif player.play_race == "Zerg":
                zerg_player = player

        if protoss_player is None or zerg_player is None:
            return []

        pid = protoss_player.pid  # 1-indexed
        opp_pid = zerg_player.pid

        state = GameState()
        training_pairs = []

        for event in replay.events:

            # Resource snapshot — update minerals/vespene/supply from stats
            if isinstance(event, PlayerStatsEvent):
                if event.player.pid == pid:
                    state.update_from_stats(event)
                elif event.player.pid == opp_pid:
                    state.update_opp_from_stats(event)

            # Increment counters when a unit/structure is created
            #    UnitDoneEvent fires when a structure completes construction.
            #    UnitBornEvent fires when a unit exits a production building.
            elif isinstance(event, (UnitBornEvent, UnitDoneEvent)):
                # Only track Protoss player's own units
                unit = event.unit
                owner = getattr(unit, "owner", None)
                if owner is None or owner.pid != pid:
                    continue
                state.unit_born_or_done(unit.name)

            # Decrement counters when a unit/structure is killed
            elif isinstance(event, UnitDiedEvent):
                unit = event.unit
                owner = getattr(unit, "owner", None)
                if owner is None or owner.pid != pid:
                    continue
                state.unit_died(unit.name)

            # Command issued — snapshot state + record action label
            elif isinstance(event, BasicCommandEvent):
                if event.player.pid != pid:
                    continue

                ability_name = event.ability_name
                action_id = self.EVENT_TO_ACTION.get(ability_name)

                if action_id is not None:
                    # Mapped action - record training pair
                    obs = state.to_obs()
                    training_pairs.append((obs, action_id))
                    self.mapped_actions[ability_name] += 1
                else:
                    # Unmapped action - track for debugging
                    self.unmapped_abilities[ability_name] += 1
                    if self.debug:
                        # Only log first occurrence to avoid spam
                        if self.unmapped_abilities[ability_name] == 1:
                            print(f"    [UNMAPPED] {ability_name}")

        return training_pairs

    def print_statistics(self):
        """Print parsing statistics for debugging."""
        print("\n" + "="*60)
        print("PARSING STATISTICS")
        print("="*60)

        print("\nMapped Actions (included in dataset):")
        for ability, count in sorted(self.mapped_actions.items(), key=lambda x: -x[1]):
            action_id = self.EVENT_TO_ACTION[ability]
            print(f"  [{action_id:2d}] {ability:30s}: {count:5d} samples")

        print(f"\nTotal mapped samples: {sum(self.mapped_actions.values())}")

        if self.unmapped_abilities:
            print("\nUnmapped Abilities (omitted from training data):")
            for ability, count in sorted(self.unmapped_abilities.items(), key=lambda x: -x[1]):
                print(f"  {ability:30s}: {count:5d} occurrences")
            print(f"\nTotal unmapped: {sum(self.unmapped_abilities.values())}")
        else:
            print("\nNo unmapped abilities found!")

    def parse_replay_folder(self):
        """
        Parse all replays in self.replay_folder, filter to PvZ on AbyssalReefLE,
        and save the dataset to self.output_file.
        """
        all_obs = []
        all_actions = []
        skipped = 0
        failed = 0

        replay_files = [f for f in os.listdir(
            self.replay_folder) if f.endswith(".SC2Replay")]
        print(f"Found {len(replay_files)} replay(s) to process.")

        for fname in replay_files:
            path = os.path.join(self.replay_folder, fname)
            try:
                # load_level=4 is required for tracker events (unit births, stats, etc.)
                replay = sc2reader.load_replay(path, load_level=4)

                # PvZ replays only
                races = {p.play_race for p in replay.players}
                if races != {"Protoss", "Zerg"}:
                    skipped += 1
                    continue

                # AbyssalReefLE replays only
                if "Abyssal Reef" not in replay.map_name:
                    skipped += 1
                    continue

                pairs = self.parse_replay(replay)
                for obs, action in pairs:
                    all_obs.append(obs)
                    all_actions.append(action)

                print(f"  {fname}: {len(pairs)} samples")

            except Exception as e:
                print(f"  FAILED {fname}: {e}")
                failed += 1

        if not all_obs:
            print("No training data collected. Check replay folder and filters.")
            return

        X = np.array(all_obs, dtype=np.float32)
        y = np.array(all_actions, dtype=np.int64)
        np.savez(self.output_file, X=X, y=y)

        print(
            f"\nDone. {len(X)} samples from {len(replay_files) - skipped - failed} replays.")
        print(
            f"Skipped (wrong matchup/map): {skipped}  |  Failed (parse error): {failed}")
        print(f"Saved to: {self.output_file}")
        print(f"X shape: {X.shape}  |  y shape: {y.shape}")
        print(
            f"Action distribution: { {i: int((y == i).sum()) for i in np.unique(y)} }")

        # Print statistics
        self.print_statistics()


if __name__ == "__main__":
    parser = ReplayParser()
    parser.parse_replay_folder()
