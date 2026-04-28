"""
conflict_diagnostic.py
======================
Runs the full mask/label conflict check on the parsed dataset and prints
exactly which (action_id, missing_prerequisite) pairs are causing conflicts.

This tells you precisely where the numpy mask in replay_parser.py and
the PyTorch mask in action_mask.py are disagreeing.
"""

import numpy as np
import os
import sys
import torch
from collections import defaultdict

# Ensure the parent directory (source) is in the path so we can import model and action_mask
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SequenceDataset, OBS_SIZE
from action_mask import build_training_mask, NUM_ACTIONS

DATASET_PATH = r"C:\dev\BetaStar\replays\parsed\dataset.npz"

# Mirror of action names for readable output
ACTIONS = [
    "do_nothing",               # 0
    "train_probe",              # 1
    "build_pylon",              # 2
    "build_gateway",            # 3
    "build_cyberneticscore",    # 4
    "build_assimilator",        # 5
    "build_nexus",              # 6
    "build_forge",              # 7
    "build_stargate",           # 8
    "build_robotics_facility",  # 9
    "build_twilight_council",   # 10
    "build_photon_cannon",      # 11
    "build_fleet_beacon",       # 12
    "build_templar_archive",    # 13
    "train_zealot",             # 14
    "train_stalker",            # 15
    "train_immortal",           # 16
    "train_voidray",            # 17
    "train_carrier",            # 18
    "train_high_templar",       # 19
    "warp_in_zealot",           # 20
    "warp_in_stalker",          # 21
    "warp_in_high_templar",     # 22
    "archon_warp",              # 23
    "research_charge",          # 24
    "research_warp_gate",       # 25
    "upgrade_ground_weapons",   # 26
    "upgrade_air_weapons",      # 27
    "upgrade_shields",          # 28
    "attack_enemy_base",        # 29
    "train_adept",              # 30
    "train_phoenix",            # 31
    "train_colossus",           # 32
    "warp_in_adept",            # 33
]

# Obs feature indices for completed structures (same as action_mask.py)
STRUCTURE_NAMES = [
    "NEXUS", "PYLON", "GATEWAY", "WARPGATE", "FORGE", "TWILIGHTCOUNCIL",
    "PHOTONCANNON", "SHIELDBATTERY", "TEMPLARARCHIVE", "ROBOTICSBAY",
    "ROBOTICSFACILITY", "ASSIMILATOR", "CYBERNETICSCORE", "STARGATE", "FLEETBEACON",
]
UNIT_NAMES = [
    "PROBE", "ZEALOT", "STALKER", "HIGHTEMPLAR", "ARCHON",
    "IMMORTAL", "CARRIER", "VOIDRAY",
]

# Build a lookup: obs index -> feature name
feature_names = (
    ["time", "minerals", "vespene", "supply_used", "supply_cap", "worker_sat"]
    + [f"struct_{s}" for s in STRUCTURE_NAMES]
    + [f"unit_{u}" for u in UNIT_NAMES]
    + [f"pend_{s}" for s in STRUCTURE_NAMES]
    + [f"pend_{u}" for u in UNIT_NAMES]
    + ["idle_gw_wg", "idle_sg", "idle_robo", "idle_wg"]
)


def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = SequenceDataset(DATASET_PATH)

    total = 0
    conflicts = 0
    by_action = defaultdict(int)   # action_id -> conflict count
    # For each conflicting action, which prerequisite features are zero
    # when they shouldn't be?
    prereq_missing = defaultdict(lambda: defaultdict(int))

    for obs_seq, act_seq in dataset.sequences:
        # Build mask for the entire sequence at once (training mask = ground truth)
        legal = build_training_mask(obs_seq)  # (T, NUM_ACTIONS)

        for t in range(len(act_seq)):
            a = act_seq[t].item()
            if a < 0:
                continue
            total += 1
            if not legal[t, a]:
                conflicts += 1
                by_action[a] += 1

                # Find which prerequisite structure/unit features are zero
                obs = obs_seq[t].numpy()
                # Check each completed structure feature (indices 6-20)
                for i, name in enumerate(STRUCTURE_NAMES):
                    feat_idx = 6 + i
                    if obs[feat_idx] < 0.01:
                        prereq_missing[a][f"struct_{name}=0"] += 1
                # Check each completed unit feature (indices 21-28)
                for i, name in enumerate(UNIT_NAMES):
                    feat_idx = 21 + i
                    if obs[feat_idx] < 0.01:
                        prereq_missing[a][f"unit_{name}=0"] += 1

    print(f"\nTotal real samples: {total}")
    print(
        f"Conflicts:          {conflicts} ({100*conflicts/max(total,1):.2f}%)")

    if conflicts == 0:
        print("\nNo conflicts — dataset is clean.")
        return

    print(f"\nConflicts by action (sorted by count):")
    print(f"  {'action_id':>9}  {'name':<28}  {'conflicts':>9}")
    print(f"  {'-'*9}  {'-'*28}  {'-'*9}")
    for a, count in sorted(by_action.items(), key=lambda x: -x[1]):
        name = ACTIONS[a] if a < len(ACTIONS) else f"action_{a}"
        print(f"  {a:>9}  {name:<28}  {count:>9}")

    print(f"\nFor each conflicting action, which prerequisite features were zero?")
    print(f"(Only features that were zero in >10% of that action's conflicts shown)\n")
    for a, count in sorted(by_action.items(), key=lambda x: -x[1]):
        name = ACTIONS[a] if a < len(ACTIONS) else f"action_{a}"
        print(f"  Action {a} ({name})  —  {count} conflicts:")
        missing = prereq_missing[a]
        threshold = count * 0.10
        shown = {k: v for k, v in missing.items() if v >= threshold}
        if shown:
            for feat, n in sorted(shown.items(), key=lambda x: -x[1]):
                print(
                    f"    {feat:<35} zero in {n}/{count} conflicts ({100*n/count:.0f}%)")
        else:
            print(f"    (no single feature dominates — may be a logic mismatch)")
        print()


if __name__ == "__main__":
    main()
