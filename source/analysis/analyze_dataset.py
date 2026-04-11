"""
analyze_dataset.py - Comprehensive analysis of the parsed training dataset

This script analyzes the entire dataset.npz file to show:
- Action distribution across all replays
- Class imbalance issues
- Sequence length statistics
- Temporal patterns (when actions occur in games)
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATASET_PATH = r"C:\dev\BetaStar\replays\parsed\dataset.npz"
OBS_SIZE = 57

# Action ID to name mapping
ACTION_NAMES = {
    0: "do_nothing",
    1: "train_probe",
    2: "build_pylon",
    3: "build_gateway",
    4: "build_cyberneticscore",
    5: "build_assimilator",
    6: "build_nexus",
    7: "build_forge",
    8: "build_stargate",
    9: "build_robotics_facility",
    10: "build_twilight_council",
    11: "build_photon_cannon",
    12: "build_fleet_beacon",
    13: "build_templar_archive",
    14: "train_zealot",
    15: "train_stalker",
    16: "train_immortal",
    17: "train_voidray",
    18: "train_carrier",
    19: "train_high_templar",
    20: "warp_in_zealot",
    21: "warp_in_stalker",
    22: "warp_in_high_templar",
    23: "archon_warp",
    24: "research_charge",
    25: "research_warp_gate",
    26: "upgrade_ground_weapons",
    27: "upgrade_air_weapons",
    28: "upgrade_shields",
    29: "attack_enemy_base",
    30: "train_adept",
    31: "train_phoenix",
    32: "train_colossus",
    33: "warp_in_adept",
}

print("=" * 120)
print("TRAINING DATASET ANALYSIS")
print("=" * 120)
print(f"\nLoading dataset from: {DATASET_PATH}\n")

data = np.load(DATASET_PATH, allow_pickle=True)
sequences = data["sequences"]

print(f"Total replays in dataset: {len(sequences)}")

# Sequence length statistics
lengths = [len(seq) for seq in sequences]
print(f"\nSequence length statistics:")
print(f"  Min:    {min(lengths):4d} windows")
print(f"  Max:    {max(lengths):4d} windows")
print(f"  Mean:   {np.mean(lengths):6.1f} windows")
print(f"  Median: {int(np.median(lengths)):4d} windows")
print(f"  Total:  {sum(lengths):5d} windows across all replays")

# Extract all actions
all_actions = []
for seq in sequences:
    actions = seq[:, OBS_SIZE].astype(int)
    all_actions.extend(actions)

all_actions = np.array(all_actions)
total_samples = len(all_actions)

print(f"\nTotal training samples: {total_samples:,}")

# Action distribution
print("\n" + "=" * 120)
print("ACTION DISTRIBUTION ACROSS ENTIRE DATASET")
print("=" * 120)

action_counts = {}
for action_id in range(34):
    count = (all_actions == action_id).sum()
    if count > 0:
        action_counts[action_id] = count

print(f"\n{'Action ID':<12} | {'Action Name':<35} | {'Count':<12} | {'Percentage':<12} | {'Bar':<30}")
print("-" * 120)

# Sort by count descending
for action_id in sorted(action_counts.keys(), key=lambda x: action_counts[x], reverse=True):
    count = action_counts[action_id]
    pct = 100 * count / total_samples
    action_name = ACTION_NAMES.get(action_id, f"UNKNOWN_{action_id}")
    bar_length = int(pct * 0.5)  # Scale for display
    bar = "█" * bar_length
    print(f"{action_id:<12} | {action_name:<35} | {count:<12,} | {pct:>6.2f}%      | {bar}")

# Class imbalance analysis
print("\n" + "=" * 120)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 120)

do_nothing_count = action_counts.get(0, 0)
do_nothing_pct = 100 * do_nothing_count / total_samples
macro_action_count = total_samples - do_nothing_count
macro_action_pct = 100 - do_nothing_pct

print(f"\ndo_nothing (action 0):  {do_nothing_count:,} samples ({do_nothing_pct:.1f}%)")
print(f"Macro actions (1-33):   {macro_action_count:,} samples ({macro_action_pct:.1f}%)")
print(f"\nClass imbalance ratio:  {do_nothing_count / macro_action_count:.1f}:1 (do_nothing:macro)")

# Find rarest and most common macro actions
macro_actions = {k: v for k, v in action_counts.items() if k != 0}
if macro_actions:
    rarest_id = min(macro_actions.keys(), key=lambda x: macro_actions[x])
    most_common_id = max(macro_actions.keys(), key=lambda x: macro_actions[x])
    
    print(f"\nRarest macro action:    {ACTION_NAMES[rarest_id]} (ID {rarest_id}) - {macro_actions[rarest_id]:,} samples")
    print(f"Most common macro:      {ACTION_NAMES[most_common_id]} (ID {most_common_id}) - {macro_actions[most_common_id]:,} samples")
    print(f"Macro imbalance ratio:  {macro_actions[most_common_id] / macro_actions[rarest_id]:.1f}:1")

# Actions that never appear
missing_actions = []
for action_id in range(34):
    if action_id not in action_counts:
        missing_actions.append(action_id)

if missing_actions:
    print(f"\n⚠ Actions that NEVER appear in dataset:")
    for action_id in missing_actions:
        print(f"  - Action {action_id}: {ACTION_NAMES.get(action_id, 'UNKNOWN')}")
else:
    print(f"\n✓ All 34 actions appear at least once in the dataset")

# Temporal analysis - when do actions occur in games?
print("\n" + "=" * 120)
print("TEMPORAL ANALYSIS - When Actions Occur in Games")
print("=" * 120)

# Divide game into early/mid/late phases
early_actions = {i: 0 for i in range(34)}
mid_actions = {i: 0 for i in range(34)}
late_actions = {i: 0 for i in range(34)}

for seq in sequences:
    seq_len = len(seq)
    actions = seq[:, OBS_SIZE].astype(int)
    
    early_cutoff = seq_len // 3
    mid_cutoff = 2 * seq_len // 3
    
    for i, action_id in enumerate(actions):
        if i < early_cutoff:
            early_actions[action_id] = early_actions.get(action_id, 0) + 1
        elif i < mid_cutoff:
            mid_actions[action_id] = mid_actions.get(action_id, 0) + 1
        else:
            late_actions[action_id] = late_actions.get(action_id, 0) + 1

print(f"\nGame phases (divided into thirds):")
print(f"  Early game: first 1/3 of each replay")
print(f"  Mid game:   middle 1/3")
print(f"  Late game:  final 1/3")

print(f"\n{'Action Name':<35} | {'Early':<12} | {'Mid':<12} | {'Late':<12} | {'Pattern':<20}")
print("-" * 120)

for action_id in sorted(action_counts.keys()):
    if action_id == 0:  # Skip do_nothing for clarity
        continue
    
    action_name = ACTION_NAMES[action_id]
    early = early_actions.get(action_id, 0)
    mid = mid_actions.get(action_id, 0)
    late = late_actions.get(action_id, 0)
    total = early + mid + late
    
    if total > 0:
        early_pct = 100 * early / total
        mid_pct = 100 * mid / total
        late_pct = 100 * late / total
        
        # Determine pattern
        if early_pct > 50:
            pattern = "Early-heavy"
        elif late_pct > 50:
            pattern = "Late-heavy"
        elif mid_pct > 50:
            pattern = "Mid-heavy"
        else:
            pattern = "Distributed"
        
        print(f"{action_name:<35} | {early:>5} ({early_pct:>4.1f}%) | {mid:>5} ({mid_pct:>4.1f}%) | {late:>5} ({late_pct:>4.1f}%) | {pattern:<20}")

# Per-replay action diversity
print("\n" + "=" * 120)
print("PER-REPLAY ACTION DIVERSITY")
print("=" * 120)

unique_actions_per_replay = []
for seq in sequences:
    actions = seq[:, OBS_SIZE].astype(int)
    unique = len(set(actions))
    unique_actions_per_replay.append(unique)

print(f"\nUnique actions per replay:")
print(f"  Min:    {min(unique_actions_per_replay)} different actions")
print(f"  Max:    {max(unique_actions_per_replay)} different actions")
print(f"  Mean:   {np.mean(unique_actions_per_replay):.1f} different actions")
print(f"  Median: {int(np.median(unique_actions_per_replay))} different actions")

print("\n" + "=" * 120)
print("RECOMMENDATIONS FOR MODEL TRAINING")
print("=" * 120)

print(f"\n1. CLASS IMBALANCE:")
print(f"   - do_nothing is {do_nothing_pct:.1f}% of data - use class weights in loss function")
print(f"   - Consider sqrt weighting: weight = 1/sqrt(count) to dampen extremes")

print(f"\n2. RARE ACTIONS:")
if macro_actions:
    rare_threshold = 100
    rare_actions = [k for k, v in macro_actions.items() if v < rare_threshold]
    if rare_actions:
        print(f"   - {len(rare_actions)} actions have <{rare_threshold} samples:")
        for action_id in rare_actions:
            print(f"     • {ACTION_NAMES[action_id]} (ID {action_id}): {macro_actions[action_id]} samples")
        print(f"   - Model may struggle to learn these - consider data augmentation or more replays")

print(f"\n3. MISSING ACTIONS:")
if missing_actions:
    print(f"   - {len(missing_actions)} actions never appear - model cannot learn them")
    print(f"   - Consider removing from action space or adding replays with these actions")

print(f"\n4. TEMPORAL PATTERNS:")
print(f"   - Early-heavy actions (builds) should be learned well")
print(f"   - Late-heavy actions may need more late-game replay data")

print("\n" + "=" * 120)
