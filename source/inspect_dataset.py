"""
inspect_dataset.py
==================
Interactive tool to explore the parsed replay dataset.

Usage:
    python inspect_dataset.py
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

DATASET_PATH = r"C:\dev\BetaStar\replays\parsed\dataset.npz"

# Action names
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
]

# Feature names
FEATURE_NAMES = [
    "time", "minerals", "vespene", "supply_used", "supply_cap", "worker_sat",
    # Completed structures
    "nexus", "pylon", "gateway", "warpgate", "forge", "twilight",
    "cannon", "battery", "templar_arch", "robo_bay", "robo_fac",
    "assimilator", "cyber_core", "stargate", "fleet_beacon",
    # Completed units
    "probe", "zealot", "stalker", "high_templar", "archon",
    "immortal", "carrier", "voidray",
    # Pending structures
    "pend_nexus", "pend_pylon", "pend_gateway", "pend_warpgate",
    "pend_forge", "pend_twilight", "pend_cannon", "pend_battery",
    "pend_templar_arch", "pend_robo_bay", "pend_robo_fac",
    "pend_assimilator", "pend_cyber_core", "pend_stargate", "pend_fleet_beacon",
    # Pending units
    "pend_probe", "pend_zealot", "pend_stalker", "pend_high_templar",
    "pend_archon", "pend_immortal", "pend_carrier", "pend_voidray",
    # Opponent
    "opp_supply",
]


def load_dataset():
    """Load and return the dataset."""
    print(f"Loading dataset from {DATASET_PATH}...")
    data = np.load(DATASET_PATH, allow_pickle=True)
    sequences = data["sequences"]
    print(f"Loaded {len(sequences)} sequences\n")
    return sequences


def print_dataset_stats(sequences):
    """Print overall dataset statistics."""
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    lengths = [len(seq) for seq in sequences]
    total_samples = sum(lengths)

    print(f"\nSequences: {len(sequences)}")
    print(f"Total samples: {total_samples}")
    print(
        f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    # Action distribution
    all_actions = []
    for seq in sequences:
        obs_size = seq.shape[1] - 1
        actions = seq[:, obs_size].astype(int)
        all_actions.extend(actions)

    action_counts = Counter(all_actions)

    print(f"\n{'Action Distribution:':<30} {'Count':<10} {'%':<8}")
    print("-" * 50)
    for action_id in sorted(action_counts.keys()):
        count = action_counts[action_id]
        pct = 100 * count / total_samples
        name = ACTIONS[action_id] if action_id < len(
            ACTIONS) else f"action_{action_id}"
        print(f"{name:<30} {count:<10} {pct:>6.2f}%")

    print()


def inspect_sequence(sequences, seq_idx):
    """Print detailed info about a specific sequence."""
    if seq_idx < 0 or seq_idx >= len(sequences):
        print(f"Invalid sequence index. Must be 0-{len(sequences)-1}")
        return

    seq = sequences[seq_idx]
    obs_size = seq.shape[1] - 1
    obs = seq[:, :obs_size]
    actions = seq[:, obs_size].astype(int)

    print("=" * 70)
    print(f"SEQUENCE {seq_idx}")
    print("=" * 70)
    print(f"Length: {len(seq)} timesteps")
    print(f"Observation size: {obs_size} features")
    print()

    # Show first 10 timesteps
    print("First 10 timesteps:")
    print(f"{'Step':<6} {'Time':<8} {'Min':<8} {'Gas':<8} {'Supply':<10} {'Action':<25}")
    print("-" * 70)

    for i in range(len(seq)):
        time = obs[i, 0] * 720  # Denormalize
        minerals = obs[i, 1] * 1800
        vespene = obs[i, 2] * 700
        supply = f"{obs[i, 3]*200:.0f}/{obs[i, 4]*200:.0f}"
        action_id = actions[i]
        action_name = ACTIONS[action_id] if action_id < len(
            ACTIONS) else f"action_{action_id}"

        print(
            f"{i:<6} {time:<8.0f} {minerals:<8.0f} {vespene:<8.0f} {supply:<10} {action_name:<25}")
    print()


def inspect_timestep(sequences, seq_idx, step_idx):
    """Print all features for a specific timestep."""
    if seq_idx < 0 or seq_idx >= len(sequences):
        print(f"Invalid sequence index. Must be 0-{len(sequences)-1}")
        return

    seq = sequences[seq_idx]
    if step_idx < 0 or step_idx >= len(seq):
        print(f"Invalid step index. Must be 0-{len(seq)-1}")
        return

    obs_size = seq.shape[1] - 1
    obs = seq[step_idx, :obs_size]
    action = int(seq[step_idx, obs_size])

    print("=" * 70)
    print(f"SEQUENCE {seq_idx}, STEP {step_idx}")
    print("=" * 70)

    action_name = ACTIONS[action] if action < len(
        ACTIONS) else f"action_{action}"
    print(f"Action: {action_name} (id={action})")
    print()

    print(f"{'Feature':<25} {'Raw Value':<15} {'Denormalized':<15}")
    print("-" * 60)

    # Denormalization factors
    denorm = [720, 1800, 700, 200, 200, 1] + \
        [10]*15 + [30]*8 + [10]*15 + [30]*8 + [200]

    for i, (name, raw, factor) in enumerate(zip(FEATURE_NAMES, obs, denorm)):
        denorm_val = raw * factor
        print(f"{name:<25} {raw:<15.4f} {denorm_val:<15.2f}")

    print()


def plot_action_distribution(sequences):
    """Plot action distribution as a bar chart."""
    all_actions = []
    for seq in sequences:
        obs_size = seq.shape[1] - 1
        actions = seq[:, obs_size].astype(int)
        all_actions.extend(actions)

    action_counts = Counter(all_actions)

    # Sort by action ID
    action_ids = sorted(action_counts.keys())
    counts = [action_counts[aid] for aid in action_ids]
    labels = [ACTIONS[aid] if aid < len(
        ACTIONS) else f"action_{aid}" for aid in action_ids]

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(action_ids)), counts)
    plt.xticks(range(len(action_ids)), labels, rotation=45, ha='right')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.title('Action Distribution in Training Dataset')
    plt.tight_layout()
    plt.show()


def plot_sequence_lengths(sequences):
    """Plot histogram of sequence lengths."""
    lengths = [len(seq) for seq in sequences]

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, edgecolor='black')
    plt.xlabel('Sequence Length (timesteps)')
    plt.ylabel('Count')
    plt.title('Distribution of Sequence Lengths')
    plt.axvline(np.mean(lengths), color='r', linestyle='--',
                label=f'Mean: {np.mean(lengths):.1f}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def interactive_menu(sequences):
    """Interactive menu for exploring the dataset."""
    while True:
        print("\n" + "=" * 70)
        print("DATASET INSPECTOR")
        print("=" * 70)
        print("1. Show dataset statistics")
        print("2. Inspect a specific sequence")
        print("3. Inspect a specific timestep")
        print("4. Plot action distribution")
        print("5. Plot sequence lengths")
        print("6. Exit")
        print()

        choice = input("Enter choice (1-6): ").strip()

        if choice == "1":
            print_dataset_stats(sequences)

        elif choice == "2":
            seq_idx = input(
                f"Enter sequence index (0-{len(sequences)-1}): ").strip()
            try:
                inspect_sequence(sequences, int(seq_idx))
            except ValueError:
                print("Invalid input. Please enter a number.")

        elif choice == "3":
            seq_idx = input(
                f"Enter sequence index (0-{len(sequences)-1}): ").strip()
            step_idx = input("Enter step index: ").strip()
            try:
                inspect_timestep(sequences, int(seq_idx), int(step_idx))
            except ValueError:
                print("Invalid input. Please enter numbers.")

        elif choice == "4":
            plot_action_distribution(sequences)

        elif choice == "5":
            plot_sequence_lengths(sequences)

        elif choice == "6":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1-6.")


def main():
    sequences = load_dataset()
    print_dataset_stats(sequences)
    interactive_menu(sequences)


if __name__ == "__main__":
    main()
