"""
dataset_audit.py — Dataset Quality Audit
=========================================
Run this before training to understand your data distribution.

Prints:
  - Action label distribution (counts + % of total, excluding do_nothing)
  - do_nothing rate globally and per-replay
  - Sequence length distribution
  - Obs feature statistics (mean, std, min, max) with flag for dead features
  - Per-action co-occurrence: what obs state looks like when each action fires

Usage:
    python dataset_audit.py
    python dataset_audit.py --path C:/dev/BetaStar/replays/parsed/dataset.npz
    python dataset_audit.py --path ... --top 10   (show top-N rarest actions in detail)
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---- match your project constants ----
DATASET_PATH = r"C:\dev\BetaStar\replays\parsed\dataset.npz"
OBS_SIZE = 71
NUM_ACTIONS = 35

ACTIONS = [
    "do_nothing",            # 0
    "train_probe",           # 1
    "build_pylon",           # 2
    "build_gateway",         # 3
    "build_cyberneticscore",  # 4
    "build_assimilator",     # 5
    "build_nexus",           # 6
    "build_forge",           # 7
    "build_stargate",        # 8
    "build_robotics_facility",  # 9
    "build_twilight_council",  # 10
    "build_photon_cannon",   # 11
    "build_fleet_beacon",    # 12
    "build_templar_archive",  # 13
    "train_zealot",          # 14
    "train_stalker",         # 15
    "train_immortal",        # 16
    "train_voidray",         # 17
    "train_carrier",         # 18
    "train_high_templar",    # 19
    "warp_in_zealot",        # 20
    "warp_in_stalker",       # 21
    "warp_in_high_templar",  # 22
    "archon_warp_selection",  # 23
    "research_charge",       # 24
    "research_warp_gate",    # 25
    "upgrade_ground_weapons",  # 26
    "upgrade_air_weapons",   # 27
    "upgrade_shields",       # 28
    "attack_enemy_base",     # 29
    "train_adept",           # 30
    "train_phoenix",         # 31
    "train_colossus",        # 32
    "warp_in_adept",         # 33
]

# Human-readable obs feature names matching the layout in observation_wrapper.py
OBS_FEATURE_NAMES = (
    ["time_norm"]
    + [f"minerals_bin{i}" for i in range(4)]
    + [f"gas_bin{i}" for i in range(4)]
    + ["supply_used", "supply_cap", "worker_sat"]
    + [f"struct_{s}" for s in [
        "NEXUS", "PYLON", "GATEWAY", "WARPGATE", "FORGE", "TWILIGHTCOUNCIL",
        "PHOTONCANNON", "SHIELDBATTERY", "TEMPLARARCHIVE", "ROBOTICSBAY",
        "ROBOTICSFACILITY", "ASSIMILATOR", "CYBERNETICSCORE", "STARGATE", "FLEETBEACON"
    ]]
    + [f"unit_{u}" for u in [
        "PROBE", "ZEALOT", "STALKER", "HIGHTEMPLAR", "ARCHON",
        "IMMORTAL", "CARRIER", "VOIDRAY", "ADEPT", "PHOENIX", "COLOSSUS"
    ]]
    + [f"pend_struct_{s}" for s in [
        "NEXUS", "PYLON", "GATEWAY", "WARPGATE", "FORGE", "TWILIGHTCOUNCIL",
        "PHOTONCANNON", "SHIELDBATTERY", "TEMPLARARCHIVE", "ROBOTICSBAY",
        "ROBOTICSFACILITY", "ASSIMILATOR", "CYBERNETICSCORE", "STARGATE", "FLEETBEACON"
    ]]
    + [f"pend_unit_{u}" for u in [
        "PROBE", "ZEALOT", "STALKER", "HIGHTEMPLAR", "ARCHON",
        "IMMORTAL", "CARRIER", "VOIDRAY", "ADEPT", "PHOENIX", "COLOSSUS"
    ]]
    + ["idle_gw_wg", "idle_sg", "idle_robo", "idle_wg"]
    + ["ground_weapons_lvl", "shields_lvl", "air_weapons_lvl"]
)

assert len(OBS_FEATURE_NAMES) == OBS_SIZE, (
    f"Feature name count mismatch: {len(OBS_FEATURE_NAMES)} vs {OBS_SIZE}"
)


def load_dataset(path: str):
    data = np.load(path, allow_pickle=True)
    raw = data["sequences"]
    sequences = []
    for seq in raw:
        seq = seq.astype(np.float32)
        obs = seq[:, :OBS_SIZE]
        acts = seq[:, OBS_SIZE].astype(np.int64)
        sequences.append((obs, acts))
    return sequences


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def audit_action_distribution(sequences):
    section("ACTION LABEL DISTRIBUTION")

    counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    for _, acts in sequences:
        for a in acts:
            if 0 <= a < NUM_ACTIONS:
                counts[a] += 1

    total = counts.sum()
    total_nonidle = counts[1:].sum()
    do_nothing_pct = 100.0 * counts[0] / total

    print(f"  Total timesteps:       {total:>10,}")
    print(
        f"  do_nothing count:      {counts[0]:>10,}  ({do_nothing_pct:.1f}%)")
    print(
        f"  Non-idle count:        {total_nonidle:>10,}  ({100-do_nothing_pct:.1f}%)")
    print(
        f"\n  {'ID':>4}  {'Action':<28}  {'Count':>8}  {'% total':>8}  {'% non-idle':>10}")
    print(f"  {'-'*4}  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*10}")

    # Sort by count descending, do_nothing first
    order = [0] + sorted(range(1, NUM_ACTIONS), key=lambda i: -counts[i])
    for i in order:
        name = ACTIONS[i] if i < len(ACTIONS) else f"action_{i}"
        pct_total = 100.0 * counts[i] / total
        pct_nonidle = 100.0 * \
            counts[i] / total_nonidle if (i > 0 and total_nonidle > 0) else 0.0
        marker = "  <-- ZERO SAMPLES" if counts[i] == 0 and i > 0 else ""
        print(
            f"  {i:>4}  {name:<28}  {counts[i]:>8,}  {pct_total:>7.2f}%  {pct_nonidle:>9.2f}%{marker}")

    # Imbalance warning
    nz = counts[1:][counts[1:] > 0]
    if len(nz) > 1:
        ratio = nz.max() / nz.min()
        print(f"\n  Max/min ratio among non-idle actions: {ratio:.1f}x")
        if ratio > 50:
            print("  [WARN] Extreme imbalance — consider subsampling do_nothing")
        elif ratio > 20:
            print("  [NOTE] High imbalance — class weights will help but verify")

    return counts


def audit_per_replay(sequences):
    section("PER-REPLAY STATISTICS")

    lengths = [len(acts) for _, acts in sequences]
    idle_pcts = []
    for _, acts in sequences:
        idle_pcts.append(100.0 * (acts == 0).sum() / len(acts))

    print(f"  Replays:              {len(sequences)}")
    print(f"\n  Sequence lengths:")
    print(f"    min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    # Length histogram (coarse buckets)
    buckets = [0, 100, 200, 400, 800, 99999]
    labels = ["<100", "100-200", "200-400", "400-800", "800+"]
    for lo, hi, label in zip(buckets, buckets[1:], labels):
        n = sum(1 for l in lengths if lo <= l < hi)
        bar = "#" * n
        print(f"    {label:>8}  {bar}  ({n})")

    print(f"\n  do_nothing % per replay:")
    print(f"    min={min(idle_pcts):.1f}%, max={max(idle_pcts):.1f}%, "
          f"mean={np.mean(idle_pcts):.1f}%, median={np.median(idle_pcts):.1f}%")

    # Flag replays with extremely high idle rate
    bad = [(i, p) for i, p in enumerate(idle_pcts) if p > 90]
    if bad:
        print(f"\n  [WARN] {len(bad)} replay(s) with >90% do_nothing:")
        for idx, pct in bad[:10]:
            print(f"    replay {idx}: {pct:.1f}% idle  (len={lengths[idx]})")


def audit_obs_features(sequences):
    section("OBSERVATION FEATURE STATISTICS")

    all_obs = np.concatenate([obs for obs, _ in sequences], axis=0)
    means = all_obs.mean(0)
    stds = all_obs.std(0)
    mins = all_obs.min(0)
    maxs = all_obs.max(0)

    print(f"  Total rows: {len(all_obs):,}")
    print(f"\n  {'Idx':>4}  {'Feature':<30}  {'Mean':>7}  {'Std':>7}  "
          f"{'Min':>7}  {'Max':>7}  Notes")
    print(f"  {'-'*4}  {'-'*30}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  -----")

    dead_features = []
    for i, name in enumerate(OBS_FEATURE_NAMES):
        note = ""
        if stds[i] < 1e-4:
            note = "<-- DEAD (zero variance)"
            dead_features.append(i)
        elif maxs[i] > 1.5:
            note = "<-- possible normalization issue"
        print(f"  {i:>4}  {name:<30}  {means[i]:>7.3f}  {stds[i]:>7.3f}  "
              f"{mins[i]:>7.3f}  {maxs[i]:>7.3f}  {note}")

    if dead_features:
        print(f"\n  [WARN] {len(dead_features)} dead feature(s) — always zero: "
              f"{[OBS_FEATURE_NAMES[i] for i in dead_features]}")
    else:
        print("\n  All features have non-zero variance.")


def audit_action_obs_context(sequences, top_n: int = 10):
    section(
        f"OBS CONTEXT WHEN EACH ACTION FIRES (top {top_n} non-idle actions by count)")

    # Collect obs rows per action
    action_obs = defaultdict(list)
    for obs, acts in sequences:
        for t, a in enumerate(acts):
            if 0 < a < NUM_ACTIONS:
                action_obs[a].append(obs[t])

    counts = {a: len(v) for a, v in action_obs.items()}
    top_actions = sorted(counts.keys(), key=lambda a: -counts[a])[:top_n]

    # Key obs indices to show: time, minerals, gas, supply, structures presence
    key_indices = {
        "time":     0,
        "min_bin":  slice(1, 5),   # argmax to get bin
        "gas_bin":  slice(5, 9),
        "supply%":  9,             # supply_used (normalised)
        "pylons":   13,
        "gateways": 14,
        "cybcore":  24,
    }

    print(f"  {'ID':>4}  {'Action':<28}  {'N':>6}  "
          f"{'avg_time':>9}  {'avg_supply%':>11}  {'avg_pylons':>10}  "
          f"{'avg_gateways':>12}  {'avg_cybcore':>11}")
    print(
        f"  {'-'*4}  {'-'*28}  {'-'*6}  {'-'*9}  {'-'*11}  {'-'*10}  {'-'*12}  {'-'*11}")

    for a in top_actions:
        obs_arr = np.array(action_obs[a])
        name = ACTIONS[a] if a < len(ACTIONS) else f"action_{a}"
        avg_time = obs_arr[:, 0].mean() * 720       # denorm seconds
        avg_supply = obs_arr[:, 9].mean() * 200       # denorm
        avg_pylons = obs_arr[:, 13].mean() * 10       # denorm
        avg_gws = obs_arr[:, 14].mean() * 10
        avg_cyb = obs_arr[:, 24].mean() * 10
        print(f"  {a:>4}  {name:<28}  {counts[a]:>6,}  "
              f"  {avg_time:>7.0f}s  {avg_supply:>10.1f}  {avg_pylons:>10.2f}  "
              f"  {avg_gws:>11.2f}  {avg_cyb:>11.2f}")


def audit_do_nothing_subsampling_impact(sequences):
    section("DO_NOTHING SUBSAMPLING SIMULATION")

    counts_orig = np.zeros(NUM_ACTIONS, dtype=np.int64)
    for _, acts in sequences:
        for a in acts:
            if 0 <= a < NUM_ACTIONS:
                counts_orig[a] += 1

    total_orig = counts_orig.sum()
    nonidle_orig = counts_orig[1:].sum()

    for keep_ratio in [1.0, 0.5, 0.25, 0.1]:
        kept_idle = int(counts_orig[0] * keep_ratio)
        total_new = nonidle_orig + kept_idle
        idle_pct = 100.0 * kept_idle / total_new
        print(f"  Keep {keep_ratio*100:>5.0f}% of do_nothing  →  "
              f"total={total_new:>8,}  idle%={idle_pct:>5.1f}%  "
              f"non-idle%={100-idle_pct:>5.1f}%")

    print(f"\n  Recommendation: aim for do_nothing ≤ 50% of total rows.")
    current_idle_pct = 100.0 * counts_orig[0] / total_orig
    if current_idle_pct > 70:
        target_keep = (nonidle_orig / total_orig) / (current_idle_pct / 100.0)
        print(f"  At current {current_idle_pct:.1f}% idle, keep ~{target_keep*100:.0f}% "
              f"of do_nothing rows to reach 50/50.")


def main():
    parser = argparse.ArgumentParser(description="SC2 dataset quality audit")
    parser.add_argument("--path", default=DATASET_PATH)
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top actions to show in obs-context table")
    args = parser.parse_args()

    path = args.path
    print(f"\nLoading dataset from: {path}")
    if not Path(path).exists():
        print(f"ERROR: file not found: {path}")
        return

    sequences = load_dataset(path)
    print(f"Loaded {len(sequences)} sequence(s)")

    audit_action_distribution(sequences)
    audit_per_replay(sequences)
    audit_obs_features(sequences)
    audit_action_obs_context(sequences, top_n=args.top)
    audit_do_nothing_subsampling_impact(sequences)

    print(f"\n{'='*60}")
    print("  Audit complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
