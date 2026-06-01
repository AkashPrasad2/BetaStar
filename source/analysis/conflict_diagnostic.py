"""
conflict_diagnostic.py — Action Mask Conflict & Train/Inference Gap Audit
==========================================================================
Loads your dataset and checks every non-zero label against both the
training mask and the inference mask to answer three questions:

  1. How many labels are illegal under the TRAINING mask?
     (These get silently turned to do_nothing during training.)

  2. How many labels are legal under training mask but ILLEGAL under the
     INFERENCE mask?  (These represent the train/inference distribution shift —
     the model was taught these actions are good but can never execute them.)

  3. Which actions have the worst conflict rates?

Also prints the per-action breakdown so you can see which actions are
being systematically demoted.

Usage:
    python conflict_diagnostic.py
    python conflict_diagnostic.py --path C:/dev/BetaStar/replays/parsed/dataset.npz
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# ---- add project root to path so we can import action_mask ----
# Adjust this if your project is elsewhere
PROJECT_ROOT = r"C:\dev\BetaStar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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


def load_flat_dataset(path: str):
    """Returns (all_obs, all_acts) as flat arrays over every non-idle timestep."""
    data = np.load(path, allow_pickle=True)
    raw = data["sequences"]
    obs_rows, act_rows = [], []
    for seq in raw:
        seq = seq.astype(np.float32)
        obs = seq[:, :OBS_SIZE]
        acts = seq[:, OBS_SIZE].astype(np.int64)
        # only keep non-idle labels for conflict analysis
        non_idle = acts != 0
        obs_rows.append(obs[non_idle])
        act_rows.append(acts[non_idle])
    all_obs = np.concatenate(obs_rows,  axis=0)
    all_acts = np.concatenate(act_rows, axis=0)
    return all_obs, all_acts


def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def run_conflict_check(all_obs: np.ndarray, all_acts: np.ndarray,
                       build_mask_fn, mask_name: str):
    """
    For each non-idle label, check if the obs-derived mask marks it legal.
    Returns per-action (total, conflict_count) dict.
    """
    import torch
    obs_t = torch.tensor(all_obs,  dtype=torch.float32)
    acts_t = torch.tensor(all_acts, dtype=torch.long)

    # Build mask for all rows at once (batch)
    CHUNK = 4096  # process in chunks to avoid OOM on large datasets
    legal_flags = []
    for start in range(0, len(obs_t), CHUNK):
        chunk_obs = obs_t[start:start+CHUNK]
        chunk_acts = acts_t[start:start+CHUNK]
        mask = build_mask_fn(chunk_obs)           # (N, NUM_ACTIONS) bool
        # legal[i] = True if action acts[i] is marked legal in mask[i]
        idx = chunk_acts.clamp(0, NUM_ACTIONS - 1)
        legal = mask[torch.arange(len(idx)), idx]
        legal_flags.append(legal.numpy())

    legal_arr = np.concatenate(legal_flags)  # (N,) bool

    # Per-action breakdown
    per_action = {}
    for a in range(1, NUM_ACTIONS):
        mask_a = all_acts == a
        total_a = mask_a.sum()
        if total_a == 0:
            continue
        conflicts_a = (~legal_arr[mask_a]).sum()
        per_action[a] = (int(total_a), int(conflicts_a))

    total_all = len(all_acts)
    conflict_all = int((~legal_arr).sum())
    return per_action, total_all, conflict_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DATASET_PATH)
    args = parser.parse_args()

    path = args.path
    print(f"\nLoading dataset: {path}")
    if not Path(path).exists():
        print(f"ERROR: not found: {path}")
        return

    all_obs, all_acts = load_flat_dataset(path)
    print(f"Non-idle timesteps to audit: {len(all_acts):,}")

    # Import masks — if this fails, adjust PROJECT_ROOT above
    try:
        from action_mask import build_training_mask, build_legal_mask
    except ImportError as e:
        print(f"\nERROR importing action_mask: {e}")
        print("Set PROJECT_ROOT at the top of this script to your project directory.")
        return

    # ---- Training mask ----
    section("TRAINING MASK CONFLICTS (relaxed PoC mask)")
    train_per_action, total, train_conflicts = run_conflict_check(
        all_obs, all_acts, build_training_mask, "training"
    )
    train_pct = 100.0 * train_conflicts / total
    print(f"  Total non-idle labels:     {total:>8,}")
    print(
        f"  Conflict count:            {train_conflicts:>8,}  ({train_pct:.2f}%)")
    print(f"  (These get silently demoted to do_nothing during training)")

    # ---- Inference mask ----
    section("INFERENCE MASK CONFLICTS (strict idle-check mask)")
    infer_per_action, _, infer_conflicts = run_conflict_check(
        all_obs, all_acts, build_legal_mask, "inference"
    )
    infer_pct = 100.0 * infer_conflicts / total
    print(f"  Total non-idle labels:     {total:>8,}")
    print(
        f"  Conflict count:            {infer_conflicts:>8,}  ({infer_pct:.2f}%)")
    print(f"  (These actions trained OK, but bot can NEVER execute them at inference)")

    # ---- Per-action breakdown ----
    section("PER-ACTION BREAKDOWN")
    print(f"  {'ID':>4}  {'Action':<28}  {'Total':>7}  "
          f"{'Train-conflict':>14}  {'Infer-conflict':>14}  {'Gap':>6}")
    print(f"  {'-'*4}  {'-'*28}  {'-'*7}  {'-'*14}  {'-'*14}  {'-'*6}")

    all_action_ids = sorted(
        set(list(train_per_action.keys()) + list(infer_per_action.keys()))
    )
    for a in all_action_ids:
        name = ACTIONS[a] if a < len(ACTIONS) else f"action_{a}"
        t_total, t_conf = train_per_action.get(a, (0, 0))
        _, i_conf = infer_per_action.get(a, (0, 0))
        if t_total == 0:
            continue
        t_pct = 100.0 * t_conf / t_total
        i_pct = 100.0 * i_conf / t_total
        gap = i_pct - t_pct
        gap_flag = "  <-- SHIFT" if gap > 10 else ""
        train_flag = "  <-- DEMOTED" if t_pct > 20 else ""
        print(f"  {a:>4}  {name:<28}  {t_total:>7,}  "
              f"   {t_conf:>5,} ({t_pct:>5.1f}%)  "
              f"   {i_conf:>5,} ({i_pct:>5.1f}%)  "
              f"{gap:>+5.1f}%{gap_flag}{train_flag}")

    # ---- Summary advice ----
    section("SUMMARY & RECOMMENDATIONS")
    if train_pct > 5:
        print(
            f"  [WARN] {train_pct:.1f}% of non-idle labels conflict with the training mask.")
        print(f"         These become do_nothing silently, worsening class imbalance.")
        print(f"         Check the parser's pending-structure tracking for these actions.")
    else:
        print(f"  Training mask conflicts: {train_pct:.1f}% — looks healthy.")

    shift = infer_pct - train_pct
    if shift > 10:
        print(f"\n  [WARN] {shift:.1f}% training→inference mask shift.")
        print(f"         Model learned these actions but bot will often be blocked at inference.")
        print(f"         Consider either:")
        print(f"           a) relaxing the inference mask for low-risk actions, OR")
        print(f"           b) using the strict mask during training too (more conflicts, but consistent).")
    else:
        print(
            f"\n  Train→inference mask shift: {shift:.1f}% — looks reasonable.")

    print()


if __name__ == "__main__":
    main()
