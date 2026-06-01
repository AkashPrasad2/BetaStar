"""
model_probe.py — Model Inference Quality Probe
================================================
Loads your trained checkpoint and runs it over a sample of sequences
from the dataset to answer:

  1. What % of top-1 predictions are do_nothing (before and after masking)?
  2. How often does the mask block the model's top-1 choice?
  3. What is the per-action prediction frequency vs dataset frequency?
     (detects mode collapse toward do_nothing or any other action)
  4. What does the logit distribution look like pre/post mask?
  5. Per-action: what is the average confidence (softmax prob) when the
     model correctly predicts that action?

Usage:
    python model_probe.py
    python model_probe.py --checkpoint C:/dev/BetaStar/checkpoints/best_model.pt
    python model_probe.py --checkpoint ... --n_seqs 50 --temperature 1.2
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = r"C:\dev\BetaStar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_PATH = r"C:\dev\BetaStar\replays\parsed\dataset.npz"
CHECKPOINT_PATH = r"C:\dev\BetaStar\checkpoints\best_model.pt"
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


def section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def load_sample_sequences(path: str, n: int, rng: np.random.Generator):
    data = np.load(path, allow_pickle=True)
    raw = data["sequences"]
    idxs = rng.choice(len(raw), size=min(n, len(raw)), replace=False)
    sequences = []
    for i in idxs:
        seq = raw[i].astype(np.float32)
        obs = seq[:, :OBS_SIZE]
        acts = seq[:, OBS_SIZE].astype(np.int64)
        sequences.append((obs, acts))
    return sequences


def run_probe(model, sequences, device, temperature: float):
    import torch
    from action_mask import apply_legal_mask, build_legal_mask

    # Aggregate counters
    pred_counts_raw = np.zeros(
        NUM_ACTIONS, dtype=np.int64)  # top-1 before mask
    pred_counts_masked = np.zeros(
        NUM_ACTIONS, dtype=np.int64)  # top-1 after mask
    label_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)  # ground truth

    mask_blocked_total = 0     # times mask changed the top-1 prediction
    total_steps = 0

    # Per-action: confidence when correctly predicted (after mask, temp=1.0)
    correct_probs = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for obs_np, acts_np in sequences:
            T = len(obs_np)
            # Process each timestep with its full causal context
            # (same as inference: obs_history up to t)
            obs_t = torch.tensor(
                obs_np, dtype=torch.float32).unsqueeze(0).to(device)
            # Forward pass over whole sequence
            logits = model(obs_t)  # (1, T, A)
            logits = logits[0]     # (T, A)

            for t in range(T):
                raw_logit = logits[t].unsqueeze(0)   # (1, A)
                obs_single = obs_t[0, t].unsqueeze(0)  # (1, OBS)
                label = int(acts_np[t])

                # Raw top-1 (no mask)
                raw_top1 = int(raw_logit.argmax(1).item())
                pred_counts_raw[raw_top1] += 1

                # Masked top-1 (inference mask)
                masked_logit = apply_legal_mask(raw_logit, obs_single)
                masked_top1 = int(masked_logit.argmax(1).item())
                pred_counts_masked[masked_top1] += 1

                if raw_top1 != masked_top1:
                    mask_blocked_total += 1

                if 0 <= label < NUM_ACTIONS:
                    label_counts[label] += 1

                # Confidence on correct prediction (temperature=1 for fair comparison)
                mask_bool = build_legal_mask(obs_single)[0]  # (A,)
                masked_for_conf = masked_logit[0].clone()
                masked_for_conf[~mask_bool] = float('-inf')
                probs = torch.softmax(masked_for_conf, dim=0)
                if masked_top1 == label and label > 0:
                    correct_probs[label].append(probs[label].item())

                total_steps += 1

    return (pred_counts_raw, pred_counts_masked, label_counts,
            mask_blocked_total, total_steps, correct_probs)


def print_distribution_comparison(pred_raw, pred_masked, label_counts, total):
    section("PREDICTION DISTRIBUTION vs LABEL DISTRIBUTION")

    idle_pred_pct = 100.0 * pred_masked[0] / total
    idle_label_pct = 100.0 * label_counts[0] / total
    blocked_by_raw = 100.0 * pred_raw[0] / total
    print(f"  Total steps analysed: {total:,}")
    print(f"\n  do_nothing rate:")
    print(f"    Raw model (pre-mask):  {blocked_by_raw:.1f}%")
    print(f"    After inference mask:  {idle_pred_pct:.1f}%")
    print(f"    Ground truth labels:   {idle_label_pct:.1f}%")

    if idle_pred_pct > idle_label_pct + 20:
        print(
            f"\n  [WARN] Model predicts do_nothing {idle_pred_pct - idle_label_pct:.1f}pp MORE")
        print(f"         than ground truth. Likely cause: class imbalance or mask over-blocking.")
    elif idle_pred_pct < idle_label_pct - 20:
        print(
            f"\n  [NOTE] Model predicts do_nothing {idle_label_pct - idle_pred_pct:.1f}pp LESS")
        print(f"         than ground truth. Model may be over-active (good or forced by mask).")

    print(
        f"\n  {'ID':>4}  {'Action':<28}  {'Pred% (masked)':>15}  {'Label%':>8}  {'Delta':>8}")
    print(f"  {'-'*4}  {'-'*28}  {'-'*15}  {'-'*8}  {'-'*8}")

    for a in range(NUM_ACTIONS):
        name = ACTIONS[a] if a < len(ACTIONS) else f"action_{a}"
        p_pct = 100.0 * pred_masked[a] / total
        l_pct = 100.0 * label_counts[a] / total
        delta = p_pct - l_pct
        flag = ""
        if a > 0 and abs(delta) > 5:
            flag = "  <--" + (" over-predicted" if delta >
                              0 else " under-predicted")
        print(
            f"  {a:>4}  {name:<28}  {p_pct:>14.2f}%  {l_pct:>7.2f}%  {delta:>+7.2f}%{flag}")


def print_mask_blocking_analysis(pred_raw, pred_masked, mask_blocked, total):
    section("MASK BLOCKING ANALYSIS")

    block_pct = 100.0 * mask_blocked / total
    print(f"  Steps where mask changed the top-1 prediction: "
          f"{mask_blocked:,}  ({block_pct:.1f}%)")

    if block_pct > 30:
        print(
            f"\n  [WARN] Mask blocks the model's first choice {block_pct:.1f}% of the time.")
        print(f"         This suggests the model frequently predicts actions it can't execute.")
        print(f"         The model then falls back to do_nothing or another legal action.")
        print(
            f"         Consider: aligning training mask more closely with inference mask.")
    elif block_pct > 10:
        print(
            f"\n  [NOTE] {block_pct:.1f}% blocking rate is moderate — worth monitoring.")
    else:
        print(f"\n  Blocking rate {block_pct:.1f}% looks healthy.")

    # What does the model predict when it would have been blocked?
    fallback = pred_masked - pred_raw
    # Actions that gained predictions (i.e. fallback targets)
    gained = [(a, int(fallback[a]))
              for a in range(NUM_ACTIONS) if fallback[a] > 0]
    gained.sort(key=lambda x: -x[1])
    if gained:
        print(f"\n  When blocked, model falls back to:")
        for a, n in gained[:8]:
            name = ACTIONS[a] if a < len(ACTIONS) else f"action_{a}"
            print(f"    {name:<30}  +{n:,} predictions")


def print_confidence_table(correct_probs):
    section("AVERAGE CONFIDENCE WHEN CORRECTLY PREDICTED")
    print(f"  (softmax prob of the correct action when model gets it right, T=1.0)")
    print(f"\n  {'ID':>4}  {'Action':<28}  {'N correct':>10}  {'Avg prob':>10}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*4}  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    for a in sorted(correct_probs.keys()):
        name = ACTIONS[a] if a < len(ACTIONS) else f"action_{a}"
        probs = correct_probs[a]
        if not probs:
            continue
        arr = np.array(probs)
        print(f"  {a:>4}  {name:<28}  {len(arr):>10,}  {arr.mean():>10.3f}  "
              f"{arr.min():>8.3f}  {arr.max():>8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--dataset",    default=DATASET_PATH)
    parser.add_argument("--n_seqs",     type=int,   default=30,
                        help="Number of sequences to probe (default 30)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    import torch

    ckpt_path = args.checkpoint
    if not Path(ckpt_path).exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return
    if not Path(args.dataset).exists():
        print(f"ERROR: dataset not found: {args.dataset}")
        return

    try:
        from model import load_model
    except ImportError as e:
        print(
            f"ERROR importing model: {e}\nSet PROJECT_ROOT at top of script.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading checkpoint: {ckpt_path}")
    model = load_model(ckpt_path, device=device)
    print(f"Device: {device}")

    rng = np.random.default_rng(args.seed)
    print(f"\nSampling {args.n_seqs} sequences from dataset...")
    sequences = load_sample_sequences(args.dataset, args.n_seqs, rng)
    total_steps = sum(len(a) for _, a in sequences)
    print(f"Total timesteps to probe: {total_steps:,}")

    print("\nRunning inference... (this may take a moment)")
    (pred_raw, pred_masked, label_counts,
     mask_blocked, total, correct_probs) = run_probe(
        model, sequences, device, args.temperature
    )

    print_distribution_comparison(pred_raw, pred_masked, label_counts, total)
    print_mask_blocking_analysis(pred_raw, pred_masked, mask_blocked, total)
    print_confidence_table(correct_probs)

    section("DONE")
    print(f"  Probed {total:,} steps across {len(sequences)} sequences.\n")


if __name__ == "__main__":
    main()
