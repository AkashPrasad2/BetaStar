"""
do_nothing_subsample.py — Drop-in patch for model.py
======================================================
Paste the SequenceDataset class and collate function below into model.py,
replacing the existing ones.  Set DO_NOTHING_KEEP_RATIO to control how
aggressively do_nothing rows are subsampled.

  1.0 = keep all (original behaviour)
  0.5 = keep 50% of do_nothing rows (recommended starting point)
  0.25 = keep 25% (aggressive — use if idle% > 70%)

The subsampling is done once at dataset load time, not per-epoch,
so training speed is unaffected after the first load.

To test the impact before committing to a full retrain:
  1. Set DO_NOTHING_KEEP_RATIO = 0.5
  2. Run dataset_audit.py to see the new distribution
  3. Train for 10 epochs and check if do_nothing prediction rate drops

IMPORTANT: After changing this ratio, re-run compute_class_weights() —
the class weights should reflect the new distribution, not the original.
The new SequenceDataset below passes the filtered sequences to the existing
compute_class_weights(), so this is handled automatically.
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# ---- TUNE THIS ----
DO_NOTHING_KEEP_RATIO = 0.5   # keep 50% of do_nothing timesteps


class SequenceDataset(Dataset):
    """
    Each object represents one replay: (obs_tensor (T', OBS_SIZE), action_tensor (T',)).
    T' <= T because do_nothing rows are subsampled at load time.

    Subsampling strategy: for each sequence, randomly drop do_nothing timesteps
    at rate (1 - DO_NOTHING_KEEP_RATIO).  Non-idle steps are always kept.
    The sequence structure is preserved (temporal order maintained).
    """

    def __init__(self, path: str, keep_ratio: float = DO_NOTHING_KEEP_RATIO,
                 seed: int = 42):
        from model import OBS_SIZE   # avoid circular import if used standalone

        rng = np.random.default_rng(seed)
        data = np.load(path, allow_pickle=True)
        raw = data["sequences"]

        self.sequences = []
        original_total = 0
        kept_total = 0

        for seq in raw:
            seq = seq.astype(np.float32)
            obs_full = seq[:, :OBS_SIZE]
            act_full = seq[:, OBS_SIZE].astype(np.int64)
            original_total += len(act_full)

            if keep_ratio < 1.0:
                # Build mask: always keep non-idle, randomly keep do_nothing
                is_idle = act_full == 0
                rand_vals = rng.random(len(act_full))
                keep_mask = (~is_idle) | (rand_vals < keep_ratio)
                obs_filtered = obs_full[keep_mask]
                act_filtered = act_full[keep_mask]
            else:
                obs_filtered = obs_full
                act_filtered = act_full

            # Skip sequences that become too short after filtering
            if len(act_filtered) < 5:
                continue

            kept_total += len(act_filtered)
            obs_t = torch.tensor(obs_filtered, dtype=torch.float32)
            act_t = torch.tensor(act_filtered, dtype=torch.long)
            self.sequences.append((obs_t, act_t))

        idle_kept = sum((a == 0).sum().item() for _, a in self.sequences)
        total_kept = sum(len(a) for _, a in self.sequences)
        idle_pct = 100.0 * idle_kept / total_kept if total_kept > 0 else 0.0

        lengths = [len(s[0]) for s in self.sequences]
        print(f"Loaded {len(self.sequences)} sequences | "
              f"original steps: {original_total:,}  kept: {total_kept:,} "
              f"({100*total_kept/original_total:.1f}%)")
        print(f"After subsampling: do_nothing={idle_pct:.1f}%  "
              f"lengths: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.0f}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_sequences(batch):
    """
    Pad variable-length sequences. Padding value -100 is ignored by
    CrossEntropyLoss(ignore_index=-100).
    Unchanged from original — compatible with subsampled sequences.
    """
    obs_list, act_list = zip(*batch)
    obs_pad = pad_sequence(obs_list, batch_first=True, padding_value=0.0)
    act_pad = pad_sequence(act_list, batch_first=True, padding_value=-100)
    return obs_pad, act_pad


# ---------------------------------------------------------------------------
# Standalone usage: print the distribution at a given keep_ratio
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import argparse
    sys.path.insert(0, r"C:\dev\BetaStar")
    from model import OBS_SIZE, DATASET_PATH   # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("--path",       default=DATASET_PATH)
    parser.add_argument("--ratio",      type=float,
                        default=DO_NOTHING_KEEP_RATIO)
    args = parser.parse_args()

    print(f"\nSimulating subsampling at keep_ratio={args.ratio}")
    ds = SequenceDataset(args.path, keep_ratio=args.ratio)
    print(f"\nTotal sequences after filter: {len(ds)}")
