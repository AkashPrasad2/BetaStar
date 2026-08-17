"""
SC2 Protoss Imitation Learning — Transformer Model + Training Script
=====================================================================
Architecture:
    obs (OBS_SIZE,) -> input proj (OBS_SIZE->128) -> sinusoidal pos enc
    -> 4x causal TransformerEncoderLayer (d=128, heads=4, ff=256)
    -> LayerNorm -> Linear (128->NUM_ACTIONS logits)

Legal-action masking applied consistently in BOTH the training loop
and predict_action, via the shared action_mask module.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from action_mask import apply_legal_mask, apply_training_mask
from obs_spec import OBS_SIZE, NUM_ACTIONS, ACTION_NAMES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_PATH = r"C:\dev\BetaStar\replays\parsed\dataset.npz"
CHECKPOINT_DIR = r"C:\dev\BetaStar\checkpoints"

# OBS_SIZE comes from obs_spec, the single source of truth for the layout.
# NUM_ACTIONS comes from obs_spec.ACTION_NAMES, the single source of truth. It was
# hardcoded to 35 here, which would have silently disagreed with the mask and the
# parser the moment an action was added or removed.

# Transformer hyper-params
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
MAX_SEQ_LEN = 2048   # positional encoding capacity

# Training hyper-params
BATCH_SIZE = 16
EPOCHS = 80  # rarely improves beyond this
LR = 3e-4
VAL_SPLIT = 0.15
SEED = 54

# Which validation metric decides the saved checkpoint.
#
#   "macro_f1" — mean of per-class F1 (default). Every action counts equally.
#   "accuracy" — fraction of windows predicted correctly.
#   "loss"     — the class-weighted cross-entropy.
#
# Accuracy is the wrong choice here and was the previous default. Labels are
# dominated by do_nothing (42.8% of 193k windows) and train_probe (20.8%), so a
# model that learns only those two scores ~63.5% while never building anything.
# build_cyberneticscore is 0.45% of windows, meaning a checkpoint that NEVER
# predicts a cybercore loses at most 0.45 accuracy points -- accuracy is close to
# indifferent about the building that gates the entire midgame. That is exactly
# the mode collapse observed in play: greedy argmax was do_nothing 89-99% of the
# time and only temperature sampling made the bot functional.
#
# Macro-F1 averages per-class F1 unweighted, so build_cyberneticscore contributes
# 1/32 = 3.1% of the score instead of 0.45% -- about 7x more leverage. F1 is the
# harmonic mean of precision and recall, which collapses toward zero if EITHER is
# near zero, so it cannot be gamed by never predicting a class (recall -> 0) or
# by predicting it constantly (precision -> 0).
MODEL_SELECTION = "macro_f1"

# Classes with fewer than this many validation labels are left out of the macro
# average. F1 measured on a handful of samples is mostly noise, and which rare
# actions land in the val split is an accident of the replay-level split. Set to
# 0 to include every class that appears at all.
MACRO_F1_MIN_SUPPORT = 10

# keep the decisions diverse (not applied during training, only inference)
INFERENCE_TEMPERATURE = 1.5

# Cap context window at inference to bound latency.
#
# Training feeds whole replays, so a window at index k gets positional-encoding
# position k. Inference matches that exactly until the history exceeds
# MAX_CONTEXT, after which truncation pins the current window to position
# MAX_CONTEXT-1 and PE decorrelates from game time -- a pairing that appears
# nowhere in training.
#
# At 256 that boundary hit at 17 minutes, inside the decisive part of most games.
# 512 windows = 2048s = 34 minutes, which covers essentially every game in the
# corpus. It is also the exact point where the O(T^2) attention cost equals all
# the linear (projection + FFN) cost for this model size, so it stays cheap:
# ~1.1 GFLOP per decision against a 4s budget.
#
# Must stay <= MAX_SEQ_LEN (positional encoding capacity).
MAX_CONTEXT = 512


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)"""
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProtossTransformerModel(nn.Module):
    """
    Causal (decoder-only) transformer for sequential action prediction.
    Takes a sequence of game-state observations and predicts an action at
    each timestep, attending only to the current and past observations.
    """

    def __init__(
        self,
        obs_size:        int = OBS_SIZE,
        d_model:         int = D_MODEL,
        nhead:           int = NHEAD,
        num_layers:      int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout:         float = DROPOUT,
        num_actions:     int = NUM_ACTIONS,
        max_seq_len:     int = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.d_model = d_model

        # Project flat observation to embedding space
        self.input_proj = nn.Sequential(
            nn.Linear(obs_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Sinusoidal positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)

        # Causal transformer (using TransformerEncoder with causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,   # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, obs_size) — sequence of observations
        Returns:
            logits: (B, T, num_actions)
        """
        B, T, _ = x.shape

        # Project to embedding space
        h = self.input_proj(x)              # (B, T, d_model)

        # Add positional encoding
        h = self.pos_encoding(h)            # (B, T, d_model)

        # Generate causal mask (upper-triangular = -inf)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device
        )

        # Apply transformer with causal masking
        h = self.transformer(h, mask=causal_mask, is_causal=True)

        # Output logits
        return self.output_head(h)          # (B, T, num_actions)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Each object represents one replay: (obs_tensor (T, OBS_SIZE), action_tensor (T,)).
    """

    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True)
        raw = data["sequences"]
        self.sequences = []
        for seq in raw:
            seq = seq.astype(np.float32)
            obs = torch.tensor(seq[:, :OBS_SIZE], dtype=torch.float32)
            act = torch.tensor(
                seq[:, OBS_SIZE].astype(np.int64), dtype=torch.long)
            self.sequences.append((obs, act))
        lengths = [len(s[0]) for s in self.sequences]
        print(f"Loaded {len(self.sequences)} sequences | "
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
    """
    obs_list, act_list = zip(*batch)
    obs_pad = pad_sequence(obs_list, batch_first=True, padding_value=0.0)
    act_pad = pad_sequence(act_list, batch_first=True, padding_value=-100)
    return obs_pad, act_pad


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_class_weights(
    dataset: SequenceDataset, num_classes: int
) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, act_seq in dataset.sequences:
        for a in act_seq.numpy():
            counts[int(a)] += 1
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / np.sqrt(counts)
    weights /= weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def _apply_mask_real_only(
    flat_logits: torch.Tensor,
    flat_obs:    torch.Tensor,
    flat_acts:   torch.Tensor,
) -> torch.Tensor:
    """Apply the relaxed TRAINING mask only to real (non-padded) positions."""
    real_mask = flat_acts != -100
    masked = flat_logits.clone()

    if real_mask.any():
        real_logits = apply_training_mask(
            flat_logits[real_mask], flat_obs[real_mask])
        masked[real_mask] = real_logits

    return masked


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for obs_pad, act_pad in loader:
        obs_pad = obs_pad.to(device)
        act_pad = act_pad.to(device)

        optimizer.zero_grad()
        logits = model(obs_pad)

        B, T, A = logits.shape
        flat_logits = logits.reshape(B * T, A)
        flat_obs = obs_pad.reshape(B * T, obs_pad.shape[-1])
        flat_acts = act_pad.reshape(B * T)

        flat_logits = _apply_mask_real_only(flat_logits, flat_obs, flat_acts)

        real = flat_acts != -100
        if real.any():
            real_idx = real.nonzero(as_tuple=True)[0]
            label_logits = flat_logits[real_idx].gather(
                1, flat_acts[real_idx].unsqueeze(1))
            bad = ~label_logits[:, 0].isfinite()
            if bad.any():
                n_bad = bad.sum().item()
                print(f"  [WARN] {n_bad} label/mask conflicts -- silencing. "
                      f"Run conflict_diagnostic.py.")
                flat_acts = flat_acts.clone()
                flat_acts[real_idx[bad]] = -100

        loss = criterion(flat_logits, flat_acts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        mask = real
        preds = flat_logits.argmax(1)
        correct += (preds[mask] == flat_acts[mask]).sum().item()
        total += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def macro_f1_from_counts(tp, fp, fn, min_support: int = 0):
    """
    Per-class F1 and their unweighted mean.

    Returns (macro_f1, per_class) where per_class maps class index ->
    (f1, precision, recall, support). Classes with support below min_support are
    reported but excluded from the mean; classes absent from the data entirely
    are excluded outright rather than counted as zero, since otherwise the metric
    would swing on whether a rare action happened to land in the val split.
    """
    per_class = {}
    scores = []
    for c in range(len(tp)):
        support = int(tp[c] + fn[c])
        if support == 0 and fp[c] == 0:
            continue                      # class does not occur and is never predicted
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        per_class[c] = (f1, precision, recall, support)
        if support >= max(min_support, 1):
            scores.append(f1)
    return (sum(scores) / len(scores) if scores else 0.0), per_class


def eval_epoch(model, loader, criterion, device):
    """Returns (loss, accuracy, macro_f1, per_class_f1)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    # Per-class confusion counts, accumulated on GPU then read once at the end.
    tp = torch.zeros(NUM_ACTIONS, dtype=torch.long, device=device)
    fp = torch.zeros(NUM_ACTIONS, dtype=torch.long, device=device)
    fn = torch.zeros(NUM_ACTIONS, dtype=torch.long, device=device)

    for obs_pad, act_pad in loader:
        obs_pad = obs_pad.to(device)
        act_pad = act_pad.to(device)

        logits = model(obs_pad)

        B, T, A = logits.shape
        flat_logits = logits.reshape(B * T, A)
        flat_obs = obs_pad.reshape(B * T, obs_pad.shape[-1])
        flat_acts = act_pad.reshape(B * T)

        flat_logits = _apply_mask_real_only(flat_logits, flat_obs, flat_acts)

        real = flat_acts != -100
        if real.any():
            real_idx = real.nonzero(as_tuple=True)[0]
            label_logits = flat_logits[real_idx].gather(
                1, flat_acts[real_idx].unsqueeze(1))
            bad = ~label_logits[:, 0].isfinite()
            if bad.any():
                flat_acts = flat_acts.clone()
                flat_acts[real_idx[bad]] = -100

        loss = criterion(flat_logits, flat_acts)

        preds = flat_logits.argmax(1)
        correct += (preds[real] == flat_acts[real]).sum().item()
        total += real.sum().item()
        total_loss += loss.item() * real.sum().item()

        # Confusion counts over real positions only.
        p_real = preds[real]
        y_real = flat_acts[real]
        hit = p_real == y_real
        tp += torch.bincount(y_real[hit], minlength=NUM_ACTIONS)
        fp += torch.bincount(p_real[~hit], minlength=NUM_ACTIONS)
        fn += torch.bincount(y_real[~hit], minlength=NUM_ACTIONS)

    macro_f1, per_class = macro_f1_from_counts(
        tp.tolist(), fp.tolist(), fn.tolist(), MACRO_F1_MIN_SUPPORT)
    return total_loss / total, correct / total, macro_f1, per_class


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU found)")

    dataset = SequenceDataset(DATASET_PATH)
    val_size = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"Train replays: {train_size} | Val replays: {val_size}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_sequences, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_sequences, num_workers=0)

    model = ProtossTransformerModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    class_weights = compute_class_weights(dataset, NUM_ACTIONS).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Loss is minimized; accuracy and macro-F1 are maximized.
    _LOWER_IS_BETTER = MODEL_SELECTION == "loss"
    if _LOWER_IS_BETTER:
        best_val_metric = float("inf")
        def is_better(new, best): return new < best
    else:
        best_val_metric = -1.0
        def is_better(new, best): return new > best

    best_path = Path(CHECKPOINT_DIR) / "best_model.pt"
    best_per_class = {}

    print(f"\nSelecting checkpoints on: {MODEL_SELECTION}")
    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>9} {'Val MacroF1':>12} {'LR':>10}")
    print("-" * 78)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, per_class = eval_epoch(
            model, val_loader, criterion, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>10.3%} "
              f"{val_loss:>10.4f} {val_acc:>9.3%} {val_f1:>12.4f} {lr:>10.2e}")

        current_metric = {
            "accuracy": val_acc,
            "macro_f1": val_f1,
            "loss":     val_loss,
        }[MODEL_SELECTION]

        if is_better(current_metric, best_val_metric):
            best_val_metric = current_metric
            best_per_class = per_class
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "val_loss":        val_loss,
                "val_acc":         val_acc,
                "val_macro_f1":    val_f1,
                "selection":       MODEL_SELECTION,
                "obs_size":        OBS_SIZE,
                "num_actions":     NUM_ACTIONS,
                "d_model":         D_MODEL,
                "nhead":           NHEAD,
                "num_layers":      NUM_LAYERS,
                "dim_feedforward":  DIM_FEEDFORWARD,
                "max_seq_len":     MAX_SEQ_LEN,
            }, best_path)
            shown = (f"{best_val_metric:.3%}" if MODEL_SELECTION == "accuracy"
                     else f"{best_val_metric:.4f}")
            print(f"         ^ new best ({MODEL_SELECTION}={shown})"
                  f" saved to {best_path}")

    shown = (f"{best_val_metric:.3%}" if MODEL_SELECTION == "accuracy"
             else f"{best_val_metric:.4f}")
    print(f"\nTraining complete. Best val {MODEL_SELECTION}: {shown}")

    # Per-class report for the saved checkpoint. This is the diagnostic that
    # accuracy hid: an action with F1 near zero is one the bot will essentially
    # never take, no matter how good the headline number looks.
    if best_per_class:
        print(f"\n{'action':<26}{'F1':>8}{'prec':>8}{'recall':>8}{'support':>9}")
        print("-" * 59)
        rows = sorted(best_per_class.items(), key=lambda kv: kv[1][0])
        for cid, (f1, prec, rec, sup) in rows:
            flag = ""
            if sup < MACRO_F1_MIN_SUPPORT:
                flag = "  (excluded, low support)"
            elif f1 < 0.05:
                flag = "  <-- effectively never predicted"
            name = ACTION_NAMES[cid] if cid < len(ACTION_NAMES) else str(cid)
            print(f"{name:<26}{f1:>8.3f}{prec:>8.3f}{rec:>8.3f}{sup:>9}{flag}")
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu") -> ProtossTransformerModel:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ProtossTransformerModel(
        obs_size=ckpt["obs_size"],
        num_actions=ckpt["num_actions"],
        d_model=ckpt.get("d_model", D_MODEL),
        nhead=ckpt.get("nhead", NHEAD),
        num_layers=ckpt.get("num_layers", NUM_LAYERS),
        dim_feedforward=ckpt.get("dim_feedforward", DIM_FEEDFORWARD),
        max_seq_len=ckpt.get("max_seq_len", MAX_SEQ_LEN),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device)


def predict_action(
    model:       ProtossTransformerModel,
    obs_history: list[list[float]],
    device:      str = "cpu",
    temperature: float = INFERENCE_TEMPERATURE,
    return_diagnostics: bool = False,
    top_k:       int = 5,
):
    """
    Sequence inference with legal masking + temperature sampling.

    Args:
        model:       trained ProtossTransformerModel
        obs_history: list of flat observation vectors (oldest first)
        device:      torch device string
        temperature: softmax temperature
        return_diagnostics: also return a dict describing the decision
        top_k:       how many candidates to report in the diagnostics

    Returns:
        action_id, or (action_id, diagnostics) if return_diagnostics is set.
    """
    x = torch.tensor(obs_history, dtype=torch.float32).unsqueeze(0).to(device)
    # x: (1, T, obs_size)

    with torch.no_grad():
        logits = model(x)   # (1, T, num_actions)

    # Take the last position's logits and the last obs for masking
    last_logits = logits[:, -1, :]   # (1, num_actions)
    last_obs = x[:, -1, :]           # (1, obs_size)

    masked_logits = apply_legal_mask(last_logits, last_obs)
    probs = torch.softmax(masked_logits[0] / temperature, dim=-1)
    action_id = int(torch.multinomial(probs, 1).item())

    if not return_diagnostics:
        return action_id

    # Diagnostics: what the model wanted vs what the mask permitted.
    from actions import ACTIONS

    def name_of(idx: int) -> str:
        return ACTIONS[idx] if 0 <= idx < len(ACTIONS) else str(idx)

    raw = last_logits[0]
    masked = masked_logits[0]
    raw_top1 = int(raw.argmax().item())
    masked_top1 = int(masked.argmax().item())
    k = min(top_k, probs.numel())
    top = torch.topk(probs, k)

    diagnostics = {
        "n_legal":        int(torch.isfinite(masked).sum().item()),
        "raw_top1":       raw_top1,
        "raw_top1_name":  name_of(raw_top1),
        "masked_top1":    masked_top1,
        "masked_top1_name": name_of(masked_top1),
        # True when the model's preferred action was illegal and the mask
        # forced a different choice.
        "blocked_top1":   bool(raw_top1 != masked_top1),
        "chosen_prob":    round(float(probs[action_id]), 4),
        "greedy_prob":    round(float(probs.max()), 4),
        "top_named":      [
            [name_of(int(i)), round(float(p), 4)]
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ],
    }
    return action_id, diagnostics


if __name__ == "__main__":
    train()
