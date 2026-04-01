"""
SC2 Protoss Imitation Learning — LSTM Model + Training Script
=============================================================
Architecture:
    obs (53,) -> Linear encoder (53->64) -> LSTM (64->128, 1 layer)
              -> MLP head (128->64->30 logits)

Key changes in this version:
  - Legal-action masking applied consistently in BOTH the training loop
    and predict_action, via the shared action_mask module.
    The model now learns P(action | obs, action is legal), so the
    conditional probabilities are calibrated for exactly the distribution
    seen at runtime — not the full 30-action space.
  - Temperature sampling replaces argmax in predict_action to avoid
    probability-mass collapse onto a single action.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from action_mask import apply_legal_mask

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_PATH = r"C:\dev\BetaStar\replays\parsed\dataset.npz"
CHECKPOINT_DIR = r"C:\dev\BetaStar\checkpoints"

OBS_SIZE = 57   # 6 base + 15 structures + 8 units + 15 pending structs + 8 pending units + 1 opp + 4 idle
NUM_ACTIONS = 30   # action 0 = do_nothing, kept for index stability

# Model hyper-params
ENCODER_DIM = 64
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
HEAD_HIDDEN = 64
DROPOUT = 0.3

# Training hyper-params
BATCH_SIZE = 32
EPOCHS = 80
LR = 3e-4
VAL_SPLIT = 0.15
SEED = 54

# "accuracy" = save model with best validation accuracy (better generalization)
# "loss" = save model with lowest validation loss (better imitation)
MODEL_SELECTION = "loss"  # Change to "accuracy" to switch

# Inference temperature: < 1.0 sharpens, > 1.0 softens.
# 0.8 keeps diversity without going fully random.
INFERENCE_TEMPERATURE = 0.8


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProtossLSTMModel(nn.Module):
    """
    Encodes each game-state observation, feeds it through an LSTM that carries
    context across the whole game, then decodes each hidden state into action
    logits via a small MLP.

    Training: full padded sequences via PackedSequence.
    Inference: one step at a time, (h, c) carried externally by the bot.
    """

    def __init__(
        self,
        obs_size:    int = OBS_SIZE,
        encoder_dim: int = ENCODER_DIM,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        head_hidden: int = HEAD_HIDDEN,
        num_actions: int = NUM_ACTIONS,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.encoder = nn.Sequential(
            nn.Linear(obs_size, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                n = p.size(0)
                p.data[n // 4: n // 2].fill_(1.0)  # forget gate bias = 1
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x:       torch.Tensor,
        lengths: torch.Tensor,
        hc:      tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Returns:
            logits — (batch, seq_len, num_actions)
            hc     — (h_n, c_n) final hidden/cell states
        """
        batch, seq_len, _ = x.shape
        enc = self.encoder(x.reshape(-1, x.size(-1))
                           ).reshape(batch, seq_len, -1)
        packed = pack_padded_sequence(
            enc, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hc_out = self.lstm(packed, hc)
        out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=seq_len)
        logits = self.head(out)
        return logits, hc_out

    def init_hidden(self, batch_size: int = 1, device: str = "cpu"):
        zeros = torch.zeros(
            self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        return (zeros, zeros.clone())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Each item is one replay: (obs_tensor (T, OBS_SIZE), act_tensor (T,)).
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
    CrossEntropyLoss(ignore_index=-100).  Obs padding is 0.0 — padded
    positions have all structure/unit counts at zero, but action 0
    (do_nothing) is always legal so softmax never sees all-(-inf) input.
    """
    obs_list, act_list = zip(*batch)
    lengths = torch.tensor([len(o) for o in obs_list], dtype=torch.long)
    obs_pad = pad_sequence(obs_list, batch_first=True, padding_value=0.0)
    act_pad = pad_sequence(act_list, batch_first=True, padding_value=-100)
    return obs_pad, act_pad, lengths


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
    weights = 1.0 / np.sqrt(counts)   # sqrt dampens extremes vs plain 1/n
    weights /= weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def _apply_mask_real_only(
    flat_logits: torch.Tensor,
    flat_obs:    torch.Tensor,
    flat_acts:   torch.Tensor,
) -> torch.Tensor:
    """
    Apply the legal mask only to real (non-padded) positions.

    Padded positions have flat_acts == -100.  Their logits are never used
    in the loss (ignore_index=-100) so masking them is unnecessary — and
    masking them with obs=0 (all-zero padding) can produce all-(-inf) rows
    if do_nothing somehow gets blocked, causing NaN in softmax.

    We clone the full tensor and only write -inf into real positions that
    are actually illegal, leaving padded rows completely untouched.
    """
    real_mask = flat_acts != -100                        # (B*T,) bool
    masked = flat_logits.clone()

    if real_mask.any():
        real_logits = flat_logits[real_mask]             # (N_real, A)
        real_obs = flat_obs[real_mask]                # (N_real, OBS_SIZE)
        real_logits = apply_legal_mask(real_logits, real_obs)
        masked[real_mask] = real_logits

    return masked


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for obs_pad, act_pad, lengths in loader:
        obs_pad = obs_pad.to(device)
        act_pad = act_pad.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits, _ = model(obs_pad, lengths)

        B, T, A = logits.shape
        flat_logits = logits.reshape(B * T, A)
        flat_obs = obs_pad.reshape(B * T, obs_pad.shape[-1])
        flat_acts = act_pad.reshape(B * T)

        flat_logits = _apply_mask_real_only(flat_logits, flat_obs, flat_acts)

        # Safety check — any remaining -inf on a real position means the label
        # contradicts the mask. Rather than clamping (which explodes the loss
        # via class weights * 1e9), silence those positions by setting their
        # label to -100 so CrossEntropyLoss ignores them, same as padding.
        real = flat_acts != -100
        if real.any():
            real_idx = real.nonzero(as_tuple=True)[0]
            label_logits = flat_logits[real_idx].gather(
                1, flat_acts[real_idx].unsqueeze(1))
            bad = ~label_logits[:, 0].isfinite()
            if bad.any():
                n_bad = bad.sum().item()
                print(f"  [WARN] {n_bad} label/mask conflicts remain in dataset "
                      f"— silencing those positions. Run conflict_diagnostic.py.")
                bad_idx = real_idx[bad]
                flat_acts = flat_acts.clone()
                flat_acts[bad_idx] = -100

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
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for obs_pad, act_pad, lengths in loader:
        obs_pad = obs_pad.to(device)
        act_pad = act_pad.to(device)
        lengths = lengths.to(device)

        logits, _ = model(obs_pad, lengths)

        B, T, A = logits.shape
        flat_logits = logits.reshape(B * T, A)
        flat_obs = obs_pad.reshape(B * T, obs_pad.shape[-1])
        flat_acts = act_pad.reshape(B * T)

        flat_logits = _apply_mask_real_only(flat_logits, flat_obs, flat_acts)

        # Same silence-on-conflict as train_epoch
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

    return total_loss / total, correct / total


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

    model = ProtossLSTMModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    class_weights = compute_class_weights(dataset, NUM_ACTIONS).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    if MODEL_SELECTION == "accuracy":
        best_val_metric = 0.0
        metric_name = "accuracy"
        def is_better(new, best): return new > best
    else:  # "loss"
        best_val_metric = float('inf')
        metric_name = "loss"
        def is_better(new, best): return new < best

    best_path = Path(CHECKPOINT_DIR) / "best_model.pt"

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>9} {'LR':>10}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>10.3%} "
              f"{val_loss:>10.4f} {val_acc:>9.3%} {lr:>10.2e}")

        # Select metric based on MODEL_SELECTION
        current_metric = val_acc if MODEL_SELECTION == "accuracy" else val_loss

        if is_better(current_metric, best_val_metric):
            best_val_metric = current_metric
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "obs_size":    OBS_SIZE,
                "num_actions": NUM_ACTIONS,
                "encoder_dim": ENCODER_DIM,
                "lstm_hidden": LSTM_HIDDEN,
                "lstm_layers": LSTM_LAYERS,
                "head_hidden": HEAD_HIDDEN,
            }, best_path)
            if MODEL_SELECTION == "accuracy":
                print(
                    f"         ↑ new best (acc={val_acc:.3%}) saved to {best_path}")
            else:
                print(
                    f"         ↑ new best (loss={val_loss:.4f}) saved to {best_path}")

    if MODEL_SELECTION == "accuracy":
        print(f"\nTraining complete. Best val accuracy: {best_val_metric:.3%}")
    else:
        print(f"\nTraining complete. Best val loss: {best_val_metric:.4f}")
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu") -> ProtossLSTMModel:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ProtossLSTMModel(
        obs_size=ckpt["obs_size"],
        num_actions=ckpt["num_actions"],
        encoder_dim=ckpt.get("encoder_dim", ENCODER_DIM),
        lstm_hidden=ckpt.get("lstm_hidden", LSTM_HIDDEN),
        lstm_layers=ckpt.get("lstm_layers", LSTM_LAYERS),
        head_hidden=ckpt.get("head_hidden", HEAD_HIDDEN),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device)


def predict_action(
    model:       ProtossLSTMModel,
    obs:         list[float],
    hc:          tuple | None = None,
    device:      str = "cpu",
    temperature: float = INFERENCE_TEMPERATURE,
) -> tuple[int, tuple]:
    """
    Single-step inference with legal masking + temperature sampling.

    The mask ensures only prerequisite-satisfied actions are candidates,
    matching the distribution the model was trained on.  Temperature
    sampling (default 0.8) avoids argmax probability collapse while
    staying close to the model's top preference.

    Args:
        model:       trained ProtossLSTMModel
        obs:         flat observation vector (length OBS_SIZE)
        hc:          LSTM hidden/cell state from previous step (None = zeros)
        device:      torch device string
        temperature: softmax temperature. Lower = sharper, higher = more random.

    Returns:
        (action_id, new_hc)
    """
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(
        0).unsqueeze(0).to(device)
    obs_2d = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    lengths = torch.tensor([1], dtype=torch.long)

    with torch.no_grad():
        logits, hc_out = model(x, lengths, hc=hc)

    # Apply legal mask — same logic as the training loop
    masked_logits = apply_legal_mask(logits[0, 0].unsqueeze(0), obs_2d)

    # Temperature sampling over the legal actions
    probs = torch.softmax(masked_logits[0] / temperature, dim=-1)
    action_id = int(torch.multinomial(probs, 1).item())

    return action_id, hc_out


if __name__ == "__main__":
    train()
