"""
SC2 Protoss Imitation Learning — Transformer Model + Training Script
=====================================================================
Architecture:
    obs (71,) -> input proj (71->128) -> sinusoidal pos enc
    -> 4x causal TransformerEncoderLayer (d=128, heads=4, ff=256)
    -> LayerNorm -> Linear (128->35 logits)

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_PATH = r"C:\dev\BetaStar\replays\parsed\dataset.npz"
CHECKPOINT_DIR = r"C:\dev\BetaStar\checkpoints"

OBS_SIZE = 71   # 1 time + 4 min + 4 gas + 3 base + 15 structures + 11 units + 15 pending structs + 11 pending units + 4 idle + 3 upgrade levels
NUM_ACTIONS = 35   # action 0 = do_nothing, kept for index stability

# Transformer hyper-params
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
MAX_SEQ_LEN = 2048   # positional encoding capacity

# Training hyper-params
BATCH_SIZE = 16
EPOCHS = 100
LR = 3e-4
VAL_SPLIT = 0.15
SEED = 54

# "accuracy" = save model with best validation accuracy
# "loss" = save model with lowest validation loss
MODEL_SELECTION = "accuracy"

# keep the decisions diverse (not applied during training, only inference)
INFERENCE_TEMPERATURE = 3

# Cap context window at inference to bound latency
MAX_CONTEXT = 256


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
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

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

    model = ProtossTransformerModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    class_weights = compute_class_weights(dataset, NUM_ACTIONS).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    if MODEL_SELECTION == "accuracy":
        best_val_metric = 0.0
        def is_better(new, best): return new > best
    else:
        best_val_metric = float('inf')
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

        current_metric = val_acc if MODEL_SELECTION == "accuracy" else val_loss

        if is_better(current_metric, best_val_metric):
            best_val_metric = current_metric
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "val_loss":        val_loss,
                "val_acc":         val_acc,
                "obs_size":        OBS_SIZE,
                "num_actions":     NUM_ACTIONS,
                "d_model":         D_MODEL,
                "nhead":           NHEAD,
                "num_layers":      NUM_LAYERS,
                "dim_feedforward":  DIM_FEEDFORWARD,
                "max_seq_len":     MAX_SEQ_LEN,
            }, best_path)
            if MODEL_SELECTION == "accuracy":
                print(
                    f"         ^ new best (acc={val_acc:.3%}) saved to {best_path}")
            else:
                print(
                    f"         ^ new best (loss={val_loss:.4f}) saved to {best_path}")

    if MODEL_SELECTION == "accuracy":
        print(f"\nTraining complete. Best val accuracy: {best_val_metric:.3%}")
    else:
        print(f"\nTraining complete. Best val loss: {best_val_metric:.4f}")
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
) -> int:
    """
    Sequence inference with legal masking + temperature sampling.

    Args:
        model:       trained ProtossTransformerModel
        obs_history: list of flat observation vectors (oldest first)
        device:      torch device string
        temperature: softmax temperature

    Returns:
        action_id
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
    return int(torch.multinomial(probs, 1).item())


if __name__ == "__main__":
    train()
