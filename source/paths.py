"""Project-relative paths shared by command-line entry points."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
REPLAY_DIR = PROJECT_ROOT / "replays"

BEST_IL_CHECKPOINT = CHECKPOINT_DIR / "best_model.pt"
LATEST_PPO_CHECKPOINT = CHECKPOINT_DIR / "ppo_opening.pt"
BEST_PPO_CHECKPOINT = CHECKPOINT_DIR / "ppo_opening_best.pt"
DEFAULT_DATASET = REPLAY_DIR / "parsed" / "dataset.npz"
DEFAULT_REPLAY_DIR = REPLAY_DIR / "raw"
