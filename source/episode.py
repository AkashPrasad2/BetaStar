"""Shared lifecycle for running one BetaStar StarCraft II episode.

Evaluation and reinforcement-learning collectors both need the same mechanics:
seed all randomness, construct the bot and opponent, launch SC2, enforce an
optional time limit, and return the resulting bot state and summary. Keeping
that lifecycle here prevents the future RL trainer from duplicating it.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from sc2 import maps
from sc2.data import Difficulty, Race, Result
from sc2.main import run_game
from sc2.player import Bot, Computer

from protoss_bot import CHECKPOINT_PATH, DEVICE, LOG_DIR, ProtossBot


MAP_NAME = "AbyssalReefLE"

DIFFICULTIES = {
    "very_easy": Difficulty.VeryEasy,
    "easy": Difficulty.Easy,
    "medium": Difficulty.Medium,
    "medium_hard": Difficulty.MediumHard,
    "hard": Difficulty.Hard,
    "harder": Difficulty.Harder,
    "very_hard": Difficulty.VeryHard,
    "cheat_vision": Difficulty.CheatVision,
    "cheat_money": Difficulty.CheatMoney,
    "cheat_insane": Difficulty.CheatInsane,
}


@dataclass(frozen=True)
class EpisodeConfig:
    """Everything needed to reproduce one game."""

    seed: int = 54
    difficulty: Difficulty = Difficulty.Easy
    map_name: str = MAP_NAME
    time_limit: int | None = None
    goal_deadline: float | None = None
    checkpoint_path: str = CHECKPOINT_PATH
    device: str = DEVICE
    temperature: float | None = None
    enable_decision_log: bool = True
    log_dir: str = LOG_DIR


@dataclass
class EpisodeResult:
    """Result plus the bot instance, whose rollout will later feed PPO."""

    game_result: Result
    summary: dict
    bot: ProtossBot


BotFactory = Callable[[EpisodeConfig], ProtossBot]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _default_bot(config: EpisodeConfig) -> ProtossBot:
    return ProtossBot(
        checkpoint_path=config.checkpoint_path,
        device=config.device,
        temperature=config.temperature,
        enable_decision_log=config.enable_decision_log,
        log_dir=config.log_dir,
        goal_deadline=config.goal_deadline,
    )


def run_episode(
    config: EpisodeConfig,
    bot_factory: BotFactory | None = None,
) -> EpisodeResult:
    """Run one non-realtime Protoss-versus-Zerg game."""
    _seed_everything(config.seed)
    bot_ai = (bot_factory or _default_bot)(config)
    result = run_game(
        maps.get(config.map_name),
        [Bot(Race.Protoss, bot_ai), Computer(Race.Zerg, config.difficulty)],
        realtime=False,
        game_time_limit=config.time_limit,
        random_seed=config.seed,
    )
    if not isinstance(result, Result):
        raise TypeError(f"Expected one game result, got {result!r}")

    summary = bot_ai.episode_summary(result)
    summary.update({
        "seed": config.seed,
        "cutoff_reached": (
            result == Result.Tie and config.time_limit is not None),
    })
    return EpisodeResult(game_result=result, summary=summary, bot=bot_ai)
