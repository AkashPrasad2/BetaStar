"""Launch one BetaStar StarCraft II game."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from sc2 import maps
from sc2.data import Difficulty, Race, Result
from sc2.main import run_game as run_sc2_game
from sc2.player import Bot, Computer

from gameplay.agent import ProtossBot


DEFAULT_MAP = "AbyssalReefLE"

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
class CompletedGame:
    """The outcome and measurements from one finished game."""

    outcome: Result
    summary: dict


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def play_game(
    agent: ProtossBot,
    *,
    seed: int = 54,
    difficulty: Difficulty = Difficulty.Easy,
    map_name: str = DEFAULT_MAP,
    time_limit: int | None = None,
) -> CompletedGame:
    """Run one non-realtime game with ``agent`` against a Zerg computer."""
    _seed_everything(seed)
    outcome = run_sc2_game(
        maps.get(map_name),
        [Bot(Race.Protoss, agent), Computer(Race.Zerg, difficulty)],
        realtime=False,
        game_time_limit=time_limit,
        random_seed=seed,
    )
    if not isinstance(outcome, Result):
        raise TypeError(f"Expected one game result, got {outcome!r}")

    summary = agent.game_summary(outcome)
    summary.update({
        "seed": seed,
        "cutoff_reached": outcome == Result.Tie and time_limit is not None,
    })
    return CompletedGame(outcome=outcome, summary=summary)
