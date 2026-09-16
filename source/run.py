"""Command-line evaluation runner for one or more BetaStar games."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from sc2.data import Race, Result

from episode import DIFFICULTIES, MAP_NAME, EpisodeConfig, run_episode
from gameplay.agent import DEVICE, LOG_DIR
from paths import LATEST_PPO_CHECKPOINT
from telemetry.console import configure_logging


DEFAULT_PPO_CHECKPOINT = str(LATEST_PPO_CHECKPOINT)
DEFAULT_PPO_TEMPERATURE = 1.0
DEFAULT_OPENING_LIMIT_SECONDS = 200.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a BetaStar policy against Zerg on Abyssal Reef.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument(
        "--difficulty", choices=DIFFICULTIES, default="easy")
    parser.add_argument(
        "--time-limit", type=int, default=None,
        help="Stop after this many in-game seconds (for example, 180).")
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--temperature", type=float,
                        default=DEFAULT_PPO_TEMPERATURE)
    parser.add_argument(
        "--checkpoint", default=DEFAULT_PPO_CHECKPOINT,
        help=("Policy checkpoint to evaluate (default: latest PPO opening "
              "checkpoint)."),
    )
    parser.add_argument(
        "--opening-limits-until", type=float,
        default=DEFAULT_OPENING_LIMIT_SECONDS,
        help=("Enforce PPO opening build limits through this game time; "
              "use 0 to disable them (default: 200)."),
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO", help="Console verbosity (default: INFO).",
    )
    parser.add_argument(
        "--decision-log", action=argparse.BooleanOptionalAction,
        default=False,
        help="Write a detailed JSONL trace for every policy decision.",
    )
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be at least 1")
    if args.time_limit is not None and args.time_limit < 1:
        parser.error("--time-limit must be at least 1 second")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")
    if args.opening_limits_until < 0:
        parser.error("--opening-limits-until cannot be negative")
    if not Path(args.checkpoint).is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def _median_milestone(episodes: list[dict], name: str) -> float | None:
    values = [
        episode["milestone_times"][name]
        for episode in episodes
        if name in episode["milestone_times"]
    ]
    return round(statistics.median(values), 2) if values else None


def _write_report(args: argparse.Namespace, episodes: list[dict]) -> Path:
    log_dir = Path(args.log_dir) / "evaluation"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"evaluation_{stamp}.json"

    wins = sum(e["result"] == Result.Victory.name for e in episodes)
    cutoffs = sum(e["cutoff_reached"] for e in episodes)
    goals = sum(e["goal_met"] for e in episodes)
    report = {
        "config": {
            "map": MAP_NAME,
            "opponent_race": Race.Zerg.name,
            "difficulty": args.difficulty,
            "games": args.games,
            "time_limit_seconds": args.time_limit,
            "seed": args.seed,
            "temperature": args.temperature,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "opening_limits_until": args.opening_limits_until or None,
        },
        "aggregate": {
            "wins": wins,
            "win_rate": wins / len(episodes),
            "cutoffs": cutoffs,
            "goals_met": goals,
            "goal_rate": goals / len(episodes),
            "median_milestone_seconds": {
                name: _median_milestone(episodes, name)
                for name in ("pylon", "gateway", "cybernetics_core")
            },
        },
        "episodes": episodes,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    logger = logging.getLogger("betastar.run")
    episodes = []

    for game_index in range(args.games):
        seed = args.seed + game_index
        config = EpisodeConfig(
            seed=seed,
            difficulty=DIFFICULTIES[args.difficulty],
            time_limit=args.time_limit,
            goal_deadline=args.time_limit,
            checkpoint_path=args.checkpoint,
            device=args.device,
            temperature=args.temperature,
            enable_decision_log=args.decision_log,
            log_dir=args.log_dir,
            opening_limits_until=args.opening_limits_until or None,
        )
        logger.info(
            "game %d/%d | Zerg %s | seed=%d | limit=%ss",
            game_index + 1, args.games, args.difficulty, seed,
            args.time_limit or "none",
        )
        episode = run_episode(config).summary
        episode["game_index"] = game_index
        episodes.append(episode)
        logger.info(
            "result=%s | goal_met=%s | milestones=%s",
            episode["result"], episode["goal_met"],
            episode["milestone_times"],
        )

    report_path = _write_report(args, episodes)
    goals = sum(e["goal_met"] for e in episodes)
    wins = sum(e["result"] == Result.Victory.name for e in episodes)
    logger.info(
        "evaluation complete | goals=%d/%d | wins=%d/%d | report=%s",
        goals, len(episodes), wins, len(episodes), report_path,
    )


if __name__ == "__main__":
    main()
