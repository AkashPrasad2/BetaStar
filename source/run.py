"""Command-line evaluation runner for one or more BetaStar games."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from sc2.data import Race, Result

from episode import DIFFICULTIES, MAP_NAME, EpisodeConfig, run_episode
from model import INFERENCE_TEMPERATURE
from protoss_bot import CHECKPOINT_PATH, DEVICE, LOG_DIR


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
                        default=INFERENCE_TEMPERATURE)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--no-decision-log", action="store_true")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be at least 1")
    if args.time_limit is not None and args.time_limit < 1:
        parser.error("--time-limit must be at least 1 second")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")
    return args


def _median_milestone(episodes: list[dict], name: str) -> float | None:
    values = [
        episode["milestone_times"][name]
        for episode in episodes
        if name in episode["milestone_times"]
    ]
    return round(statistics.median(values), 2) if values else None


def _write_report(args: argparse.Namespace, episodes: list[dict]) -> Path:
    log_dir = Path(args.log_dir)
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
            enable_decision_log=not args.no_decision_log,
            log_dir=args.log_dir,
        )
        print(
            f"\nEVALUATION GAME {game_index + 1}/{args.games} | "
            f"Zerg {args.difficulty} | seed={seed} | "
            f"limit={args.time_limit or 'none'}s")
        episode = run_episode(config).summary
        episode["game_index"] = game_index
        episodes.append(episode)
        print(
            f"Result={episode['result']} | goal_met={episode['goal_met']} | "
            f"milestones={episode['milestone_times']}")

    report_path = _write_report(args, episodes)
    goals = sum(e["goal_met"] for e in episodes)
    wins = sum(e["result"] == Result.Victory.name for e in episodes)
    print(f"\nEvaluation complete: goals={goals}/{len(episodes)}, "
          f"wins={wins}/{len(episodes)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
