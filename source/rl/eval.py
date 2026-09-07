"""Fixed-seed, read-only evaluation for IL and PPO opening policies.

Run from the repository root with ``python source/rl/eval.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Support running this file directly while keeping the implementation inside
# the rl package. The rest of the project treats ``source`` as its import root.
SOURCE_DIR = Path(__file__).resolve().parents[1]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import torch

from episode import DIFFICULTIES, EpisodeConfig, run_episode
from rl.bot import PPOBot
from rl.ppo import ActorCritic
from rl.reward import OpeningRewardConfig, default_opening_reward


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_CHECKPOINTS = (
    PROJECT_ROOT / "checkpoints" / "best_model.pt",
    PROJECT_ROOT / "checkpoints" / "ppo_opening.pt",
    PROJECT_ROOT / "checkpoints" / "ppo_opening_best.pt",
)
DEFAULT_MODES = ("greedy", "sampled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare IL, latest PPO, and best PPO on identical opening seeds. "
            "This command never trains or writes checkpoints."
        )
    )
    parser.add_argument("--games", type=int, default=6,
                        help="Games per checkpoint and evaluation mode.")
    parser.add_argument("--seed", type=int, default=54,
                        help="First seed; every policy sees the same sequence.")
    parser.add_argument(
        "--difficulty", choices=DIFFICULTIES, default="easy"
    )
    parser.add_argument("--time-limit", type=int, default=200)
    parser.add_argument(
        "--modes", nargs="+", choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
        help="greedy uses argmax; sampled draws actions at --temperature.",
    )
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for sampled mode.")
    parser.add_argument(
        "--checkpoints", nargs="+", type=Path,
        default=list(DEFAULT_CHECKPOINTS),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--decision-log", action="store_true")
    parser.add_argument(
        "--no-opening-limits", action="store_true",
        help="Disable the exact build-count constraints used during PPO.",
    )
    args = parser.parse_args()

    if args.games < 1:
        parser.error("--games must be at least 1")
    if args.time_limit < 1:
        parser.error("--time-limit must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")
    missing = [str(path) for path in args.checkpoints if not path.is_file()]
    if missing:
        parser.error("checkpoint does not exist: " + ", ".join(missing))
    return args


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _checkpoint_label(path: Path) -> str:
    names = {
        "best_model": "il",
        "ppo_opening": "ppo_latest",
        "ppo_opening_best": "ppo_best",
    }
    return names.get(path.stem, path.stem)


def _load_actor(path: Path, device: str) -> tuple[ActorCritic, dict]:
    payload = torch.load(path, map_location=device)
    actor_critic = ActorCritic.from_il_checkpoint(str(path), device=device)
    if "actor_critic_state" in payload:
        actor_critic.load_state_dict(payload["actor_critic_state"])
    actor_critic.eval()
    info = {
        "path": str(path.resolve()),
        "format": payload.get("format", "il"),
        "update": payload.get("update"),
        "training_goal_rate": payload.get("batch_goal_rate"),
    }
    return actor_critic, info


def _target_time(episode: dict, milestone) -> float | None:
    field = (
        "opening_started_times"
        if milestone.required_phase == "started"
        else "opening_completion_times"
    )
    value = episode[field].get(milestone.name)
    return float(value) if value is not None else None


def summarize_episodes(
    episodes: list[dict], reward_config: OpeningRewardConfig
) -> dict:
    """Aggregate exactly the objective used by the PPO reward tracker."""
    targets = {}
    for milestone in reward_config.milestones:
        observed_times = [
            value for episode in episodes
            if (value := _target_time(episode, milestone)) is not None
        ]
        on_time = [value for value in observed_times
                   if value <= milestone.deadline]
        targets[milestone.name] = {
            "required_phase": milestone.required_phase,
            "deadline_seconds": milestone.deadline,
            "on_time": len(on_time),
            "on_time_rate": len(on_time) / len(episodes),
            "observed": len(observed_times),
            "median_seconds": (
                round(statistics.median(observed_times), 2)
                if observed_times else None
            ),
            "median_on_time_seconds": (
                round(statistics.median(on_time), 2) if on_time else None
            ),
        }

    rewards = [float(episode["episode_reward"]) for episode in episodes]
    goals = sum(bool(episode["reward_goal_met"]) for episode in episodes)
    return {
        "episodes": len(episodes),
        "goals_met": goals,
        "goal_rate": goals / len(episodes),
        "mean_reward": statistics.fmean(rewards),
        "median_reward": statistics.median(rewards),
        "targets": targets,
    }


def _format_episode_targets(episode: dict, reward_config) -> str:
    parts = []
    for milestone in reward_config.milestones:
        value = _target_time(episode, milestone)
        if value is None:
            shown = "missing"
        else:
            marker = "ok" if value <= milestone.deadline else "late"
            shown = f"{value:.0f}s/{marker}"
        parts.append(f"{milestone.name}={shown}")
    return " ".join(parts)


def _write_report(args, device: str, reward_config, policies, runs) -> Path:
    if args.output is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output = Path(args.log_dir) / f"rl_evaluation_{stamp}.json"
    else:
        output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "difficulty": args.difficulty,
            "games_per_run": args.games,
            "first_seed": args.seed,
            "seeds": list(range(args.seed, args.seed + args.games)),
            "time_limit_seconds": args.time_limit,
            "modes": args.modes,
            "sampled_temperature": args.temperature,
            "opening_limits": not args.no_opening_limits,
            "device": device,
            "reward": asdict(reward_config),
        },
        "policies": policies,
        "runs": runs,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    reward_config = default_opening_reward()
    policies = []
    runs = []
    total_games = len(args.checkpoints) * len(args.modes) * args.games
    print(
        f"Opening evaluation: {total_games} games | device={device} | "
        f"Zerg {args.difficulty} | horizon={args.time_limit}s"
    )

    for checkpoint_path in args.checkpoints:
        actor_critic, checkpoint_info = _load_actor(checkpoint_path, device)
        label = _checkpoint_label(checkpoint_path)
        checkpoint_info["label"] = label
        policies.append(checkpoint_info)

        for mode in args.modes:
            deterministic = mode == "greedy"
            episodes = []
            print(f"\n{label} | {mode} | {checkpoint_path}")
            for game_index in range(args.games):
                seed = args.seed + game_index
                config = EpisodeConfig(
                    seed=seed,
                    difficulty=DIFFICULTIES[args.difficulty],
                    time_limit=args.time_limit,
                    goal_deadline=float(args.time_limit),
                    checkpoint_path=str(checkpoint_path),
                    device=device,
                    temperature=args.temperature,
                    enable_decision_log=args.decision_log,
                    log_dir=args.log_dir,
                )

                def make_bot(_config: EpisodeConfig) -> PPOBot:
                    return PPOBot(
                        actor_critic,
                        reward_config,
                        device=device,
                        temperature=args.temperature,
                        enable_decision_log=args.decision_log,
                        log_dir=args.log_dir,
                        goal_deadline=float(args.time_limit),
                        deterministic=deterministic,
                        enforce_opening_limits=not args.no_opening_limits,
                    )

                episode = run_episode(config, bot_factory=make_bot).summary
                episode["seed"] = seed
                episodes.append(episode)
                print(
                    f"  seed={seed} reward={episode['episode_reward']:+.2f} "
                    f"goal={'yes' if episode['reward_goal_met'] else 'no'} | "
                    f"{_format_episode_targets(episode, reward_config)}"
                )

            aggregate = summarize_episodes(episodes, reward_config)
            runs.append({
                "policy": label,
                "checkpoint": str(checkpoint_path.resolve()),
                "mode": mode,
                "aggregate": aggregate,
                "episodes": episodes,
            })
            target_rates = " ".join(
                f"{name}={stats['on_time_rate']:.0%}"
                for name, stats in aggregate["targets"].items()
            )
            print(
                f"  RESULT goal={aggregate['goal_rate']:.0%} "
                f"mean_reward={aggregate['mean_reward']:+.3f} | "
                f"{target_rates}"
            )

    report_path = _write_report(
        args, device, reward_config, policies, runs
    )
    print(f"\nEvaluation complete. Report: {report_path}")


if __name__ == "__main__":
    main()
