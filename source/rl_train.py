"""Train BetaStar's IL policy with finite-horizon PPO rollouts."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from episode import DIFFICULTIES, EpisodeConfig, run_episode
from model import INFERENCE_TEMPERATURE
from protoss_bot import CHECKPOINT_PATH, LOG_DIR
from rl.bot import PPOBot
from rl.ppo import ActorCritic, PPOConfig, PPOTrainer, frozen_policy_copy
from rl.reward import OpeningRewardConfig, default_opening_reward


DEFAULT_OUTPUT = r"C:\dev\BetaStar\checkpoints\ppo_opening.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the IL checkpoint on a 180-second opening goal."
    )
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=180)
    parser.add_argument(
        "--difficulty", choices=DIFFICULTIES, default="easy"
    )
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--temperature", type=float,
                        default=INFERENCE_TEMPERATURE)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                        help="IL checkpoint used to initialize and anchor PPO.")
    parser.add_argument("--resume", default=None,
                        help="Resume an RL checkpoint created by this script.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, or another torch device.")
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--decision-log", action="store_true",
                        help="Write the verbose per-decision diagnostic log.")

    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--reference-kl-coef", type=float, default=0.02)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)

    parser.add_argument("--success-bonus", type=float, default=2.0)
    parser.add_argument("--failure-penalty", type=float, default=-1.0)
    parser.add_argument("--execution-failure-penalty", type=float,
                        default=-0.02)
    parser.add_argument("--pylon-start-reward", type=float, default=0.05)
    parser.add_argument("--pylon-complete-reward", type=float, default=0.15)
    parser.add_argument("--gateway-start-reward", type=float, default=0.10)
    parser.add_argument("--gateway-complete-reward", type=float, default=0.30)
    parser.add_argument("--cybercore-start-reward", type=float, default=0.20)
    parser.add_argument("--cybercore-complete-reward", type=float, default=0.60)
    parser.add_argument("--pylon-deadline", type=float, default=None)
    parser.add_argument("--gateway-deadline", type=float, default=None)
    parser.add_argument("--cybercore-deadline", type=float, default=None)
    args = parser.parse_args()

    positive_ints = {
        "--updates": args.updates,
        "--episodes-per-update": args.episodes_per_update,
        "--time-limit": args.time_limit,
        "--ppo-epochs": args.ppo_epochs,
        "--minibatch-size": args.minibatch_size,
    }
    for flag, value in positive_ints.items():
        if value < 1:
            parser.error(f"{flag} must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")
    if not 0 <= args.clip_ratio < 1:
        parser.error("--clip-ratio must be in [0, 1)")
    if not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        parser.error("--gamma and --gae-lambda must be in [0, 1]")
    for flag in ("pylon_deadline", "gateway_deadline", "cybercore_deadline"):
        value = getattr(args, flag)
        if value is not None and value <= 0:
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    return args


def _device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def _reward_config(args: argparse.Namespace) -> OpeningRewardConfig:
    base = default_opening_reward(float(args.time_limit))
    deadlines = {
        "pylon": args.pylon_deadline,
        "gateway": args.gateway_deadline,
        "cybernetics_core": args.cybercore_deadline,
    }
    rewards = {
        "pylon": (args.pylon_start_reward, args.pylon_complete_reward),
        "gateway": (
            args.gateway_start_reward, args.gateway_complete_reward
        ),
        "cybernetics_core": (
            args.cybercore_start_reward, args.cybercore_complete_reward
        ),
    }
    milestones = tuple(
        replace(
            milestone,
            deadline=(deadlines[milestone.name]
                      if deadlines[milestone.name] is not None
                      else milestone.deadline),
            started_reward=rewards[milestone.name][0],
            completed_reward=rewards[milestone.name][1],
        )
        for milestone in base.milestones
    )
    return replace(
        base,
        milestones=milestones,
        success_bonus=args.success_bonus,
        failure_penalty=args.failure_penalty,
        execution_failure_penalty=args.execution_failure_penalty,
    )


def _ppo_config(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coefficient=args.value_coef,
        entropy_coefficient=args.entropy_coef,
        reference_kl_coefficient=args.reference_kl_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        target_kl=args.target_kl,
        temperature=args.temperature,
    )


def _model_metadata(checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    keys = (
        "obs_size", "num_actions", "d_model", "nhead", "num_layers",
        "dim_feedforward", "max_seq_len",
    )
    return {key: checkpoint[key] for key in keys if key in checkpoint}


def _save_checkpoint(
    path: Path,
    actor_critic: ActorCritic,
    trainer: PPOTrainer,
    update: int,
    source_il_checkpoint: str,
    model_metadata: dict,
    ppo_config: PPOConfig,
    reward_config: OpeningRewardConfig,
    goal_rate: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "betastar_ppo_v1",
        "update": update,
        # Keeping model_state makes this checkpoint directly compatible with
        # source/run.py and model.load_model for ordinary evaluation.
        "model_state": actor_critic.policy.state_dict(),
        "actor_critic_state": actor_critic.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "source_il_checkpoint": str(Path(source_il_checkpoint).resolve()),
        "ppo_config": asdict(ppo_config),
        "reward_config": reward_config.to_dict(),
        "batch_goal_rate": goal_rate,
        **model_metadata,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    torch.manual_seed(args.seed)

    source_il_checkpoint = args.checkpoint
    resume_data = None
    if args.resume:
        resume_data = torch.load(args.resume, map_location=device)
        if resume_data.get("format") != "betastar_ppo_v1":
            raise ValueError("--resume is not a BetaStar PPO checkpoint")
        source_il_checkpoint = resume_data.get(
            "source_il_checkpoint", source_il_checkpoint
        )

    actor_critic = ActorCritic.from_il_checkpoint(
        source_il_checkpoint, device=device
    )
    reference_policy = frozen_policy_copy(actor_critic)
    ppo_config = _ppo_config(args)
    reward_config = _reward_config(args)
    trainer = PPOTrainer(
        actor_critic, reference_policy, ppo_config, device=device
    )
    first_update = 1
    if resume_data is not None:
        actor_critic.load_state_dict(resume_data["actor_critic_state"])
        trainer.optimizer.load_state_dict(resume_data["optimizer_state"])
        first_update = int(resume_data["update"]) + 1

    output_path = Path(args.output)
    best_path = output_path.with_name(
        f"{output_path.stem}_best{output_path.suffix}"
    )
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    training_log_path = log_dir / f"rl_training_{stamp}.jsonl"
    metadata = _model_metadata(source_il_checkpoint)
    best_goal_rate = -1.0

    print(f"Device: {device}")
    print(f"IL initialization: {source_il_checkpoint}")
    print(f"Opening horizon: {args.time_limit}s")
    print(f"PPO checkpoint: {output_path}")
    print(f"Training log: {training_log_path}")

    with training_log_path.open("w", encoding="utf-8") as training_log:
        training_log.write(json.dumps({
            "_meta": {
                "ppo_config": asdict(ppo_config),
                "reward_config": reward_config.to_dict(),
                "difficulty": args.difficulty,
                "episodes_per_update": args.episodes_per_update,
                "source_il_checkpoint": str(
                    Path(source_il_checkpoint).resolve()
                ),
            }
        }) + "\n")
        training_log.flush()

        final_update = first_update + args.updates - 1
        for update in range(first_update, final_update + 1):
            rollouts = []
            summaries = []
            print(f"\nPPO UPDATE {update}/{final_update}")
            for episode_index in range(args.episodes_per_update):
                seed = args.seed + (
                    (update - 1) * args.episodes_per_update + episode_index
                )
                config = EpisodeConfig(
                    seed=seed,
                    difficulty=DIFFICULTIES[args.difficulty],
                    time_limit=args.time_limit,
                    goal_deadline=float(args.time_limit),
                    checkpoint_path=source_il_checkpoint,
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
                    )

                result = run_episode(config, bot_factory=make_bot)
                rollouts.append(result.bot.rollout)
                summary = result.summary
                summary["seed"] = seed
                summaries.append(summary)
                print(
                    f"  episode {episode_index + 1}/{args.episodes_per_update} "
                    f"seed={seed} reward={summary['episode_reward']:+.3f} "
                    f"goal={'yes' if summary['reward_goal_met'] else 'no'} "
                    f"milestones={summary['milestone_times']}"
                )

            metrics = trainer.update(rollouts)
            goal_rate = sum(
                summary["reward_goal_met"] for summary in summaries
            ) / len(summaries)
            mean_reward = sum(
                summary["episode_reward"] for summary in summaries
            ) / len(summaries)
            record = {
                "update": update,
                "goal_rate": goal_rate,
                "mean_episode_reward": mean_reward,
                "metrics": metrics,
                "episodes": summaries,
            }
            training_log.write(json.dumps(record) + "\n")
            training_log.flush()

            _save_checkpoint(
                output_path, actor_critic, trainer, update,
                source_il_checkpoint, metadata, ppo_config, reward_config,
                goal_rate,
            )
            if goal_rate > best_goal_rate:
                best_goal_rate = goal_rate
                _save_checkpoint(
                    best_path, actor_critic, trainer, update,
                    source_il_checkpoint, metadata, ppo_config,
                    reward_config, goal_rate,
                )

            print(
                f"  goal_rate={goal_rate:.1%} "
                f"mean_reward={mean_reward:+.3f} "
                f"policy_loss={metrics['policy_loss']:+.4f} "
                f"value_loss={metrics['value_loss']:.4f} "
                f"KL(old)={metrics['approx_kl']:.5f} "
                f"KL(IL)={metrics['reference_kl']:.5f}"
            )
            print(f"  saved {output_path}")

    print(f"\nTraining complete. Best batch checkpoint: {best_path}")


if __name__ == "__main__":
    main()
