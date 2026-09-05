"""Actor-critic model, rollout records, GAE, and clipped PPO updates."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from action_mask import build_legal_mask
from model import ProtossTransformerModel, load_model
from obs_spec import ACTION_NAMES


DEFAULT_PPO_TEMPERATURE = 1.0


@dataclass
class RolloutStep:
    """One sampled policy decision and the outcome assigned to it later."""

    obs_history: np.ndarray
    action: int
    old_log_prob: float
    old_value: float
    reward: float = 0.0
    done: bool = False
    # The exact mask used when sampling.  Opening rollouts add live-game caps
    # (including worker-en-route builds that are absent from the observation),
    # so PPO must retain the mask to reproduce the old policy probability.
    legal_mask: np.ndarray | None = None


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 1.0e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_coefficient: float = 0.50
    entropy_coefficient: float = 0.01
    reference_kl_coefficient: float = 0.02
    max_grad_norm: float = 0.50
    update_epochs: int = 4
    minibatch_size: int = 64
    target_kl: float = 0.03
    temperature: float = DEFAULT_PPO_TEMPERATURE


class ActorCritic(nn.Module):
    """The IL policy plus a new scalar value head on shared hidden states."""

    def __init__(self, policy: ProtossTransformerModel):
        super().__init__()
        self.policy = policy
        self.value_head = nn.Sequential(
            nn.LayerNorm(policy.d_model),
            nn.Linear(policy.d_model, 1),
        )
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    @classmethod
    def from_il_checkpoint(cls, path: str, device: str) -> "ActorCritic":
        return cls(load_model(path, device=device)).to(device)

    def forward(self, observations: torch.Tensor):
        hidden = self.policy.encode(observations)
        return self.policy.output_head(hidden), self.value_head(hidden).squeeze(-1)

    @torch.no_grad()
    def sample_action(
        self,
        obs_history: list[list[float]],
        device: str,
        temperature: float,
        top_k: int = 5,
        legal_mask: np.ndarray | None = None,
    ) -> tuple[int, float, float, dict]:
        """Sample exactly once and retain the probability PPO must compare."""
        observations = torch.as_tensor(
            np.asarray(obs_history, dtype=np.float32), device=device
        ).unsqueeze(0)
        logits, values = self(observations)
        raw_logits = logits[:, -1, :]
        last_obs = observations[:, -1, :]
        legal = build_legal_mask(last_obs)
        if legal_mask is not None:
            rollout_mask = torch.as_tensor(
                legal_mask, dtype=torch.bool, device=device
            ).reshape(1, -1)
            if rollout_mask.shape != legal.shape:
                raise ValueError(
                    "legal_mask must contain exactly one value per action"
                )
            legal &= rollout_mask
        masked_logits = raw_logits.masked_fill(~legal, float("-inf"))
        distribution = Categorical(logits=masked_logits / temperature)
        action = distribution.sample()
        action_id = int(action.item())
        probabilities = distribution.probs[0]

        raw_top1 = int(raw_logits[0].argmax().item())
        masked_top1 = int(masked_logits[0].argmax().item())
        k = min(top_k, probabilities.numel())
        top = torch.topk(probabilities, k)
        diagnostics = {
            "n_legal": int(torch.isfinite(masked_logits[0]).sum().item()),
            "raw_top1": raw_top1,
            "raw_top1_name": ACTION_NAMES[raw_top1],
            "masked_top1": masked_top1,
            "masked_top1_name": ACTION_NAMES[masked_top1],
            "blocked_top1": raw_top1 != masked_top1,
            "chosen_prob": round(float(probabilities[action_id]), 4),
            "greedy_prob": round(float(probabilities.max()), 4),
            "top_named": [
                [ACTION_NAMES[int(i)], round(float(p), 4)]
                for p, i in zip(top.values.tolist(), top.indices.tolist())
            ],
        }
        return (
            action_id,
            float(distribution.log_prob(action).item()),
            float(values[0, -1].item()),
            diagnostics,
        )


def frozen_policy_copy(actor_critic: ActorCritic) -> ProtossTransformerModel:
    """Keep the original IL policy as a fixed anti-forgetting reference."""
    reference = copy.deepcopy(actor_critic.policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference


def compute_gae(
    rollout: list[RolloutStep], gamma: float, gae_lambda: float
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate delayed rewards backward through one episode."""
    advantages = np.zeros(len(rollout), dtype=np.float32)
    next_advantage = 0.0
    next_value = 0.0
    for index in range(len(rollout) - 1, -1, -1):
        step = rollout[index]
        not_done = 0.0 if step.done else 1.0
        delta = step.reward + gamma * next_value * not_done - step.old_value
        next_advantage = (
            delta + gamma * gae_lambda * not_done * next_advantage
        )
        advantages[index] = next_advantage
        next_value = step.old_value
    returns = advantages + np.asarray(
        [step.old_value for step in rollout], dtype=np.float32
    )
    return advantages, returns


def _pad_histories(samples: list[RolloutStep], device: str):
    lengths = torch.tensor(
        [len(step.obs_history) for step in samples],
        dtype=torch.long,
        device=device,
    )
    max_length = int(lengths.max().item())
    obs_size = samples[0].obs_history.shape[1]
    padded = torch.zeros(
        len(samples), max_length, obs_size, dtype=torch.float32, device=device
    )
    for row, step in enumerate(samples):
        history = torch.as_tensor(step.obs_history, device=device)
        padded[row, :len(history)] = history
    return padded, lengths


class PPOTrainer:
    """Perform clipped, on-policy updates over complete SC2 rollouts."""

    def __init__(
        self,
        actor_critic: ActorCritic,
        reference_policy: ProtossTransformerModel,
        config: PPOConfig,
        device: str,
    ):
        self.actor_critic = actor_critic
        self.reference_policy = reference_policy
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            actor_critic.parameters(), lr=config.learning_rate, weight_decay=1e-4
        )

    def _evaluate(self, samples: list[RolloutStep]):
        observations, lengths = _pad_histories(samples, self.device)
        rows = torch.arange(len(samples), device=self.device)
        positions = lengths - 1
        logits, all_values = self.actor_critic(observations)
        selected_logits = logits[rows, positions]
        values = all_values[rows, positions]
        last_observations = observations[rows, positions]
        legal = build_legal_mask(last_observations)
        for row, sample in enumerate(samples):
            if sample.legal_mask is not None:
                rollout_mask = torch.as_tensor(
                    sample.legal_mask, dtype=torch.bool, device=self.device
                )
                if rollout_mask.numel() != legal.shape[1]:
                    raise ValueError(
                        "stored legal_mask must contain one value per action"
                    )
                legal[row] &= rollout_mask.reshape(-1)
        masked_logits = selected_logits.masked_fill(
            ~legal, float("-inf")
        )
        distribution = Categorical(
            logits=masked_logits / self.config.temperature
        )

        with torch.no_grad():
            reference_logits = self.reference_policy(observations)[rows, positions]
            reference_masked = reference_logits.masked_fill(
                ~legal, float("-inf")
            ) / self.config.temperature

        current_log_all = torch.log_softmax(
            masked_logits / self.config.temperature, dim=-1
        )
        reference_log_all = torch.log_softmax(reference_masked, dim=-1)
        safe_current = torch.where(legal, current_log_all, 0.0)
        safe_reference = torch.where(legal, reference_log_all, 0.0)
        reference_kl = (
            current_log_all.exp() * (safe_current - safe_reference)
        ).sum(dim=-1)
        return distribution, values, reference_kl

    def update(self, rollouts: list[list[RolloutStep]]) -> dict[str, float]:
        samples: list[RolloutStep] = []
        advantages_parts = []
        returns_parts = []
        for rollout in rollouts:
            if not rollout:
                continue
            advantages, returns = compute_gae(
                rollout, self.config.gamma, self.config.gae_lambda
            )
            samples.extend(rollout)
            advantages_parts.append(advantages)
            returns_parts.append(returns)
        if not samples:
            raise ValueError("PPO update received no rollout decisions")

        advantages = torch.as_tensor(
            np.concatenate(advantages_parts), device=self.device
        )
        returns = torch.as_tensor(np.concatenate(returns_parts), device=self.device)
        old_log_probs = torch.tensor(
            [step.old_log_prob for step in samples], device=self.device
        )
        old_values = torch.tensor(
            [step.old_value for step in samples], device=self.device
        )
        actions = torch.tensor(
            [step.action for step in samples], dtype=torch.long, device=self.device
        )
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "reference_kl": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        examples_seen = 0
        epochs_completed = 0
        self.actor_critic.eval()  # gradients stay enabled; dropout stays off

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(len(samples)).tolist()
            epoch_kl_weighted = 0.0
            epoch_examples = 0
            for start in range(0, len(samples), self.config.minibatch_size):
                indices = permutation[start:start + self.config.minibatch_size]
                batch = [samples[index] for index in indices]
                index_tensor = torch.tensor(
                    indices, dtype=torch.long, device=self.device
                )
                distribution, values, reference_kl = self._evaluate(batch)
                new_log_probs = distribution.log_prob(actions[index_tensor])
                log_ratio = new_log_probs - old_log_probs[index_tensor]
                ratio = log_ratio.exp()

                batch_advantages = advantages[index_tensor]
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * batch_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()

                batch_old_values = old_values[index_tensor]
                value_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values,
                    -self.config.clip_ratio,
                    self.config.clip_ratio,
                )
                value_error = (values - returns[index_tensor]).square()
                clipped_value_error = (
                    value_clipped - returns[index_tensor]
                ).square()
                value_loss = 0.5 * torch.maximum(
                    value_error, clipped_value_error
                ).mean()
                entropy = distribution.entropy().mean()
                reference_kl_mean = reference_kl.mean()

                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                    + self.config.reference_kl_coefficient * reference_kl_mean
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.config.clip_ratio
                    ).float().mean()
                count = len(indices)
                values_to_add = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "reference_kl": reference_kl_mean,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                }
                for name, value in values_to_add.items():
                    totals[name] += float(value.item()) * count
                examples_seen += count
                epoch_kl_weighted += float(approx_kl.item()) * count
                epoch_examples += count

            epochs_completed += 1
            epoch_kl = epoch_kl_weighted / max(epoch_examples, 1)
            if self.config.target_kl > 0 and epoch_kl > self.config.target_kl:
                break

        metrics = {
            name: total / max(examples_seen, 1)
            for name, total in totals.items()
        }
        metrics.update({
            "decisions": float(len(samples)),
            "epochs_completed": float(epochs_completed),
            "mean_raw_advantage": float(
                np.concatenate(advantages_parts).mean()
            ),
            "mean_return": float(np.concatenate(returns_parts).mean()),
        })
        return metrics
