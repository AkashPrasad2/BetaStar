# ProtossBot specialization that collects on-policy PPO rollouts.

from __future__ import annotations

import numpy as np

from protoss_bot import ProtossBot
from rl.ppo import ActorCritic, RolloutStep
from rl.reward import (
    OpeningRewardConfig,
    OpeningRewardTracker,
    snapshot_opening_state,
)


class PPOBot(ProtossBot):
    """Use the shared game logic while recording every decision"""

    def __init__(
        self,
        actor_critic: ActorCritic,
        reward_config: OpeningRewardConfig,
        *,
        device: str,
        temperature: float,
        enable_decision_log: bool = False,
        log_dir: str,
        goal_deadline: float,
    ):
        super().__init__(
            device=device,
            temperature=temperature,
            enable_decision_log=enable_decision_log,
            log_dir=log_dir,
            goal_deadline=goal_deadline,
            policy_model=actor_critic.policy,
        )
        self.actor_critic = actor_critic
        self.actor_critic.eval()
        self.reward_tracker = OpeningRewardTracker(reward_config)
        self.rollout: list[RolloutStep] = []
        self._last_execution_result = None

    def _before_policy_decision(self, obs: list[float]) -> None:
        snapshot = snapshot_opening_state(self)
        if not self.reward_tracker.initialized:
            self.reward_tracker.reset(snapshot)
            return
        if self.rollout:
            self.rollout[-1].reward += self.reward_tracker.observe(
                snapshot, self._last_execution_result
            )
        self._last_execution_result = None

    def _select_policy_action(self):
        action, log_prob, value, diagnostics = self.actor_critic.sample_action(
            self.obs_history,
            device=self.device,
            temperature=self.temperature,
        )
        self.rollout.append(RolloutStep(
            obs_history=np.asarray(self.obs_history, dtype=np.float32).copy(),
            action=action,
            old_log_prob=log_prob,
            old_value=value,
        ))
        return action, diagnostics

    def _after_action_execution(self, action_id: int, result) -> None:
        self._last_execution_result = result

    def _on_policy_episode_end(self, game_result) -> None:
        snapshot = snapshot_opening_state(self)
        reward = self.reward_tracker.observe(
            snapshot, self._last_execution_result, terminal=True
        )
        if self.rollout:
            self.rollout[-1].reward += reward
            self.rollout[-1].done = True

    def episode_summary(self, game_result=None) -> dict:
        summary = super().episode_summary(game_result)
        summary.update({
            "episode_reward": round(self.reward_tracker.total_reward, 4),
            "ppo_decisions": len(self.rollout),
            "reward_goal_met": self.reward_tracker.goal_met,
            "reward_breakdown": {
                name: round(value, 4)
                for name, value in self.reward_tracker.breakdown.items()
            },
        })
        return summary
