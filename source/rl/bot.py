# ProtossBot specialization that collects on-policy PPO rollouts.

from __future__ import annotations

import numpy as np
import torch
from sc2.ids.unit_typeid import UnitTypeId

from action_mask import build_legal_mask
from obs_spec import ACTION_ID
from protoss_bot import ProtossBot
from rl.ppo import ActorCritic, RolloutStep
from rl.reward import (
    OpeningRewardConfig,
    OpeningRewardTracker,
    snapshot_opening_state,
)


# This is an opening curriculum, not a full-game production limit.  Counts
# include completed structures, structures under construction, and worker
# build orders that have not broken ground yet.
OPENING_STRUCTURE_LIMITS = {
    "PYLON": 1,
    "GATEWAY": 1,
    "ASSIMILATOR": 1,
    "NEXUS": 2,  # starting Nexus plus the requested expansion
    "CYBERNETICSCORE": 1,
}

_OPENING_BUILD_ACTIONS = {
    "PYLON": "build_pylon",
    "GATEWAY": "build_gateway",
    "ASSIMILATOR": "build_assimilator",
    "NEXUS": "build_nexus",
    "CYBERNETICSCORE": "build_cyberneticscore",
}


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
        legal_mask = self._opening_legal_mask()
        action, log_prob, value, diagnostics = self.actor_critic.sample_action(
            self.obs_history,
            device=self.device,
            temperature=self.temperature,
            legal_mask=legal_mask,
        )
        self.rollout.append(RolloutStep(
            obs_history=np.asarray(self.obs_history, dtype=np.float32).copy(),
            action=action,
            old_log_prob=log_prob,
            old_value=value,
            legal_mask=legal_mask.copy(),
        ))
        return action, diagnostics

    def _opening_legal_mask(self) -> np.ndarray:
        """Apply exact opening build limits on top of the normal legal mask."""
        observation = torch.as_tensor(
            self.obs_history[-1], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        legal = build_legal_mask(observation)[0]

        for structure_name, target_count in OPENING_STRUCTURE_LIMITS.items():
            structure = getattr(UnitTypeId, structure_name)
            completed = self.structures(structure).ready.amount
            pending = self.already_pending(structure)
            if completed + pending >= target_count:
                action_name = _OPENING_BUILD_ACTIONS[structure_name]
                legal[ACTION_ID[action_name]] = False

        return legal.cpu().numpy()

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
            "opening_started_times": {
                name: round(value, 2)
                for name, value in self.reward_tracker.started_times.items()
            },
            "opening_completion_times": {
                name: round(value, 2)
                for name, value in self.reward_tracker.completion_times.items()
            },
            "reward_breakdown": {
                name: round(value, 4)
                for name, value in self.reward_tracker.breakdown.items()
            },
        })
        return summary
