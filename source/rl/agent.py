"""PPO training agent with opening rewards and rollout collection."""

from __future__ import annotations

import numpy as np
from sc2.ids.unit_typeid import UnitTypeId

from agent import ProtossAgent
from model import predict_action
from obs_spec import ACTION_ID, NUM_ACTIONS
from rl.ppo import ActorCritic, RolloutStep
from rl.reward import (
    OpeningRewardConfig,
    OpeningRewardTracker,
    snapshot_opening_state,
)


OPENING_STRUCTURE_LIMITS = {
    "PYLON": 1,
    "GATEWAY": 1,
    "ASSIMILATOR": 1,
    "NEXUS": 2,
    "CYBERNETICSCORE": 1,
}

_OPENING_BUILD_ACTIONS = {
    "PYLON": "build_pylon",
    "GATEWAY": "build_gateway",
    "ASSIMILATOR": "build_assimilator",
    "NEXUS": "build_nexus",
    "CYBERNETICSCORE": "build_cyberneticscore",
}

_OPENING_MILESTONES = {
    "pylon": (UnitTypeId.PYLON, 1),
    "gateway": (UnitTypeId.GATEWAY, 1),
    "assimilator": (UnitTypeId.ASSIMILATOR, 1),
    "nexus": (UnitTypeId.NEXUS, 2),
    "cybernetics_core": (UnitTypeId.CYBERNETICSCORE, 1),
}


class PPOAgent(ProtossAgent):
    """Play a trained PPO policy with its opening-specific constraints."""

    def __init__(self, *args, opening_limits_until: float | None = 200.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.opening_limits_until = opening_limits_until

    def _opening_action_mask(self) -> np.ndarray | None:
        """Block duplicate target structures during the opening curriculum."""
        if (self.opening_limits_until is None
                or self.time >= self.opening_limits_until):
            return None

        legal = np.ones(NUM_ACTIONS, dtype=np.bool_)
        for structure_name, target_count in OPENING_STRUCTURE_LIMITS.items():
            structure = getattr(UnitTypeId, structure_name)
            completed = self.structures(structure).ready.amount
            pending = self.already_pending(structure)
            if completed + pending >= target_count:
                action_name = _OPENING_BUILD_ACTIONS[structure_name]
                legal[ACTION_ID[action_name]] = False
        return legal

    def _select_policy_action(self):
        predict_kwargs = {
            "device": self.device,
            "return_diagnostics": self.decision_log is not None,
            "legal_mask": self._opening_action_mask(),
        }
        if self.temperature is not None:
            predict_kwargs["temperature"] = self.temperature
        selected = predict_action(
            self.model, self.obs_history, **predict_kwargs)
        if self.decision_log is not None:
            return selected
        return selected, {}


class PPOTrainingAgent(PPOAgent):
    """Play with an actor-critic while collecting PPO training data."""

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
        deterministic: bool = False,
        enforce_opening_limits: bool = True,
    ):
        super().__init__(
            device=device,
            temperature=temperature,
            enable_decision_log=enable_decision_log,
            log_dir=log_dir,
            policy=actor_critic.policy,
            opening_limits_until=(
                goal_deadline if enforce_opening_limits else None
            ),
        )
        self.goal_deadline = goal_deadline
        self.milestone_times: dict[str, float] = {}
        self.actor_critic = actor_critic
        self.actor_critic.eval()
        self.reward_tracker = OpeningRewardTracker(reward_config)
        self.deterministic = deterministic
        self.rollout: list[RolloutStep] = []
        self._last_action_result = None

    def _update_milestones(self) -> None:
        """Record the first completion time for each opening objective."""
        for name, (unit_type, target_count) in _OPENING_MILESTONES.items():
            if (name not in self.milestone_times
                    and self.structures(unit_type).ready.amount >= target_count):
                self.milestone_times[name] = float(self.time)

    async def on_step(self, iteration: int):
        self._update_milestones()
        await super().on_step(iteration)

    def _settle_previous_reward(self) -> None:
        snapshot = snapshot_opening_state(self)
        if not self.reward_tracker.initialized:
            self.reward_tracker.reset(snapshot)
            return
        if self.rollout:
            self.rollout[-1].reward += self.reward_tracker.observe(
                snapshot, self._last_action_result
            )
        self._last_action_result = None

    def _select_policy_action(self):
        self._settle_previous_reward()
        legal_mask = self._opening_action_mask()
        action, log_prob, value, diagnostics = self.actor_critic.sample_action(
            self.obs_history,
            device=self.device,
            temperature=self.temperature,
            legal_mask=legal_mask,
            deterministic=self.deterministic,
        )
        self.rollout.append(RolloutStep(
            obs_history=np.asarray(self.obs_history, dtype=np.float32).copy(),
            action=action,
            old_log_prob=log_prob,
            old_value=value,
            legal_mask=(legal_mask.copy() if legal_mask is not None else None),
        ))
        return action, diagnostics

    async def _execute_selected_action(self, action_id: int):
        result = await super()._execute_selected_action(action_id)
        self._last_action_result = result
        return result

    async def on_end(self, game_result):
        self._update_milestones()
        snapshot = snapshot_opening_state(self)
        reward = self.reward_tracker.observe(
            snapshot, self._last_action_result, terminal=True
        )
        if self.rollout:
            self.rollout[-1].reward += reward
            self.rollout[-1].done = True
        await super().on_end(game_result)

    def game_summary(self, game_result=None) -> dict:
        summary = super().game_summary(game_result)
        deadline = self.goal_deadline
        required = ("pylon", "gateway", "cybernetics_core")
        goal_met = all(
            name in self.milestone_times
            and (deadline is None or self.milestone_times[name] <= deadline)
            for name in required
        )
        summary.update({
            "goal_deadline_seconds": deadline,
            "goal_met": goal_met,
            "milestone_times": {
                name: round(value, 2)
                for name, value in self.milestone_times.items()
            },
            "total_reward": round(self.reward_tracker.total_reward, 4),
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
