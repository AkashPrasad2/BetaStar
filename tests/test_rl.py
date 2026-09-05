from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

from model import ProtossTransformerModel  # noqa: E402
from obs_spec import OBS_SIZE  # noqa: E402
from rl.ppo import (  # noqa: E402
    ActorCritic,
    PPOConfig,
    PPOTrainer,
    RolloutStep,
    compute_gae,
    frozen_policy_copy,
)
from rl.reward import (  # noqa: E402
    OpeningRewardTracker,
    OpeningSnapshot,
    default_opening_reward,
)


class RewardTests(unittest.TestCase):
    def test_rewards_state_transitions_once_and_terminal_success(self):
        tracker = OpeningRewardTracker(default_opening_reward(180.0))
        tracker.reset(OpeningSnapshot(0.0, frozenset(), frozenset()))

        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            24.0, frozenset({"pylon"}), frozenset()
        )), 0.05)
        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            48.0, frozenset({"pylon"}), frozenset({"pylon"})
        )), 0.15)
        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            52.0, frozenset({"pylon"}), frozenset({"pylon"})
        )), 0.0)
        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            100.0,
            frozenset({"pylon", "gateway", "cybernetics_core"}),
            frozenset({"pylon", "gateway", "cybernetics_core"}),
        ), terminal=True), 3.2)
        self.assertTrue(tracker.goal_met)
        self.assertAlmostEqual(tracker.total_reward, 3.4)

    def test_failed_execution_and_terminal_failure_are_penalized(self):
        tracker = OpeningRewardTracker(default_opening_reward(180.0))
        tracker.reset(OpeningSnapshot(0.0, frozenset(), frozenset()))
        reward = tracker.observe(
            OpeningSnapshot(180.0, frozenset(), frozenset()),
            "unaffordable",
            terminal=True,
        )
        self.assertAlmostEqual(reward, -1.02)
        self.assertFalse(tracker.goal_met)


class PPOTests(unittest.TestCase):
    def _model(self):
        return ProtossTransformerModel(
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            max_seq_len=16,
        )

    def test_actor_critic_preserves_policy_logits(self):
        policy = self._model().eval()
        actor_critic = ActorCritic(copy.deepcopy(policy)).eval()
        observations = torch.zeros(2, 3, OBS_SIZE)
        expected = policy(observations)
        actual, values = actor_critic(observations)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(values, torch.zeros_like(values))

    def test_gae_propagates_terminal_reward_backwards(self):
        observations = np.zeros((1, OBS_SIZE), dtype=np.float32)
        rollout = [
            RolloutStep(observations, 0, 0.0, 0.0),
            RolloutStep(observations, 0, 0.0, 0.0, reward=1.0, done=True),
        ]
        advantages, returns = compute_gae(rollout, 0.99, 0.95)
        self.assertAlmostEqual(float(advantages[1]), 1.0)
        self.assertAlmostEqual(float(advantages[0]), 0.99 * 0.95, places=5)
        np.testing.assert_allclose(advantages, returns)

    def test_ppo_update_accepts_variable_length_histories(self):
        actor_critic = ActorCritic(self._model()).eval()
        reference = frozen_policy_copy(actor_critic)
        config = PPOConfig(
            update_epochs=1,
            minibatch_size=2,
            temperature=1.0,
        )
        trainer = PPOTrainer(actor_critic, reference, config, device="cpu")

        rollout = []
        history = []
        for index in range(3):
            history.append(np.zeros(OBS_SIZE, dtype=np.float32))
            action, log_prob, value, _ = actor_critic.sample_action(
                history, "cpu", temperature=1.0
            )
            rollout.append(RolloutStep(
                np.asarray(history, dtype=np.float32).copy(),
                action,
                log_prob,
                value,
                reward=1.0 if index == 2 else 0.0,
                done=index == 2,
            ))
        metrics = trainer.update([rollout])
        self.assertEqual(metrics["decisions"], 3.0)
        for value in metrics.values():
            self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()

