from __future__ import annotations

import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

from model import ProtossTransformerModel  # noqa: E402
from obs_spec import ACTION_ID, NUM_ACTIONS, OBS_SIZE  # noqa: E402
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
    timing_multiplier,
)
from rl_eval import summarize_episodes  # noqa: E402


class RewardTests(unittest.TestCase):
    def test_rewards_state_transitions_once_and_terminal_success(self):
        tracker = OpeningRewardTracker(default_opening_reward())
        tracker.reset(OpeningSnapshot(0.0, frozenset(), frozenset()))

        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            28.0, frozenset({"pylon"}), frozenset()
        )), 0.05)
        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            44.0, frozenset({"pylon"}), frozenset({"pylon"})
        )), 0.15)
        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            64.0,
            frozenset({"pylon", "gateway"}),
            frozenset({"pylon"}),
        )), 0.10)
        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            72.0,
            frozenset({"pylon", "gateway", "assimilator"}),
            frozenset({"pylon"}),
        )), 0.10)
        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            96.0,
            frozenset({"pylon", "gateway", "assimilator"}),
            frozenset({"pylon", "assimilator"}),
            completion_times={"pylon": 44.0, "assimilator": 96.0},
        )), 0.30)
        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            110.0,
            frozenset({"pylon", "gateway", "assimilator"}),
            frozenset({"pylon", "gateway", "assimilator"}),
            completion_times={
                "pylon": 44.0, "gateway": 110.0,
                "assimilator": 96.0,
            },
        )), 0.30)
        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            128.0,
            frozenset({"pylon", "gateway", "assimilator", "nexus"}),
            frozenset({"pylon", "gateway", "assimilator"}),
        )), 0.40)
        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            144.0,
            frozenset({
                "pylon", "gateway", "assimilator", "nexus",
                "cybernetics_core",
            }),
            frozenset({"pylon", "gateway", "assimilator"}),
        )), 0.20)
        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            180.0,
            frozenset({
                "pylon", "gateway", "assimilator", "nexus",
                "cybernetics_core",
            }),
            frozenset({
                "pylon", "gateway", "assimilator", "cybernetics_core",
            }),
            completion_times={
                "pylon": 44.0, "gateway": 110.0,
                "assimilator": 96.0, "cybernetics_core": 180.0,
            },
        )), 0.60)
        self.assertAlmostEqual(tracker.observe(OpeningSnapshot(
            200.0,
            frozenset({
                "pylon", "gateway", "assimilator", "nexus",
                "cybernetics_core",
            }),
            frozenset({
                "pylon", "gateway", "assimilator", "cybernetics_core",
            }),
        ), terminal=True), 3.0)
        self.assertTrue(tracker.goal_met)
        self.assertAlmostEqual(tracker.total_reward, 5.2)

    def test_late_cybercore_misses_terminal_timing_goal(self):
        tracker = OpeningRewardTracker(default_opening_reward())
        tracker.reset(OpeningSnapshot(0.0, frozenset(), frozenset()))
        all_started = frozenset({
            "pylon", "gateway", "assimilator", "nexus",
            "cybernetics_core",
        })
        ready_without_nexus = frozenset({
            "pylon", "gateway", "assimilator", "cybernetics_core",
        })
        tracker.observe(OpeningSnapshot(
            136.0,
            all_started,
            frozenset({"pylon", "gateway", "assimilator"}),
            completion_times={
                "pylon": 50.0,
                "gateway": 122.0,
                "assimilator": 96.0,
            },
        ))
        tracker.observe(OpeningSnapshot(
            204.0,
            all_started,
            ready_without_nexus,
            completion_times={
                "pylon": 50.0,
                "gateway": 122.0,
                "assimilator": 96.0,
                "cybernetics_core": 201.0,
            },
        ), terminal=True)
        self.assertFalse(tracker.goal_met)
        self.assertAlmostEqual(
            tracker.breakdown["terminal:missed:cybernetics_core"], -0.4
        )

    def test_failed_execution_and_terminal_failure_are_penalized(self):
        config = replace(
            default_opening_reward(), execution_failure_penalty=-0.02
        )
        tracker = OpeningRewardTracker(config)
        tracker.reset(OpeningSnapshot(0.0, frozenset(), frozenset()))
        reward = tracker.observe(
            OpeningSnapshot(180.0, frozenset(), frozenset()),
            "unaffordable",
            terminal=True,
        )
        self.assertAlmostEqual(reward, -2.02)
        self.assertFalse(tracker.goal_met)

    def test_timing_reward_decays_linearly_after_target(self):
        self.assertEqual(timing_multiplier(100.0, 110.0, 132.0), 1.0)
        self.assertAlmostEqual(
            timing_multiplier(121.0, 110.0, 132.0), 0.5
        )
        self.assertEqual(timing_multiplier(132.0, 110.0, 132.0), 0.0)
        self.assertEqual(timing_multiplier(150.0, 110.0, 132.0), 0.0)

    def test_idle_penalties_repeat_while_the_opportunity_remains(self):
        tracker = OpeningRewardTracker(default_opening_reward())
        tracker.reset(OpeningSnapshot(0.0, frozenset(), frozenset()))
        snapshot = OpeningSnapshot(
            4.0,
            frozenset(),
            frozenset(),
            idle_unsaturated_nexuses=1,
            affordable_idle_production=2,
        )

        self.assertAlmostEqual(tracker.observe(snapshot), -0.05)
        self.assertAlmostEqual(tracker.observe(snapshot), -0.05)
        self.assertAlmostEqual(tracker.total_reward, -0.10)


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

    def test_reset_value_head_for_new_reward_scale(self):
        actor_critic = ActorCritic(self._model()).eval()
        with torch.no_grad():
            actor_critic.value_head[0].weight.fill_(2.0)
            actor_critic.value_head[0].bias.fill_(3.0)
            actor_critic.value_head[-1].weight.fill_(4.0)
            actor_critic.value_head[-1].bias.fill_(5.0)

        actor_critic.reset_value_head()

        torch.testing.assert_close(
            actor_critic.value_head[0].weight,
            torch.ones_like(actor_critic.value_head[0].weight),
        )
        torch.testing.assert_close(
            actor_critic.value_head[0].bias,
            torch.zeros_like(actor_critic.value_head[0].bias),
        )
        torch.testing.assert_close(
            actor_critic.value_head[-1].weight,
            torch.zeros_like(actor_critic.value_head[-1].weight),
        )
        torch.testing.assert_close(
            actor_critic.value_head[-1].bias,
            torch.zeros_like(actor_critic.value_head[-1].bias),
        )

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

    def test_actor_critic_respects_and_records_opening_mask_probability(self):
        actor_critic = ActorCritic(self._model()).eval()
        observation = np.zeros(OBS_SIZE, dtype=np.float32)
        legal_mask = np.ones(NUM_ACTIONS, dtype=np.bool_)
        legal_mask[ACTION_ID["build_pylon"]] = False

        action, log_prob, _, diagnostics = actor_critic.sample_action(
            [observation], "cpu", temperature=1.0, legal_mask=legal_mask
        )

        self.assertNotEqual(action, ACTION_ID["build_pylon"])
        self.assertTrue(np.isfinite(log_prob))
        self.assertEqual(diagnostics["n_legal"], 2)

    def test_deterministic_sampling_selects_masked_top_action(self):
        actor_critic = ActorCritic(self._model()).eval()
        observation = np.zeros(OBS_SIZE, dtype=np.float32)

        action, _, _, diagnostics = actor_critic.sample_action(
            [observation], "cpu", temperature=1.0, deterministic=True
        )

        self.assertEqual(action, diagnostics["masked_top1"])

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


class EvaluatorTests(unittest.TestCase):
    def test_summary_uses_required_phase_and_individual_deadlines(self):
        episode = {
            "episode_reward": 0.6,
            "reward_goal_met": False,
            "opening_started_times": {
                "pylon": 24.0,
                "gateway": 64.0,
                "assimilator": 120.0,
                "nexus": 124.0,
                "cybernetics_core": 140.0,
            },
            "opening_completion_times": {
                "pylon": 41.0,
                "gateway": 109.0,
                "assimilator": 139.0,
                "cybernetics_core": 175.0,
            },
        }

        summary = summarize_episodes(
            [episode], default_opening_reward()
        )

        self.assertEqual(summary["targets"]["nexus"]["on_time"], 1)
        self.assertEqual(summary["targets"]["gateway"]["on_time"], 1)
        self.assertEqual(summary["targets"]["assimilator"]["on_time"], 0)
        self.assertEqual(summary["goal_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
