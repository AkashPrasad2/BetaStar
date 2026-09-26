import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from agent import ProtossAgent  # noqa: E402
from sc2.data import Difficulty, Result  # noqa: E402
from sc2.ids.unit_typeid import UnitTypeId  # noqa: E402

from game_runner import play_game  # noqa: E402
from obs_spec import ACTION_ID  # noqa: E402
from rl.agent import PPOAgent, PPOTrainingAgent  # noqa: E402


class GameRunnerTests(unittest.TestCase):
    @patch("game_runner.Computer", return_value="opponent")
    @patch("game_runner.Bot", return_value="player")
    @patch("game_runner.maps.get", return_value="map")
    @patch("game_runner.run_sc2_game", return_value=Result.Tie)
    def test_play_game_returns_outcome_and_summary(
        self, run_sc2_game, get_map, make_bot, make_computer
    ):
        agent = Mock()
        agent.game_summary.return_value = {"result": "Tie"}

        game = play_game(
            agent,
            seed=123,
            difficulty=Difficulty.Medium,
            map_name="TestMap",
            time_limit=180,
        )

        self.assertEqual(game.outcome, Result.Tie)
        self.assertEqual(game.summary["seed"], 123)
        self.assertTrue(game.summary["cutoff_reached"])
        get_map.assert_called_once_with("TestMap")
        make_bot.assert_called_once()
        make_computer.assert_called_once()
        run_sc2_game.assert_called_once()
        agent.game_summary.assert_called_once_with(Result.Tie)


class AgentSeparationTests(unittest.TestCase):
    def test_ppo_agents_extend_the_base_agent(self):
        self.assertTrue(issubclass(PPOAgent, ProtossAgent))
        self.assertTrue(issubclass(PPOTrainingAgent, PPOAgent))

    def test_base_summary_contains_no_opening_or_reward_fields(self):
        agent = SimpleNamespace(
            final_game_result=Result.Victory,
            final_game_time=123.456,
        )

        summary = ProtossAgent.game_summary(agent)

        self.assertEqual(summary, {
            "result": "Victory",
            "game_time_seconds": 123.46,
        })

    def test_opening_limits_belong_to_ppo_agent(self):
        completed = {UnitTypeId.PYLON: 1}

        def structures(unit_type):
            return SimpleNamespace(
                ready=SimpleNamespace(amount=completed.get(unit_type, 0))
            )

        agent = SimpleNamespace(
            opening_limits_until=200.0,
            time=100.0,
            structures=structures,
            already_pending=lambda _unit_type: 0,
        )

        mask = PPOAgent._opening_action_mask(agent)

        self.assertFalse(mask[ACTION_ID["build_pylon"]])
        self.assertTrue(mask[ACTION_ID["build_gateway"]])


if __name__ == "__main__":
    unittest.main()
