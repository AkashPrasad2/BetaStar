import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from sc2.data import Difficulty, Result  # noqa: E402

from game_runner import play_game  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
