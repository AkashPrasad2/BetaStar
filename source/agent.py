"""Live SC2 agent that executes decisions from a trained policy."""

from paths import BEST_IL_CHECKPOINT, DEFAULT_LOG_DIR
from sc2.bot_ai import BotAI

import logging
import math

from observation_wrapper import ObservationWrapper
from obs_spec import DECISION_INTERVAL_SECONDS
from model import load_model, predict_action, MAX_CONTEXT
from telemetry.decision_log import DecisionLogger
from helpers import (
    ArmyState,
    auto_saturate_assimilators,
    set_production_rally_points,
    manage_army,
    auto_merge_archons,
    chrono_boost_production
)
import actions


logger = logging.getLogger(__name__)


CHECKPOINT_PATH = str(BEST_IL_CHECKPOINT)
DEVICE = "cpu"

# Per-decision JSON tracing is opt-in. Normal runs have concise summaries.
ENABLE_DECISION_LOG = False
LOG_DIR = str(DEFAULT_LOG_DIR)


class ProtossAgent(BotAI):

    def __init__(
        self,
        checkpoint_path: str = CHECKPOINT_PATH,
        device: str = DEVICE,
        temperature: float | None = None,
        enable_decision_log: bool = ENABLE_DECISION_LOG,
        log_dir: str = LOG_DIR,
        policy=None,
    ):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.obs_wrapper = ObservationWrapper()
        self.model = (
            policy if policy is not None
            else load_model(checkpoint_path, device=device)
        )
        self.obs_history: list = []  # rolling window of observation vectors

        self.final_game_time: float = 0.0
        self.final_game_result = None

        # Next game-time (seconds) at which to query the model. Scheduled on
        # game time rather than by counting on_step iterations (it was off by a bit)
        self.next_decision_time: float = 0.0

        # troop management
        self.army_state = ArmyState.RALLY
        self.army_state_since: float = 0.0
        self.enemy_bases_cleared: set = set()

        self.structure_hp_snapshot: dict = {}   # tag -> (hp+shield, position)
        self.last_damage_time: float = -1.0e9
        self.last_damage_pos = None
        self.threat_position = None

        self.last_army_command_time: float = 0.0
        self.last_rally_time: float = 0.0

        # workers reserved for a build (mutex-like system)
        self.reserved_workers: dict = {}

        self.rally_tags_set: set = set()  # for production buildings

        # High templar tag -> game time a merge was last commanded to avoid spam
        self.archon_merge_issued: dict = {}

        # Per-decision introspection (None when disabled)
        self.decision_log = (
            DecisionLogger(log_dir) if enable_decision_log else None)

    def game_summary(self, game_result=None) -> dict:
        """Return basic JSON-serializable measurements for one game."""
        result = game_result if game_result is not None else self.final_game_result
        result_name = getattr(result, "name", str(
            result) if result is not None else None)
        return {
            "result": result_name,
            "game_time_seconds": round(float(self.final_game_time), 2),
        }

    def _select_policy_action(self):
        """Sample from the loaded policy."""
        predict_kwargs = {
            "device": self.device,
            "return_diagnostics": self.decision_log is not None,
        }
        if self.temperature is not None:
            predict_kwargs["temperature"] = self.temperature
        selected = predict_action(
            self.model, self.obs_history, **predict_kwargs)
        if self.decision_log is not None:
            return selected
        return selected, {}

    async def _execute_selected_action(self, action_id: int):
        return await actions.execute_action(action_id, self)

    async def on_step(self, iteration: int):
        self.final_game_time = float(self.time)

        # Always-on behaviours
        await self.distribute_workers()
        await auto_saturate_assimilators(self)
        await set_production_rally_points(self)
        await auto_merge_archons(self)
        await manage_army(self)
        await chrono_boost_production(self)

        # Query the model on the training grid (every DECISION_INTERVAL_SECONDS)
        if self.time < self.next_decision_time:
            return

        self.next_decision_time += DECISION_INTERVAL_SECONDS
        if self.next_decision_time <= self.time:
            # Fell behind (e.g. a long stall). Resync to the next grid boundary
            # ahead of now instead of firing repeatedly to catch up.
            slots_elapsed = math.floor(self.time / DECISION_INTERVAL_SECONDS)
            self.next_decision_time = (
                slots_elapsed + 1) * DECISION_INTERVAL_SECONDS

        # Settle the previous decision's outcome now that the game has advanced.
        if self.decision_log is not None:
            self.decision_log.resolve_previous(self)

        obs = self.obs_wrapper.get_observation(self)
        self.obs_history.append(obs)

        # Cap context window to bound inference latency
        if len(self.obs_history) > MAX_CONTEXT:
            self.obs_history = self.obs_history[-MAX_CONTEXT:]

        action_id, diagnostics = self._select_policy_action()

        if self.decision_log is not None:
            self.decision_log.log_decision(
                self, iteration, obs, action_id, diagnostics)

        logger.debug(
            "t=%.1fs step=%d action=%s (%d)",
            self.time, iteration, actions.ACTIONS[action_id], action_id,
        )

        # The execution layer reports why it did or did not act, so a dropped
        # decision is visible in the log instead of showing up as a mystery no-op.
        result = await self._execute_selected_action(action_id)
        if self.decision_log is not None:
            self.decision_log.note_execution(result)

    async def on_end(self, game_result):
        self.final_game_result = game_result
        self.final_game_time = max(self.final_game_time, float(self.time))
        if self.decision_log is not None:
            self.decision_log.finish(
                self, game_result,
                game_summary=self.game_summary(game_result),
            )
