from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId

import math

from observation_wrapper import ObservationWrapper
from obs_spec import DECISION_INTERVAL_SECONDS
from model import load_model, predict_action, MAX_CONTEXT
from helpers import (
    ArmyState,
    auto_saturate_assimilators,
    set_production_rally_points,
    manage_army,
    auto_merge_archons,
    chrono_boost_production
)
import actions

CHECKPOINT_PATH = r"C:\dev\BetaStar\checkpoints\best_model.pt"
DEVICE = "cpu"


class ProtossBot(BotAI):

    def __init__(self):
        super().__init__()
        self.obs_wrapper = ObservationWrapper()
        self.model = load_model(CHECKPOINT_PATH, device=DEVICE)
        self.obs_history: list = []  # rolling window of observation vectors

        # Next game-time (seconds) at which to query the model. Scheduled on
        # game time rather than by counting on_step iterations: the old
        # `action_cooldown = 22` actually produced a 23-iteration period, which
        # at a 4-frame game step is 4.107s against the parser's 4s training
        # window -- 2.7% slow, accumulating over the game. Counting iterations
        # also silently breaks if the game step changes.
        self.next_decision_time: float = 0.0

        # Army state machine
        self.army_state = ArmyState.RALLY
        self.enemy_bases_cleared: set = set()

        # Timing
        self.last_army_command_time: float = 0.0
        self.last_rally_time: float = 0.0

        # Production buildings that have had rally points set
        self.rally_tags_set: set = set()

        # High templar tag -> game time a merge was last commanded, so a
        # rejected merge cannot become a per-step retry loop.
        self.archon_merge_issued: dict = {}

    async def on_step(self, iteration: int):
        # Always-on behaviours
        await self.distribute_workers()
        await auto_saturate_assimilators(self)
        await set_production_rally_points(self)
        await auto_merge_archons(self)
        await manage_army(self)
        await chrono_boost_production(self)

        # Query the model on the training grid (every DECISION_INTERVAL_SECONDS
        # of game time). on_step granularity is ~0.18s, so decisions land at the
        # first step at or after each grid boundary: jitter under one iteration
        # and, unlike a step counter, no accumulating drift.
        if self.time < self.next_decision_time:
            return

        self.next_decision_time += DECISION_INTERVAL_SECONDS
        if self.next_decision_time <= self.time:
            # Fell behind (e.g. a long stall). Resync to the next grid boundary
            # ahead of now instead of firing repeatedly to catch up.
            slots_elapsed = math.floor(self.time / DECISION_INTERVAL_SECONDS)
            self.next_decision_time = (
                slots_elapsed + 1) * DECISION_INTERVAL_SECONDS

        obs = self.obs_wrapper.get_observation(self)
        self.obs_history.append(obs)

        # Cap context window to bound inference latency
        if len(self.obs_history) > MAX_CONTEXT:
            self.obs_history = self.obs_history[-MAX_CONTEXT:]

        action_id = predict_action(
            self.model,
            self.obs_history,
            device=DEVICE,
        )

        print(
            f"[{self.time:.1f}s] step={iteration}  action={actions.ACTIONS[action_id]} ({action_id})")

        # Illegal actions fail silently (just continue to next step)
        await actions.execute_action(action_id, self)

    async def on_end(self, game_result):
        pass


run_game(
    maps.get("AbyssalReefLE"),
    [Bot(Race.Protoss, ProtossBot()), Computer(Race.Zerg, Difficulty.Easy)],
    realtime=False,
)
