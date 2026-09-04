from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId

import math

from observation_wrapper import ObservationWrapper
from obs_spec import DECISION_INTERVAL_SECONDS
from model import load_model, predict_action, MAX_CONTEXT
from decision_log import DecisionLogger
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

# Per-decision introspection log. Set to False to turn it off.
ENABLE_DECISION_LOG = True
LOG_DIR = r"C:\dev\BetaStar\logs"


class ProtossBot(BotAI):

    def __init__(
        self,
        checkpoint_path: str = CHECKPOINT_PATH,
        device: str = DEVICE,
        temperature: float | None = None,
        enable_decision_log: bool = ENABLE_DECISION_LOG,
        log_dir: str = LOG_DIR,
        goal_deadline: float | None = None,
    ):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.goal_deadline = goal_deadline
        self.obs_wrapper = ObservationWrapper()
        self.model = load_model(checkpoint_path, device=device)
        self.obs_history: list = []  # rolling window of observation vectors

        # Baseline/RL episode measurements. These are observed on every SC2
        # step (not just every 4-second policy decision), so milestone timing is
        # as precise as the game-step interval permits.
        self.milestone_times: dict[str, float] = {}
        self.final_game_time: float = 0.0
        self.final_game_result = None

        # Next game-time (seconds) at which to query the model. Scheduled on
        # game time rather than by counting on_step iterations: the old
        # `action_cooldown = 22` actually produced a 23-iteration period, which
        # at a 4-frame game step is 4.107s against the parser's 4s training
        # window -- 2.7% slow, accumulating over the game. Counting iterations
        # also silently breaks if the game step changes.
        self.next_decision_time: float = 0.0

        # Army state machine
        self.army_state = ArmyState.RALLY
        self.army_state_since: float = 0.0
        self.enemy_bases_cleared: set = set()

        # Threat tracking. Protoss structure health never regenerates, so
        # "is anything damaged" is a permanent condition and cannot be used to
        # decide whether to keep defending. Instead we watch for hp/shield to
        # DROP between steps, and for enemies near our structures.
        self.structure_hp_snapshot: dict = {}   # tag -> (hp+shield, position)
        self.last_damage_time: float = -1.0e9
        self.last_damage_pos = None
        self.threat_position = None

        # Timing
        self.last_army_command_time: float = 0.0
        self.last_rally_time: float = 0.0

        # Workers reserved for a build (tag -> game time the hold expires), so
        # auto_saturate_assimilators and friends cannot steal a probe that is
        # walking to a build site. See helpers.reserve_worker.
        self.reserved_workers: dict = {}

        # Production buildings that have had rally points set
        self.rally_tags_set: set = set()

        # High templar tag -> game time a merge was last commanded, so a
        # rejected merge cannot become a per-step retry loop.
        self.archon_merge_issued: dict = {}

        # Per-decision introspection (None when disabled)
        self.decision_log = (
            DecisionLogger(log_dir) if enable_decision_log else None)

    def _update_milestones(self):
        """Record the first observed completion time for the opening goal."""
        milestones = {
            "pylon": UnitTypeId.PYLON,
            "gateway": UnitTypeId.GATEWAY,
            "cybernetics_core": UnitTypeId.CYBERNETICSCORE,
        }
        for name, unit_type in milestones.items():
            if name not in self.milestone_times and self.structures(unit_type).ready:
                self.milestone_times[name] = float(self.time)

    def episode_summary(self, game_result=None) -> dict:
        """Return JSON-serializable measurements for baseline/RL tooling."""
        result = game_result if game_result is not None else self.final_game_result
        result_name = getattr(result, "name", str(result) if result is not None else None)
        deadline = self.goal_deadline
        required = ("pylon", "gateway", "cybernetics_core")
        goal_met = all(
            name in self.milestone_times
            and (deadline is None or self.milestone_times[name] <= deadline)
            for name in required
        )
        return {
            "result": result_name,
            "game_time_seconds": round(float(self.final_game_time), 2),
            "goal_deadline_seconds": deadline,
            "goal_met": goal_met,
            "milestone_times": {
                name: round(value, 2)
                for name, value in self.milestone_times.items()
            },
        }

    async def on_step(self, iteration: int):
        self.final_game_time = float(self.time)
        self._update_milestones()

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

        # Settle the previous decision's outcome now that the game has advanced.
        if self.decision_log is not None:
            self.decision_log.resolve_previous(self)

        obs = self.obs_wrapper.get_observation(self)
        self.obs_history.append(obs)

        # Cap context window to bound inference latency
        if len(self.obs_history) > MAX_CONTEXT:
            self.obs_history = self.obs_history[-MAX_CONTEXT:]

        if self.decision_log is not None:
            predict_kwargs = {
                "device": self.device,
                "return_diagnostics": True,
            }
            if self.temperature is not None:
                predict_kwargs["temperature"] = self.temperature
            action_id, diagnostics = predict_action(
                self.model, self.obs_history, **predict_kwargs)
            self.decision_log.log_decision(
                self, iteration, obs, action_id, diagnostics)
        else:
            predict_kwargs = {"device": self.device}
            if self.temperature is not None:
                predict_kwargs["temperature"] = self.temperature
            action_id = predict_action(
                self.model, self.obs_history, **predict_kwargs)

        print(
            f"[{self.time:.1f}s] step={iteration}  action={actions.ACTIONS[action_id]} ({action_id})")

        # The execution layer reports why it did or did not act, so a dropped
        # decision is visible in the log instead of showing up as a mystery no-op.
        result = await actions.execute_action(action_id, self)
        if self.decision_log is not None:
            self.decision_log.note_execution(result)

    async def on_end(self, game_result):
        self.final_game_result = game_result
        self.final_game_time = max(self.final_game_time, float(self.time))
        self._update_milestones()
        if self.decision_log is not None:
            self.decision_log.finish(
                self, game_result,
                episode_summary=self.episode_summary(game_result),
            )
