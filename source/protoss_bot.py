from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId

from observation_wrapper import ObservationWrapper
from model import load_model, predict_action, MAX_CONTEXT
from helpers import auto_saturate_assimilators, set_production_rally_points, rally_idle_army, auto_attack, defend_structures
import actions

CHECKPOINT_PATH = r"C:\dev\BetaStar\checkpoints\best_model.pt"
DEVICE = "cpu"


class ProtossBot(BotAI):

    def __init__(self):
        super().__init__()
        self.obs_wrapper = ObservationWrapper()
        self.model = load_model(CHECKPOINT_PATH, device=DEVICE)
        self.action_cooldown = 0  # start at 0 so we act at step 0
        self.obs_history = []   # rolling window of observation vectors

    async def on_step(self, iteration: int):
        # Always-on behaviours
        await self.distribute_workers()
        await auto_saturate_assimilators(self)
        await set_production_rally_points(self)
        await defend_structures(self)  # Check defense first (higher priority)
        await rally_idle_army(self)
        await auto_attack(self)

        # Model cooldown (subtract 1 at each frame)
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
            return

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
            f"[{self.time:.0f}s] step={iteration}  action={actions.ACTIONS[action_id]} ({action_id})")

        # illegal actions will faily silently (just continue to next step)
        await actions.execute_action(action_id, self)
        self.action_cooldown = 22

    async def on_end(self, game_result):
        pass


run_game(
    maps.get("AbyssalReefLE"),
    [Bot(Race.Protoss, ProtossBot()), Computer(Race.Zerg, Difficulty.Easy)],
    realtime=False,
)
