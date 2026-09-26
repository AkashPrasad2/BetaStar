# POC for a bot, getting famliar with the API.

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId


class ZergRushBot(BotAI):
    async def on_step(self, iteration: int):
        if (self.supply_used == self.supply_cap) and self.larva:
            if self.can_afford(UnitTypeId.OVERLORD) and self.already_pending(UnitTypeId.OVERLORD) == 0:
                self.larva.first.train(UnitTypeId.OVERLORD)
                return
        # 12 pool
        if self.can_afford(UnitTypeId.SPAWNINGPOOL) and self.already_pending(UnitTypeId.SPAWNINGPOOL) + self.structures.filter(lambda structure: structure.type_id == UnitTypeId.SPAWNINGPOOL and structure.is_ready).amount == 0:
            map_center = self.game_info.map_center
            position_towards_map_center = self.start_location.towards(
                map_center, distance=5)
            await self.build(UnitTypeId.SPAWNINGPOOL, near=position_towards_map_center, placement_step=1)

        # drone to 14
        if self.supply_used < 14 and self.can_afford(UnitTypeId.DRONE) and self.structures(UnitTypeId.SPAWNINGPOOL).exists and self.larva:
            self.larva.first.train(UnitTypeId.DRONE)
            return

        # overlord at 14
        if self.supply_used == 14 and self.can_afford(UnitTypeId.OVERLORD) and self.already_pending(UnitTypeId.OVERLORD) == 0 and self.units(UnitTypeId.OVERLORD).amount < 2 and self.larva:
            self.larva.first.train(UnitTypeId.OVERLORD)
            return

        if self.structures(UnitTypeId.SPAWNINGPOOL).exists:
            for loop_larva in self.larva:
                if self.can_afford(UnitTypeId.ZERGLING):
                    loop_larva.train(UnitTypeId.ZERGLING)
            for zergling in self.units(UnitTypeId.ZERGLING).idle:
                zergling.attack(self.enemy_start_locations[0])


run_game(
    maps.get("CyberForestLE"),
    [Bot(Race.Zerg, ZergRushBot()), Computer(Race.Zerg, Difficulty.VeryEasy)],
    realtime=False,
)
