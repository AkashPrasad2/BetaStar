from sc2.bot_ai import BotAI  # for our bot
from sc2.player import Bot, Computer  # for opponent
from sc2.ids.unit_typeid import UnitTypeId

PROTOSS_STRUCTURES = [
    UnitTypeId.NEXUS,
    UnitTypeId.PYLON,
    UnitTypeId.GATEWAY,
    UnitTypeId.WARPGATE,
    UnitTypeId.FORGE,
    UnitTypeId.TWILIGHTCOUNCIL,
    UnitTypeId.PHOTONCANNON,
    UnitTypeId.SHIELDBATTERY,
    UnitTypeId.TEMPLARARCHIVE,
    UnitTypeId.ROBOTICSBAY,
    UnitTypeId.ROBOTICSFACILITY,
    UnitTypeId.ASSIMILATOR,
    UnitTypeId.CYBERNETICSCORE,
    UnitTypeId.STARGATE,
    UnitTypeId.FLEETBEACON,
]

PROTOSS_UNITS = [
    UnitTypeId.PROBE,
    UnitTypeId.ZEALOT,
    UnitTypeId.STALKER,
    UnitTypeId.HIGHTEMPLAR,
    UnitTypeId.ARCHON,
    UnitTypeId.IMMORTAL,
    UnitTypeId.CARRIER,
    UnitTypeId.VOIDRAY,
]


class ObservationWrapper:
    """
    Converts game into a vector for neural network input, storing resources, supply, etc.
    """

    def __init__(self):
        self.observation_size = self.calculate_obs_size()

    def calculate_obs_size(self):
        return len(PROTOSS_UNITS) + len(PROTOSS_STRUCTURES) + 8

    def get_observation(self, bot, opponent):
        obs = []

        # Game time (normalized to 12 min average game)
        obs.append(bot.time / 720.0)

        # Player resources (normalized)
        obs.append(bot.minerals / 1800.0)
        obs.append(bot.vespene / 700.0)
        obs.append(bot.supply_used / 200.0)
        obs.append(bot.supply_cap / 200.0)
        obs.append(bot.supply_left / 200.0)

        # Worker saturation (for macro)
        worker_supply = bot.units(UnitTypeId.PROBE).amount
        ideal_workers = bot.townhalls.amount * 22  # 16 for minerals, 6 for vespene
        obs.append(worker_supply / max(ideal_workers, 1))  # Avoid div by 0

        # Player structures
        for structure in PROTOSS_STRUCTURES:
            obs.append(bot.structures(structure).amount / 10.0)

        # Player units
        for unit in PROTOSS_UNITS:
            obs.append(bot.units(unit).amount / 30.0)

        # Opponent info (normalized)
        obs.append(opponent.supply_used / 200.0)

        return obs
