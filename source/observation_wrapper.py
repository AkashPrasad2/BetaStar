from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId

# 15 structures
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

# 8 units
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
    Converts game state into a flat vector for neural network input.

    Feature layout (56 total):
        [0]     game time (normalized)
        [1]     minerals
        [2]     vespene
        [3]     supply_used
        [4]     supply_cap
        [5]     worker saturation
        [6:21]  completed structure counts   (15)
        [21:29] completed unit counts        (8)
        [29:44] in-progress structure counts (15)
        [44:52] in-progress unit counts      (8)
        [52]    idle gateway+warpgate count  (normalised /5)
        [53]    idle stargate count          (normalised /5)
        [54]    idle robotics facility count (normalised /5)
        [55]    idle warpgate count          (normalised /5)
    """

    def __init__(self):
        self.observation_size = self.calculate_obs_size()

    def calculate_obs_size(self):
        # 6 base + 15 structs + 8 units + 15 structs_pending + 8 units_pending
        # + 4 idle production buildings
        return (6
                + len(PROTOSS_STRUCTURES)
                + len(PROTOSS_UNITS)
                + len(PROTOSS_STRUCTURES)
                + len(PROTOSS_UNITS)
                + 4)

    def get_observation(self, bot: BotAI, opponent=None):
        obs = []

        # Base features
        obs.append(bot.time / 720.0)
        obs.append(bot.minerals / 1800.0)
        obs.append(bot.vespene / 700.0)
        obs.append(bot.supply_used / 200.0)
        obs.append(bot.supply_cap / 200.0)

        worker_supply = bot.units(UnitTypeId.PROBE).amount
        ideal_workers = bot.townhalls.amount * 22
        obs.append(worker_supply / max(ideal_workers, 1))

        # Completed structures
        for structure in PROTOSS_STRUCTURES:
            obs.append(bot.structures(structure).ready.amount / 10.0)

        # Completed units
        for unit in PROTOSS_UNITS:
            obs.append(bot.units(unit).amount / 30.0)

        # In-progress structures (under construction)
        for structure in PROTOSS_STRUCTURES:
            obs.append(bot.structures(structure).not_ready.amount / 10.0)

        # In-progress units (queued in production buildings)
        for unit in PROTOSS_UNITS:
            obs.append(bot.already_pending(unit) / 30.0)

        # Idle production buildings (indices 52-55)
        # Gateway + Warpgate combined pool: idle if building count exceeds
        # the number of gateway-type units currently in production.
        gw_count = bot.structures(UnitTypeId.GATEWAY).ready.amount
        wg_count = bot.structures(UnitTypeId.WARPGATE).ready.amount
        gw_wg_busy = (bot.already_pending(UnitTypeId.ZEALOT)
                      + bot.already_pending(UnitTypeId.STALKER)
                      + bot.already_pending(UnitTypeId.HIGHTEMPLAR))
        idle_gw_wg = max(0, (gw_count + wg_count) - gw_wg_busy)

        # Stargate: idle if stargate count exceeds air units in production.
        sg_count = bot.structures(UnitTypeId.STARGATE).ready.amount
        sg_busy = (bot.already_pending(UnitTypeId.VOIDRAY)
                   + bot.already_pending(UnitTypeId.CARRIER))
        idle_sg = max(0, sg_count - sg_busy)

        # Robotics Facility: idle if count exceeds immortals in production.
        robo_count = bot.structures(UnitTypeId.ROBOTICSFACILITY).ready.amount
        robo_busy = bot.already_pending(UnitTypeId.IMMORTAL)
        idle_robo = max(0, robo_count - robo_busy)

        # Warpgate-specific idle: warpgates whose warp cooldown has expired.
        # already_pending counts units mid-warp, so idle warpgates are those
        # not currently warping anything.
        idle_wg = max(0, wg_count - max(0, gw_wg_busy - gw_count))

        obs.append(idle_gw_wg / 5.0)   # index 52
        obs.append(idle_sg / 5.0)   # index 53
        obs.append(idle_robo / 5.0)   # index 54
        obs.append(idle_wg / 5.0)   # index 55

        return obs
