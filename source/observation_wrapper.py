from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

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

# 11 units
PROTOSS_UNITS = [
    UnitTypeId.PROBE,
    UnitTypeId.ZEALOT,
    UnitTypeId.STALKER,
    UnitTypeId.HIGHTEMPLAR,
    UnitTypeId.ARCHON,
    UnitTypeId.IMMORTAL,
    UnitTypeId.CARRIER,
    UnitTypeId.VOIDRAY,
    UnitTypeId.ADEPT,
    UnitTypeId.PHOENIX,
    UnitTypeId.COLOSSUS,
]


class ObservationWrapper:
    """
    Converts game state into a flat, normalized vector for neural network input. Runs at every game step.

    Feature layout (71 total):
        [0]     game time (normalized)
        [1:5]   minerals one-hot (4 bins)
        [5:9]   vespene one-hot (4 bins)
        [9]     supply_used
        [10]    supply_cap
        [11]    worker saturation
        [12:27] completed structure counts   (15)
        [27:38] completed unit counts        (11)
        [38:53] in-progress structure counts (15)
        [53:64] in-progress unit counts      (11)
        [64]    idle gateway+warpgate count  (normalised /5)
        [65]    idle stargate count          (normalised /5)
        [66]    idle robotics facility count (normalised /5)
        [67]    idle warpgate count          (normalised /5)
        [68]    ground weapons level         (normalised /3)
        [69]    shields level                (normalised /3)
        [70]    air weapons level            (normalised /3)
    """

    def __init__(self):
        self.observation_size = self.calculate_obs_size()

    def calculate_obs_size(self) -> int:
        # 12 base + 15 structs + 8 units + 15 structs_pending + 8 units_pending
        # + 4 idle production buildings + 3 upgrade levels
        return (12
                + len(PROTOSS_STRUCTURES)
                + len(PROTOSS_UNITS)
                + len(PROTOSS_STRUCTURES)
                + len(PROTOSS_UNITS)
                + 4
                + 3)

    def get_observation(self, bot: BotAI) -> list[float]:
        obs = []

        obs.append(bot.time / 720.0)

        # Minerals one-hot (4 bins)
        if bot.minerals < 100:
            obs.extend([1.0, 0.0, 0.0, 0.0])
        elif bot.minerals < 300:
            obs.extend([0.0, 1.0, 0.0, 0.0])
        elif bot.minerals < 700:
            obs.extend([0.0, 0.0, 1.0, 0.0])
        else:
            obs.extend([0.0, 0.0, 0.0, 1.0])

        # Gas one-hot (4 bins)
        if bot.vespene < 25:
            obs.extend([1.0, 0.0, 0.0, 0.0])
        elif bot.vespene < 100:
            obs.extend([0.0, 1.0, 0.0, 0.0])
        elif bot.vespene < 300:
            obs.extend([0.0, 0.0, 1.0, 0.0])
        else:
            obs.extend([0.0, 0.0, 0.0, 1.0])

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

        # Idle production buildings (indices 58-61)
        # Gateway + Warpgate combined pool: idle if building count exceeds
        # the number of gateway-type units currently in production.
        gw_count = bot.structures(UnitTypeId.GATEWAY).ready.amount
        wg_count = bot.structures(UnitTypeId.WARPGATE).ready.amount
        gw_wg_busy = (bot.already_pending(UnitTypeId.ZEALOT)
                      + bot.already_pending(UnitTypeId.STALKER)
                      + bot.already_pending(UnitTypeId.HIGHTEMPLAR)
                      + bot.already_pending(UnitTypeId.ADEPT))
        idle_gw_wg = max(0, (gw_count + wg_count) - gw_wg_busy)

        # Stargate: idle if stargate count exceeds air units in production.
        sg_count = bot.structures(UnitTypeId.STARGATE).ready.amount
        sg_busy = (bot.already_pending(UnitTypeId.VOIDRAY)
                   + bot.already_pending(UnitTypeId.CARRIER)
                   + bot.already_pending(UnitTypeId.PHOENIX))
        idle_sg = max(0, sg_count - sg_busy)

        # Robotics Facility: idle if count exceeds immortals in production.
        robo_count = bot.structures(UnitTypeId.ROBOTICSFACILITY).ready.amount
        robo_busy = (bot.already_pending(UnitTypeId.IMMORTAL)
                     + bot.already_pending(UnitTypeId.COLOSSUS))
        idle_robo = max(0, robo_count - robo_busy)

        # Warpgate-specific idle: warpgates whose warp cooldown has expired.
        # already_pending counts units mid-warp, so idle warpgates are those
        # not currently warping anything.
        idle_wg = max(0, wg_count - max(0, gw_wg_busy - gw_count))

        obs.append(idle_gw_wg / 5.0)   # index 64
        obs.append(idle_sg / 5.0)   # index 65
        obs.append(idle_robo / 5.0)   # index 66
        obs.append(idle_wg / 5.0)   # index 67

        # Upgrade levels: committed = completed OR currently being researched.
        # Matches the pending-or-complete convention used in the replay parser.
        def committed_upgrade_level(upgrade_ids):
            lvl = 0
            for i, uid in enumerate(upgrade_ids, start=1):
                if uid in bot.state.upgrades:
                    lvl = i          # level is done, keep checking higher
                elif bot.already_pending_upgrade(uid) > 0:
                    lvl = i          # level is in progress
                    break            # can't have higher levels pending yet
            return lvl

        gw_lvl = committed_upgrade_level([
            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
        ])
        sh_lvl = committed_upgrade_level([
            UpgradeId.PROTOSSSHIELDSLEVEL1,
            UpgradeId.PROTOSSSHIELDSLEVEL2,
            UpgradeId.PROTOSSSHIELDSLEVEL3,
        ])
        aw_lvl = committed_upgrade_level([
            UpgradeId.PROTOSSAIRWEAPONSLEVEL1,
            UpgradeId.PROTOSSAIRWEAPONSLEVEL2,
            UpgradeId.PROTOSSAIRWEAPONSLEVEL3,
        ])

        obs.append(gw_lvl / 3.0)   # index 68
        obs.append(sh_lvl / 3.0)   # index 69
        obs.append(aw_lvl / 3.0)   # index 70

        return obs
