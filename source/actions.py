from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from helpers import build_structure, warp_in_unit

# output layer will be an array of numbers corresponding to the differnet actions the model can take
ACTIONS = [
    "do_nothing",               # 0
    "train_probe",              # 1
    "build_pylon",              # 2
    "build_gateway",            # 3
    "build_cyberneticscore",    # 4
    "build_assimilator",        # 5
    "build_nexus",              # 6
    "build_forge",              # 7
    "build_stargate",           # 8
    "build_robotics_facility",  # 9
    "build_twilight_council",   # 10
    "build_photon_cannon",      # 11
    "build_fleet_beacon",       # 12
    "build_templar_archive",    # 13
    "train_zealot",             # 14
    "train_stalker",            # 15
    "train_immortal",           # 16
    "train_voidray",            # 17
    "train_carrier",            # 18
    "train_high_templar",       # 19
    "warp_in_zealot",           # 20
    "warp_in_stalker",          # 21
    "warp_in_high_templar",     # 22
    "archon_warp_selection",    # 23
    "research_charge",          # 24
    "research_warp_gate",       # 25
    "upgrade_ground_weapons",   # 26
    "upgrade_air_weapons",      # 27
    "upgrade_shields",          # 28
    "attack_enemy_base",        # 29
    "train_adept",              # 30
    "train_phoenix",            # 31
    "train_colossus",           # 32
    "warp_in_adept",            # 33
]

ARMY = [
    UnitTypeId.ZEALOT,
    UnitTypeId.STALKER,
    UnitTypeId.ADEPT,
    UnitTypeId.HIGHTEMPLAR,
    UnitTypeId.ARCHON,
    UnitTypeId.IMMORTAL,
    UnitTypeId.COLOSSUS,
    UnitTypeId.VOIDRAY,
    UnitTypeId.PHOENIX,
    UnitTypeId.CARRIER,
]


async def execute_action(action_id: int, bot: BotAI):
    """Execute an action with minimal safety checks for faster training"""
    action_name = ACTIONS[action_id]

    if action_name == "do_nothing":
        pass

    elif action_name == "train_probe":
        if bot.can_afford(UnitTypeId.PROBE) and bot.townhalls.ready:
            bot.townhalls.ready.first.train(UnitTypeId.PROBE)

    elif action_name == "build_pylon":
        await build_structure(bot, UnitTypeId.PYLON)

    elif action_name == "build_gateway":
        await build_structure(bot, UnitTypeId.GATEWAY)

    elif action_name == "build_cyberneticscore":
        await build_structure(bot, UnitTypeId.CYBERNETICSCORE)

    elif action_name == "build_assimilator":
        await build_structure(bot, UnitTypeId.ASSIMILATOR)

    elif action_name == "build_nexus":
        await build_structure(bot, UnitTypeId.NEXUS)

    elif action_name == "build_forge":
        await build_structure(bot, UnitTypeId.FORGE)

    elif action_name == "build_stargate":
        await build_structure(bot, UnitTypeId.STARGATE)

    elif action_name == "build_robotics_facility":
        await build_structure(bot, UnitTypeId.ROBOTICSFACILITY)

    elif action_name == "build_twilight_council":
        await build_structure(bot, UnitTypeId.TWILIGHTCOUNCIL)

    elif action_name == "build_photon_cannon":
        await build_structure(bot, UnitTypeId.PHOTONCANNON)

    elif action_name == "build_fleet_beacon":
        await build_structure(bot, UnitTypeId.FLEETBEACON)

    elif action_name == "build_templar_archive":
        await build_structure(bot, UnitTypeId.TEMPLARARCHIVE)

    elif action_name == "train_zealot":
        if bot.can_afford(UnitTypeId.ZEALOT) and bot.structures(UnitTypeId.GATEWAY).ready.idle:
            bot.structures(UnitTypeId.GATEWAY).ready.idle.first.train(
                UnitTypeId.ZEALOT)

    elif action_name == "train_stalker":
        if bot.can_afford(UnitTypeId.STALKER) and bot.structures(UnitTypeId.CYBERNETICSCORE).ready:
            if bot.structures(UnitTypeId.GATEWAY).ready.idle:
                bot.structures(UnitTypeId.GATEWAY).ready.idle.first.train(
                    UnitTypeId.STALKER)

    elif action_name == "train_immortal":
        if bot.can_afford(UnitTypeId.IMMORTAL) and bot.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
            bot.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle.first.train(
                UnitTypeId.IMMORTAL)

    elif action_name == "train_voidray":
        if bot.can_afford(UnitTypeId.VOIDRAY) and bot.structures(UnitTypeId.STARGATE).ready.idle:
            bot.structures(UnitTypeId.STARGATE).ready.idle.first.train(
                UnitTypeId.VOIDRAY)

    elif action_name == "train_carrier":
        if bot.can_afford(UnitTypeId.CARRIER) and bot.structures(UnitTypeId.FLEETBEACON).ready and bot.structures(UnitTypeId.STARGATE).ready.idle:
            bot.structures(UnitTypeId.STARGATE).ready.idle.first.train(
                UnitTypeId.CARRIER)

    elif action_name == "train_high_templar":
        if bot.can_afford(UnitTypeId.HIGHTEMPLAR) and bot.structures(UnitTypeId.TEMPLARARCHIVE).ready:
            bot.structures(UnitTypeId.GATEWAY).ready.idle.first.train(
                UnitTypeId.HIGHTEMPLAR)

    elif action_name == "warp_in_zealot":
        if bot.can_afford(UnitTypeId.ZEALOT) and bot.structures(UnitTypeId.WARPGATE).ready:
            await warp_in_unit(bot, UnitTypeId.ZEALOT, AbilityId.WARPGATETRAIN_ZEALOT)

    elif action_name == "warp_in_stalker":
        if bot.can_afford(UnitTypeId.STALKER) and bot.structures(UnitTypeId.WARPGATE).ready:
            await warp_in_unit(bot, UnitTypeId.STALKER, AbilityId.WARPGATETRAIN_STALKER)

    elif action_name == "warp_in_high_templar":
        if bot.can_afford(UnitTypeId.HIGHTEMPLAR) and bot.structures(UnitTypeId.WARPGATE).ready and bot.structures(UnitTypeId.TEMPLARARCHIVE).ready:
            await warp_in_unit(bot, UnitTypeId.HIGHTEMPLAR, AbilityId.WARPGATETRAIN_HIGHTEMPLAR)

    elif action_name == "archon_warp_selection":
        # Merge 2 High Templars into an Archon
        if bot.units(UnitTypeId.HIGHTEMPLAR).idle.amount >= 2:
            # Get 2 idle High Templars
            templars = bot.units(UnitTypeId.HIGHTEMPLAR).idle.take(2)
            # Command them to morph into Archon
            templars.first(AbilityId.MORPH_ARCHON)
            templars[1](AbilityId.MORPH_ARCHON)

    elif action_name == "research_charge":
        if bot.structures(UnitTypeId.TWILIGHTCOUNCIL).ready and bot.can_afford(AbilityId.RESEARCH_CHARGE):
            bot.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.first(
                AbilityId.RESEARCH_CHARGE)

    elif action_name == "research_warp_gate":
        if bot.structures(UnitTypeId.CYBERNETICSCORE).ready and bot.can_afford(AbilityId.RESEARCH_WARPGATE):
            bot.structures(UnitTypeId.CYBERNETICSCORE).ready.first(
                AbilityId.RESEARCH_WARPGATE)

    elif action_name == "upgrade_ground_weapons":
        if bot.structures(UnitTypeId.FORGE).ready.idle:
            forge = bot.structures(UnitTypeId.FORGE).ready.idle.first
            # Check current upgrade level and upgrade to next
            if bot.can_afford(AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1):
                if bot.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) == 0:
                    forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1)
            elif bot.can_afford(AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2):
                if bot.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2) == 0:
                    forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2)
            elif bot.can_afford(AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3):
                if bot.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3) == 0:
                    forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3)

    elif action_name == "upgrade_air_weapons":
        if bot.structures(UnitTypeId.CYBERNETICSCORE).ready.idle:
            cyber = bot.structures(UnitTypeId.CYBERNETICSCORE).ready.idle.first
            if bot.can_afford(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1):
                if bot.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL1) == 0:
                    cyber.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL1)
            elif bot.can_afford(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2):
                if bot.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL2) == 0:
                    cyber.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL2)
            elif bot.can_afford(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3):
                if bot.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL3) == 0:
                    cyber.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL3)

    elif action_name == "upgrade_shields":
        if bot.structures(UnitTypeId.FORGE).ready.idle:
            forge = bot.structures(UnitTypeId.FORGE).ready.idle.first
            if bot.can_afford(AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL1):
                if bot.already_pending_upgrade(UpgradeId.PROTOSSSHIELDSLEVEL1) == 0:
                    forge.research(UpgradeId.PROTOSSSHIELDSLEVEL1)
            elif bot.can_afford(AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL2):
                if bot.already_pending_upgrade(UpgradeId.PROTOSSSHIELDSLEVEL2) == 0:
                    forge.research(UpgradeId.PROTOSSSHIELDSLEVEL2)
            elif bot.can_afford(AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL3):
                if bot.already_pending_upgrade(UpgradeId.PROTOSSSHIELDSLEVEL3) == 0:
                    forge.research(UpgradeId.PROTOSSSHIELDSLEVEL3)

    elif action_name == "attack_enemy_base":
        for unit in bot.units.of_type([UnitTypeId.ZEALOT, UnitTypeId.STALKER]).idle:
            unit.attack(bot.enemy_start_locations[0])

    elif action_name == "train_adept":
        if bot.can_afford(UnitTypeId.ADEPT) and bot.structures(UnitTypeId.CYBERNETICSCORE).ready:
            if bot.structures(UnitTypeId.GATEWAY).ready.idle:
                bot.structures(UnitTypeId.GATEWAY).ready.idle.first.train(
                    UnitTypeId.ADEPT)

    elif action_name == "train_phoenix":
        if bot.can_afford(UnitTypeId.PHOENIX) and bot.structures(UnitTypeId.STARGATE).ready.idle:
            bot.structures(UnitTypeId.STARGATE).ready.idle.first.train(
                UnitTypeId.PHOENIX)

    elif action_name == "train_colossus":
        if bot.can_afford(UnitTypeId.COLOSSUS) and bot.structures(UnitTypeId.ROBOTICSBAY).ready:
            if bot.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                bot.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle.first.train(
                    UnitTypeId.COLOSSUS)

    elif action_name == "warp_in_adept":
        if bot.can_afford(UnitTypeId.ADEPT) and bot.structures(UnitTypeId.WARPGATE).ready and bot.structures(UnitTypeId.CYBERNETICSCORE).ready:
            await warp_in_unit(bot, UnitTypeId.ADEPT, AbilityId.TRAINWARP_ADEPT)
