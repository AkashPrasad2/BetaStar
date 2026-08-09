from __future__ import annotations

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from helpers import build_structure, warp_in_unit, ActionResult

# Action names are defined once in obs_spec (index == action id) and imported
# here so execution dispatch, the parser, and the analysis scripts can never
# disagree about which id means what.
from obs_spec import ACTION_NAMES as ACTIONS


ARMY = [
    UnitTypeId.ZEALOT,
    UnitTypeId.STALKER,
    UnitTypeId.ADEPT,
    UnitTypeId.ARCHON,
    UnitTypeId.IMMORTAL,
    UnitTypeId.COLOSSUS,
    UnitTypeId.VOIDRAY,
    UnitTypeId.PHOENIX,
    UnitTypeId.CARRIER,
]



def _train(bot: BotAI, unit: UnitTypeId, building: UnitTypeId,
           requires: UnitTypeId | None = None) -> ActionResult:
    """
    Train `unit` from an idle `building`, optionally gated on `requires` being
    complete. Returns why it did or did not happen.
    """
    if requires is not None and not bot.structures(requires).ready:
        return ActionResult.NO_PREREQ
    if not bot.can_afford(unit):
        return ActionResult.UNAFFORDABLE
    idle = bot.structures(building).ready.idle
    if not idle:
        return ActionResult.NO_PRODUCTION
    idle.first.train(unit)
    return ActionResult.ISSUED


def _train_probe(bot: BotAI) -> ActionResult:
    """
    Train a probe, preferring an IDLE nexus.

    This used to be `bot.townhalls.ready.first.train(...)`, which always targeted
    the same nexus. Its production queue filled up (max 5) and further orders were
    rejected outright -- 24 silent no-ops across the logged games -- while the
    other nexuses sat idle. Preferring an idle townhall spreads production and
    removes the rejection.
    """
    if not bot.can_afford(UnitTypeId.PROBE):
        return ActionResult.UNAFFORDABLE
    halls = bot.townhalls.ready.idle or bot.townhalls.ready
    if not halls:
        return ActionResult.NO_PRODUCTION
    halls.first.train(UnitTypeId.PROBE)
    return ActionResult.ISSUED


async def execute_action(action_id: int, bot: BotAI):
    """Execute an action. All branches are fully guarded — no .first on empty collections."""
    action_name = ACTIONS[action_id]

    if action_name == "do_nothing":
        pass

    elif action_name == "train_probe":
        return _train_probe(bot)

    elif action_name == "build_pylon":
        return await build_structure(bot, UnitTypeId.PYLON)

    elif action_name == "build_gateway":
        return await build_structure(bot, UnitTypeId.GATEWAY)

    elif action_name == "build_cyberneticscore":
        return await build_structure(bot, UnitTypeId.CYBERNETICSCORE)

    elif action_name == "build_assimilator":
        return await build_structure(bot, UnitTypeId.ASSIMILATOR)

    elif action_name == "build_nexus":
        return await build_structure(bot, UnitTypeId.NEXUS)

    elif action_name == "build_forge":
        return await build_structure(bot, UnitTypeId.FORGE)

    elif action_name == "build_stargate":
        return await build_structure(bot, UnitTypeId.STARGATE)

    elif action_name == "build_robotics_facility":
        return await build_structure(bot, UnitTypeId.ROBOTICSFACILITY)

    elif action_name == "build_twilight_council":
        return await build_structure(bot, UnitTypeId.TWILIGHTCOUNCIL)

    elif action_name == "build_photon_cannon":
        return await build_structure(bot, UnitTypeId.PHOTONCANNON)

    elif action_name == "build_fleet_beacon":
        return await build_structure(bot, UnitTypeId.FLEETBEACON)

    elif action_name == "build_templar_archive":
        return await build_structure(bot, UnitTypeId.TEMPLARARCHIVE)

    elif action_name == "build_robotics_bay":
        return await build_structure(bot, UnitTypeId.ROBOTICSBAY)

    elif action_name == "build_shield_battery":
        return await build_structure(bot, UnitTypeId.SHIELDBATTERY)

    elif action_name == "train_zealot":
        return _train(bot, UnitTypeId.ZEALOT, UnitTypeId.GATEWAY)

    elif action_name == "train_stalker":
        return _train(bot, UnitTypeId.STALKER, UnitTypeId.GATEWAY, requires=UnitTypeId.CYBERNETICSCORE)

    elif action_name == "train_immortal":
        return _train(bot, UnitTypeId.IMMORTAL, UnitTypeId.ROBOTICSFACILITY)

    elif action_name == "train_voidray":
        return _train(bot, UnitTypeId.VOIDRAY, UnitTypeId.STARGATE)

    elif action_name == "train_carrier":
        return _train(bot, UnitTypeId.CARRIER, UnitTypeId.STARGATE, requires=UnitTypeId.FLEETBEACON)

    elif action_name == "train_high_templar":
        return _train(bot, UnitTypeId.HIGHTEMPLAR, UnitTypeId.GATEWAY, requires=UnitTypeId.TEMPLARARCHIVE)

    elif action_name == "warp_in_zealot":
        return await warp_in_unit(bot, UnitTypeId.ZEALOT, AbilityId.WARPGATETRAIN_ZEALOT)

    elif action_name == "warp_in_stalker":
        return await warp_in_unit(bot, UnitTypeId.STALKER, AbilityId.WARPGATETRAIN_STALKER)

    elif action_name == "warp_in_high_templar":
        return await warp_in_unit(bot, UnitTypeId.HIGHTEMPLAR, AbilityId.WARPGATETRAIN_HIGHTEMPLAR, requires=UnitTypeId.TEMPLARARCHIVE)

    elif action_name == "research_charge":
        if (bot.structures(UnitTypeId.TWILIGHTCOUNCIL).ready
                and bot.can_afford(AbilityId.RESEARCH_CHARGE)):
            bot.structures(UnitTypeId.TWILIGHTCOUNCIL).ready.first(
                AbilityId.RESEARCH_CHARGE)

    elif action_name == "research_warp_gate":
        if (bot.structures(UnitTypeId.CYBERNETICSCORE).ready
                and bot.can_afford(AbilityId.RESEARCH_WARPGATE)):
            bot.structures(UnitTypeId.CYBERNETICSCORE).ready.first(
                AbilityId.RESEARCH_WARPGATE)

    elif action_name == "upgrade_ground_weapons":
        # FIX: check idle before calling .first
        if bot.structures(UnitTypeId.FORGE).ready.idle:
            forge = bot.structures(UnitTypeId.FORGE).ready.idle.first
            if bot.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) == 0 and \
               UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1 not in bot.state.upgrades:
                if bot.can_afford(AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1):
                    forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1)
            elif bot.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2) == 0 and \
                    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2 not in bot.state.upgrades:
                if bot.can_afford(AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2):
                    forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2)
            elif bot.already_pending_upgrade(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3) == 0 and \
                    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3 not in bot.state.upgrades:
                if bot.can_afford(AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3):
                    forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3)

    elif action_name == "upgrade_air_weapons":
        if bot.structures(UnitTypeId.CYBERNETICSCORE).ready.idle:
            cyber = bot.structures(UnitTypeId.CYBERNETICSCORE).ready.idle.first
            if bot.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL1) == 0 and \
               UpgradeId.PROTOSSAIRWEAPONSLEVEL1 not in bot.state.upgrades:
                if bot.can_afford(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1):
                    cyber.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL1)
            elif bot.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL2) == 0 and \
                    UpgradeId.PROTOSSAIRWEAPONSLEVEL2 not in bot.state.upgrades:
                if bot.can_afford(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2):
                    cyber.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL2)
            elif bot.already_pending_upgrade(UpgradeId.PROTOSSAIRWEAPONSLEVEL3) == 0 and \
                    UpgradeId.PROTOSSAIRWEAPONSLEVEL3 not in bot.state.upgrades:
                if bot.can_afford(AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3):
                    cyber.research(UpgradeId.PROTOSSAIRWEAPONSLEVEL3)

    elif action_name == "upgrade_shields":
        if bot.structures(UnitTypeId.FORGE).ready.idle:
            forge = bot.structures(UnitTypeId.FORGE).ready.idle.first
            if bot.already_pending_upgrade(UpgradeId.PROTOSSSHIELDSLEVEL1) == 0 and \
               UpgradeId.PROTOSSSHIELDSLEVEL1 not in bot.state.upgrades:
                if bot.can_afford(AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL1):
                    forge.research(UpgradeId.PROTOSSSHIELDSLEVEL1)
            elif bot.already_pending_upgrade(UpgradeId.PROTOSSSHIELDSLEVEL2) == 0 and \
                    UpgradeId.PROTOSSSHIELDSLEVEL2 not in bot.state.upgrades:
                if bot.can_afford(AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL2):
                    forge.research(UpgradeId.PROTOSSSHIELDSLEVEL2)
            elif bot.already_pending_upgrade(UpgradeId.PROTOSSSHIELDSLEVEL3) == 0 and \
                    UpgradeId.PROTOSSSHIELDSLEVEL3 not in bot.state.upgrades:
                if bot.can_afford(AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL3):
                    forge.research(UpgradeId.PROTOSSSHIELDSLEVEL3)

    elif action_name == "attack_enemy_base":
        for unit in bot.units.of_type(ARMY).idle:
            unit.attack(bot.enemy_start_locations[0])

    elif action_name == "train_adept":
        return _train(bot, UnitTypeId.ADEPT, UnitTypeId.GATEWAY, requires=UnitTypeId.CYBERNETICSCORE)

    elif action_name == "train_phoenix":
        return _train(bot, UnitTypeId.PHOENIX, UnitTypeId.STARGATE)

    elif action_name == "train_colossus":
        return _train(bot, UnitTypeId.COLOSSUS, UnitTypeId.ROBOTICSFACILITY, requires=UnitTypeId.ROBOTICSBAY)

    elif action_name == "warp_in_adept":
        return await warp_in_unit(bot, UnitTypeId.ADEPT, AbilityId.TRAINWARP_ADEPT, requires=UnitTypeId.CYBERNETICSCORE)

    return ActionResult.NOT_LABELLED
