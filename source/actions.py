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


# (The army unit list lives in helpers.ARMY_TYPES, which the state machine uses.
# The copy that used to be here existed only for the removed attack action.)


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


def _research(
    bot: BotAI,
    building: UnitTypeId,
    ability: AbilityId,
    upgrade: UpgradeId,
) -> ActionResult:
    """Issue one research command and report why it could not be issued."""
    structures = bot.structures(building).ready
    if not structures:
        return ActionResult.NO_PREREQ
    if upgrade in bot.state.upgrades or bot.already_pending_upgrade(upgrade) > 0:
        return ActionResult.SUPPRESSED
    if not structures.idle:
        return ActionResult.NO_PRODUCTION
    if not bot.can_afford(ability):
        return ActionResult.UNAFFORDABLE
    structures.idle.first(ability)
    return ActionResult.ISSUED


def _research_next_level(
    bot: BotAI,
    building: UnitTypeId,
    levels: tuple[tuple[UpgradeId, AbilityId], ...],
) -> ActionResult:
    """Research the first incomplete level in an ordered upgrade chain."""
    for upgrade, ability in levels:
        if upgrade in bot.state.upgrades:
            continue
        return _research(bot, building, ability, upgrade)
    return ActionResult.SUPPRESSED


async def execute_action(action_id: int, bot: BotAI):
    """Execute an action. All branches are fully guarded — no .first on empty collections."""
    action_name = ACTIONS[action_id]

    if action_name == "do_nothing":
        return ActionResult.NO_OP

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

    elif action_name == "warp_in_zealot":
        return await warp_in_unit(bot, UnitTypeId.ZEALOT, AbilityId.WARPGATETRAIN_ZEALOT)

    elif action_name == "warp_in_stalker":
        return await warp_in_unit(bot, UnitTypeId.STALKER, AbilityId.WARPGATETRAIN_STALKER)

    elif action_name == "warp_in_high_templar":
        return await warp_in_unit(bot, UnitTypeId.HIGHTEMPLAR, AbilityId.WARPGATETRAIN_HIGHTEMPLAR, requires=UnitTypeId.TEMPLARARCHIVE)

    elif action_name == "research_charge":
        return _research(
            bot, UnitTypeId.TWILIGHTCOUNCIL,
            AbilityId.RESEARCH_CHARGE, UpgradeId.CHARGE)

    elif action_name == "research_warp_gate":
        return _research(
            bot, UnitTypeId.CYBERNETICSCORE,
            AbilityId.RESEARCH_WARPGATE, UpgradeId.WARPGATERESEARCH)

    elif action_name == "upgrade_ground_weapons":
        return _research_next_level(bot, UnitTypeId.FORGE, (
            (UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
             AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1),
            (UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
             AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2),
            (UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
             AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3),
        ))

    elif action_name == "upgrade_air_weapons":
        return _research_next_level(bot, UnitTypeId.CYBERNETICSCORE, (
            (UpgradeId.PROTOSSAIRWEAPONSLEVEL1,
             AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1),
            (UpgradeId.PROTOSSAIRWEAPONSLEVEL2,
             AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2),
            (UpgradeId.PROTOSSAIRWEAPONSLEVEL3,
             AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3),
        ))

    elif action_name == "upgrade_shields":
        return _research_next_level(bot, UnitTypeId.FORGE, (
            (UpgradeId.PROTOSSSHIELDSLEVEL1,
             AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL1),
            (UpgradeId.PROTOSSSHIELDSLEVEL2,
             AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL2),
            (UpgradeId.PROTOSSSHIELDSLEVEL3,
             AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL3),
        ))

    elif action_name == "train_adept":
        return _train(bot, UnitTypeId.ADEPT, UnitTypeId.GATEWAY, requires=UnitTypeId.CYBERNETICSCORE)

    elif action_name == "train_phoenix":
        return _train(bot, UnitTypeId.PHOENIX, UnitTypeId.STARGATE)

    elif action_name == "train_colossus":
        return _train(bot, UnitTypeId.COLOSSUS, UnitTypeId.ROBOTICSFACILITY, requires=UnitTypeId.ROBOTICSBAY)

    # Note: there is deliberately no attack action. Attacking is owned by the
    # army state machine in helpers.manage_army(); the parser never labelled it
    # (106,449 Attack events, zero labels), so as a model output it could only
    # ever fire from sampling noise.
    return ActionResult.NOT_LABELLED
