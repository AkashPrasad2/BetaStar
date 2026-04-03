from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
import random


async def build_structure(bot: BotAI, building: UnitTypeId):
    """Helper to systematically builds sturctures depending on the type"""

    if building == UnitTypeId.ASSIMILATOR:
        if bot.can_afford(UnitTypeId.ASSIMILATOR) and bot.townhalls:
            for vespene in bot.vespene_geyser.closer_than(15, bot.townhalls.first):
                # Check if already built here
                if bot.gas_buildings.filter(lambda u: u.distance_to(vespene) < 1):
                    continue
                await bot.build(UnitTypeId.ASSIMILATOR, vespene)
                return

    elif building == UnitTypeId.PYLON:
        if bot.can_afford(UnitTypeId.PYLON) and bot.townhalls:
            # First pylon: place near townhall but towards map center
            if bot.structures(UnitTypeId.PYLON).amount == 0:
                townhall_pos = bot.townhalls.first.position
                map_center = bot.game_info.map_center
                # Direction vector from townhall to map center
                direction = (map_center - townhall_pos).normalized
                # Place towards map center to avoid blocking probes
                target_pos = townhall_pos + direction * 5
                placement = await bot.find_placement(
                    UnitTypeId.PYLON,
                    near=target_pos,
                    placement_step=2
                )
            else:
                # Subsequent pylons: spread them out to maximize buildable surface area
                placement = await bot.find_placement(
                    UnitTypeId.PYLON,
                    near=bot.townhalls.first.position,
                    placement_step=8
                )
            if placement:
                await bot.build(UnitTypeId.PYLON, placement)
                return

    elif building == UnitTypeId.NEXUS:
        if bot.can_afford(UnitTypeId.NEXUS):
            # Expand to nearest mineral field
            location = await bot.get_next_expansion()
            if location:
                await bot.build(UnitTypeId.NEXUS, location)
                return

    # all other buildings just need to be powered by pylon
    else:
        if bot.can_afford(building) and bot.structures(UnitTypeId.PYLON).ready:
            placement = await bot.find_placement(
                building,
                near=bot.townhalls.first.position,
                placement_step=2
            )
            if placement:
                worker = bot.select_build_worker(placement)
                if worker:
                    bot.do(worker.build(building, placement))

                    # Set rally point for production buildings
                    if building in [UnitTypeId.GATEWAY, UnitTypeId.STARGATE, UnitTypeId.ROBOTICSFACILITY]:
                        # Calculate center of all nexuses
                        if bot.townhalls:
                            nexus_center = bot.townhalls.center
                            # Wait a frame for the building to be registered, then set rally
                            # We'll set it in the next on_step when the building exists
                            bot._pending_rally_buildings = getattr(
                                bot, '_pending_rally_buildings', [])
                            bot._pending_rally_buildings.append(
                                (building, placement, nexus_center))
                return


async def auto_saturate_assimilators(bot: BotAI):
    """Checks if there are assimilators with less than 3 workers and assigns them to it"""
    for assimilator in bot.structures(UnitTypeId.ASSIMILATOR).ready:
        if assimilator.assigned_harvesters < 3:
            # select closest probe and assign to assimilator
            probe = bot.workers.closest_to(assimilator)
            probe.gather(assimilator)
    return


async def set_production_rally_points(bot: BotAI):
    """Set rally points for production buildings to the center of all nexuses"""
    if not bot.townhalls:
        return

    nexus_center = bot.townhalls.center

    # Set rally for gateways and warpgates
    for gateway in bot.structures(UnitTypeId.GATEWAY).ready:
        if not hasattr(gateway, '_rally_set'):
            gateway(AbilityId.RALLY_UNITS, nexus_center)
            gateway._rally_set = True

    # Set rally for stargates
    for stargate in bot.structures(UnitTypeId.STARGATE).ready:
        if not hasattr(stargate, '_rally_set'):
            stargate(AbilityId.RALLY_UNITS, nexus_center)
            stargate._rally_set = True

    # Set rally for robotics facilities
    for robo in bot.structures(UnitTypeId.ROBOTICSFACILITY).ready:
        if not hasattr(robo, '_rally_set'):
            robo(AbilityId.RALLY_UNITS, nexus_center)
            robo._rally_set = True


async def warp_in_unit(bot: BotAI, unit_type: UnitTypeId, ability_id: AbilityId) -> bool:
    """
    Attempt to warp in a unit near a pylon with multiple placement attempts.
    After warping, the unit will automatically move to nexus center due to rally points.

    Returns True if warp was successful, False otherwise.
    """
    if not bot.structures(UnitTypeId.WARPGATE).ready:
        return False

    warpgate = bot.structures(UnitTypeId.WARPGATE).ready.first

    # Check if warpgate has the ability available
    abilities = await bot.get_available_abilities(warpgate)
    if ability_id not in abilities:
        return False

    # Find a pylon to warp near
    if not bot.structures(UnitTypeId.PYLON).ready:
        return False

    pylon = bot.structures(UnitTypeId.PYLON).ready.closest_to(warpgate)

    # Try multiple positions around the pylon
    # Pylon power radius is 6.5, we'll try positions within that range
    placement_radius = 6.0
    max_attempts = 10

    for attempt in range(max_attempts):
        # Generate random offset within pylon power radius
        angle = random.uniform(0, 6.28)  # 0 to 2*pi
        distance = random.uniform(2, placement_radius)

        offset_x = distance * (angle % 3.14)  # Simple angle to x conversion
        offset_y = distance * (angle / 3.14)  # Simple angle to y conversion

        target_pos = Point2((
            pylon.position.x + offset_x,
            pylon.position.y + offset_y
        ))

        # Check if position is valid for placement
        placement = await bot.find_placement(
            AbilityId.WARPGATETRAIN_ZEALOT,  # Use any warp ability for placement check
            target_pos,
            max_distance=0,
            placement_step=1
        )

        if placement:
            # Warp in the unit
            warpgate.warp_in(unit_type, placement)
            return True

    # If all attempts failed, try at pylon position directly
    warpgate.warp_in(unit_type, pylon.position)
    return True


async def rally_idle_army(bot: BotAI):
    """Move idle army units to nexus center every 10 seconds"""
    # Only run every 10 seconds
    if not hasattr(bot, '_last_rally_time'):
        bot._last_rally_time = 0

    if bot.time - bot._last_rally_time < 10:
        return

    bot._last_rally_time = bot.time

    if not bot.townhalls:
        return

    nexus_center = bot.townhalls.center

    # Army unit types to rally
    army_types = [
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

    for unit_type in army_types:
        for unit in bot.units(unit_type).idle:
            unit.move(nexus_center)


async def auto_attack(bot: BotAI):
    """Automatically attack when army supply > 70 or time > 28 minutes"""
    # Calculate army supply
    army_types = [
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

    army_supply = sum(bot.units(unit_type).amount * bot.calculate_supply_cost(unit_type)
                      for unit_type in army_types)

    # Check if we should attack (supply > 70 or time > 28 minutes)
    should_attack = army_supply > 70 or bot.time > 1680  # 28 minutes = 1680 seconds

    if not should_attack:
        return

    # Mark that we've initiated auto-attack
    if not hasattr(bot, '_auto_attack_initiated'):
        bot._auto_attack_initiated = False

    if should_attack and not bot._auto_attack_initiated:
        bot._auto_attack_initiated = True

    # Get all army units
    army = bot.units.of_type(army_types)

    if not army:
        return

    # Find next target: enemy bases in order
    target = None

    # Check enemy start locations first
    for enemy_start in bot.enemy_start_locations:
        # If there's an enemy base there, attack it
        enemy_structures = bot.enemy_structures.closer_than(20, enemy_start)
        if enemy_structures:
            target = enemy_start
            break

    # If no enemy base at start location, check all expansion locations
    if not target:
        for expansion in bot.expansion_locations_list:
            enemy_structures = bot.enemy_structures.closer_than(20, expansion)
            if enemy_structures:
                target = expansion
                break

    # If still no target, just attack enemy start location
    if not target and bot.enemy_start_locations:
        target = bot.enemy_start_locations[0]

    # Send all army units to attack
    if target:
        for unit in army:
            unit.attack(target)


async def defend_structures(bot: BotAI):
    """Send all troops to defend if any structure is under attack (unless auto-attack already initiated)"""
    # If auto-attack has already been initiated, don't interrupt it
    if hasattr(bot, '_auto_attack_initiated') and bot._auto_attack_initiated:
        return

    # Check if any of our structures are under attack by checking if health is less than max
    structures_under_attack = []
    for structure in bot.structures:
        if structure.health < structure.health_max:
            structures_under_attack.append(structure)

    if not structures_under_attack:
        return

    # Get the first structure under attack as rally point
    defend_target = structures_under_attack[0].position

    # Army unit types to send
    army_types = [
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

    # Send all army units to defend
    for unit_type in army_types:
        for unit in bot.units(unit_type):
            unit.attack(defend_target)
