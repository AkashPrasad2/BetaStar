from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from enum import Enum
import random

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARMY_TYPES = [
    UnitTypeId.ZEALOT,
    UnitTypeId.STALKER,
    UnitTypeId.ADEPT,
    # High templar excluded so auto-merge won't be interrupted by other army commands
    UnitTypeId.ARCHON,
    UnitTypeId.IMMORTAL,
    UnitTypeId.COLOSSUS,
    UnitTypeId.VOIDRAY,
    UnitTypeId.PHOENIX,
    UnitTypeId.CARRIER,
]

PRODUCTION_BUILDINGS = [
    UnitTypeId.GATEWAY,
    UnitTypeId.STARGATE,
    UnitTypeId.ROBOTICSFACILITY,
]

# trigger defense if any completed structure drops below this HP %
DEFEND_HEALTH_THRESHOLD = 0.85
ATTACK_SUPPLY_THRESHOLD = 70    # army supply needed to initiate an attack
RETREAT_SUPPLY_THRESHOLD = 30   # army supply floor — retreat below this
ATTACK_TIME_CAP = 1680          # hard attack at 28 min regardless of supply
RALLY_INTERVAL = 30             # seconds between passive rally commands
# seconds between re-issuing army orders (avoid spam)
ARMY_COMMAND_INTERVAL = 5


class ArmyState(Enum):
    RALLY = "RALLY"
    DEFEND = "DEFEND"
    ATTACK = "ATTACK"


def get_army_supply(bot: BotAI) -> int:
    army_types_supply = [
        (UnitTypeId.ZEALOT,      2),
        (UnitTypeId.STALKER,     2),
        (UnitTypeId.ADEPT,       2),
        (UnitTypeId.HIGHTEMPLAR, 2),
        (UnitTypeId.ARCHON,      4),
        (UnitTypeId.IMMORTAL,    4),
        (UnitTypeId.COLOSSUS,    6),
        (UnitTypeId.VOIDRAY,     4),
        (UnitTypeId.PHOENIX,     2),
        (UnitTypeId.CARRIER,     6),
    ]
    return sum(bot.units(ut).amount * cost for ut, cost in army_types_supply)


def _structures_under_attack(bot: BotAI) -> list:
    """
    Returns completed structures whose health has dropped below the defend
    threshold. Excludes structures still under construction — their health
    naturally starts at 1% and climbs to 100%, which would otherwise
    trigger a false defense response on every new build.
    """
    return [
        s for s in bot.structures
        if s.is_ready and s.health_percentage < DEFEND_HEALTH_THRESHOLD
    ]


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------

async def build_structure(bot: BotAI, building: UnitTypeId):
    """Helper to systematically build structures depending on the type."""

    starting_nexus = bot.townhalls.closest_to(
        bot.start_location) if bot.townhalls else None

    if building == UnitTypeId.ASSIMILATOR:
        if bot.can_afford(UnitTypeId.ASSIMILATOR) and starting_nexus:
            for vespene in bot.vespene_geyser.closer_than(15, starting_nexus):
                if bot.gas_buildings.filter(lambda u: u.distance_to(vespene) < 1):
                    continue
                await bot.build(UnitTypeId.ASSIMILATOR, vespene)
                return

    elif building == UnitTypeId.PYLON:
        if bot.can_afford(UnitTypeId.PYLON) and starting_nexus:
            if bot.structures(UnitTypeId.PYLON).amount == 0:
                townhall_pos = starting_nexus.position
                map_center = bot.game_info.map_center
                direction = (map_center - townhall_pos).normalized
                target_pos = townhall_pos + direction * 5
                placement = await bot.find_placement(
                    UnitTypeId.PYLON, near=target_pos, placement_step=2)
            else:
                placement = await bot.find_placement(
                    UnitTypeId.PYLON,
                    near=starting_nexus.position,
                    placement_step=8)
            if placement:
                await bot.build(UnitTypeId.PYLON, placement)
                return

    elif building == UnitTypeId.NEXUS:
        if bot.can_afford(UnitTypeId.NEXUS):
            location = await bot.get_next_expansion()
            if location:
                await bot.build(UnitTypeId.NEXUS, location)
                return

    else:
        if bot.can_afford(building) and bot.structures(UnitTypeId.PYLON).ready and starting_nexus:
            placement = await bot.find_placement(
                building,
                near=starting_nexus.position,
                placement_step=1)
            if placement:
                worker = bot.select_build_worker(placement)
                if worker:
                    bot.do(worker.build(building, placement))
                    return


# ---------------------------------------------------------------------------
# Worker saturation
# ---------------------------------------------------------------------------

async def auto_saturate_assimilators(bot: BotAI):
    """Assign workers to under-staffed assimilators."""
    for assimilator in bot.structures(UnitTypeId.ASSIMILATOR).ready:
        if assimilator.assigned_harvesters < 3:
            probe = bot.workers.closest_to(assimilator)
            probe.gather(assimilator)


# ---------------------------------------------------------------------------
# Production rally points
# ---------------------------------------------------------------------------

async def set_production_rally_points(bot: BotAI):
    """Set rally points for production buildings to the army staging area."""
    if not bot.townhalls:
        return

    rally_point = bot.townhalls.center

    for unit_type in PRODUCTION_BUILDINGS:
        for building in bot.structures(unit_type).ready:
            if building.tag not in bot.rally_tags_set:
                building(AbilityId.RALLY_UNITS, rally_point)
                bot.rally_tags_set.add(building.tag)
                print(f"[{bot.time:.0f}s] Rally point set for {unit_type.name} "
                      f"→ {rally_point}")


# ---------------------------------------------------------------------------
# Army management
# ---------------------------------------------------------------------------

async def manage_army(bot: BotAI):
    """
    Single entry point for all army behaviour. Evaluates state transitions
    first, then issues exactly one set of orders based on the current state.

    State machine:
        RALLY  →  ATTACK  if army supply >= ATTACK_SUPPLY_THRESHOLD or time cap
        RALLY  →  DEFEND  if a completed structure is taking health damage
        DEFEND →  RALLY   if the threat clears (no more damaged structures)
        ATTACK →  RALLY   if army supply drops below RETREAT_SUPPLY_THRESHOLD
    """
    army = bot.units.of_type(ARMY_TYPES)

    _transition_state(bot)

    # Throttle actual unit commands to avoid spam, but always evaluate state
    if bot.time - bot.last_army_command_time < ARMY_COMMAND_INTERVAL:
        return
    bot.last_army_command_time = bot.time

    if not army:
        return

    if bot.army_state == ArmyState.RALLY:
        await _do_rally(bot, army)

    elif bot.army_state == ArmyState.DEFEND:
        _do_defend(bot, army)

    elif bot.army_state == ArmyState.ATTACK:
        _do_attack(bot, army)


def _transition_state(bot: BotAI):
    """
    Evaluate and apply state transitions. Called every frame so reactions
    are immediate even if unit commands are throttled.
    """
    supply = get_army_supply(bot)

    # never retreat :)
    # if bot.army_state == ArmyState.ATTACK:
    #     if supply < RETREAT_SUPPLY_THRESHOLD:
    #         print(
    #             f"[{bot.time:.0f}s] ARMY: Supply dropped to {supply}, retreating to RALLY.")
    #         bot.army_state = ArmyState.RALLY
    #     return

    if bot.army_state == ArmyState.DEFEND:
        if not _structures_under_attack(bot):
            print(f"[{bot.time:.0f}s] ARMY: Threat cleared, returning to RALLY.")
            bot.army_state = ArmyState.RALLY
        return

    # RALLY: check for transitions to DEFEND or ATTACK (DEFEND takes priority)
    under_attack = _structures_under_attack(bot)
    if under_attack:
        target = min(under_attack, key=lambda s: s.health_percentage)
        print(f"[{bot.time:.0f}s] ARMY: {target.name} under attack "
              f"({target.health_percentage:.0%} HP) — switching to DEFEND.")
        bot.army_state = ArmyState.DEFEND
        return

    if supply >= ATTACK_SUPPLY_THRESHOLD or bot.time >= ATTACK_TIME_CAP:
        print(f"[{bot.time:.0f}s] ARMY: Supply={supply}, switching to ATTACK.")
        bot.army_state = ArmyState.ATTACK


async def _do_rally(bot: BotAI, army):
    """Move idle army units to the staging area near our townhalls."""
    if bot.time - bot.last_rally_time < RALLY_INTERVAL:
        return
    bot.last_rally_time = bot.time

    if not bot.townhalls:
        return

    staging = bot.townhalls.center
    idle_army = [u for u in army if u.is_idle]
    if idle_army:
        for unit in idle_army:
            unit.attack(staging)  # attack-move so they engage anything nearby
        print(
            f"[{bot.time:.0f}s] ARMY: Rallying {len(idle_army)} idle unit(s) → {staging}")


def _do_defend(bot: BotAI, army):
    """Send army to defend the most damaged completed structure."""
    under_attack = _structures_under_attack(bot)
    if not under_attack:
        return  # transition will handle this next frame

    target = min(under_attack, key=lambda s: s.health_percentage)
    _issue_attack(bot, army, target.position,
                  f"defending {target.name} ({target.health_percentage:.0%} HP)")


def _do_attack(bot: BotAI, army):
    """
    Systematically attack enemy structures and bases.
    Priority: visible structures → enemy start locations → expansions.
    Clears locations as they are confirmed empty.
    """
    # Priority 1: visible enemy structures
    if bot.enemy_structures:
        target_pos = bot.enemy_structures.closest_to(
            bot.start_location).position
        _issue_attack(bot, army, target_pos, "visible enemy structure")
        return

    # Priority 2: enemy start locations
    for loc in bot.enemy_start_locations:
        if not isinstance(loc, Point2):
            loc = Point2(loc)
        if loc in bot.enemy_bases_cleared:
            continue
        if not bot.is_visible(loc):
            _issue_attack(bot, army, loc, "enemy start (unscouted)")
            return
        if not bot.enemy_structures.closer_than(10, loc):
            bot.enemy_bases_cleared.add(loc)
            print(f"[{bot.time:.0f}s] ARMY: Marked enemy start {loc} as cleared.")
        else:
            _issue_attack(bot, army, loc, "enemy start (structures present)")
            return

    # Priority 3: all expansion locations
    for loc in bot.expansion_locations_list:
        if loc in bot.enemy_bases_cleared:
            continue
        if not bot.is_visible(loc):
            _issue_attack(bot, army, loc, "expansion (unscouted)")
            return
        if not bot.enemy_structures.closer_than(10, loc):
            bot.enemy_bases_cleared.add(loc)
            continue
        else:
            _issue_attack(bot, army, loc, "expansion (structures present)")
            return

    # Everything cleared — reset and sweep again
    print(f"[{bot.time:.0f}s] ARMY: All known locations cleared, resetting.")
    bot.enemy_bases_cleared.clear()


def _issue_attack(bot: BotAI, army, target_pos: Point2, reason: str):
    """Issue attack-move to all army units."""
    print(f"[{bot.time:.0f}s] ARMY [{bot.army_state.value}]: "
          f"{army.amount} unit(s) → {target_pos} ({reason})")
    for unit in army:
        unit.attack(target_pos)


# ---------------------------------------------------------------------------
# Auto-merge High Templars into Archons
# ---------------------------------------------------------------------------

async def auto_merge_archons(bot: BotAI):
    """Only issue merge command if HTs aren't already mergeing (avoid spam)"""
    hts = bot.units(UnitTypeId.HIGHTEMPLAR)

    if hts.amount >= 2:
        # Check if any HT already has a morph order
        for ht in hts:
            # If any HT is already morphing, don't issue new commands
            if ht.orders:  # Has active orders
                for order in ht.orders:
                    if order.ability.id == AbilityId.MORPH_ARCHON:
                        return  # Already merging, don't interfere

        # No merge in progress, start one
        ht1 = hts[0]
        ht2 = hts[1]
        ht1(AbilityId.MORPH_ARCHON, ht2)
        print(f"[{bot.time:.0f}s] AUTO-MERGE: Initiated archon merge")

# ---------------------------------------------------------------------------
# Warp-in helper
# ---------------------------------------------------------------------------


async def warp_in_unit(bot: BotAI, unit_type: UnitTypeId, ability_id: AbilityId) -> bool:
    """
    Attempt to warp in a unit near a pylon.
    Returns True if the warp command was issued.
    """
    warpgates = bot.structures(UnitTypeId.WARPGATE).ready
    if not warpgates:
        return False

    warpgate = warpgates.first
    abilities = await bot.get_available_abilities(warpgate)
    if ability_id not in abilities:
        return False

    pylons = bot.structures(UnitTypeId.PYLON).ready
    if not pylons:
        return False

    pylon = pylons.closest_to(bot.game_info.map_center)

    placement_radius = 6.0
    for _ in range(12):
        angle = random.uniform(0, 6.2832)
        distance = random.uniform(1.5, placement_radius)
        offset = Point2((distance * __import__("math").cos(angle),
                         distance * __import__("math").sin(angle)))
        target_pos = pylon.position + offset

        placement = await bot.find_placement(
            AbilityId.WARPGATETRAIN_ZEALOT,
            target_pos,
            max_distance=2,
            placement_step=1,
        )
        if placement:
            warpgate.warp_in(unit_type, placement)
            return True

    # Fallback: warp at pylon
    warpgate.warp_in(unit_type, pylon.position)
    return True
