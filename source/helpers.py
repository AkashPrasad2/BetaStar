from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
import random

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARMY_TYPES = [
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

PRODUCTION_BUILDINGS = [
    UnitTypeId.GATEWAY,
    UnitTypeId.STARGATE,
    UnitTypeId.ROBOTICSFACILITY,
]

# How long a structure must be at reduced health before triggering defense
# (avoids false positives from shield regen etc.)
DEFEND_HEALTH_THRESHOLD = 0.85   # trigger if health_pct drops below this
RALLY_INTERVAL = 30              # seconds between passive army rallies
DEFEND_RECHECK_INTERVAL = 5     # seconds between defense rechecks


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------

async def build_structure(bot: BotAI, building: UnitTypeId):
    """Helper to systematically build structures depending on the type."""

    if building == UnitTypeId.ASSIMILATOR:
        if bot.can_afford(UnitTypeId.ASSIMILATOR) and bot.townhalls:
            for vespene in bot.vespene_geyser.closer_than(15, bot.townhalls.first):
                if bot.gas_buildings.filter(lambda u: u.distance_to(vespene) < 1):
                    continue
                await bot.build(UnitTypeId.ASSIMILATOR, vespene)
                return

    elif building == UnitTypeId.PYLON:
        if bot.can_afford(UnitTypeId.PYLON) and bot.townhalls:
            if bot.structures(UnitTypeId.PYLON).amount == 0:
                townhall_pos = bot.townhalls.first.position
                map_center = bot.game_info.map_center
                direction = (map_center - townhall_pos).normalized
                target_pos = townhall_pos + direction * 5
                placement = await bot.find_placement(
                    UnitTypeId.PYLON, near=target_pos, placement_step=2)
            else:
                placement = await bot.find_placement(
                    UnitTypeId.PYLON,
                    near=bot.townhalls.first.position,
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
        if bot.can_afford(building) and bot.structures(UnitTypeId.PYLON).ready:
            placement = await bot.find_placement(
                building,
                near=bot.townhalls.first.position,
                placement_step=2)
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

# We track rally-set tags in a bot-level set so we only issue the command
# once per building (persistent across frames unlike hasattr on sc2 units).
def _get_rally_tag_set(bot: BotAI) -> set:
    if not hasattr(bot, "_rally_tags_set"):
        bot._rally_tags_set = set()
    return bot._rally_tags_set


async def set_production_rally_points(bot: BotAI):
    """Set rally points for production buildings to the army staging area."""
    if not bot.townhalls:
        return

    rally_point = bot.townhalls.center
    tag_set = _get_rally_tag_set(bot)

    for unit_type in PRODUCTION_BUILDINGS:
        for building in bot.structures(unit_type).ready:
            if building.tag not in tag_set:
                building(AbilityId.RALLY_UNITS, rally_point)
                tag_set.add(building.tag)
                print(f"[{bot.time:.0f}s] Rally point set for {unit_type.name} "
                      f"→ {rally_point}")


# ---------------------------------------------------------------------------
# Passive army rally (move idle troops to staging area)
# ---------------------------------------------------------------------------

async def rally_idle_army(bot: BotAI):
    """
    Every RALLY_INTERVAL seconds, move idle army units to the nexus center.
    Skipped once auto-attack is active so we don't interrupt attacks.
    """
    if getattr(bot, "_auto_attack_initiated", False):
        return

    if not hasattr(bot, "_last_rally_time"):
        bot._last_rally_time = 0.0

    if bot.time - bot._last_rally_time < RALLY_INTERVAL:
        return

    bot._last_rally_time = bot.time

    if not bot.townhalls:
        return

    staging = bot.townhalls.center
    army = bot.units.of_type(ARMY_TYPES)

    if not army:
        return

    # Use move (not attack-move) so units don't wander chasing targets
    for unit in army:
        unit.move(staging)

    print(f"[{bot.time:.0f}s] Rallying {army.amount} unit(s) → staging at {staging}")


# ---------------------------------------------------------------------------
# Defense
# ---------------------------------------------------------------------------

async def defend_structures(bot: BotAI):
    """
    Send army to defend if any structure is taking significant HP damage.
    Resets once no structures are under threat.
    Uses a cooldown so we don't spam commands every frame.
    """
    if getattr(bot, "_auto_attack_initiated", False):
        return

    if not hasattr(bot, "_last_defend_check"):
        bot._last_defend_check = 0.0
    if not hasattr(bot, "_defending"):
        bot._defending = False

    if bot.time - bot._last_defend_check < DEFEND_RECHECK_INTERVAL:
        return
    bot._last_defend_check = bot.time

    # Find structures that are meaningfully damaged (not just shield chip)
    under_attack = [
        s for s in bot.structures
        if s.health_percentage < DEFEND_HEALTH_THRESHOLD
    ]

    if under_attack:
        # Pick the most damaged structure as the rally point
        target = min(under_attack, key=lambda s: s.health_percentage)
        army = bot.units.of_type(ARMY_TYPES)

        if army and not bot._defending:
            print(f"[{bot.time:.0f}s] DEFEND: {len(under_attack)} structure(s) under attack. "
                  f"Sending {army.amount} unit(s) to defend {target.name} "
                  f"({target.health_percentage:.0%} HP) at {target.position}")
            bot._defending = True

        for unit in army:
            unit.attack(target.position)

    elif bot._defending:
        # Threat cleared — stand down and re-rally
        staging = bot.townhalls.center if bot.townhalls else None
        army = bot.units.of_type(ARMY_TYPES)
        if staging and army:
            for unit in army:
                unit.move(staging)
        bot._defending = False
        print(f"[{bot.time:.0f}s] DEFEND: Threat cleared. Returning {army.amount if army else 0} "
              f"unit(s) to staging.")


# ---------------------------------------------------------------------------
# Auto-attack
# ---------------------------------------------------------------------------

async def auto_attack(bot: BotAI):
    """
    Attack when army supply exceeds threshold or game time is long.
    Systematically walks enemy base locations and known expansion spots.
    """
    if not hasattr(bot, "_auto_attack_initiated"):
        bot._auto_attack_initiated = False
    if not hasattr(bot, "_cleared_bases"):
        bot._cleared_bases = set()
    if not hasattr(bot, "_last_attack_order_time"):
        bot._last_attack_order_time = 0.0

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
    army_supply = sum(
        bot.units(ut).amount * cost for ut, cost in army_types_supply
    )

    should_attack = army_supply >= 30 or bot.time >= 1680  # 28 min hard cap

    if not should_attack:
        return

    if not bot._auto_attack_initiated:
        bot._auto_attack_initiated = True
        print(f"[{bot.time:.0f}s] AUTO-ATTACK: Threshold reached "
              f"(army supply ≈ {army_supply}). Beginning attack run.")

    # Only re-issue attack orders every few seconds to avoid command spam
    if bot.time - bot._last_attack_order_time < 5.0:
        return
    bot._last_attack_order_time = bot.time

    army = bot.units.of_type(ARMY_TYPES)
    if not army:
        return

    # ---- Find best target ----
    # Priority 1: enemy structures we can currently see
    if bot.enemy_structures:
        target_pos = bot.enemy_structures.closest_to(
            bot.start_location).position
        _issue_attack(bot, army, target_pos, "visible enemy structure")
        return

    # Priority 2: enemy start locations not yet confirmed cleared
    for enemy_start in bot.enemy_start_locations:
        loc = Point2(enemy_start) if not isinstance(
            enemy_start, Point2) else enemy_start
        if loc in bot._cleared_bases:
            continue
        # Check if we have intel that it's empty
        if not bot.is_visible(loc):
            _issue_attack(bot, army, loc, "enemy start (scouting)")
            return
        # Visible and no structures → mark cleared
        if not bot.enemy_structures.closer_than(20, loc):
            bot._cleared_bases.add(loc)
            print(f"[{bot.time:.0f}s] AUTO-ATTACK: Marked {loc} as cleared.")
        else:
            _issue_attack(bot, army, loc, "enemy start (structures present)")
            return

    # Priority 3: expansion locations not cleared
    for expansion in bot.expansion_locations_list:
        if expansion in bot._cleared_bases:
            continue
        if bot.is_visible(expansion) and not bot.enemy_structures.closer_than(20, expansion):
            bot._cleared_bases.add(expansion)
            continue
        _issue_attack(bot, army, expansion, "expansion (clearing)")
        return

    # Everything appears cleared — reset and start over
    print(f"[{bot.time:.0f}s] AUTO-ATTACK: All known locations cleared, resetting.")
    bot._cleared_bases.clear()


def _issue_attack(bot: BotAI, army, target_pos: Point2, reason: str):
    """Issue an attack-move to all army units and log it."""
    print(f"[{bot.time:.0f}s] AUTO-ATTACK: {army.amount} unit(s) → {target_pos} ({reason})")
    for unit in army:
        unit.attack(target_pos)


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

    # Prefer pylon closest to map center so warped units are near the front
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
