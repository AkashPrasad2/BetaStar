from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
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
# Bounds how often a merge can be re-commanded for the same templar. Defensive:
# if the game ever rejects a merge, this caps the retry rate instead of letting
# it fire on every on_step.
ARCHON_MERGE_RETRY_SECONDS = 5.0


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

# Max number of each structure type allowed to be in flight (under construction
# OR with a worker walking to the site) at any one time.
#
# Why this is needed: the model re-decides every few seconds, but the pending-
# structure feature it sees comes from `not_ready.amount`, which only counts
# structures that physically EXIST and are being built. Between issuing the build
# order and the worker arriving at the site, nothing exists yet -- so the
# observation is identical to "nothing is happening" and the policy samples the
# same build again. And again. The model also gets no action history in its
# input, so it cannot know it just ordered one. Result: a stream of pylons until
# the first one finally breaks ground, then the same for gateways, etc.
#
# already_pending() counts worker-en-route builds too, so it is the correct
# signal for suppressing duplicates at the execution layer. Caps above 1 are
# allowed where pros genuinely build in parallel.
MAX_CONCURRENT_BUILDS = {
    UnitTypeId.PYLON:       2,
    UnitTypeId.ASSIMILATOR: 2,
    UnitTypeId.GATEWAY:     2,
}
DEFAULT_MAX_CONCURRENT_BUILDS = 1

# How far from a townhall a vespene geyser is considered to belong to that base.
GEYSER_SEARCH_RADIUS = 15


async def build_structure(bot: BotAI, building: UnitTypeId):
    """Helper to systematically build structures depending on the type."""

    # Suppress duplicate builds already in flight (see MAX_CONCURRENT_BUILDS).
    cap = MAX_CONCURRENT_BUILDS.get(building, DEFAULT_MAX_CONCURRENT_BUILDS)
    if bot.already_pending(building) >= cap:
        return

    starting_nexus = bot.townhalls.closest_to(
        bot.start_location) if bot.townhalls else None

    if building == UnitTypeId.ASSIMILATOR:
        if not bot.can_afford(UnitTypeId.ASSIMILATOR):
            return
        # Check every base, not just the main. This used to search only
        # `starting_nexus` (the townhall closest to start_location), so once the
        # two main geysers were taken the loop matched nothing and returned
        # silently -- expansions never got gas no matter what the model chose.
        # Nearest-to-start ordering keeps the main saturated first.
        for townhall in bot.townhalls.sorted(
                lambda th: th.distance_to(bot.start_location)):
            for vespene in bot.vespene_geyser.closer_than(
                    GEYSER_SEARCH_RADIUS, townhall):
                if bot.gas_buildings.filter(
                        lambda u, v=vespene: u.distance_to(v) < 1):
                    continue      # already has an assimilator
                await bot.build(UnitTypeId.ASSIMILATOR, vespene)
                return
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
    """
    Merge pairs of High Templars into Archons.

    MORPH_ARCHON is a NO-TARGET ability issued to a *group* of templars: the raw
    API expects a single action with ability_id=MORPH_ARCHON carrying the tags of
    both units. The previous version passed the second templar as a target
    (`ht1(MORPH_ARCHON, ht2)`), which the game rejects.

    That rejection was self-sustaining: because the command never landed, no
    templar ever acquired a MORPH_ARCHON order, so the "already merging" guard
    never tripped and this ran again on every on_step -- console spam while the
    templars stood still.

    burnysc2 combines UnitCommands sharing (ability, target, queue) into one raw
    action, so issuing the no-target ability to both templars in the same step
    produces exactly the action the game expects.
    """
    hts = bot.units(UnitTypeId.HIGHTEMPLAR).ready
    if hts.amount < 2:
        return

    now = bot.time
    recent = bot.archon_merge_issued

    # Expire old entries so tags cannot leak for the rest of the game.
    for tag in [t for t, when in recent.items()
                if now - when > ARCHON_MERGE_RETRY_SECONDS]:
        del recent[tag]

    candidates = []
    for ht in hts:
        if ht.tag in recent:
            continue                      # commanded very recently
        if any(order.ability.id == AbilityId.MORPH_ARCHON
               for order in ht.orders):
            continue                      # already merging
        candidates.append(ht)

    if len(candidates) < 2:
        return

    ht1, ht2 = candidates[0], candidates[1]

    # Confirm the game actually offers the merge right now. Without this, any
    # future rejection would silently become a retry loop again.
    abilities = await bot.get_available_abilities([ht1, ht2])
    if not all(AbilityId.MORPH_ARCHON in available for available in abilities):
        return

    # No target — issued to both templars, combined into one action.
    ht1(AbilityId.MORPH_ARCHON)
    ht2(AbilityId.MORPH_ARCHON)

    recent[ht1.tag] = now
    recent[ht2.tag] = now
    print(f"[{now:.0f}s] AUTO-MERGE: merging 2 high templars into an archon")

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


# ---------------------------------------------------------------------------
# Chrono boost helper
# ---------------------------------------------------------------------------

async def chrono_boost_production(bot: BotAI):
    """
    Automatically chrono boost a production structure if it is currently building
    something. Priority order: robo facility, then stargate, then warpgate.

    Note: has_buff() takes a BuffId, NOT an AbilityId. Passing an AbilityId
    raises inside burnysc2 (it asserts on the type), which crashed the game the
    first time any production building was busy. Because Robotics Facility is
    checked first, the crash typically surfaced at the Stargate in games with no
    robo. The cast ability is an AbilityId; only the buff lookup takes a BuffId.
    """
    CHRONO_BUFF = BuffId.CHRONOBOOSTENERGYCOST
    CHRONO_ABILITY = AbilityId.EFFECT_CHRONOBOOSTENERGYCOST
    CHRONO_ENERGY_COST = 50

    priority = (
        UnitTypeId.ROBOTICSFACILITY,
        UnitTypeId.STARGATE,
        UnitTypeId.WARPGATE,
    )

    for nexus in bot.townhalls.ready:
        if nexus.energy < CHRONO_ENERGY_COST:
            continue

        for structure_type in priority:
            for structure in bot.structures(structure_type).ready:
                if structure.is_idle:
                    continue
                if structure.has_buff(CHRONO_BUFF):
                    continue
                nexus(CHRONO_ABILITY, structure)
                return
