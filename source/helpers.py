from __future__ import annotations

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.position import Point2
from enum import Enum
import logging
import math
import random


logger = logging.getLogger(__name__)

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

# trigger defense if:
#   - a structure lost health/shields since the last step (recent damage), or
#   - a visible enemy combat unit is near one of our structures.

DEFEND_DAMAGE_MEMORY = 12.0   # seconds a damage event keeps us in DEFEND
DEFEND_ENEMY_RADIUS = 25.0    # enemy within this of a structure = threat
MIN_STATE_DWELL = 6.0         # min seconds in DEFEND/ATTACK before switching out
DAMAGE_EPSILON = 1.0          # ignore sub-unit jitter in hp+shield readings

ATTACK_SUPPLY_THRESHOLD = 70
ATTACK_TIME_CAP = 1680          # hard attack at 28 min regardless of supply

RALLY_INTERVAL = 30
# seconds between re-issuing army orders (avoid spam)
ARMY_COMMAND_INTERVAL = 5
ARCHON_MERGE_RETRY_SECONDS = 5.0

# Use state machine to manage army


class ArmyState(Enum):
    RALLY = "RALLY"
    DEFEND = "DEFEND"
    ATTACK = "ATTACK"


class ActionResult(Enum):
    """Why an execution attempt did or did not happen."""

    ISSUED = "issued"               # order successfully sent to the game
    SUPPRESSED = "suppressed"      # refused on purpose (MAX_CONCURRENT_BUILDS)
    UNAFFORDABLE = "unaffordable"   # not enough minerals/gas
    NO_PLACEMENT = "no_placement"   # find_placement found nowhere legal
    NO_WORKER = "no_worker"         # no worker free to build
    NO_PREREQ = "no_prereq"         # missing powered pylon / townhall / tech
    NO_TARGET = "no_target"         # no expansion site or free geyser
    NO_PRODUCTION = "no_production"  # no idle production building
    NO_OP = "no_op"                 # intentional do_nothing action
    NOT_LABELLED = "not_labelled"   # path not yet instrumented


# probe mutex: due to auto_saturate and other probe management code,
# we lock a probe for a set amount of time so actions do not silently fail due to this probe being reassigned an action
WORKER_RESERVATION_SECONDS = 20.0

# Warp-in placement search around the chosen pylon.
WARP_PLACEMENT_RADIUS = 5.0
WARP_PLACEMENT_ATTEMPTS = 20


def _reservations(bot: BotAI) -> dict:
    """Reserved worker tags -> game time the reservation expires. Self-pruning."""
    if not hasattr(bot, "reserved_workers"):
        bot.reserved_workers = {}
    expired = [tag for tag, until in bot.reserved_workers.items()
               if until <= bot.time]
    for tag in expired:
        del bot.reserved_workers[tag]
    return bot.reserved_workers


def reserve_worker(bot: BotAI, tag: int,
                   seconds: float = WORKER_RESERVATION_SECONDS):
    _reservations(bot)[tag] = bot.time + seconds


def release_worker(bot: BotAI, tag: int):
    _reservations(bot).pop(tag, None)


def _worker_is_available(bot: BotAI, worker, reserved: dict) -> bool:
    """
    A worker we may command. `is_idle or is_collecting` is the key test
    """
    if worker.tag in reserved:
        return False
    if worker.tag in bot.unit_tags_received_action:
        return False   # already given an order this step
    return worker.is_idle or worker.is_collecting


def pick_builder(bot: BotAI, near):
    """
    Closest available worker to `near`, preferring one that is actively
    harvesting (proven able to move) over one that is merely idle.
    """
    reserved = _reservations(bot)
    pool = [w for w in bot.workers if _worker_is_available(bot, w, reserved)]
    if not pool:
        return None
    harvesting = [w for w in pool if w.is_collecting]
    return min(harvesting or pool, key=lambda w: w.distance_to(near))


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


def _ensure_threat_state(bot: BotAI):
    """Lazily create the threat-tracking attributes (safe if __init__ predates them)."""
    if not hasattr(bot, "structure_hp_snapshot"):
        bot.structure_hp_snapshot = {}       # tag -> (hp+shield, position)
        bot.last_damage_time = -1.0e9
        bot.last_damage_pos = None
        bot.threat_position = None
    if not hasattr(bot, "army_state_since"):
        bot.army_state_since = 0.0


def _detect_damage(bot: BotAI):
    """
    Compare this step's structure hp+shield against the previous step and record
    the position of the worst loss. A structure that disappeared entirely counts
    as damage at its last known position.
    """
    _ensure_threat_state(bot)
    snapshot = bot.structure_hp_snapshot
    worst_pos = None
    worst_loss = DAMAGE_EPSILON

    seen = set()
    for structure in bot.structures:
        seen.add(structure.tag)
        current = structure.health + structure.shield
        previous = snapshot.get(structure.tag)
        snapshot[structure.tag] = (current, structure.position)
        if previous is None:
            continue
        loss = previous[0] - current
        if loss > worst_loss:
            worst_loss = loss
            worst_pos = structure.position

    # Structures that vanished were destroyed -- treat as damage where they stood.
    for tag in [t for t in snapshot if t not in seen]:
        _, position = snapshot.pop(tag)
        if worst_pos is None:
            worst_pos = position

    if worst_pos is not None:
        bot.last_damage_time = bot.time
        bot.last_damage_pos = worst_pos


def _current_threat(bot: BotAI) -> tuple[bool, object | None]:
    """
    (threatened, position_to_defend).

    Both conditions are transient, so DEFEND always ends: enemies leave or die,
    and the damage memory expires.
    """
    _ensure_threat_state(bot)

    structures = bot.structures.ready
    if structures:
        for enemy in bot.enemy_units:
            if getattr(enemy, "is_structure", False):
                continue
            if getattr(enemy, "is_hallucination", False):
                continue
            for structure in structures:
                if enemy.position.distance_to(structure.position) < DEFEND_ENEMY_RADIUS:
                    return True, enemy.position

    if bot.time - bot.last_damage_time < DEFEND_DAMAGE_MEMORY:
        fallback = bot.last_damage_pos
        if fallback is None and bot.townhalls:
            fallback = bot.townhalls.center
        return True, fallback

    return False, None


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

# Max number of each structure type allowed to be in flight (under construction
# or with a worker walking to the site) at any one time.
MAX_CONCURRENT_BUILDS = {
    UnitTypeId.PYLON:       2,
    UnitTypeId.GATEWAY:     6,
    UnitTypeId.STARGATE:    4
}
DEFAULT_MAX_CONCURRENT_BUILDS = 1

# How far from a townhall a vespene geyser is considered to belong to that base.
GEYSER_SEARCH_RADIUS = 15


# Placement search
PLACEMENT_STEP = 2
PLACEMENT_RADIUS = 12          # search radius per anchor
MAX_PLACEMENT_ANCHORS = 5      # bounds placement queries per decision
PYLON_PLACEMENT_STEP = 8

# Structures that belong on the outside of the base.
FORWARD_STRUCTURES = {UnitTypeId.SHIELDBATTERY}
FORWARD_NUDGE = 3.0            # units toward the enemy from the anchor pylon


async def _find_placement_near_any(bot: BotAI, building: UnitTypeId,
                                   anchors: list, step: int = PLACEMENT_STEP):
    """First legal placement found around any anchor, best anchor first."""
    for anchor in anchors[:MAX_PLACEMENT_ANCHORS]:
        placement = await bot.find_placement(
            building, near=anchor, placement_step=step,
            max_distance=PLACEMENT_RADIUS)
        if placement:
            return placement
    return None


def _enemy_anchor(bot: BotAI):
    """The point to face when placing forward structures."""
    if bot.enemy_start_locations:
        return bot.enemy_start_locations[0]
    return bot.game_info.map_center


def _powered_anchors(bot: BotAI, forward: bool = False) -> list:
    """
    Anchors for buildings needing power from a pylon
    """
    if forward:
        target = _enemy_anchor(bot)
        pylons = sorted(bot.structures(UnitTypeId.PYLON).ready,
                        key=lambda p: p.distance_to(target))
        anchors = []
        for pylon in pylons:
            direction = (target - pylon.position).normalized
            anchors.append(pylon.position + direction * FORWARD_NUDGE)
        anchors += [th.position for th in sorted(
            bot.townhalls.ready, key=lambda th: th.distance_to(target))]
        return anchors

    pylons = sorted(bot.structures(UnitTypeId.PYLON).ready,
                    key=lambda p: p.distance_to(bot.start_location))
    return ([p.position for p in pylons]
            + [th.position for th in bot.townhalls.ready])


async def build_structure(bot: BotAI, building: UnitTypeId) -> ActionResult:
    """
    Try to build, returning an ActionResult describing what happened.
    """
    cap = MAX_CONCURRENT_BUILDS.get(building, DEFAULT_MAX_CONCURRENT_BUILDS)
    if bot.already_pending(building) >= cap:
        return ActionResult.SUPPRESSED

    if not bot.can_afford(building):
        return ActionResult.UNAFFORDABLE

    if not bot.townhalls:
        return ActionResult.NO_PREREQ

    starting_nexus = bot.townhalls.closest_to(bot.start_location)

    # Assimilator: any free geyser at any base
    if building == UnitTypeId.ASSIMILATOR:
        for townhall in sorted(bot.townhalls,
                               key=lambda th: th.distance_to(bot.start_location)):
            for geyser in bot.vespene_geyser.closer_than(
                    GEYSER_SEARCH_RADIUS, townhall):
                if bot.gas_buildings.filter(
                        lambda u, g=geyser: u.distance_to(g) < 1):
                    continue      # already has an assimilator
                worker = pick_builder(bot, geyser)
                if worker is None:
                    return ActionResult.NO_WORKER
                worker.build(building, geyser)
                reserve_worker(bot, worker.tag)
                return ActionResult.ISSUED
        return ActionResult.NO_TARGET

    # Nexus expo
    if building == UnitTypeId.NEXUS:
        location = await bot.get_next_expansion()
        if not location:
            return ActionResult.NO_TARGET
        worker = pick_builder(bot, location)
        if worker is None:
            return ActionResult.NO_WORKER
        worker.build(building, location)
        reserve_worker(bot, worker.tag)
        return ActionResult.ISSUED

    # Pylon: spread out for power coverage
    if building == UnitTypeId.PYLON:
        if not bot.structures(UnitTypeId.PYLON):
            direction = (bot.game_info.map_center
                         - starting_nexus.position).normalized
            anchors = [starting_nexus.position + direction * 5]
        else:
            anchors = [th.position for th in sorted(
                bot.townhalls.ready,
                key=lambda th: th.distance_to(bot.start_location))]
            anchors += [p.position
                        for p in bot.structures(UnitTypeId.PYLON).ready]
        placement = await _find_placement_near_any(
            bot, building, anchors, step=PYLON_PLACEMENT_STEP)
        if not placement:
            return ActionResult.NO_PLACEMENT
        worker = pick_builder(bot, placement)
        if worker is None:
            return ActionResult.NO_WORKER
        worker.build(building, placement)
        reserve_worker(bot, worker.tag)
        return ActionResult.ISSUED

    # Everything else (pylon needed for power)
    if not bot.structures(UnitTypeId.PYLON).ready:
        return ActionResult.NO_PREREQ

    placement = await _find_placement_near_any(
        bot, building, _powered_anchors(
            bot, forward=building in FORWARD_STRUCTURES))
    if not placement:
        return ActionResult.NO_PLACEMENT

    worker = pick_builder(bot, placement)
    if worker is None:
        return ActionResult.NO_WORKER

    worker.build(building, placement)
    reserve_worker(bot, worker.tag)
    return ActionResult.ISSUED


# ---------------------------------------------------------------------------
# Worker saturation
# ---------------------------------------------------------------------------

async def auto_saturate_assimilators(bot: BotAI):
    """
    Assign workers to under-staffed assimilators.

    Only considers workers that are idle or collecting (never a builder),
    skips reserved workers, and skips workers already assigned to that geyser so
    it stops re-issuing the same order several times a second.
    """
    reserved = _reservations(bot)
    for assimilator in bot.structures(UnitTypeId.ASSIMILATOR).ready:
        if assimilator.assigned_harvesters >= 3:
            continue
        candidates = [
            w for w in bot.workers
            if _worker_is_available(bot, w, reserved)
            and getattr(w, "order_target", None) != assimilator.tag
        ]
        if not candidates:
            continue
        probe = min(candidates, key=lambda w: w.distance_to(assimilator))
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
                logger.debug(
                    "t=%.0fs rally set | building=%s target=%s",
                    bot.time, unit_type.name, rally_point,
                )


# ---------------------------------------------------------------------------
# Army management
# ---------------------------------------------------------------------------

async def manage_army(bot: BotAI):
    """
    Single entry point for all army behaviour. Evaluates state transitions
    first, then issues exactly one set of orders based on the current state.

    State machine:
        RALLY  →  DEFEND  if a base threat is detected (recent damage, or an
                          enemy near our structures)
        DEFEND →  RALLY   once the threat is gone and MIN_STATE_DWELL passed
        RALLY  →  ATTACK  at ATTACK_SUPPLY_THRESHOLD army supply, or the time cap
        ATTACK →  (none)  absorbing: we push until everything is dead or we are.
                          Base threats do not recall the army.
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


def _set_army_state(bot: BotAI, new_state: "ArmyState", reason: str):
    """
    Set army to desired state and log
    """
    if bot.army_state == new_state:
        return
    logger.info(
        "t=%.0fs army state | %s -> %s | %s",
        bot.time, bot.army_state.value, new_state.value, reason,
    )
    bot.army_state = new_state
    bot.army_state_since = bot.time


def _transition_state(bot: BotAI):
    """
    Evaluate and apply state transitions. Called every frame so reactions
    are immediate even if unit commands are throttled.
    """
    _ensure_threat_state(bot)
    _detect_damage(bot)

    supply = get_army_supply(bot)
    threatened, threat_pos = _current_threat(bot)
    bot.threat_position = threat_pos
    dwell = bot.time - bot.army_state_since

    if bot.army_state == ArmyState.DEFEND:
        # Leaving DEFEND requires the threat to be gone AND a minimum dwell, so
        # the army does not ping-pong on a single stray scout.
        if not threatened and dwell >= MIN_STATE_DWELL:
            _set_army_state(bot, ArmyState.RALLY, "threat cleared")
        return

    if bot.army_state == ArmyState.ATTACK:
        # Once commited to attack, do not turn back. Either everything dies or we win
        return

    # A live threat interrupts rallying.
    if threatened:
        _set_army_state(bot, ArmyState.DEFEND, "base under threat")
        return

    if supply >= ATTACK_SUPPLY_THRESHOLD:
        _set_army_state(bot, ArmyState.ATTACK, f"army supply {supply}")
    elif bot.time >= ATTACK_TIME_CAP:
        _set_army_state(bot, ArmyState.ATTACK, "attack time cap reached")


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
        logger.debug(
            "t=%.0fs rallying %d idle army units to %s",
            bot.time, len(idle_army), staging,
        )


def _do_defend(bot: BotAI, army):
    """
    Send the army to the current threat: the nearest enemy to our base, or the
    site of the most recent damage. _transition_state has already computed it.
    """
    target = getattr(bot, "threat_position", None)
    if target is None and bot.townhalls:
        target = bot.townhalls.center
    if target is None:
        return  # nothing left to defend
    _issue_attack(bot, army, target, "defending")


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
            logger.info("t=%.0fs enemy start cleared | %s", bot.time, loc)
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
    logger.info("t=%.0fs all known enemy locations cleared; resetting", bot.time)
    bot.enemy_bases_cleared.clear()


def _issue_attack(bot: BotAI, army, target_pos: Point2, reason: str):
    """Issue attack-move to all army units."""
    logger.debug(
        "t=%.0fs army command | state=%s units=%d target=%s reason=%s",
        bot.time, bot.army_state.value, army.amount, target_pos, reason,
    )
    for unit in army:
        unit.attack(target_pos)


# ---------------------------------------------------------------------------
# Auto-merge High Templars into Archons
# ---------------------------------------------------------------------------

async def auto_merge_archons(bot: BotAI):
    """
    Merge pairs of High Templars into Archons.
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
    # future rejection would silently fail
    abilities = await bot.get_available_abilities([ht1, ht2])
    if not all(AbilityId.MORPH_ARCHON in available for available in abilities):
        return

    # No target: issued to both templars, combined into one action.
    ht1(AbilityId.MORPH_ARCHON)
    ht2(AbilityId.MORPH_ARCHON)

    recent[ht1.tag] = now
    recent[ht2.tag] = now
    logger.debug("t=%.0fs merging two high templars into an archon", now)

# ---------------------------------------------------------------------------
# Warp-in helper
# ---------------------------------------------------------------------------


async def warp_in_unit(bot: BotAI, unit_type: UnitTypeId,
                       ability_id: AbilityId,
                       requires: UnitTypeId | None = None) -> ActionResult:
    """
    Warp a unit in near a pylon. Returns why it did or did not happen.

    NO_PRODUCTION here means every warpgate is on warp cooldown
    """
    if requires is not None and not bot.structures(requires).ready:
        return ActionResult.NO_PREREQ

    if not bot.can_afford(unit_type):
        return ActionResult.UNAFFORDABLE

    warpgates = bot.structures(UnitTypeId.WARPGATE).ready
    if not warpgates:
        return ActionResult.NO_PREREQ

    pylons = bot.structures(UnitTypeId.PYLON).ready
    if not pylons:
        return ActionResult.NO_PREREQ

    # Any warpgate that is off cooldown will do.
    ready_gate = None
    for warpgate in warpgates:
        abilities = await bot.get_available_abilities(warpgate)
        if ability_id in abilities:
            ready_gate = warpgate
            break
    if ready_gate is None:
        return ActionResult.NO_PRODUCTION      # all warpgates on cooldown

    pylon = pylons.closest_to(bot.game_info.map_center)
    for _ in range(WARP_PLACEMENT_ATTEMPTS):
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(1.5, WARP_PLACEMENT_RADIUS)
        offset = Point2((distance * math.cos(angle),
                         distance * math.sin(angle)))
        placement = await bot.find_placement(
            ability_id, pylon.position + offset,
            max_distance=2, placement_step=1)
        if placement:
            ready_gate.warp_in(unit_type, placement)
            return ActionResult.ISSUED

    # Last resort: try warping onto the pylon itself
    ready_gate.warp_in(unit_type, pylon.position)
    return ActionResult.ISSUED


# ---------------------------------------------------------------------------
# Chrono boost helper
# ---------------------------------------------------------------------------

async def chrono_boost_production(bot: BotAI):
    """
    Automatically chrono boost a production structure if it is currently building
    something. Priority order: robo facility, then stargate, then warpgate.
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
