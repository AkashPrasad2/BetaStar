"""
The single source of truth for the observation vector.

Both sides build observations by calling build_obs_vector():

    training  : replay_parser.py       (state reconstructed from a replay)
    inference : observation_wrapper.py (state queried from the live SC2 API)

Feature layout (76 total)
-------------------------
    [0]      game time (clipped, normalized by TIME_NORM)
    [1:6]    minerals one-hot (5 bins, edges 100/300/700/1500)
    [6]      minerals magnitude (sqrt-scaled, monotonic)
    [7:12]   vespene one-hot (5 bins, edges 25/100/250/600)
    [12]     vespene magnitude (sqrt-scaled, monotonic)
    [13]     supply used
    [14]     supply cap
    [15]     supply remaining (clipped 0..16, normalized by 16)
    [16]     supply blocked flag (1.0 when used >= cap)
    [17]     worker saturation
    [18:33]  completed structure counts   (15)
    [33:44]  completed unit counts        (11)
    [44:58]  pending structure counts     (14, excludes WARPGATE)
    [58:69]  pending unit counts          (11)
    [69]     idle gateway+warpgate count
    [70]     idle stargate count
    [71]     idle robotics facility count
    [72]     idle warpgate count
    [73]     ground weapons level
    [74]     shields level
    [75]     air weapons level
"""

from __future__ import annotations

from typing import Mapping, Sequence

STRUCTURES: list[str] = [
    "NEXUS", "PYLON", "GATEWAY", "FORGE", "TWILIGHTCOUNCIL", "PHOTONCANNON",
    "SHIELDBATTERY", "TEMPLARARCHIVE", "ROBOTICSBAY", "ROBOTICSFACILITY",
    "ASSIMILATOR", "CYBERNETICSCORE", "STARGATE", "FLEETBEACON", "WARPGATE",
]

UNITS: list[str] = [
    "PROBE", "ZEALOT", "STALKER", "HIGHTEMPLAR", "ARCHON", "IMMORTAL",
    "CARRIER", "VOIDRAY", "ADEPT", "PHOENIX", "COLOSSUS",
]

# A Warpgate is never "under construction" to the API, so no pending slot
PENDING_STRUCTURES: list[str] = STRUCTURES[:-1]
UPGRADE_KEYS: list[str] = ["GROUND_WEAPONS", "SHIELDS", "AIR_WEAPONS"]


# Action space
# every other module resolves ids through ACTION_ID or ACTION_NAMES
ACTION_NAMES: list[str] = [
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
    "build_robotics_bay",       # 14
    "build_shield_battery",     # 15
    "train_zealot",             # 16
    "train_stalker",            # 17
    "train_immortal",           # 18
    "train_voidray",            # 19
    "train_carrier",            # 20
    "warp_in_zealot",           # 21
    "warp_in_stalker",          # 22
    "warp_in_high_templar",     # 23
    "research_charge",          # 24
    "research_warp_gate",       # 25
    "upgrade_ground_weapons",   # 26
    "upgrade_air_weapons",      # 27
    "upgrade_shields",          # 28
    "train_adept",              # 29
    "train_phoenix",            # 30
    "train_colossus",           # 31
]

# name -> action id map
ACTION_ID: dict[str, int] = {n: i for i, n in enumerate(ACTION_NAMES)}
NUM_ACTIONS = len(ACTION_NAMES)

# Supply each action consumes (only applies to units)
# Lives here rather than in action_mask.py because three places need to
# agree on it: the strict inference mask, the relaxed training mask, and the
# parser's _action_legal_numpy label check
ACTION_SUPPLY_COST = {
    ACTION_ID["train_probe"]:          1,
    ACTION_ID["train_zealot"]:         2,
    ACTION_ID["train_stalker"]:        2,
    ACTION_ID["train_immortal"]:       4,
    ACTION_ID["train_voidray"]:        4,
    ACTION_ID["train_carrier"]:        6,
    ACTION_ID["train_adept"]:          2,
    ACTION_ID["train_phoenix"]:        2,
    ACTION_ID["train_colossus"]:       6,
    ACTION_ID["warp_in_zealot"]:       2,
    ACTION_ID["warp_in_stalker"]:      2,
    ACTION_ID["warp_in_high_templar"]: 2,
}

# Supply values are integers, so a 0.5 tolerance absorbs float round-trip error
# from the /200 normalization without ever changing a comparison.
SUPPLY_EPS = 0.5

# give some lenience due to the 4s window
TRAINING_SUPPLY_SLACK = 8.0

# Seconds of game time between policy decisions. The parser bins replays into
# windows of this width, and the live bot must query the model on the same schedule
DECISION_INTERVAL_SECONDS = 4

# Normalization constants: we aim to keep weights within [0,1]
TIME_NORM = 1800.0
SUPPLY_NORM = 200.0
STRUCT_NORM = 10.0
UNIT_NORM = 30.0
IDLE_NORM = 5.0
UPGRADE_NORM = 3.0
WORKERS_PER_BASE = 22

# one-hot minerals and gas since training/inference is off by too much otherwise and the signal gets weak
# given the rate of game state checks, the resources can be stale
MINERAL_BIN_EDGES = (100.0, 300.0, 700.0, 1500.0)
GAS_BIN_EDGES = (25.0, 100.0, 250.0, 600.0)

# derive bins from edges for robustness
N_MINERAL_BINS = len(MINERAL_BIN_EDGES) + 1
N_GAS_BINS = len(GAS_BIN_EDGES) + 1

# also use continuous versions of resources, 4000 chosen based on observed data
RESOURCE_MAG_NORM = 4000.0
SUPPLY_REMAINING_NORM = 16.0

# observation index contants
IDX_TIME = 0
IDX_MINERAL_BINS = 1                                   # 1
IDX_MINERAL_MAG = IDX_MINERAL_BINS + N_MINERAL_BINS    # 6
IDX_GAS_BINS = IDX_MINERAL_MAG + 1                     # 7
IDX_GAS_MAG = IDX_GAS_BINS + N_GAS_BINS                # 12
IDX_SUPPLY_USED = IDX_GAS_MAG + 1                      # 13
IDX_SUPPLY_CAP = IDX_SUPPLY_USED + 1                   # 14
IDX_SUPPLY_REMAINING = IDX_SUPPLY_CAP + 1              # 15
IDX_SUPPLY_BLOCKED = IDX_SUPPLY_REMAINING + 1          # 16
IDX_WORKER_SAT = IDX_SUPPLY_BLOCKED + 1                # 17
IDX_STRUCT_BASE = IDX_WORKER_SAT + 1                   # 18
IDX_UNIT_BASE = IDX_STRUCT_BASE + len(STRUCTURES)            # 33
IDX_PEND_STRUCT_BASE = IDX_UNIT_BASE + len(UNITS)            # 44
IDX_PEND_UNIT_BASE = (IDX_PEND_STRUCT_BASE
                      + len(PENDING_STRUCTURES))             # 58
IDX_IDLE_GW_WG = IDX_PEND_UNIT_BASE + len(UNITS)             # 69
IDX_IDLE_SG = IDX_IDLE_GW_WG + 1                             # 70
IDX_IDLE_ROBO = IDX_IDLE_SG + 1                              # 71
IDX_IDLE_WG = IDX_IDLE_ROBO + 1                              # 72
IDX_GROUND_WEAPONS_LVL = IDX_IDLE_WG + 1                     # 73
IDX_SHIELDS_LVL = IDX_GROUND_WEAPONS_LVL + 1                 # 74
IDX_AIR_WEAPONS_LVL = IDX_SHIELDS_LVL + 1                    # 75
OBS_SIZE = IDX_AIR_WEAPONS_LVL + 1                           # 76

# Name -> feature index. For callers that need a specific feature.
STRUCT_IDX = {n: IDX_STRUCT_BASE + i for i, n in enumerate(STRUCTURES)}
UNIT_IDX = {n: IDX_UNIT_BASE + i for i, n in enumerate(UNITS)}
PEND_STRUCT_IDX = {n: IDX_PEND_STRUCT_BASE + i
                   for i, n in enumerate(PENDING_STRUCTURES)}
PEND_UNIT_IDX = {n: IDX_PEND_UNIT_BASE + i for i, n in enumerate(UNITS)}

# Production building -> units trainable from it
GATEWAY_UNITS = ("ZEALOT", "STALKER", "HIGHTEMPLAR", "ADEPT")
STARGATE_UNITS = ("VOIDRAY", "CARRIER", "PHOENIX")
ROBOTICS_UNITS = ("IMMORTAL", "COLOSSUS")


def _one_hot_bins(value: float, edges: Sequence[float]) -> list[float]:
    """Return a one-hot over len(edges)+1 bins, bin i = value < edges[i]."""
    out = [0.0] * (len(edges) + 1)
    slot = len(edges)
    for i, edge in enumerate(edges):
        if value < edge:
            slot = i
            break
    out[slot] = 1.0
    return out


def _magnitude(value: float, cap: float = RESOURCE_MAG_NORM) -> float:
    """
    Monotonic, order-preserving magnitude in [0, 1], sqrt-scaled.
    Unlike the one-hot bins this never saturates within the cap, so the model can
    distinguish 600 minerals from 3000
    """
    return min(max(value, 0.0) / cap, 1.0) ** 0.5


def compute_idle_counts(
    structures_done: Mapping[str, float],
    units_pending: Mapping[str, float],
) -> tuple[float, float, float, float]:
    """return number of idle buildings (not training anything)"""

    gw = structures_done.get("GATEWAY", 0)
    wg = structures_done.get("WARPGATE", 0)
    gw_wg_busy = sum(units_pending.get(u, 0) for u in GATEWAY_UNITS)
    idle_gw_wg = max(0, (gw + wg) - gw_wg_busy)

    sg_busy = sum(units_pending.get(u, 0) for u in STARGATE_UNITS)
    idle_sg = max(0, structures_done.get("STARGATE", 0) - sg_busy)

    robo_busy = sum(units_pending.get(u, 0) for u in ROBOTICS_UNITS)
    idle_robo = max(0, structures_done.get("ROBOTICSFACILITY", 0) - robo_busy)

    # Warpgates specifically: gateway-type production beyond what the plain
    # Gateways can absorb is assumed to be occupying Warpgates.
    idle_wg = max(0, wg - max(0, gw_wg_busy - gw))

    return idle_gw_wg, idle_sg, idle_robo, idle_wg


def build_obs_vector(
    *,
    time_s: float,
    minerals: float,
    vespene: float,
    supply_used: float,
    supply_cap: float,
    structures_done: Mapping[str, float],
    units_done: Mapping[str, float],
    structures_pending: Mapping[str, float],
    units_pending: Mapping[str, float],
    upgrade_lvls: Mapping[str, float],
) -> list[float]:
    """
    Build the observation vector from raw game-state values.

    All arguments are un-normalized.
    Keyword-only to prevent positional mix-ups between the two call sites.
    """
    obs: list[float] = []

    # [0] time — clipped so the feature can never leave [0, 1]
    obs.append(min(max(time_s, 0.0) / TIME_NORM, 1.0))

    # minerals: one-hot affordability bins + magnitude
    obs.extend(_one_hot_bins(minerals, MINERAL_BIN_EDGES))
    obs.append(_magnitude(minerals))
    # gas: same
    obs.extend(_one_hot_bins(vespene, GAS_BIN_EDGES))
    obs.append(_magnitude(vespene))

    # supply _ remaining supply. Derived here rather than taken as an
    # argument so the parser and the live wrapper cannot disagree about it.
    obs.append(supply_used / SUPPLY_NORM)
    obs.append(supply_cap / SUPPLY_NORM)
    remaining = supply_cap - supply_used
    obs.append(min(max(remaining, 0.0), SUPPLY_REMAINING_NORM)
               / SUPPLY_REMAINING_NORM)
    obs.append(1.0 if remaining <= 0.0 else 0.0)

    # [11] worker saturation
    nexus_count = structures_done.get("NEXUS", 0)
    ideal_workers = max(nexus_count, 1) * WORKERS_PER_BASE
    obs.append(units_done.get("PROBE", 0) / ideal_workers)

    # [12:27] completed structures
    for name in STRUCTURES:
        obs.append(structures_done.get(name, 0) / STRUCT_NORM)

    # [27:38] completed units
    for name in UNITS:
        obs.append(units_done.get(name, 0) / UNIT_NORM)

    # [38:52] pending structures
    for name in PENDING_STRUCTURES:
        obs.append(structures_pending.get(name, 0) / STRUCT_NORM)

    # [52:63] pending units
    for name in UNITS:
        obs.append(units_pending.get(name, 0) / UNIT_NORM)

    # [63:67] idle production buildings
    idle_gw_wg, idle_sg, idle_robo, idle_wg = compute_idle_counts(
        structures_done, units_pending)
    obs.append(idle_gw_wg / IDLE_NORM)
    obs.append(idle_sg / IDLE_NORM)
    obs.append(idle_robo / IDLE_NORM)
    obs.append(idle_wg / IDLE_NORM)

    # [67:70] upgrade levels
    for key in UPGRADE_KEYS:
        obs.append(upgrade_lvls.get(key, 0) / UPGRADE_NORM)

    if len(obs) != OBS_SIZE:
        raise AssertionError(
            f"obs size mismatch: built {len(obs)}, expected {OBS_SIZE}")
    return obs


def feature_names() -> list[str]:
    """Human-readable name per index (for logging output)"""
    names = ["time_norm"]
    names += [f"minerals_bin{i}" for i in range(N_MINERAL_BINS)]
    names += ["minerals_mag"]
    names += [f"gas_bin{i}" for i in range(N_GAS_BINS)]
    names += ["gas_mag"]
    names += ["supply_used", "supply_cap", "supply_remaining",
              "supply_blocked", "worker_sat"]
    names += [f"struct_{s}" for s in STRUCTURES]
    names += [f"unit_{u}" for u in UNITS]
    names += [f"pend_struct_{s}" for s in PENDING_STRUCTURES]
    names += [f"pend_unit_{u}" for u in UNITS]
    names += ["idle_gw_wg", "idle_sg", "idle_robo", "idle_wg"]
    names += ["ground_weapons_lvl", "shields_lvl", "air_weapons_lvl"]
    if len(names) != OBS_SIZE:
        raise AssertionError(
            f"feature name count {len(names)} != OBS_SIZE {OBS_SIZE}")
    return names
