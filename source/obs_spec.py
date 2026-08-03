"""
obs_spec.py — canonical definition of the observation vector
============================================================
This module is the SINGLE source of truth for the 70-dim observation.

Both sides of the pipeline build observations by calling build_obs_vector():

    training  : replay_parser.py       (state reconstructed from a replay)
    inference : observation_wrapper.py (state queried from the live SC2 API)

Previously each side had its own independent implementation of the layout,
binning and normalization. They disagreed, which meant the model saw a
different input distribution at inference than it was trained on. Routing both
through this module makes the transformation identical by construction; the
only remaining difference is the accuracy of the raw inputs handed in.

Feature layout (70 total)
-------------------------
    [0]      game time (clipped, normalized by TIME_NORM)
    [1:5]    minerals one-hot (4 bins)
    [5:9]    vespene one-hot (4 bins)
    [9]      supply used
    [10]     supply cap
    [11]     worker saturation
    [12:27]  completed structure counts   (15)
    [27:38]  completed unit counts        (11)
    [38:52]  pending structure counts     (14, excludes WARPGATE)
    [52:63]  pending unit counts          (11)
    [63]     idle gateway+warpgate count
    [64]     idle stargate count
    [65]     idle robotics facility count
    [66]     idle warpgate count
    [67]     ground weapons level
    [68]     shields level
    [69]     air weapons level
"""

from __future__ import annotations

from typing import Mapping, Sequence

# ---------------------------------------------------------------------------
# Entity vocabularies — order defines the feature layout
# ---------------------------------------------------------------------------

STRUCTURES: list[str] = [
    "NEXUS", "PYLON", "GATEWAY", "FORGE", "TWILIGHTCOUNCIL", "PHOTONCANNON",
    "SHIELDBATTERY", "TEMPLARARCHIVE", "ROBOTICSBAY", "ROBOTICSFACILITY",
    "ASSIMILATOR", "CYBERNETICSCORE", "STARGATE", "FLEETBEACON", "WARPGATE",
]

UNITS: list[str] = [
    "PROBE", "ZEALOT", "STALKER", "HIGHTEMPLAR", "ARCHON", "IMMORTAL",
    "CARRIER", "VOIDRAY", "ADEPT", "PHOENIX", "COLOSSUS",
]

# A Warpgate is never "under construction" — it is a morph of a finished
# Gateway — so it has no pending slot.
PENDING_STRUCTURES: list[str] = STRUCTURES[:-1]

UPGRADE_KEYS: list[str] = ["GROUND_WEAPONS", "SHIELDS", "AIR_WEAPONS"]

# ---------------------------------------------------------------------------
# Decision cadence
# ---------------------------------------------------------------------------

# Seconds of game time between policy decisions. The parser bins replays into
# windows of this width, and the live bot must query the model on the same
# schedule -- the transformer's positional encoding assumes one step == one
# window, so a cadence mismatch stretches the model's sense of time.
# Shared here so training and inference cannot disagree.
DECISION_INTERVAL_SECONDS = 4

# Number of decision windows the model sees at once.
#
# The live bot truncates its observation history to this length, which resets the
# positional encoding to 0 for the trailing window. Training must therefore also
# feed crops of this length, otherwise a mid-game observation gets PE position
# ~255 at inference but its true absolute index (up to 1181) during training --
# and PE position and the time feature, perfectly correlated in training, would
# decorrelate at inference. Shared here so the two cannot drift apart.
CONTEXT_WINDOW = 256

# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------

# Games routinely exceed 12 minutes: an audit of 40 replays found 31% of rows
# past 720s (longest 30.8 min). The old /720 with no clipping fed time_norm up
# to 2.57 into the model. 1800s covers essentially every game, and the clip
# guarantees the feature stays in [0, 1].
TIME_NORM = 1800.0

SUPPLY_NORM = 200.0
STRUCT_NORM = 10.0
UNIT_NORM = 30.0
IDLE_NORM = 5.0
UPGRADE_NORM = 3.0
WORKERS_PER_BASE = 22

MINERAL_BIN_EDGES = (100.0, 300.0, 500.0)
GAS_BIN_EDGES = (25.0, 100.0, 200.0)

# ---------------------------------------------------------------------------
# Index layout — derived from the vocabularies so it cannot drift
# ---------------------------------------------------------------------------

IDX_TIME = 0
IDX_MINERAL_BINS = 1                      # 1..4
IDX_GAS_BINS = IDX_MINERAL_BINS + 4       # 5..8
IDX_SUPPLY_USED = IDX_GAS_BINS + 4        # 9
IDX_SUPPLY_CAP = IDX_SUPPLY_USED + 1      # 10
IDX_WORKER_SAT = IDX_SUPPLY_CAP + 1       # 11
IDX_STRUCT_BASE = IDX_WORKER_SAT + 1      # 12
IDX_UNIT_BASE = IDX_STRUCT_BASE + len(STRUCTURES)            # 27
IDX_PEND_STRUCT_BASE = IDX_UNIT_BASE + len(UNITS)            # 38
IDX_PEND_UNIT_BASE = (IDX_PEND_STRUCT_BASE
                      + len(PENDING_STRUCTURES))             # 52
IDX_IDLE_GW_WG = IDX_PEND_UNIT_BASE + len(UNITS)             # 63
IDX_IDLE_SG = IDX_IDLE_GW_WG + 1                             # 64
IDX_IDLE_ROBO = IDX_IDLE_SG + 1                              # 65
IDX_IDLE_WG = IDX_IDLE_ROBO + 1                              # 66
IDX_GROUND_WEAPONS_LVL = IDX_IDLE_WG + 1                     # 67
IDX_SHIELDS_LVL = IDX_GROUND_WEAPONS_LVL + 1                 # 68
IDX_AIR_WEAPONS_LVL = IDX_SHIELDS_LVL + 1                    # 69

OBS_SIZE = IDX_AIR_WEAPONS_LVL + 1                           # 70

# Name -> absolute feature index, for callers that need a specific feature.
STRUCT_IDX = {n: IDX_STRUCT_BASE + i for i, n in enumerate(STRUCTURES)}
UNIT_IDX = {n: IDX_UNIT_BASE + i for i, n in enumerate(UNITS)}
PEND_STRUCT_IDX = {n: IDX_PEND_STRUCT_BASE + i
                   for i, n in enumerate(PENDING_STRUCTURES)}
PEND_UNIT_IDX = {n: IDX_PEND_UNIT_BASE + i for i, n in enumerate(UNITS)}

# ---------------------------------------------------------------------------
# Production building -> the units that occupy it
# Used to derive the idle-building features identically on both sides.
# ---------------------------------------------------------------------------

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


def compute_idle_counts(
    structures_done: Mapping[str, float],
    units_pending: Mapping[str, float],
) -> tuple[float, float, float, float]:
    """
    Derive idle production-building counts.

    A building is treated as idle when the number of completed buildings of
    that class exceeds the number of its units currently in production.
    Kept here so training and inference use the exact same derivation.
    """
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
    Build the canonical observation vector from raw game-state values.

    All arguments are raw (un-normalized) counts/amounts. Mapping keys are the
    names in STRUCTURES / UNITS / UPGRADE_KEYS; missing keys count as 0.
    Keyword-only to prevent positional mix-ups between the two call sites.
    """
    obs: list[float] = []

    # [0] time — clipped so the feature can never leave [0, 1]
    obs.append(min(max(time_s, 0.0) / TIME_NORM, 1.0))

    # [1:5] minerals, [5:9] gas
    obs.extend(_one_hot_bins(minerals, MINERAL_BIN_EDGES))
    obs.extend(_one_hot_bins(vespene, GAS_BIN_EDGES))

    # [9], [10] supply
    obs.append(supply_used / SUPPLY_NORM)
    obs.append(supply_cap / SUPPLY_NORM)

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
    """Human-readable name per index — for audits and debugging output."""
    names = ["time_norm"]
    names += [f"minerals_bin{i}" for i in range(4)]
    names += [f"gas_bin{i}" for i in range(4)]
    names += ["supply_used", "supply_cap", "worker_sat"]
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
