"""
action_mask.py — Shared mask for training and inference
====================================================================
Determins if an action is legal or not. Added so that the model will only make legal moves
Imported by model.py (for training) and protoss_bot.py (for inference).

Applying the same mask in both places ensures the model learns to
distribute probability only over legal actions — so the conditional
probabilities are calibrated for exactly the distribution seen at runtime.

Observation layout (53 features, matching observation_wrapper.py):
    [0]     game time
    [1]     minerals
    [2]     vespene
    [3]     supply_used
    [4]     supply_cap
    [5]     worker saturation
    [6:21]  completed structure counts  (15 structures, normalised /10)
    [21:29] completed unit counts       (8 units,       normalised /30)
    [29:44] pending structure counts    (15 structures, normalised /10)
    [44:52] pending unit counts         (8 units,       normalised /30)
    [52]    opponent supply_used
"""

import torch

NUM_ACTIONS = 30

# ---------------------------------------------------------------------------
# Obs feature indices — completed structure counts
# ---------------------------------------------------------------------------
IDX_NEXUS = 6
IDX_PYLON = 7
IDX_GATEWAY = 8
IDX_WARPGATE = 9
IDX_FORGE = 10
IDX_TWILIGHTCOUNCIL = 11
IDX_PHOTONCANNON = 12
IDX_SHIELDBATTERY = 13
IDX_TEMPLARARCHIVE = 14
IDX_ROBOTICSBAY = 15
IDX_ROBOTICSFACILITY = 16
IDX_ASSIMILATOR = 17
IDX_CYBERNETICSCORE = 18
IDX_STARGATE = 19
IDX_FLEETBEACON = 20

# ---------------------------------------------------------------------------
# Obs feature indices — completed unit counts
# ---------------------------------------------------------------------------
IDX_PROBE = 21
IDX_ZEALOT = 22
IDX_STALKER = 23
IDX_HIGHTEMPLAR = 24
IDX_ARCHON = 25
IDX_IMMORTAL = 26
IDX_CARRIER = 27
IDX_VOIDRAY = 28

# ---------------------------------------------------------------------------
# Obs feature indices — pending unit counts
# ---------------------------------------------------------------------------
IDX_PENDING_PROBE = 44

# ---------------------------------------------------------------------------
# Obs feature indices — pending structure counts
# ---------------------------------------------------------------------------
IDX_PENDING_NEXUS = 29  # First in pending structures section

# Threshold: structures are normalised /10, units /30.
# 0.01 is safely above zero but below 0.1 (= 1 structure).
EPS = 0.01

# Supply costs for units (unnormalized)
SUPPLY_COSTS = {
    'PROBE': 1,
    'ZEALOT': 2,
    'STALKER': 2,
    'HIGHTEMPLAR': 2,
    'ARCHON': 4,
    'IMMORTAL': 4,
    'CARRIER': 6,
    'VOIDRAY': 4,
}


def build_legal_mask(obs: torch.Tensor) -> torch.Tensor:
    """
    Compute a boolean legal-action mask from a batch of observations.

    Args:
        obs: (N, OBS_SIZE) — works for (B*T, OBS_SIZE) in training
             or (1, OBS_SIZE) in single-step inference.

    Returns:
        mask: (N, NUM_ACTIONS) bool tensor.  True = action is legal.

    Notes:
        - do_nothing (0) and build_pylon (2) are always legal.
        - Padded training positions have obs=0, but action 0 is always
          True so softmax never receives an all-(-inf) input.
        - We check only *completed* structure/unit counts (indices 6-28),
          not pending, because a structure under construction cannot yet
          unlock dependent buildings or units.
    """
    N = obs.shape[0]
    device = obs.device
    mask = torch.zeros(N, NUM_ACTIONS, dtype=torch.bool, device=device)

    # Supply check: obs indices 3 and 4 are supply_used and supply_cap (normalized /200)
    supply_used = obs[:, 3] * 200.0
    supply_cap = obs[:, 4] * 200.0
    supply_available = supply_cap - supply_used

    has_nexus = obs[:, IDX_NEXUS] > EPS
    nexus_count = obs[:, IDX_NEXUS] * 10.0  # Denormalize
    pending_nexus = obs[:, IDX_PENDING_NEXUS] * 10.0  # Denormalize
    total_nexus = nexus_count + pending_nexus
    pending_probes = obs[:, IDX_PENDING_PROBE] * 30.0  # Denormalize
    has_pylon = obs[:, IDX_PYLON] > EPS
    has_gateway = obs[:, IDX_GATEWAY] > EPS
    has_warpgate = obs[:, IDX_WARPGATE] > EPS
    has_forge = obs[:, IDX_FORGE] > EPS
    has_twilight = obs[:, IDX_TWILIGHTCOUNCIL] > EPS
    has_templar_archive = obs[:, IDX_TEMPLARARCHIVE] > EPS
    has_robofac = obs[:, IDX_ROBOTICSFACILITY] > EPS
    has_cybcore = obs[:, IDX_CYBERNETICSCORE] > EPS
    has_stargate = obs[:, IDX_STARGATE] > EPS
    has_fleetbeacon = obs[:, IDX_FLEETBEACON] > EPS

    # 2+ idle high templar needed to merge into archon (units normalised /30)
    has_2_hightemplar = obs[:, IDX_HIGHTEMPLAR] > (1.5 / 30.0)

    # Any ground or air combat unit counts as "has army"
    has_army = (
        (obs[:, IDX_ZEALOT] > EPS) |
        (obs[:, IDX_STALKER] > EPS) |
        (obs[:, IDX_IMMORTAL] > EPS) |
        (obs[:, IDX_VOIDRAY] > EPS) |
        (obs[:, IDX_CARRIER] > EPS) |
        (obs[:, IDX_ARCHON] > EPS)
    )

    # ------------------------------------------------------------------
    # Action 0: do_nothing — always legal
    mask[:, 0] = True

    # Action 1: train_probe — needs Nexus + supply + not already queued at every Nexus
    # Only allow training if pending_probes < nexus_count (max 1 probe per Nexus)
    mask[:, 1] = has_nexus & (supply_available >= SUPPLY_COSTS['PROBE']) & (
        pending_probes < 2*nexus_count)

    # Action 2: build_pylon — always legal (just needs minerals)
    mask[:, 2] = True

    # Action 3: build_gateway — needs at least one Pylon for power
    mask[:, 3] = has_pylon

    # Action 4: build_cyberneticscore — needs a Gateway
    mask[:, 4] = has_gateway

    # Action 5: build_assimilator — needs a Nexus (geysers must exist nearby,
    #           but we can't check that from obs; Nexus is the binding req)
    mask[:, 5] = has_nexus

    # Action 6: build_nexus — cap at 3 total (completed + in-progress)
    mask[:, 6] = total_nexus < 3
    # Action 7: build_forge — needs Pylon for power
    mask[:, 7] = has_pylon

    # Action 8: build_stargate — needs Cybernetics Core
    mask[:, 8] = has_cybcore

    # Action 9: build_robotics_facility — needs Cybernetics Core
    mask[:, 9] = has_cybcore

    # Action 10: build_twilight_council — needs Cybernetics Core
    mask[:, 10] = has_cybcore

    # Action 11: build_photon_cannon — needs Forge
    mask[:, 11] = has_forge

    # Action 12: build_fleet_beacon — needs Stargate
    mask[:, 12] = has_stargate

    # Action 13: build_templar_archive — needs Twilight Council
    mask[:, 13] = has_twilight

    # Action 14: train_zealot — needs a ready Gateway + supply
    mask[:, 14] = has_gateway & (supply_available >= SUPPLY_COSTS['ZEALOT'])

    # Action 15: train_stalker — needs Gateway + Cybernetics Core + supply
    mask[:, 15] = has_gateway & has_cybcore & (
        supply_available >= SUPPLY_COSTS['STALKER'])

    # Action 16: train_immortal — needs Robotics Facility + supply
    mask[:, 16] = has_robofac & (supply_available >= SUPPLY_COSTS['IMMORTAL'])

    # Action 17: train_voidray — needs Stargate + supply
    mask[:, 17] = has_stargate & (supply_available >= SUPPLY_COSTS['VOIDRAY'])

    # Action 18: train_carrier — needs Stargate + Fleet Beacon + supply
    mask[:, 18] = has_stargate & has_fleetbeacon & (
        supply_available >= SUPPLY_COSTS['CARRIER'])

    # Action 19: train_high_templar — needs Gateway + Templar Archive + supply
    mask[:, 19] = has_gateway & has_templar_archive & (
        supply_available >= SUPPLY_COSTS['HIGHTEMPLAR'])

    # Action 20: warp_in_zealot — needs a ready Warpgate + supply
    mask[:, 20] = has_warpgate & (supply_available >= SUPPLY_COSTS['ZEALOT'])

    # Action 21: warp_in_stalker — needs Warpgate + Cybernetics Core + supply
    mask[:, 21] = has_warpgate & has_cybcore & (
        supply_available >= SUPPLY_COSTS['STALKER'])

    # Action 22: warp_in_high_templar — needs Warpgate + Templar Archive + supply
    mask[:, 22] = has_warpgate & has_templar_archive & (
        supply_available >= SUPPLY_COSTS['HIGHTEMPLAR'])

    # Action 23: archon_warp — needs 2+ idle High Templars to merge
    mask[:, 23] = has_2_hightemplar

    # Action 24: research_charge — needs Twilight Council
    mask[:, 24] = has_twilight

    # Action 25: research_warp_gate — needs Cybernetics Core
    mask[:, 25] = has_cybcore

    # Action 26: upgrade_ground_weapons — needs Forge
    mask[:, 26] = has_forge

    # Action 27: upgrade_air_weapons — needs Cybernetics Core
    mask[:, 27] = has_cybcore

    # Action 28: upgrade_shields — needs Forge
    mask[:, 28] = has_forge

    # Action 29: attack_enemy_base — needs at least one combat unit
    mask[:, 29] = has_army

    return mask


def apply_legal_mask(logits: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """
    Set logits for illegal actions to -inf so they receive zero probability
    after softmax, and CrossEntropyLoss treats them as impossible.

    Args:
        logits: (N, NUM_ACTIONS)
        obs:    (N, OBS_SIZE)

    Returns:
        masked_logits: (N, NUM_ACTIONS)  — same shape, illegal entries = -inf
    """
    mask = build_legal_mask(obs)
    masked = logits.clone()
    masked[~mask] = float('-inf')
    return masked
