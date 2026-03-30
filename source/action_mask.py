"""
action_mask.py — Shared prerequisite mask for training and inference
====================================================================
Single source of truth for which actions are legal given an observation.
Imported by model.py (training loop) and protoss_bot.py (inference).

Applying the same mask in both places ensures the model learns to
distribute probability only over legal actions — so the conditional
probabilities are calibrated for exactly the distribution seen at runtime.

Observation layout (57 features, matching observation_wrapper.py):
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
    [53]    idle gateway+warpgate count (normalised /5)
    [54]    idle stargate count         (normalised /5)
    [55]    idle robotics facility count(normalised /5)
    [56]    idle warpgate count         (normalised /5)
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
# Obs feature indices — idle production building counts
# ---------------------------------------------------------------------------
IDX_IDLE_GW_WG = 53   # idle gateway + warpgate combined
IDX_IDLE_SG = 54   # idle stargates
IDX_IDLE_ROBO = 55   # idle robotics facilities
IDX_IDLE_WG = 56   # idle warpgates specifically

# Threshold: structures are normalised /10, units /30.
# 0.01 is safely above zero but below 0.1 (= 1 structure).
EPS = 0.01


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

    has_nexus = obs[:, IDX_NEXUS] > EPS
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

    # Probe queue cap: allow train_probe only if fewer than 2 are pending
    # per nexus. pending probe count is at IDX_PENDING_PROBE, normalised /30.
    # nexus count is at IDX_NEXUS, normalised /10.
    # Cap: pending_probes < 2 * nexus_count  (max 2 queued per nexus)
    pending_probes = obs[:, IDX_PENDING_PROBE] * 30.0
    nexus_count = obs[:, IDX_NEXUS] * 10.0
    probe_queue_ok = pending_probes < (2.0 * nexus_count)

    # Idle building checks — unit training only makes sense if a building
    # is actually available (not already busy). Uses the derived idle counts.
    # idle > 0 means (idle_count / 5) > 0, i.e. > 1/5 of a building.
    # We use a small epsilon here because floating point rounding on the
    # normalised value may produce something like 0.199 instead of 0.2.
    has_idle_gw_wg = obs[:, IDX_IDLE_GW_WG] > (0.5 / 5.0)   # at least 1 idle
    has_idle_sg = obs[:, IDX_IDLE_SG] > (0.5 / 5.0)
    has_idle_robo = obs[:, IDX_IDLE_ROBO] > (0.5 / 5.0)
    has_idle_wg = obs[:, IDX_IDLE_WG] > (0.5 / 5.0)

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

    # Action 1: train_probe — needs Nexus, queue must have room
    mask[:, 1] = has_nexus & probe_queue_ok

    # Action 2: build_pylon — always legal (just needs minerals)
    mask[:, 2] = True

    # Action 3: build_gateway — needs at least one Pylon for power
    mask[:, 3] = has_pylon

    # Action 4: build_cyberneticscore — needs a Gateway
    mask[:, 4] = has_gateway

    # Action 5: build_assimilator — needs a Nexus
    mask[:, 5] = has_nexus

    # Action 6: build_nexus — always legal
    mask[:, 6] = True

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

    # Action 14: train_zealot — needs idle Gateway
    mask[:, 14] = has_idle_gw_wg

    # Action 15: train_stalker — needs idle Gateway + Cybernetics Core
    mask[:, 15] = has_idle_gw_wg & has_cybcore

    # Action 16: train_immortal — needs idle Robotics Facility
    mask[:, 16] = has_idle_robo

    # Action 17: train_voidray — needs idle Stargate
    mask[:, 17] = has_idle_sg

    # Action 18: train_carrier — needs idle Stargate + Fleet Beacon
    mask[:, 18] = has_idle_sg & has_fleetbeacon

    # Action 19: train_high_templar — needs idle Gateway + Templar Archive
    mask[:, 19] = has_idle_gw_wg & has_templar_archive

    # Action 20: warp_in_zealot — needs idle Warpgate
    mask[:, 20] = has_idle_wg

    # Action 21: warp_in_stalker — needs idle Warpgate + Cybernetics Core
    mask[:, 21] = has_idle_wg & has_cybcore

    # Action 22: warp_in_high_templar — needs idle Warpgate + Templar Archive
    mask[:, 22] = has_idle_wg & has_templar_archive

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
