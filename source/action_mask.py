"""
action_mask.py — Prerequisite masks for training and inference
==============================================================
Two masks are provided:

  build_legal_mask      — STRICT inference mask. Uses completed-only
                          prerequisites and idle-building counts. Governs
                          what the bot is allowed to do right now.

  build_training_mask   — RELAXED training mask. Mirrors the parser's
                          _action_legal_numpy semantics:
                            * Pending-or-complete for all structure prereqs
                              (pro players queue the next building before the
                              previous one finishes; the 4-second grid window
                              means gateway+cybcore can land in the same slot).
                            * No idle-building checks for unit-train actions
                              (idle counts drift in the parser, same reason
                              as the probe queue cap removal).
                          This eliminates false label/mask conflicts during
                          training without changing inference behaviour.
"""

import torch

import obs_spec
from obs_spec import STRUCT_IDX, UNIT_IDX, PEND_STRUCT_IDX, PEND_UNIT_IDX

NUM_ACTIONS = 35  # all indices 0-34 are active actions (see ACTIONS in actions.py)

# ---------------------------------------------------------------------------
# Feature indices — imported from obs_spec (the single source of truth for the
# observation layout) rather than restated here, so they cannot drift out of
# sync with the parser and the live observation wrapper.
# ---------------------------------------------------------------------------
IDX_NEXUS = STRUCT_IDX["NEXUS"]
IDX_PYLON = STRUCT_IDX["PYLON"]
IDX_GATEWAY = STRUCT_IDX["GATEWAY"]
IDX_FORGE = STRUCT_IDX["FORGE"]
IDX_TWILIGHTCOUNCIL = STRUCT_IDX["TWILIGHTCOUNCIL"]
IDX_PHOTONCANNON = STRUCT_IDX["PHOTONCANNON"]
IDX_SHIELDBATTERY = STRUCT_IDX["SHIELDBATTERY"]
IDX_TEMPLARARCHIVE = STRUCT_IDX["TEMPLARARCHIVE"]
IDX_ROBOTICSBAY = STRUCT_IDX["ROBOTICSBAY"]
IDX_ROBOTICSFACILITY = STRUCT_IDX["ROBOTICSFACILITY"]
IDX_ASSIMILATOR = STRUCT_IDX["ASSIMILATOR"]
IDX_CYBERNETICSCORE = STRUCT_IDX["CYBERNETICSCORE"]
IDX_STARGATE = STRUCT_IDX["STARGATE"]
IDX_FLEETBEACON = STRUCT_IDX["FLEETBEACON"]
IDX_WARPGATE = STRUCT_IDX["WARPGATE"]

IDX_PROBE = UNIT_IDX["PROBE"]
IDX_ZEALOT = UNIT_IDX["ZEALOT"]
IDX_STALKER = UNIT_IDX["STALKER"]
IDX_HIGHTEMPLAR = UNIT_IDX["HIGHTEMPLAR"]
IDX_ARCHON = UNIT_IDX["ARCHON"]
IDX_IMMORTAL = UNIT_IDX["IMMORTAL"]
IDX_CARRIER = UNIT_IDX["CARRIER"]
IDX_VOIDRAY = UNIT_IDX["VOIDRAY"]
IDX_ADEPT = UNIT_IDX["ADEPT"]
IDX_PHOENIX = UNIT_IDX["PHOENIX"]
IDX_COLOSSUS = UNIT_IDX["COLOSSUS"]

IDX_PEND_NEXUS = PEND_STRUCT_IDX["NEXUS"]
IDX_PEND_PYLON = PEND_STRUCT_IDX["PYLON"]
IDX_PEND_GATEWAY = PEND_STRUCT_IDX["GATEWAY"]
IDX_PEND_FORGE = PEND_STRUCT_IDX["FORGE"]
IDX_PEND_TWILIGHTCOUNCIL = PEND_STRUCT_IDX["TWILIGHTCOUNCIL"]
IDX_PEND_PHOTONCANNON = PEND_STRUCT_IDX["PHOTONCANNON"]
IDX_PEND_SHIELDBATTERY = PEND_STRUCT_IDX["SHIELDBATTERY"]
IDX_PEND_TEMPLARARCHIVE = PEND_STRUCT_IDX["TEMPLARARCHIVE"]
IDX_PEND_ROBOTICSBAY = PEND_STRUCT_IDX["ROBOTICSBAY"]
IDX_PEND_ROBOTICSFACILITY = PEND_STRUCT_IDX["ROBOTICSFACILITY"]
IDX_PEND_ASSIMILATOR = PEND_STRUCT_IDX["ASSIMILATOR"]
IDX_PEND_CYBERNETICSCORE = PEND_STRUCT_IDX["CYBERNETICSCORE"]
IDX_PEND_STARGATE = PEND_STRUCT_IDX["STARGATE"]
IDX_PEND_FLEETBEACON = PEND_STRUCT_IDX["FLEETBEACON"]

IDX_PENDING_PROBE = PEND_UNIT_IDX["PROBE"]

IDX_IDLE_GW_WG = obs_spec.IDX_IDLE_GW_WG
IDX_IDLE_SG = obs_spec.IDX_IDLE_SG
IDX_IDLE_ROBO = obs_spec.IDX_IDLE_ROBO
IDX_IDLE_WG = obs_spec.IDX_IDLE_WG

IDX_GROUND_WEAPONS_LVL = obs_spec.IDX_GROUND_WEAPONS_LVL
IDX_SHIELDS_LVL = obs_spec.IDX_SHIELDS_LVL
IDX_AIR_WEAPONS_LVL = obs_spec.IDX_AIR_WEAPONS_LVL

EPS = 0.01


def build_legal_mask(obs: torch.Tensor) -> torch.Tensor:
    """
    Compute a boolean legal-action mask from a batch of observations.

    Args:
        obs: (N, OBS_SIZE)

    Returns:
        mask: (N, NUM_ACTIONS) bool tensor. True = action is legal.
    """
    N = obs.shape[0]
    device = obs.device
    mask = torch.zeros(N, NUM_ACTIONS, dtype=torch.bool, device=device)

    # --- Structure presence ---
    has_nexus = obs[:, IDX_NEXUS] > EPS
    has_pylon = obs[:, IDX_PYLON] > EPS
    has_gateway = obs[:, IDX_GATEWAY] > EPS
    has_forge = obs[:, IDX_FORGE] > EPS
    has_twilight = obs[:, IDX_TWILIGHTCOUNCIL] > EPS
    has_temparch = obs[:, IDX_TEMPLARARCHIVE] > EPS
    has_cybcore = obs[:, IDX_CYBERNETICSCORE] > EPS
    has_stargate = obs[:, IDX_STARGATE] > EPS
    has_fleet = obs[:, IDX_FLEETBEACON] > EPS
    has_robobay = obs[:, IDX_ROBOTICSBAY] > EPS
    has_robo = obs[:, IDX_ROBOTICSFACILITY] > EPS

    # --- Building caps ---
    under_cybcore_cap = obs[:, IDX_CYBERNETICSCORE] < (1.5 / 10.0)
    no_twilight = obs[:, IDX_TWILIGHTCOUNCIL] < EPS
    no_fleet = obs[:, IDX_FLEETBEACON] < EPS
    no_temparch = obs[:, IDX_TEMPLARARCHIVE] < EPS
    no_robobay = obs[:, IDX_ROBOTICSBAY] < EPS

    # --- Probe queue cap (inference only — not in parser mirror) ---
    pending_probes = obs[:, IDX_PENDING_PROBE] * 30.0
    nexus_count = obs[:, IDX_NEXUS] * 10.0

    # --- Idle building checks ---
    _IDLE_EPS = 0.5 / 5.0
    has_idle_gw_wg = obs[:, IDX_IDLE_GW_WG] > _IDLE_EPS
    has_idle_sg = obs[:, IDX_IDLE_SG] > _IDLE_EPS
    has_idle_robo = obs[:, IDX_IDLE_ROBO] > _IDLE_EPS
    has_idle_wg = obs[:, IDX_IDLE_WG] > _IDLE_EPS

    # 2+ idle high templar to merge into archon
    has_2_hightemplar = obs[:, IDX_HIGHTEMPLAR] > (1.5 / 30.0)

    # Any combat unit = "has army"
    has_army = (
        (obs[:, IDX_ZEALOT] > EPS) |
        (obs[:, IDX_STALKER] > EPS) |
        (obs[:, IDX_IMMORTAL] > EPS) |
        (obs[:, IDX_VOIDRAY] > EPS) |
        (obs[:, IDX_CARRIER] > EPS) |
        (obs[:, IDX_ARCHON] > EPS) |
        (obs[:, IDX_ADEPT] > EPS) |
        (obs[:, IDX_PHOENIX] > EPS) |
        (obs[:, IDX_COLOSSUS] > EPS)
    )

    # ------------------------------------------------------------------
    # Action 0: do_nothing — always legal
    mask[:, 0] = True

    # Action 1: train_probe — needs Nexus + queue room
    mask[:, 1] = has_nexus

    # Action 2: build_pylon — always legal
    mask[:, 2] = True

    # Action 3: build_gateway — needs Pylon
    mask[:, 3] = has_pylon

    # Action 4: build_cyberneticscore — needs Gateway, max 2 allowed
    mask[:, 4] = has_gateway & under_cybcore_cap

    # Action 5: build_assimilator — needs Nexus
    mask[:, 5] = has_nexus

    # Action 6: build_nexus — always legal
    mask[:, 6] = True

    # Action 7: build_forge — needs Pylon
    mask[:, 7] = has_pylon

    # Action 8: build_stargate — needs Cybernetics Core
    mask[:, 8] = has_cybcore

    # Action 9: build_robotics_facility — needs Cybernetics Core
    mask[:, 9] = has_cybcore

    # Action 10: build_twilight_council — needs Cybernetics Core, must not have one
    mask[:, 10] = has_cybcore & no_twilight

    # Action 11: build_photon_cannon — needs Forge
    mask[:, 11] = has_forge

    # Action 12: build_fleet_beacon — needs Stargate, must not have one
    mask[:, 12] = has_stargate & no_fleet

    # Action 13: build_templar_archive — needs Twilight Council, must not have one
    mask[:, 13] = has_twilight & no_temparch

    # Action 14: build_robotics_bay — needs Robotics Facility, must not have one
    mask[:, 14] = has_robo & no_robobay

    # Action 15: build_shield_battery — needs Cybernetics Core
    mask[:, 15] = has_cybcore

    # Action 16: train_zealot — needs idle Gateway
    mask[:, 16] = has_idle_gw_wg

    # Action 17: train_stalker — needs idle Gateway + Cybernetics Core
    mask[:, 17] = has_idle_gw_wg & has_cybcore

    # Action 18: train_immortal — needs idle Robotics Facility
    mask[:, 18] = has_idle_robo

    # Action 19: train_voidray — needs idle Stargate
    mask[:, 19] = has_idle_sg

    # Action 20: train_carrier — needs idle Stargate + Fleet Beacon
    mask[:, 20] = has_idle_sg & has_fleet

    # Action 21: train_high_templar — needs idle Gateway + Templar Archive
    mask[:, 21] = has_idle_gw_wg & has_temparch

    # Action 22: warp_in_zealot — needs idle Warpgate
    mask[:, 22] = has_idle_wg

    # Action 23: warp_in_stalker — needs idle Warpgate + Cybernetics Core
    mask[:, 23] = has_idle_wg & has_cybcore

    # Action 24: warp_in_high_templar — needs idle Warpgate + Templar Archive
    mask[:, 24] = has_idle_wg & has_temparch

    # Action 25: research_charge — needs Twilight Council
    mask[:, 25] = has_twilight

    # Action 26: research_warp_gate — needs Cybernetics Core
    mask[:, 26] = has_cybcore

    # Action 27: upgrade_ground_weapons — needs Forge, level < 3
    mask[:, 27] = has_forge & (obs[:, IDX_GROUND_WEAPONS_LVL] < (1.0 - EPS))

    # Action 28: upgrade_air_weapons — needs Cybernetics Core, level < 3
    mask[:, 28] = has_cybcore & (obs[:, IDX_AIR_WEAPONS_LVL] < (1.0 - EPS))

    # Action 29: upgrade_shields — needs Forge, level < 3
    mask[:, 29] = has_forge & (obs[:, IDX_SHIELDS_LVL] < (1.0 - EPS))

    # Action 30: attack_enemy_base — needs at least one combat unit
    mask[:, 30] = has_army

    # Action 31: train_adept — needs idle Gateway + Cybernetics Core
    mask[:, 31] = has_idle_gw_wg & has_cybcore

    # Action 32: train_phoenix — needs idle Stargate
    mask[:, 32] = has_idle_sg

    # Action 33: train_colossus — needs idle Robotics Facility + Robotics Bay (1-of)
    mask[:, 33] = has_idle_robo & has_robobay

    # Action 34: warp_in_adept — needs idle Warpgate + Cybernetics Core
    mask[:, 34] = has_idle_wg & has_cybcore

    return mask


def apply_legal_mask(logits: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """
    Set logits for illegal actions to -inf (strict inference mask).

    Args:
        logits: (N, NUM_ACTIONS)
        obs:    (N, OBS_SIZE)

    Returns:
        masked_logits: (N, NUM_ACTIONS)
    """
    mask = build_legal_mask(obs)
    masked = logits.clone()
    masked[~mask] = float('-inf')
    return masked


def build_training_mask(obs: torch.Tensor) -> torch.Tensor:
    """
    Compute a relaxed boolean legal-action mask for use during training only.

    Mirrors the parser's _action_legal_numpy semantics:
      - Pending-or-complete for all structure prerequisites.
      - No idle-building checks for unit-train/warp actions.
      - 1-of caps and army checks are unchanged.

    Args:
        obs: (N, OBS_SIZE)

    Returns:
        mask: (N, NUM_ACTIONS) bool tensor. True = action is legal.
    """
    N = obs.shape[0]
    device = obs.device
    mask = torch.zeros(N, NUM_ACTIONS, dtype=torch.bool, device=device)

    # --- Completed structure presence ---
    has_nexus   = obs[:, IDX_NEXUS]           > EPS
    has_pylon   = obs[:, IDX_PYLON]           > EPS
    has_gateway = obs[:, IDX_GATEWAY]         > EPS
    has_forge   = obs[:, IDX_FORGE]           > EPS
    has_twilight = obs[:, IDX_TWILIGHTCOUNCIL] > EPS
    has_temparch = obs[:, IDX_TEMPLARARCHIVE]  > EPS
    has_cybcore  = obs[:, IDX_CYBERNETICSCORE] > EPS
    has_stargate = obs[:, IDX_STARGATE]        > EPS
    has_fleet    = obs[:, IDX_FLEETBEACON]     > EPS
    has_robobay  = obs[:, IDX_ROBOTICSBAY]     > EPS
    has_robo     = obs[:, IDX_ROBOTICSFACILITY] > EPS
    has_warpgate = obs[:, IDX_WARPGATE]        > EPS

    # --- Pending structure presence ---
    pend_pylon    = obs[:, IDX_PEND_PYLON]           > EPS
    pend_gateway  = obs[:, IDX_PEND_GATEWAY]         > EPS
    pend_cybcore  = obs[:, IDX_PEND_CYBERNETICSCORE] > EPS
    pend_stargate = obs[:, IDX_PEND_STARGATE]        > EPS
    pend_robo     = obs[:, IDX_PEND_ROBOTICSFACILITY] > EPS
    pend_twilight = obs[:, IDX_PEND_TWILIGHTCOUNCIL] > EPS
    pend_temparch = obs[:, IDX_PEND_TEMPLARARCHIVE]  > EPS
    pend_forge    = obs[:, IDX_PEND_FORGE]           > EPS

    # --- Pending-or-complete: player has committed to building this ---
    poc_pylon    = has_pylon    | pend_pylon
    poc_gateway  = has_gateway  | pend_gateway
    poc_warpgate = has_warpgate
    poc_cybcore  = has_cybcore  | pend_cybcore
    poc_stargate = has_stargate | pend_stargate
    poc_robo     = has_robo     | pend_robo
    poc_twilight = has_twilight | pend_twilight
    poc_temparch = has_temparch | pend_temparch
    poc_forge    = has_forge    | pend_forge

    # Gateway-type production: a Warpgate is a morphed Gateway and gates the same
    # units/tech. After Warp Gate research pros morph every Gateway, so the plain
    # GATEWAY count drops to zero while production continues. Must mirror
    # _action_legal_numpy in replay_parser.py, or labels the parser kept would be
    # masked out by the loss (and vice versa). The strict inference mask already
    # combines the two via the idle_gw_wg feature.
    poc_gateway_type = poc_gateway | has_warpgate

    # --- Building caps ---
    under_cybcore_cap = obs[:, IDX_CYBERNETICSCORE] < (1.5 / 10.0)
    no_twilight = ~has_twilight
    no_fleet    = ~has_fleet
    no_temparch = ~has_temparch

    # --- 2+ high templar to merge into archon ---
    has_2_hightemplar = obs[:, IDX_HIGHTEMPLAR] > (1.5 / 30.0)

    # --- Any combat unit = "has army" ---
    has_army = (
        (obs[:, IDX_ZEALOT]   > EPS) |
        (obs[:, IDX_STALKER]  > EPS) |
        (obs[:, IDX_IMMORTAL] > EPS) |
        (obs[:, IDX_VOIDRAY]  > EPS) |
        (obs[:, IDX_CARRIER]  > EPS) |
        (obs[:, IDX_ARCHON]   > EPS) |
        (obs[:, IDX_ADEPT]    > EPS) |
        (obs[:, IDX_PHOENIX]  > EPS) |
        (obs[:, IDX_COLOSSUS] > EPS)
    )

    # ------------------------------------------------------------------
    # Action 0: do_nothing — always legal
    mask[:, 0] = True

    # Action 1: train_probe — needs Nexus (no queue cap in training)
    mask[:, 1] = has_nexus

    # Action 2: build_pylon — always legal
    mask[:, 2] = True

    # Action 3: build_gateway — needs Pylon
    mask[:, 3] = poc_pylon

    # Action 4: build_cyberneticscore — gateway poc, max 2 allowed
    mask[:, 4] = poc_gateway_type & under_cybcore_cap

    # Action 5: build_assimilator — needs Nexus
    mask[:, 5] = has_nexus

    # Action 6: build_nexus — always legal
    mask[:, 6] = True

    # Action 7: build_forge — needs Pylon
    mask[:, 7] = poc_pylon

    # Action 8: build_stargate — cybcore poc
    mask[:, 8] = poc_cybcore

    # Action 9: build_robotics_facility — cybcore poc
    mask[:, 9] = poc_cybcore

    # Action 10: build_twilight_council — cybcore poc, no existing twilight
    mask[:, 10] = poc_cybcore & no_twilight

    # Action 11: build_photon_cannon — needs completed Forge
    mask[:, 11] = has_forge

    # Action 12: build_fleet_beacon — stargate poc, no existing fleet beacon
    mask[:, 12] = poc_stargate & no_fleet

    # Action 13: build_templar_archive — twilight poc, no existing templar archive
    mask[:, 13] = poc_twilight & no_temparch

    # Action 14: build_robotics_bay — robo poc, no existing robotics bay
    mask[:, 14] = poc_robo & ~has_robobay

    # Action 15: build_shield_battery — cybcore poc
    mask[:, 15] = poc_cybcore

    # Action 16: train_zealot — gateway poc (no idle check)
    mask[:, 16] = poc_gateway_type

    # Action 17: train_stalker — gateway + cybcore both poc (no idle check)
    mask[:, 17] = poc_gateway_type & poc_cybcore

    # Action 18: train_immortal — robo poc (no idle check)
    mask[:, 18] = poc_robo

    # Action 19: train_voidray — stargate poc (no idle check)
    mask[:, 19] = poc_stargate

    # Action 20: train_carrier — stargate poc + fleet beacon complete
    mask[:, 20] = poc_stargate & has_fleet

    # Action 21: train_high_templar — gateway + templar archive both poc (no idle)
    mask[:, 21] = poc_gateway_type & poc_temparch

    # Action 22: warp_in_zealot — warpgate poc (no idle check)
    mask[:, 22] = poc_warpgate

    # Action 23: warp_in_stalker — warpgate + cybcore both poc (no idle check)
    mask[:, 23] = poc_warpgate & poc_cybcore

    # Action 24: warp_in_high_templar — warpgate + templar archive both poc (no idle)
    mask[:, 24] = poc_warpgate & poc_temparch

    # Action 25: research_charge — twilight poc
    mask[:, 25] = poc_twilight

    # Action 26: research_warp_gate — cybcore poc
    mask[:, 26] = poc_cybcore

    # Action 27: upgrade_ground_weapons — forge poc (no level cap in training to handle lag)
    mask[:, 27] = poc_forge

    # Action 28: upgrade_air_weapons — cybcore poc (no level cap in training)
    mask[:, 28] = poc_cybcore

    # Action 29: upgrade_shields — forge poc (no level cap in training)
    mask[:, 29] = poc_forge

    # Action 30: attack_enemy_base — needs army
    mask[:, 30] = has_army

    # Action 31: train_adept — gateway + cybcore both poc (no idle check)
    mask[:, 31] = poc_gateway_type & poc_cybcore

    # Action 32: train_phoenix — stargate poc (no idle check)
    mask[:, 32] = poc_stargate

    # Action 33: train_colossus — robo poc + robobay complete (no idle check)
    mask[:, 33] = poc_robo & has_robobay

    # Action 34: warp_in_adept — warpgate + cybcore both poc (no idle check)
    mask[:, 34] = poc_warpgate & poc_cybcore

    return mask


def apply_training_mask(logits: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """
    Set logits for illegal actions to -inf using the relaxed training mask.

    Args:
        logits: (N, NUM_ACTIONS)
        obs:    (N, OBS_SIZE)

    Returns:
        masked_logits: (N, NUM_ACTIONS)
    """
    mask = build_training_mask(obs)
    masked = logits.clone()
    masked[~mask] = float('-inf')
    return masked
