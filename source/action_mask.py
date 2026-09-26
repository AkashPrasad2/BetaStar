"""
Prerequisite masks for training and inference

Two masks provided:
  build_legal_mask      : STRICT inference mask. Uses completed-only
                          prerequisites and idle-building counts. Governs
                          what the bot is allowed to do right now.

  build_training_mask   : RELAXED training mask. Mirrors the parser's
                          _action_legal_numpy semantics:
                            * Pending-or-complete for all structure prereqs
                            * No idle-building checks for unit-train actions
                          This exists because of the 4s grid that could create deviances.
                          We give some lenience to still capture training actions despite pro players having strict timings
"""

import torch

import obs_spec as spec

EPS = 0.01
LEVEL_ONE_COMMITTED = (1.0 / spec.UPGRADE_NORM) - EPS


def _has_structure(obs: torch.Tensor, name: str) -> torch.Tensor:
    return obs[:, spec.STRUCT_IDX[name]] > EPS


def _has_pending_structure(obs: torch.Tensor, name: str) -> torch.Tensor:
    return obs[:, spec.PEND_STRUCT_IDX[name]] > EPS


def _apply_supply_gate(mask: torch.Tensor, obs: torch.Tensor,
                       slack: float = 0.0) -> None:
    """
    Clear unit-production actions that our supply headroom cannot pay for.

    Mutates `mask` in place. Headroom is recovered from the supply_used and
    supply_cap channels rather than the clipped supply_remaining feature, so the
    comparison is exact even above the clip point.

    `slack` relaxes the check for the training mask; see
    spec.TRAINING_SUPPLY_SLACK for why that is necessary.
    """
    remaining = (obs[:, spec.IDX_SUPPLY_CAP]
                 - obs[:, spec.IDX_SUPPLY_USED]) * spec.SUPPLY_NORM
    for action_id, cost in spec.ACTION_SUPPLY_COST.items():
        affordable = remaining >= (cost - slack - spec.SUPPLY_EPS)
        mask[:, action_id] &= affordable


def build_legal_mask(obs: torch.Tensor) -> torch.Tensor:
    """
    Takes an entire observation and returns the action mask for the observation.

    Args:
        obs: (N, OBS_SIZE)

    Returns:
        mask: (N, NUM_ACTIONS) bool tensor. True = action is legal.
    """
    N = obs.shape[0]
    device = obs.device
    mask = torch.zeros(N, spec.NUM_ACTIONS, dtype=torch.bool, device=device)

    # --- Structure presence ---
    has_nexus = _has_structure(obs, "NEXUS")
    has_pylon = _has_structure(obs, "PYLON")
    has_gateway = _has_structure(obs, "GATEWAY")
    has_forge = _has_structure(obs, "FORGE")
    has_twilight = _has_structure(obs, "TWILIGHTCOUNCIL")
    has_temparch = _has_structure(obs, "TEMPLARARCHIVE")
    has_cybcore = _has_structure(obs, "CYBERNETICSCORE")
    has_stargate = _has_structure(obs, "STARGATE")
    has_fleet = _has_structure(obs, "FLEETBEACON")
    has_robobay = _has_structure(obs, "ROBOTICSBAY")
    has_robo = _has_structure(obs, "ROBOTICSFACILITY")

    # --- Building caps ---
    under_cybcore_cap = (
        obs[:, spec.STRUCT_IDX["CYBERNETICSCORE"]]
        < (1.5 / spec.STRUCT_NORM)
    )
    no_twilight = ~has_twilight
    no_fleet = ~has_fleet
    no_temparch = ~has_temparch
    no_robobay = ~has_robobay

    # --- Idle building checks ---
    idle_eps = 0.5 / spec.IDLE_NORM
    has_idle_gw_wg = obs[:, spec.IDX_IDLE_GW_WG] > idle_eps
    has_idle_sg = obs[:, spec.IDX_IDLE_SG] > idle_eps
    has_idle_robo = obs[:, spec.IDX_IDLE_ROBO] > idle_eps
    has_idle_wg = obs[:, spec.IDX_IDLE_WG] > idle_eps

    # Action 0: do_nothing : always legal
    mask[:, spec.ACTION_ID["do_nothing"]] = True

    # Action 1: train_probe : needs Nexus + queue room
    mask[:, spec.ACTION_ID["train_probe"]] = has_nexus

    # Action 2: build_pylon : always legal
    mask[:, spec.ACTION_ID["build_pylon"]] = True

    # Action 3: build_gateway : needs Pylon
    mask[:, spec.ACTION_ID["build_gateway"]] = has_pylon

    # Action 4: build_cyberneticscore : needs Gateway, max 2 allowed
    mask[:, spec.ACTION_ID["build_cyberneticscore"]
         ] = has_gateway & under_cybcore_cap

    # Action 5: build_assimilator : needs Nexus
    mask[:, spec.ACTION_ID["build_assimilator"]] = has_nexus

    # Action 6: build_nexus : always legal
    mask[:, spec.ACTION_ID["build_nexus"]] = True

    # Action 7: build_forge : needs Pylon
    mask[:, spec.ACTION_ID["build_forge"]] = has_pylon

    # Action 8: build_stargate : needs Cybernetics Core
    mask[:, spec.ACTION_ID["build_stargate"]] = has_cybcore

    # Action 9: build_robotics_facility : needs Cybernetics Core
    mask[:, spec.ACTION_ID["build_robotics_facility"]] = has_cybcore

    # Action 10: build_twilight_council : needs Cybernetics Core, must not have one
    mask[:, spec.ACTION_ID["build_twilight_council"]] = has_cybcore & no_twilight

    # Action 11: build_photon_cannon : needs Forge
    mask[:, spec.ACTION_ID["build_photon_cannon"]] = has_forge

    # Action 12: build_fleet_beacon : needs Stargate, must not have one
    mask[:, spec.ACTION_ID["build_fleet_beacon"]] = has_stargate & no_fleet

    # Action 13: build_templar_archive : needs Twilight Council, must not have one
    mask[:, spec.ACTION_ID["build_templar_archive"]] = has_twilight & no_temparch

    # Action 14: build_robotics_bay : needs Robotics Facility, must not have one
    mask[:, spec.ACTION_ID["build_robotics_bay"]] = has_robo & no_robobay

    # Action 15: build_shield_battery : needs Cybernetics Core
    mask[:, spec.ACTION_ID["build_shield_battery"]] = has_cybcore

    # Action 16: train_zealot : needs idle Gateway
    mask[:, spec.ACTION_ID["train_zealot"]] = has_idle_gw_wg

    # Action 17: train_stalker : needs idle Gateway + Cybernetics Core
    mask[:, spec.ACTION_ID["train_stalker"]] = has_idle_gw_wg & has_cybcore

    # Action 18: train_immortal : needs idle Robotics Facility
    mask[:, spec.ACTION_ID["train_immortal"]] = has_idle_robo

    # Action 19: train_voidray : needs idle Stargate
    mask[:, spec.ACTION_ID["train_voidray"]] = has_idle_sg

    # Action 20: train_carrier : needs idle Stargate + Fleet Beacon
    mask[:, spec.ACTION_ID["train_carrier"]] = has_idle_sg & has_fleet

    # Action 22: warp_in_zealot : needs idle Warpgate
    mask[:, spec.ACTION_ID["warp_in_zealot"]] = has_idle_wg

    # Action 23: warp_in_stalker : needs idle Warpgate + Cybernetics Core
    mask[:, spec.ACTION_ID["warp_in_stalker"]] = has_idle_wg & has_cybcore

    # Action 24: warp_in_high_templar : needs idle Warpgate + Templar Archive
    mask[:, spec.ACTION_ID["warp_in_high_templar"]] = has_idle_wg & has_temparch

    # Action 25: research_charge : needs Twilight Council
    mask[:, spec.ACTION_ID["research_charge"]] = has_twilight

    # Action 26: research_warp_gate : needs Cybernetics Core
    mask[:, spec.ACTION_ID["research_warp_gate"]] = has_cybcore

    # Level 1 needs only a Forge. Levels 2-3 also need Twilight Council.
    ground_level = obs[:, spec.IDX_GROUND_WEAPONS_LVL]
    mask[:, spec.ACTION_ID["upgrade_ground_weapons"]] = has_forge & (
        ground_level < (1.0 - EPS)) & (
        (ground_level < LEVEL_ONE_COMMITTED) | has_twilight)

    # Level 1 needs only a Cybernetics Core. Levels 2-3 need Fleet Beacon.
    air_level = obs[:, spec.IDX_AIR_WEAPONS_LVL]
    mask[:, spec.ACTION_ID["upgrade_air_weapons"]] = has_cybcore & (
        air_level < (1.0 - EPS)) & (
        (air_level < LEVEL_ONE_COMMITTED) | has_fleet)

    # Level 1 needs only a Forge. Levels 2-3 also need Twilight Council.
    shields_level = obs[:, spec.IDX_SHIELDS_LVL]
    mask[:, spec.ACTION_ID["upgrade_shields"]] = has_forge & (
        shields_level < (1.0 - EPS)) & (
        (shields_level < LEVEL_ONE_COMMITTED) | has_twilight)

    # Action 31: train_adept : needs idle Gateway + Cybernetics Core
    mask[:, spec.ACTION_ID["train_adept"]] = has_idle_gw_wg & has_cybcore

    # Action 32: train_phoenix : needs idle Stargate
    mask[:, spec.ACTION_ID["train_phoenix"]] = has_idle_sg

    # Action 33: train_colossus : needs idle Robotics Facility + Robotics Bay (1-of)
    mask[:, spec.ACTION_ID["train_colossus"]] = has_idle_robo & has_robobay

    # Supply headroom. Exact at inference: the game rejects these orders outright
    # when we are capped, so leaving them legal only wasted decisions.
    _apply_supply_gate(mask, obs)

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

    Args:
        obs: (N, OBS_SIZE)

    Returns:
        mask: (N, NUM_ACTIONS) bool tensor. True = action is legal.
    """
    N = obs.shape[0]
    device = obs.device
    mask = torch.zeros(N, spec.NUM_ACTIONS, dtype=torch.bool, device=device)

    # --- Completed structure presence ---
    has_nexus = _has_structure(obs, "NEXUS")
    has_pylon = _has_structure(obs, "PYLON")
    has_gateway = _has_structure(obs, "GATEWAY")
    has_forge = _has_structure(obs, "FORGE")
    has_twilight = _has_structure(obs, "TWILIGHTCOUNCIL")
    has_temparch = _has_structure(obs, "TEMPLARARCHIVE")
    has_cybcore = _has_structure(obs, "CYBERNETICSCORE")
    has_stargate = _has_structure(obs, "STARGATE")
    has_fleet = _has_structure(obs, "FLEETBEACON")
    has_robobay = _has_structure(obs, "ROBOTICSBAY")
    has_robo = _has_structure(obs, "ROBOTICSFACILITY")
    has_warpgate = _has_structure(obs, "WARPGATE")

    # --- Pending structure presence ---
    pend_pylon = _has_pending_structure(obs, "PYLON")
    pend_gateway = _has_pending_structure(obs, "GATEWAY")
    pend_cybcore = _has_pending_structure(obs, "CYBERNETICSCORE")
    pend_stargate = _has_pending_structure(obs, "STARGATE")
    pend_robo = _has_pending_structure(obs, "ROBOTICSFACILITY")
    pend_twilight = _has_pending_structure(obs, "TWILIGHTCOUNCIL")
    pend_temparch = _has_pending_structure(obs, "TEMPLARARCHIVE")
    pend_forge = _has_pending_structure(obs, "FORGE")
    pend_fleet = _has_pending_structure(obs, "FLEETBEACON")

    # --- Pending-or-complete: player has committed to building this ---
    poc_pylon = has_pylon | pend_pylon
    poc_gateway = has_gateway | pend_gateway
    poc_warpgate = has_warpgate
    poc_cybcore = has_cybcore | pend_cybcore
    poc_stargate = has_stargate | pend_stargate
    poc_robo = has_robo | pend_robo
    poc_twilight = has_twilight | pend_twilight
    poc_temparch = has_temparch | pend_temparch
    poc_forge = has_forge | pend_forge
    poc_fleet = has_fleet | pend_fleet
    poc_gateway_type = poc_gateway | has_warpgate

    # --- Building caps ---
    under_cybcore_cap = (
        obs[:, spec.STRUCT_IDX["CYBERNETICSCORE"]]
        < (1.5 / spec.STRUCT_NORM)
    )
    no_twilight = ~has_twilight
    no_fleet = ~has_fleet
    no_temparch = ~has_temparch

    # Action 0: do_nothing : always legal
    mask[:, spec.ACTION_ID["do_nothing"]] = True

    # Action 1: train_probe : needs Nexus (no queue cap in training)
    mask[:, spec.ACTION_ID["train_probe"]] = has_nexus

    # Action 2: build_pylon : always legal
    mask[:, spec.ACTION_ID["build_pylon"]] = True

    # Action 3: build_gateway : needs Pylon
    mask[:, spec.ACTION_ID["build_gateway"]] = poc_pylon

    # Action 4: build_cyberneticscore : gateway poc, max 2 allowed
    mask[:, spec.ACTION_ID["build_cyberneticscore"]
         ] = poc_gateway_type & under_cybcore_cap

    # Action 5: build_assimilator : needs Nexus
    mask[:, spec.ACTION_ID["build_assimilator"]] = has_nexus

    # Action 6: build_nexus : always legal
    mask[:, spec.ACTION_ID["build_nexus"]] = True

    # Action 7: build_forge : needs Pylon
    mask[:, spec.ACTION_ID["build_forge"]] = poc_pylon

    # Action 8: build_stargate : cybcore poc
    mask[:, spec.ACTION_ID["build_stargate"]] = poc_cybcore

    # Action 9: build_robotics_facility : cybcore poc
    mask[:, spec.ACTION_ID["build_robotics_facility"]] = poc_cybcore

    # Action 10: build_twilight_council : cybcore poc, no existing twilight
    mask[:, spec.ACTION_ID["build_twilight_council"]] = poc_cybcore & no_twilight

    # Action 11: build_photon_cannon : needs completed Forge
    mask[:, spec.ACTION_ID["build_photon_cannon"]] = has_forge

    # Action 12: build_fleet_beacon : stargate poc, no existing fleet beacon
    mask[:, spec.ACTION_ID["build_fleet_beacon"]] = poc_stargate & no_fleet

    # Action 13: build_templar_archive : twilight poc, no existing templar archive
    mask[:, spec.ACTION_ID["build_templar_archive"]] = poc_twilight & no_temparch

    # Action 14: build_robotics_bay : robo poc, no existing robotics bay
    mask[:, spec.ACTION_ID["build_robotics_bay"]] = poc_robo & ~has_robobay

    # Action 15: build_shield_battery : cybcore poc
    mask[:, spec.ACTION_ID["build_shield_battery"]] = poc_cybcore

    # Action 16: train_zealot : gateway poc (no idle check)
    mask[:, spec.ACTION_ID["train_zealot"]] = poc_gateway_type

    # Action 17: train_stalker : gateway + cybcore both poc (no idle check)
    mask[:, spec.ACTION_ID["train_stalker"]] = poc_gateway_type & poc_cybcore

    # Action 18: train_immortal : robo poc (no idle check)
    mask[:, spec.ACTION_ID["train_immortal"]] = poc_robo

    # Action 19: train_voidray : stargate poc (no idle check)
    mask[:, spec.ACTION_ID["train_voidray"]] = poc_stargate

    # Action 20: train_carrier : stargate poc + fleet beacon complete
    mask[:, spec.ACTION_ID["train_carrier"]] = poc_stargate & has_fleet

    # Action 22: warp_in_zealot : warpgate poc (no idle check)
    mask[:, spec.ACTION_ID["warp_in_zealot"]] = poc_warpgate

    # Action 23: warp_in_stalker : warpgate + cybcore both poc (no idle check)
    mask[:, spec.ACTION_ID["warp_in_stalker"]] = poc_warpgate & poc_cybcore

    # Action 24: warp_in_high_templar : warpgate + templar archive both poc (no idle)
    mask[:, spec.ACTION_ID["warp_in_high_templar"]] = poc_warpgate & poc_temparch

    # Action 25: research_charge : twilight poc
    mask[:, spec.ACTION_ID["research_charge"]] = poc_twilight

    # Action 26: research_warp_gate : cybcore poc
    mask[:, spec.ACTION_ID["research_warp_gate"]] = poc_cybcore

    # Higher Forge levels require a Twilight Council. Pending counts as
    # committed here because this is the relaxed replay-training mask.
    ground_level = obs[:, spec.IDX_GROUND_WEAPONS_LVL]
    mask[:, spec.ACTION_ID["upgrade_ground_weapons"]] = poc_forge & (
        (ground_level < LEVEL_ONE_COMMITTED) | poc_twilight)

    # Higher air-weapon levels require a Fleet Beacon.
    air_level = obs[:, spec.IDX_AIR_WEAPONS_LVL]
    mask[:, spec.ACTION_ID["upgrade_air_weapons"]] = poc_cybcore & (
        (air_level < LEVEL_ONE_COMMITTED) | poc_fleet)

    shields_level = obs[:, spec.IDX_SHIELDS_LVL]
    mask[:, spec.ACTION_ID["upgrade_shields"]] = poc_forge & (
        (shields_level < LEVEL_ONE_COMMITTED) | poc_twilight)

    # Action 31: train_adept : gateway + cybcore both poc (no idle check)
    mask[:, spec.ACTION_ID["train_adept"]] = poc_gateway_type & poc_cybcore

    # Action 32: train_phoenix : stargate poc (no idle check)
    mask[:, spec.ACTION_ID["train_phoenix"]] = poc_stargate

    # Action 33: train_colossus : robo poc + robobay complete (no idle check)
    mask[:, spec.ACTION_ID["train_colossus"]] = poc_robo & has_robobay

    # Supply headroom, relaxed by one pylon. same 4s window reason
    _apply_supply_gate(mask, obs, slack=spec.TRAINING_SUPPLY_SLACK)

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
