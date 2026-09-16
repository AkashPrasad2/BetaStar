"""
observation_wrapper.py — live (inference-time) observation builder
==================================================================
Queries the live SC2 API for raw game state and hands it to
obs_spec.build_obs_vector(), which is the same function the replay parser uses
to build training observations. All layout/binning/normalization lives in
obs_spec so the two paths cannot diverge.
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

import obs_spec
from obs_spec import (
    STRUCTURES, UNITS, PENDING_STRUCTURES, OBS_SIZE, build_obs_vector,
)

# Map the canonical names in obs_spec to burnysc2 type ids.
STRUCTURE_IDS = {
    "NEXUS":            UnitTypeId.NEXUS,
    "PYLON":            UnitTypeId.PYLON,
    "GATEWAY":          UnitTypeId.GATEWAY,
    "FORGE":            UnitTypeId.FORGE,
    "TWILIGHTCOUNCIL":  UnitTypeId.TWILIGHTCOUNCIL,
    "PHOTONCANNON":     UnitTypeId.PHOTONCANNON,
    "SHIELDBATTERY":    UnitTypeId.SHIELDBATTERY,
    "TEMPLARARCHIVE":   UnitTypeId.TEMPLARARCHIVE,
    "ROBOTICSBAY":      UnitTypeId.ROBOTICSBAY,
    "ROBOTICSFACILITY": UnitTypeId.ROBOTICSFACILITY,
    "ASSIMILATOR":      UnitTypeId.ASSIMILATOR,
    "CYBERNETICSCORE":  UnitTypeId.CYBERNETICSCORE,
    "STARGATE":         UnitTypeId.STARGATE,
    "FLEETBEACON":      UnitTypeId.FLEETBEACON,
    "WARPGATE":         UnitTypeId.WARPGATE,
}

UNIT_IDS = {
    "PROBE":       UnitTypeId.PROBE,
    "ZEALOT":      UnitTypeId.ZEALOT,
    "STALKER":     UnitTypeId.STALKER,
    "HIGHTEMPLAR": UnitTypeId.HIGHTEMPLAR,
    "ARCHON":      UnitTypeId.ARCHON,
    "IMMORTAL":    UnitTypeId.IMMORTAL,
    "CARRIER":     UnitTypeId.CARRIER,
    "VOIDRAY":     UnitTypeId.VOIDRAY,
    "ADEPT":       UnitTypeId.ADEPT,
    "PHOENIX":     UnitTypeId.PHOENIX,
    "COLOSSUS":    UnitTypeId.COLOSSUS,
}

UPGRADE_CHAINS = {
    "GROUND_WEAPONS": (
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
        UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
    ),
    "SHIELDS": (
        UpgradeId.PROTOSSSHIELDSLEVEL1,
        UpgradeId.PROTOSSSHIELDSLEVEL2,
        UpgradeId.PROTOSSSHIELDSLEVEL3,
    ),
    "AIR_WEAPONS": (
        UpgradeId.PROTOSSAIRWEAPONSLEVEL1,
        UpgradeId.PROTOSSAIRWEAPONSLEVEL2,
        UpgradeId.PROTOSSAIRWEAPONSLEVEL3,
    ),
}

# Backwards-compatible aliases (older code imported these lists from here).
PROTOSS_STRUCTURES = [STRUCTURE_IDS[n] for n in STRUCTURES]
PROTOSS_UNITS = [UNIT_IDS[n] for n in UNITS]


class ObservationWrapper:
    """Converts live game state into the canonical observation vector."""

    def __init__(self):
        self.observation_size = OBS_SIZE

    def calculate_obs_size(self) -> int:
        return OBS_SIZE

    def get_observation(self, bot: BotAI) -> list[float]:
        structures_done = {
            name: bot.structures(tid).ready.amount
            for name, tid in STRUCTURE_IDS.items()
        }
        units_done = {
            name: bot.units(tid).amount
            for name, tid in UNIT_IDS.items()
        }
        # not_ready == physically under construction, which is the same
        # semantics the parser reconstructs from UnitInit..UnitDone.
        structures_pending = {
            name: bot.structures(STRUCTURE_IDS[name]).not_ready.amount
            for name in PENDING_STRUCTURES
        }
        # already_pending == ordered but not yet delivered, matching the
        # parser's command->birth matching.
        units_pending = {
            name: bot.already_pending(UNIT_IDS[name])
            for name in UNITS
        }
        upgrade_lvls = {
            key: self._committed_upgrade_level(bot, chain)
            for key, chain in UPGRADE_CHAINS.items()
        }

        return build_obs_vector(
            time_s=bot.time,
            minerals=bot.minerals,
            vespene=bot.vespene,
            supply_used=bot.supply_used,
            supply_cap=bot.supply_cap,
            structures_done=structures_done,
            units_done=units_done,
            structures_pending=structures_pending,
            units_pending=units_pending,
            upgrade_lvls=upgrade_lvls,
        )

    @staticmethod
    def _committed_upgrade_level(bot: BotAI, upgrade_ids) -> int:
        """
        Highest upgrade level that is complete OR currently researching.
        Matches the parser's pending-or-complete convention.
        """
        level = 0
        for i, uid in enumerate(upgrade_ids, start=1):
            if uid in bot.state.upgrades:
                level = i
            elif bot.already_pending_upgrade(uid) > 0:
                level = i
                break
        return level
