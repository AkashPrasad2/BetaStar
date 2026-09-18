from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

import observation_wrapper  # noqa: E402
from action_mask import build_legal_mask, build_training_mask  # noqa: E402
from actions import _research  # noqa: E402
from gameplay.helpers import ActionResult  # noqa: E402
from obs_spec import (  # noqa: E402
    ACTION_ID,
    IDX_AIR_WEAPONS_LVL,
    IDX_GROUND_WEAPONS_LVL,
    IDX_SHIELDS_LVL,
    OBS_SIZE,
    PEND_STRUCT_IDX,
    STRUCT_IDX,
    STRUCT_NORM,
    UPGRADE_NORM,
)
from replay_parser import _action_legal_numpy  # noqa: E402


class UpgradeMaskTests(unittest.TestCase):
    @staticmethod
    def _observation(*completed: str) -> torch.Tensor:
        obs = torch.zeros(1, OBS_SIZE)
        for name in completed:
            obs[0, STRUCT_IDX[name]] = 1.0 / STRUCT_NORM
        return obs

    def test_level_one_does_not_require_advanced_tech(self):
        obs = self._observation("FORGE", "CYBERNETICSCORE")
        mask = build_legal_mask(obs)[0]

        self.assertTrue(mask[ACTION_ID["upgrade_ground_weapons"]])
        self.assertTrue(mask[ACTION_ID["upgrade_shields"]])
        self.assertTrue(mask[ACTION_ID["upgrade_air_weapons"]])

    def test_higher_levels_require_advanced_tech(self):
        obs = self._observation("FORGE", "CYBERNETICSCORE")
        obs[0, IDX_GROUND_WEAPONS_LVL] = 1.0 / UPGRADE_NORM
        obs[0, IDX_SHIELDS_LVL] = 1.0 / UPGRADE_NORM
        obs[0, IDX_AIR_WEAPONS_LVL] = 1.0 / UPGRADE_NORM

        mask = build_legal_mask(obs)[0]
        self.assertFalse(mask[ACTION_ID["upgrade_ground_weapons"]])
        self.assertFalse(mask[ACTION_ID["upgrade_shields"]])
        self.assertFalse(mask[ACTION_ID["upgrade_air_weapons"]])

        obs[0, STRUCT_IDX["TWILIGHTCOUNCIL"]] = 1.0 / STRUCT_NORM
        obs[0, STRUCT_IDX["FLEETBEACON"]] = 1.0 / STRUCT_NORM
        mask = build_legal_mask(obs)[0]
        self.assertTrue(mask[ACTION_ID["upgrade_ground_weapons"]])
        self.assertTrue(mask[ACTION_ID["upgrade_shields"]])
        self.assertTrue(mask[ACTION_ID["upgrade_air_weapons"]])

    def test_training_mask_accepts_pending_advanced_tech(self):
        obs = self._observation("FORGE", "CYBERNETICSCORE")
        obs[0, IDX_GROUND_WEAPONS_LVL] = 1.0 / UPGRADE_NORM
        obs[0, IDX_SHIELDS_LVL] = 1.0 / UPGRADE_NORM
        obs[0, IDX_AIR_WEAPONS_LVL] = 1.0 / UPGRADE_NORM
        obs[0, PEND_STRUCT_IDX["TWILIGHTCOUNCIL"]] = 1.0 / STRUCT_NORM
        obs[0, PEND_STRUCT_IDX["FLEETBEACON"]] = 1.0 / STRUCT_NORM

        mask = build_training_mask(obs)[0]
        self.assertTrue(mask[ACTION_ID["upgrade_ground_weapons"]])
        self.assertTrue(mask[ACTION_ID["upgrade_shields"]])
        self.assertTrue(mask[ACTION_ID["upgrade_air_weapons"]])


class ResearchExecutionTests(unittest.TestCase):
    def test_specific_research_checks_extra_prerequisite(self):
        class Structures:
            def __init__(self, exists: bool):
                self.exists = exists

            @property
            def ready(self):
                return self

            def __bool__(self):
                return self.exists

        class Bot:
            def structures(self, unit_type):
                return Structures(unit_type == UnitTypeId.FORGE)

        result = _research(
            Bot(),
            UnitTypeId.FORGE,
            AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2,
            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
            requires=UnitTypeId.TWILIGHTCOUNCIL,
        )
        self.assertEqual(result, ActionResult.NO_PREREQ)


class ReplayLabelRuleTests(unittest.TestCase):
    def test_replay_labels_use_the_same_advanced_tech_rules(self):
        obs = [0.0] * OBS_SIZE
        obs[STRUCT_IDX["FORGE"]] = 1.0 / STRUCT_NORM
        obs[STRUCT_IDX["CYBERNETICSCORE"]] = 1.0 / STRUCT_NORM
        obs[IDX_GROUND_WEAPONS_LVL] = 1.0 / UPGRADE_NORM
        obs[IDX_SHIELDS_LVL] = 1.0 / UPGRADE_NORM
        obs[IDX_AIR_WEAPONS_LVL] = 1.0 / UPGRADE_NORM

        for action in (
            "upgrade_ground_weapons", "upgrade_shields",
            "upgrade_air_weapons",
        ):
            legal, _ = _action_legal_numpy(obs, ACTION_ID[action])
            self.assertFalse(legal)

        obs[PEND_STRUCT_IDX["TWILIGHTCOUNCIL"]] = 1.0 / STRUCT_NORM
        obs[PEND_STRUCT_IDX["FLEETBEACON"]] = 1.0 / STRUCT_NORM
        for action in (
            "upgrade_ground_weapons", "upgrade_shields",
            "upgrade_air_weapons",
        ):
            legal, _ = _action_legal_numpy(obs, ACTION_ID[action])
            self.assertTrue(legal)


class ObservationWrapperTests(unittest.TestCase):
    def test_obsolete_compatibility_aliases_are_removed(self):
        self.assertFalse(hasattr(observation_wrapper, "PROTOSS_STRUCTURES"))
        self.assertFalse(hasattr(observation_wrapper, "PROTOSS_UNITS"))


if __name__ == "__main__":
    unittest.main()
