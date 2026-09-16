from __future__ import annotations

import sys
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))

from sc2reader.events import (  # noqa: E402
    BasicCommandEvent,
    TargetUnitCommandEvent,
)

from replay_parser import is_command_event  # noqa: E402


class CommandEventTests(unittest.TestCase):
    def test_real_command_class_is_accepted(self):
        event = object.__new__(BasicCommandEvent)
        self.assertTrue(is_command_event(event))

    def test_sc2reader_update_subclass_is_rejected(self):
        # This mirrors UpdateTargetUnitCommandEvent: isinstance(event,
        # TargetUnitCommandEvent) is true, but it is an internal delta update,
        # not another player-issued command.
        update_type = type(
            "UpdateTargetUnitCommandEvent",
            (TargetUnitCommandEvent,),
            {},
        )
        event = object.__new__(update_type)
        self.assertIsInstance(event, TargetUnitCommandEvent)
        self.assertFalse(is_command_event(event))


if __name__ == "__main__":
    unittest.main()
