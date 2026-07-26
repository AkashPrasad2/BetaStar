"""
replay_capture_log.py

Prints a human-readable, per-window log of what the parser extracts from a single
replay: resources, key state, and the action captured in each 4s grid window.
Useful for eyeballing whether labels line up with the state the model will see.
"""

import os
import sys
from pathlib import Path

import sc2reader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from replay_parser import (  # noqa: E402
    ReplayParser, WindowedState, GRID_INTERVAL_SECONDS,
)
from actions import ACTIONS  # noqa: E402

DEFAULT_REPLAY_DIR = r"C:\dev\BetaStar\replays\raw"


def analyze_replay(replay_path: str, max_windows: int | None = None):
    print(f"Analyzing replay: {replay_path}")
    try:
        replay = sc2reader.load_replay(replay_path, load_level=4)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load replay: {exc}")
        return

    parser = ReplayParser(debug=False)
    pid = parser.find_protoss_pid(replay)
    if pid is None:
        print("No Protoss player found in this replay.")
        return

    state = WindowedState(replay, pid)
    grid_actions, action_last = parser._collect_actions(replay, pid)
    last_window = max(len(state) - 1, action_last)
    if max_windows is not None:
        last_window = min(last_window, max_windows - 1)

    print(f"\n--- Replay Capture Log ---")
    print(f"Grid interval: {GRID_INTERVAL_SECONDS}s | fps={state.fps:.1f} | "
          f"windows={len(state)}")
    print(f"{'Win':<5} | {'Time':<7} | {'Mins':<5} | {'Gas':<5} | "
          f"{'Nex':<3} | {'Gate':<4} | {'WG':<3} | {'PendS':<5} | "
          f"{'Action'}")
    print("-" * 88)

    for w in range(last_window + 1):
        idx = min(w, len(state) - 1)
        win = state.windows[idx]
        action_id = grid_actions.get(w, 0)
        name = ACTIONS[action_id] if action_id < len(ACTIONS) else f"?{action_id}"
        pend_total = sum(win["structures_pending"].values())
        print(f"{w:<5} | {win['time']:<7.0f} | {int(win['minerals']):<5} | "
              f"{int(win['vespene']):<5} | "
              f"{win['structures_done'].get('NEXUS', 0):<3} | "
              f"{win['structures_done'].get('GATEWAY', 0):<4} | "
              f"{win['structures_done'].get('WARPGATE', 0):<3} | "
              f"{pend_total:<5} | {name}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_replay(sys.argv[1])
    else:
        if not Path(DEFAULT_REPLAY_DIR).is_dir():
            print(f"No replay dir {DEFAULT_REPLAY_DIR}; pass a path instead.")
        else:
            replays = [f for f in os.listdir(DEFAULT_REPLAY_DIR)
                       if f.endswith(".SC2Replay")]
            if replays:
                pick = next((f for f in replays if len(f) > 12), replays[0])
                analyze_replay(os.path.join(DEFAULT_REPLAY_DIR, pick))
            else:
                print(f"No replays found in {DEFAULT_REPLAY_DIR}")
