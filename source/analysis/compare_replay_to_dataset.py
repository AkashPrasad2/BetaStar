"""
compare_replay_to_dataset.py - Comprehensive analysis of replay parsing

This script shows exactly what happens when a replay is parsed into training data:
- Which abilities are mapped to training actions
- Which abilities are ignored (unmapped)
- How actions are queued into 4-second windows
- What conflicts occur (actions illegal at snapshot time)
- Final action distribution in the training data
"""
import sc2reader
import numpy as np
import sys
import os

# Add parent directory to path to import replay_parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from replay_parser import (
    ReplayParser, GRID_INTERVAL_SECONDS, OBS_SIZE, is_command_event,
)

# Load the same replay
replay_path = r"C:\dev\BetaStar\replays\raw\Railgan v ShaDoWn - Abyssal Reef LE.SC2Replay"
#C:\dev\BetaStar\replays\raw\Classic v Rogue_ Game 2 - Tokamak LE.SC2Replay
print(f"Loading replay: {replay_path}")
print("=" * 120)
replay = sc2reader.load_replay(replay_path, load_level=4)

# Find Protoss player
protoss_player = None
for p in replay.players:
    if p.play_race == "Protoss":
        protoss_player = p
        break

if not protoss_player:
    print("No Protoss player found!")
    exit()

print(f"Protoss player: {protoss_player.name} (PID {protoss_player.pid})")
print(f"Grid interval: {GRID_INTERVAL_SECONDS}s windows")
print("=" * 120)

# Collect all command events from replay
all_commands = []
for event in replay.events:
    if is_command_event(event):
        if event.player.pid == protoss_player.pid:
            all_commands.append((event.second, event.ability_name))

print(f"\nTotal command events from Protoss player: {len(all_commands)}")

# Now show what the parser maps
print("\n" + "=" * 120)
print("STEP 1: EVENT TO ACTION MAPPING")
print("=" * 120)
print("\nThe parser uses EVENT_TO_ACTION dictionary to map replay abilities to action IDs.")
print("Only mapped abilities become training labels. Unmapped abilities are ignored.\n")

# Import the actual EVENT_TO_ACTION from replay_parser to ensure accuracy
parser = ReplayParser(debug=False)
EVENT_TO_ACTION = parser.EVENT_TO_ACTION

# Action names come from obs_spec, never a local copy: the private lists that
# used to live here silently went stale when the action space changed.
from obs_spec import ACTION_NAMES as _ACTION_LIST   # noqa: E402
ACTION_NAMES = dict(enumerate(_ACTION_LIST))

# Categorize all commands
mapped_commands = []
unmapped_commands = []

for t, ability in all_commands:
    action_id = EVENT_TO_ACTION.get(ability)
    if action_id is not None:
        mapped_commands.append((t, ability, action_id))
    else:
        unmapped_commands.append((t, ability))

print(f"Mapped commands (will be in training data):   {len(mapped_commands)}")
print(f"Unmapped commands (ignored):                   {len(unmapped_commands)}")

# Show mapped commands
if mapped_commands:
    print(f"\n{'Time':>8} | {'Replay Ability':<35} | {'→':<3} | {'Action ID':<10} | {'Training Action Name':<30}")
    print("-" * 120)
    for t, ability, action_id in mapped_commands[:30]:
        action_name = ACTION_NAMES[action_id]
        print(f"{t:7.1f}s | {ability:<35} | {'→':<3} | {action_id:<10} | {action_name:<30}")
    if len(mapped_commands) > 30:
        print(f"... and {len(mapped_commands) - 30} more mapped commands")

# Show unmapped commands
if unmapped_commands:
    print(f"\n\nUNMAPPED ABILITIES (these are IGNORED and will NOT appear in training data):")
    print("-" * 120)
    unmapped_set = {}
    for t, ability in unmapped_commands:
        if ability not in unmapped_set:
            unmapped_set[ability] = []
        unmapped_set[ability].append(t)
    
    for ability in sorted(unmapped_set.keys()):
        times = unmapped_set[ability]
        print(f"  {ability:<40} - {len(times):3d} occurrences (e.g., at t={times[0]:.1f}s)")
    print(f"\nThese {len(unmapped_commands)} commands are filtered out during parsing.")
else:
    print("\n✓ All commands are mapped!")

# Now actually parse the replay to see what ends up in the dataset
print("\n" + "=" * 120)
print("STEP 2: PARSING REPLAY INTO TRAINING DATA")
print("=" * 120)
print("\nNow running the actual parser to see what ends up in the final dataset...\n")

parser_with_debug = ReplayParser(debug=True)
seq = parser_with_debug.parse_replay(replay)

if seq is None:
    print("\nReplay was too short and was skipped.")
    exit()

print(f"\n{'='*120}")
print("STEP 3: FINAL TRAINING DATA ANALYSIS")
print("=" * 120)

actions = seq[:, OBS_SIZE].astype(int)  # action is the last column
print(f"\nTotal windows in parsed sequence: {len(actions)}")
print(f"Grid interval: {GRID_INTERVAL_SECONDS}s per window")
print(f"Game duration covered: {len(actions) * GRID_INTERVAL_SECONDS}s ({len(actions) * GRID_INTERVAL_SECONDS / 60:.1f} minutes)\n")

# Count action distribution in final dataset
action_distribution = {}
for action_id in actions:
    action_distribution[action_id] = action_distribution.get(action_id, 0) + 1

print("ACTION DISTRIBUTION IN FINAL TRAINING DATA:")
print("-" * 120)
print(f"{'Action ID':<12} | {'Action Name':<35} | {'Count':<10} | {'Percentage':<12}")
print("-" * 120)

for action_id in sorted(action_distribution.keys()):
    count = action_distribution[action_id]
    pct = 100 * count / len(actions)
    action_name = ACTION_NAMES.get(action_id, f"UNKNOWN_{action_id}")
    print(f"{action_id:<12} | {action_name:<35} | {count:<10} | {pct:>6.2f}%")

print("\n" + "=" * 120)
print("KEY INSIGHTS:")
print("=" * 120)

do_nothing_count = action_distribution.get(0, 0)
do_nothing_pct = 100 * do_nothing_count / len(actions)
print(f"\n1. do_nothing (action 0) represents {do_nothing_pct:.1f}% of training data")
print(f"   This is expected - most 4s windows have no macro action.")

action_count = len(actions) - do_nothing_count
print(f"\n2. Actual macro actions: {action_count} out of {len(actions)} windows ({100*action_count/len(actions):.1f}%)")

print(f"\n3. Commands in replay vs training data:")
print(f"   - Mapped commands in replay: {len(mapped_commands)}")
print(f"   - Actions in training data:  {action_count}")
if len(mapped_commands) > action_count:
    diff = len(mapped_commands) - action_count
    print(f"   - Difference: {diff} commands were demoted to do_nothing due to conflicts")
    print(f"     (action was illegal at the snapshot time - see CONFLICT messages above)")
elif len(mapped_commands) < action_count:
    print(f"   - Note: Some windows may have multiple commands queued")

print(f"\n4. Unmapped abilities: {len(unmapped_commands)} commands ignored (not in EVENT_TO_ACTION)")

print("\n" + "=" * 120)
