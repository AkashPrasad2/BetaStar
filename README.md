# BetaStar

Training a bot to play StarCraft II.

2 stages: imitation learning from professional human replays, then PPO reinforcement learning.

## Project layout

```text
source/
  run.py                 Play games with a trained policy
  replay_parser.py       Convert SC2 replays into IL training data
  model.py               Transformer policy
  protoss_bot.py         Shared live bot and action execution
  episode.py             Shared SC2 episode lifecycle
  rl/
    train.py             PPO training entry point
    eval.py              Fixed-seed IL/PPO benchmark
    bot.py               Rollout collection
    ppo.py               Actor-critic model, GAE, and PPO updates
    reward.py            Opening reward definition and tracking
  analysis/              Replay and dataset inspection tools
tests/                   Automated tests, separate from runtime code
checkpoints/             IL and PPO model checkpoints
logs/                    Training and evaluation output
```

## Common commands

From the repository root:

```powershell
# Play one unrestricted game with the latest PPO policy.
.\venv\Scripts\python.exe source\run.py

# Continue PPO training from the latest PPO checkpoint.
.\venv\Scripts\python.exe source\rl\train.py --updates 10 --episodes-per-update 6

# Compare IL, latest PPO, and best PPO on held-out seeds.
.\venv\Scripts\python.exe source\rl\eval.py --games 12 --modes sampled --seed 1000

# Run the test suite.
.\venv\Scripts\python.exe -m unittest discover -s tests
```
