# BetaStar

A fully autonomous StarCraft II agent whose imitation-learning policy
consistently defeats the built-in Medium bot. It learns from professional
replays with a causal Transformer, then improves through PPO reinforcement
learning in live games.

The repository includes replay parsing, state and action design, supervised
training, live inference, reinforcement learning, evaluation, and debugging.

## Making StarCraft learnable

The game's real-time action space is massive: arbitrary unit selections,
map coordinates, commands, and timings. Learning directly from those raw
actions would require far more data and compute than I could afford.

Inspired by DeepMind's AlphaStar, the project focuses on macro decision-making
while abstracting away mechanical tasks. The model chooses between 32 strategic
actions, such as building a structure or training a unit. The execution layer
handles details such as choosing a worker and finding a legal build location,
greatly reducing the amount of training needed.

Rule-based controllers handle routine behaviours such as worker distribution
and army management, while the learned policy makes strategic decisions.

## Replay-to-game data pipeline

Built a Python pipeline that turns 700+ professional replays into 193,000
training decisions. Reconstructs historical game states and maps player
commands into the same 32 actions used by the live bot.

The parser and bot share the same input format and decision timing, keeping
training data aligned with live gameplay.

## Transformer policy

```text
76 game-state features (input)
        -> 128-dimensional input projection
        -> sinusoidal positional encoding
        -> 4 causal Transformer encoder layers (4 attention heads)
        -> 32 action logits
        -> state-dependent legality mask
        -> decided action (output)
```

- Causal attention uses the match history without leaking future information.
- Class-weighted loss and macro-F1 stop common actions from drowning out rare
  but important decisions.
- Action masking removes invalid choices based on prerequisites and supply.

## Two-stage learning

### 1. Imitation learning

The Transformer first learns from decisions made in professional replays. This
gives the agent a useful starting policy for PPO fine-tuning in live games.

The IL policy performs well on familiar states but struggles when its own
mistakes push the game outside the replay distribution. PPO addresses this by
letting the agent learn from states produced by its own actions.

### 2. PPO reinforcement learning

PPO fine-tunes the policy through live games against the built-in Zerg bot.
Structured rewards guide opening timings, production, and economy management
while exposing the policy to states that do not appear in the replay dataset.

The PPO system uses an actor-critic model, collects
on-policy game rollouts, and uses GAE to connect
delayed rewards to earlier decisions. Clipped updates, entropy, gradient
clipping, and KL regularization against the original imitation policy keep
fine-tuning stable and limit catastrophic forgetting.

## Evaluation and observability

Optional JSONL traces record action probabilities, masking interventions, and
execution outcomes. Concise console summaries are shown by default; use
`--decision-log` for full traces and `--log-level DEBUG` for detailed console
output. A fixed-seed evaluator compares IL and PPO checkpoints under the same
conditions.

## Running the project

### Requirements

- Python 3.12
- StarCraft II installed
- The `AbyssalReefLE` map installed locally
- A CUDA-capable GPU is optional but recommended for training

Create an environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Play one unrestricted game with the latest PPO policy:

```powershell
python source\run.py --checkpoint checkpoints\ppo_opening.pt
```

Continue PPO training from the latest checkpoint:

```powershell
python source\rl\train.py --updates 10 --episodes-per-update 6
```

Compare IL, latest PPO, and best PPO on held-out seeds:

```powershell
python source\rl\eval.py --games 12 --modes sampled --seed 1000
```

PPO training resumes from `checkpoints/ppo_opening.pt` by default. Pass
`--fresh` to restart from the imitation-learning checkpoint.

## Project layout

```text
source/
  run.py                 Play games with a trained policy
  replay_parser.py       Build IL sequences from SC2 replays
  model.py               Transformer model and IL training
  episode.py             Shared SC2 episode lifecycle
  gameplay/
    agent.py             Live policy and SC2 integration
    helpers.py           Economy, construction, and army controllers
  rl/
    train.py             PPO training entry point
    eval.py              Fixed-seed IL/PPO benchmark
    rollout.py           On-policy rollout collection
    ppo.py               Actor-critic model, GAE, and PPO updates
    reward.py            Opening reward definition and tracking
  telemetry/             Console and structured diagnostic logging
  analysis/              Replay, dataset, and model diagnostics
tests/                   Automated parser and RL tests
checkpoints/             IL and PPO model checkpoints
```
