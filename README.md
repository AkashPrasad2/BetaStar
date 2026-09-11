# BetaStar

A fully autonomous agent that plays StarCraft II and
consistently defeats the built-in Medium bot. First learns from human
replays with a causal Transformer, then improves through PPO reinforcement
learning in live games.

Includes replay parsing, state and
action design, supervised training, live inference, reinforcement learning,
evaluation, and debugging.

## Making StarCraft learnable

The game's real-time action space is massive: arbitrary unit selections,
map coordinates, commands, and timings. Learning directly from those raw
actions would require far more data and compute than I could afford.

Inspired by Google's AplhaStar, the project simplifies the problem and focuses on macro decision making while abstracting more tedious tasks. The model chooses between 32 strategic actions such as building a structure, training a unit, etc.. The execution layer handles the mechanical details: choosing a worker, finding a legal build location, etc. greatly reducing the amount of training needed.

Rule-based controllers handle routine behaviours such as worker distribution, army management, etc.. Learned poilcy is decision-based.

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

The Transformer first learns from the decisions made in professional replays.
This gives the agent a useful starting policy, to be tuned by self-play later.

With IL alone, it could beat easy bots consistently but is prone to a mismatch in training/inference distributions causing very poor performance. As soon as it goes down a bad path it doesn't know what to do, leading to RL.

### 2. PPO reinforcement learning

With RL we can structure rewards to guide the agent and through self play, it becomes more robust, less prone to distribution shifts and behaves better in cases that diverge from what was seen in training data.

The PPO system uses an actor-critic model, collects
on-policy game rollouts, and uses GAE to connect
delayed rewards to earlier decisions. Clipped updates, entropy, gradient
clipping, and KL regularization against the original imitation policy keep
fine-tuning stable and limit catastrophic forgetting.

## Evaluation and observability

Comprehensive logging system recording preferred actions, probabilities, mask interventions etc.. Also has an evaluator comparing IL and RL checkpoints.

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
  protoss_bot.py         Live policy and action execution
  episode.py             Shared SC2 episode lifecycle
  rl/
    train.py             PPO training entry point
    eval.py              Fixed-seed IL/PPO benchmark
    bot.py               On-policy rollout collection
    ppo.py               Actor-critic model, GAE, and PPO updates
    reward.py            Opening reward definition and tracking
  analysis/              Replay, dataset, and model diagnostics
tests/                   Automated parser and RL tests
checkpoints/             IL and PPO model checkpoints
```
