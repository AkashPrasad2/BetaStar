# BetaStar

Training a model to play StarCraft II.

## Policy evaluation

Run the current imitation-learning policy against the built-in Zerg AI on
Abyssal Reef:

```powershell
.\venv\Scripts\python.exe source\run.py --games 10 --difficulty easy
```

For the initial reinforcement-learning opening task, stop each game after 180
in-game seconds and measure whether a completed Pylon, Gateway, and Cybernetics
Core were observed by the cutoff:

```powershell
.\venv\Scripts\python.exe source\run.py --games 10 --difficulty easy --time-limit 180
```

Each run writes an aggregate `evaluation_*.json` report to `logs/`. Per-decision
JSONL logs can be disabled with `--no-decision-log`.
