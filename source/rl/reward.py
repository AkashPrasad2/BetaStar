"""State-based reward shaping for the first three minutes of a game."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from sc2.ids.unit_typeid import UnitTypeId


@dataclass(frozen=True)
class MilestoneReward:
    """One structure objective and its one-time shaping rewards."""

    name: str
    deadline: float
    started_reward: float
    completed_reward: float
    required_phase: str = "completed"


@dataclass(frozen=True)
class OpeningRewardConfig:
    """Reward configuration for a finite opening-build episode."""

    milestones: tuple[MilestoneReward, ...]
    success_bonus: float = 2.0
    failure_penalty: float = -1.0
    execution_failure_penalty: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def default_opening_reward() -> OpeningRewardConfig:
    """Data-informed targets for the one-gate expansion opening.

    These deadlines are based on the MaNa-v-Solar reference sequence plus
    roughly 8-15 seconds of slack. The Nexus is only required to be started:
    its reference completion (219s) occurs after the Cyber Core objective.
    """
    return OpeningRewardConfig(milestones=(
        MilestoneReward("pylon", 60.0, 0.05, 0.15),
        MilestoneReward("gateway", 132.0, 0.10, 0.30),
        MilestoneReward("assimilator", 108.0, 0.05, 0.15),
        MilestoneReward(
            "nexus", 136.0, 0.20, 0.0, required_phase="started"
        ),
        MilestoneReward("cybernetics_core", 200.0, 0.20, 0.60),
    ))


@dataclass(frozen=True)
class OpeningSnapshot:
    """The reward-relevant portion of the live SC2 state."""

    time_seconds: float
    started: frozenset[str]
    ready: frozenset[str]
    completion_times: Mapping[str, float] = field(default_factory=dict)


_UNIT_TYPES = {
    "pylon": UnitTypeId.PYLON,
    "gateway": UnitTypeId.GATEWAY,
    "assimilator": UnitTypeId.ASSIMILATOR,
    "nexus": UnitTypeId.NEXUS,
    "cybernetics_core": UnitTypeId.CYBERNETICSCORE,
}


def snapshot_opening_state(bot) -> OpeningSnapshot:
    """Read actual pending/completed structures rather than model intentions."""
    started: set[str] = set()
    ready: set[str] = set()
    for name, unit_type in _UNIT_TYPES.items():
        ready_count = bot.structures(unit_type).ready.amount
        any_count = bot.structures(unit_type).amount
        pending = float(bot.already_pending(unit_type))
        # The starting Nexus is baseline state. For this milestone, "started"
        # means an expansion has been ordered and "ready" means two townhalls.
        target_count = 2 if name == "nexus" else 1
        if any_count >= target_count or pending > 0:
            started.add(name)
        if ready_count >= target_count:
            ready.add(name)
    return OpeningSnapshot(
        time_seconds=float(bot.time),
        started=frozenset(started),
        ready=frozenset(ready),
        # ProtossBot observes these on every game step, so deadlines are judged
        # at sub-decision precision rather than rounded up to the next 4s tick.
        completion_times=dict(bot.milestone_times),
    )


_NON_FAILURE_RESULTS = {None, "issued", "no_op"}


def _execution_name(result) -> str | None:
    value = getattr(result, "value", result)
    return str(value) if value is not None else None


class OpeningRewardTracker:
    """Turn state transitions into non-repeatable rewards."""

    def __init__(self, config: OpeningRewardConfig):
        self.config = config
        self.initialized = False
        self.finalized = False
        self.seen_started: set[str] = set()
        self.seen_completed: set[str] = set()
        self.started_times: dict[str, float] = {}
        self.completion_times: dict[str, float] = {}
        self.total_reward = 0.0
        self.breakdown: dict[str, float] = {}
        self.goal_met = False

    def reset(self, snapshot: OpeningSnapshot) -> None:
        """Establish a baseline without rewarding pre-existing structures."""
        self.initialized = True
        self.seen_started = set(snapshot.started)
        self.seen_completed = set(snapshot.ready)
        self.started_times = {
            name: snapshot.time_seconds for name in snapshot.started
        }
        self.completion_times = {
            name: snapshot.completion_times.get(name, snapshot.time_seconds)
            for name in snapshot.ready
        }

    def _add(self, key: str, amount: float) -> float:
        self.breakdown[key] = self.breakdown.get(key, 0.0) + amount
        self.total_reward += amount
        return amount

    def observe(
        self,
        snapshot: OpeningSnapshot,
        execution_result=None,
        *,
        terminal: bool = False,
    ) -> float:
        """Return reward earned since the preceding policy decision."""
        if not self.initialized:
            self.reset(snapshot)
            if not terminal:
                return 0.0

        reward = 0.0
        execution_name = _execution_name(execution_result)
        if execution_name not in _NON_FAILURE_RESULTS:
            reward += self._add(
                f"execution_failure:{execution_name}",
                self.config.execution_failure_penalty,
            )

        for milestone in self.config.milestones:
            name = milestone.name
            if name in snapshot.started and name not in self.seen_started:
                self.seen_started.add(name)
                self.started_times[name] = snapshot.time_seconds
                reward += self._add(
                    f"{name}:started", milestone.started_reward)

            if name in snapshot.ready and name not in self.seen_completed:
                self.seen_completed.add(name)
                self.completion_times[name] = snapshot.completion_times.get(
                    name, snapshot.time_seconds
                )
                reward += self._add(
                    f"{name}:completed", milestone.completed_reward)

        if terminal and not self.finalized:
            self.finalized = True
            def milestone_met(milestone: MilestoneReward) -> bool:
                if milestone.required_phase == "started":
                    return (
                        milestone.name in snapshot.started
                        and self.started_times.get(
                            milestone.name, float("inf"))
                        <= milestone.deadline
                    )
                return (
                    milestone.name in snapshot.ready
                    and self.completion_times.get(
                        milestone.name, float("inf"))
                    <= milestone.deadline
                )

            self.goal_met = all(
                milestone_met(milestone)
                for milestone in self.config.milestones
            )
            terminal_reward = (
                self.config.success_bonus
                if self.goal_met else self.config.failure_penalty
            )
            key = "terminal:success" if self.goal_met else "terminal:failure"
            reward += self._add(key, terminal_reward)

        return reward
