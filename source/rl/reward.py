"""State-based, timing-aware reward shaping for the opening curriculum."""

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
    started_target: float | None = None
    started_deadline: float | None = None
    completed_target: float | None = None


@dataclass(frozen=True)
class OpeningRewardConfig:
    """Reward configuration for a finite opening-build episode."""

    milestones: tuple[MilestoneReward, ...]
    success_bonus: float = 3.0
    # Optional blanket failure penalty, retained as a CLI escape hatch. The
    # default is zero because per-target penalties are more informative.
    failure_penalty: float = 0.0
    missed_milestone_penalty: float = -0.40
    execution_failure_penalty: float = 0.0
    idle_nexus_penalty: float = -0.02
    idle_production_penalty: float = -0.015
    idle_production_penalty_cap: float = 0.06

    def to_dict(self) -> dict:
        return asdict(self)


def default_opening_reward() -> OpeningRewardConfig:
    """Data-informed targets for the one-gate expansion opening.

    These deadlines are based on the MaNa-v-Solar reference sequence plus
    roughly 8-15 seconds of slack. The Nexus is only required to be started:
    its reference completion (219s) occurs after the Cyber Core objective.
    """
    return OpeningRewardConfig(milestones=(
        MilestoneReward(
            "pylon", 60.0, 0.05, 0.15,
            started_target=28.0, started_deadline=44.0,
            completed_target=44.0,
        ),
        MilestoneReward(
            "gateway", 132.0, 0.10, 0.30,
            started_target=64.0, started_deadline=88.0,
            completed_target=110.0,
        ),
        MilestoneReward(
            "assimilator", 108.0, 0.10, 0.30,
            started_target=72.0, started_deadline=90.0,
            completed_target=96.0,
        ),
        MilestoneReward(
            "nexus", 136.0, 0.40, 0.0, required_phase="started",
            started_target=128.0, started_deadline=136.0,
        ),
        MilestoneReward(
            "cybernetics_core", 200.0, 0.20, 0.60,
            started_target=144.0, started_deadline=166.0,
            completed_target=180.0,
        ),
    ))


@dataclass(frozen=True)
class OpeningSnapshot:
    """The reward-relevant portion of the live SC2 state."""

    time_seconds: float
    started: frozenset[str]
    ready: frozenset[str]
    completion_times: Mapping[str, float] = field(default_factory=dict)
    idle_unsaturated_nexuses: int = 0
    affordable_idle_production: int = 0


_UNIT_TYPES = {
    "pylon": UnitTypeId.PYLON,
    "gateway": UnitTypeId.GATEWAY,
    "assimilator": UnitTypeId.ASSIMILATOR,
    "nexus": UnitTypeId.NEXUS,
    "cybernetics_core": UnitTypeId.CYBERNETICSCORE,
}

_PRODUCTION_OPPORTUNITIES = (
    (UnitTypeId.GATEWAY, UnitTypeId.ZEALOT),
    (UnitTypeId.STARGATE, UnitTypeId.PHOENIX),
    (UnitTypeId.ROBOTICSFACILITY, UnitTypeId.IMMORTAL),
)


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
    idle_unsaturated_nexuses = sum(
        1 for nexus in bot.townhalls.ready.idle
        if nexus.assigned_harvesters < nexus.ideal_harvesters
    )
    affordable_idle_production = sum(
        bot.structures(structure).ready.idle.amount
        for structure, unit in _PRODUCTION_OPPORTUNITIES
        if bot.can_afford(unit)
    )

    return OpeningSnapshot(
        time_seconds=float(bot.time),
        started=frozenset(started),
        ready=frozenset(ready),
        # ProtossBot observes these on every game step, so deadlines are judged
        # at sub-decision precision rather than rounded up to the next 4s tick.
        completion_times=dict(bot.milestone_times),
        idle_unsaturated_nexuses=idle_unsaturated_nexuses,
        affordable_idle_production=affordable_idle_production,
    )


_NON_FAILURE_RESULTS = {None, "issued", "no_op"}


def _execution_name(result) -> str | None:
    value = getattr(result, "value", result)
    return str(value) if value is not None else None


def timing_multiplier(
    event_time: float,
    target_time: float | None,
    deadline: float | None,
) -> float:
    """Full credit through target, then linear decay to zero at deadline."""
    if target_time is None or deadline is None:
        return 1.0
    if event_time <= target_time:
        return 1.0
    if event_time >= deadline:
        return 0.0
    if deadline <= target_time:
        return 0.0
    return (deadline - event_time) / (deadline - target_time)


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

    def _milestone_met(
        self, milestone: MilestoneReward, snapshot: OpeningSnapshot
    ) -> bool:
        if milestone.required_phase == "started":
            return (
                milestone.name in snapshot.started
                and self.started_times.get(milestone.name, float("inf"))
                <= milestone.deadline
            )
        return (
            milestone.name in snapshot.ready
            and self.completion_times.get(milestone.name, float("inf"))
            <= milestone.deadline
        )

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

        if snapshot.idle_unsaturated_nexuses:
            reward += self._add(
                "idle:nexus",
                self.config.idle_nexus_penalty
                * snapshot.idle_unsaturated_nexuses,
            )
        if snapshot.affordable_idle_production:
            idle_penalty = min(
                self.config.idle_production_penalty_cap,
                abs(self.config.idle_production_penalty)
                * snapshot.affordable_idle_production,
            )
            reward += self._add("idle:production", -idle_penalty)

        for milestone in self.config.milestones:
            name = milestone.name
            if name in snapshot.started and name not in self.seen_started:
                self.seen_started.add(name)
                self.started_times[name] = snapshot.time_seconds
                start_deadline = (
                    milestone.started_deadline
                    if milestone.started_deadline is not None
                    else milestone.deadline
                )
                amount = milestone.started_reward * timing_multiplier(
                    snapshot.time_seconds,
                    milestone.started_target,
                    start_deadline,
                )
                reward += self._add(
                    f"{name}:started", amount)

            if name in snapshot.ready and name not in self.seen_completed:
                self.seen_completed.add(name)
                self.completion_times[name] = snapshot.completion_times.get(
                    name, snapshot.time_seconds
                )
                completion_time = self.completion_times[name]
                amount = milestone.completed_reward * timing_multiplier(
                    completion_time,
                    milestone.completed_target,
                    milestone.deadline,
                )
                reward += self._add(
                    f"{name}:completed", amount)

        if terminal and not self.finalized:
            self.finalized = True
            missed = [
                milestone for milestone in self.config.milestones
                if not self._milestone_met(milestone, snapshot)
            ]
            self.goal_met = not missed
            if self.goal_met:
                reward += self._add(
                    "terminal:success", self.config.success_bonus
                )
            else:
                if self.config.failure_penalty:
                    reward += self._add(
                        "terminal:failure", self.config.failure_penalty
                    )
                for milestone in missed:
                    reward += self._add(
                        f"terminal:missed:{milestone.name}",
                        self.config.missed_milestone_penalty,
                    )

        return reward
