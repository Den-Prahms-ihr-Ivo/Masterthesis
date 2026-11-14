"""Data objects used throughout the experiment.

The dataclasses are frozen so a repair cannot change an earlier instance
or schedule by accident. Solver objects and file handling belong elsewhere.

Volunteer and shift ids start at zero and match their tuple positions.
A Schedule is a frozenset of (volunteer_id, shift_id) pairs. An absent pair
means that assignment is not used. A shift with no assignments is cancelled.
Events produce a new Instance through perturbations.apply_event().
"""

from __future__ import annotations

import hashlib
import json

from enum import StrEnum
from dataclasses import dataclass, field, fields, is_dataclass, replace
from fractions import Fraction
from typing import NamedTuple, Union

VolunteerId = int
ShiftId = int
Cell = tuple[VolunteerId, ShiftId]
Schedule = frozenset[Cell]


def _replace_at(items: tuple, index: int, item, label: str) -> tuple:
    """Return a tuple with one position replaced; negative indices are rejected."""
    if not 0 <= index < len(items):
        raise ValueError(f"no {label} with id {index} (valid ids: 0..{len(items) - 1})")
    return items[:index] + (item,) + items[index + 1 :]


class SystematicError(Exception):
    """An error serious enough to stop starting further jobs.

    This is for problems with the code or run setup, such as an oracle mismatch
    or an incompatible resume. The worker writes ABORT for these errors.
    A new check can inherit from this class to use the same stopping rule.
    """


# Instance


@dataclass(frozen=True)
class UtilityWeights:
    """Integer quality weights from Table A1, including the buffer limit."""

    beta_mentor: int = 1
    beta_buffer: int = 5
    buffer_cap: int = 2  # Maximum rewarded surplus per shift (b_s).
    gamma_extra: int = 10


@dataclass(frozen=True)
class Volunteer:
    """One volunteer, with availability, skills, preferences and a workload cap."""

    id: VolunteerId
    available: frozenset[int]  # Available period ids, where a_wp = 1.
    skill: tuple[int, ...]  # Skill k_wr for each role, from 0 to 3.
    seniority: int  # Seniority sigma_w, from 0 to 2.
    cap: int  # Maximum number of shifts, L_w.
    pref_role: tuple[int, ...]  # Role preferences p^role_wr, from 0 to 2.
    pref_time: tuple[int, ...]  # Period preferences p^time_wp, from 0 to 2.


@dataclass(frozen=True)
class Shift:
    """One active period-role combination and its current staffing requirements."""

    id: ShiftId
    period: int
    role: int
    demand: int  # Current demand d_s; it must stay at least 1.
    min_skill: int  # Required skill level kappa_s, either 1 or 2.
    expert: bool  # Whether a skill-3 expert is required (e_s).


@dataclass(frozen=True)
class InstanceMeta:
    """Keep the generation seed and experiment class alongside the instance."""

    size_class: int  # demand class (20/40/80); 40 for the under-supplied class
    scale: int  # perturbation scale k
    seed: int  # Candidate-generation seed, also used for tie-break ranks.
    under_supplied: bool = False


@dataclass(frozen=True)
class Instance:
    """A complete scheduling problem, checked when it is constructed."""

    volunteers: tuple[Volunteer, ...]
    shifts: tuple[Shift, ...]
    n_periods: int
    n_roles: int
    meta: InstanceMeta
    weights: UtilityWeights = field(default_factory=UtilityWeights)

    MAX_SKILL: int = 3
    MAX_SENIORITY: int = 2
    MAX_PREF: int = 2
    KNOWN_SIZE_CLASSES = frozenset({20, 40, 80})
    KNOWN_SCALES = frozenset({1, 2, 4})

    def __post_init__(self) -> None:
        """Check the instance structure and allowed values from exposé 4.2 and A1.

        The checks also catch wrong tuple lengths and ids before they reach the solver.
        Small test instances may use different dimensions from the experiment defaults.
        This validates the input data; it does not prove that full coverage is possible.
        """
        if self.MAX_SKILL < 2:
            raise ValueError(
                f"MAX_SKILL has to be greater than 2, got {self.MAX_SKILL}"
            )

        expected_tuple_sets = ["shifts", "volunteers"]
        for e in expected_tuple_sets:
            if not isinstance(getattr(self, e), tuple):
                raise ValueError(f"{e} is supposed to be a tuple.")

        if not all(
            volunteer.id == idx for idx, volunteer in enumerate(self.volunteers)
        ):
            raise ValueError(
                f"Volunteer IDs are not in the correct order: f{[v.id for v in self.volunteers]}"
            )

        if not all(shift.id == idx for idx, shift in enumerate(self.shifts)):
            raise ValueError(
                f"Shift IDs are not in the correct order: f{[v.id for v in self.shifts]}"
            )

        _seen_slots: set[tuple[int, int]] = set()
        for shift in self.shifts:
            # Period ids must fit the availability arrays (exposé 4.2; default: 28).
            if not 0 <= shift.period < self.n_periods:
                raise ValueError(
                    f"Shift #{shift.id}'s period index ({shift.period}) is out of range (0 - {self.n_periods})"
                )

            # Role ids must fit the skill and preference tuples (default: 5 roles).
            if not 0 <= shift.role < self.n_roles:
                raise ValueError(
                    f"Shift #{shift.id}'s role index ({shift.role}) is out of range (0 - {self.n_roles})"
                )

            # §4.2: κ_s ∈ {1, 2}
            if shift.min_skill not in range(1, self.MAX_SKILL):
                raise ValueError(
                    f"Shift #{shift.id}'s minimum required skill ({shift.min_skill}) is out of range (1 - {self.MAX_SKILL-1})"
                )

            # P1 may lower demand, but it must stay at least 1 (exposé 4.4).
            if not shift.demand > 0:
                raise ValueError(
                    f"Shift #{shift.id}'s demand has to be greater than 0."
                )

            combo = (shift.period, shift.role)
            if combo in _seen_slots:
                raise ValueError(f"Duplicate period and role combination: {combo}")

            _seen_slots.add(combo)

        for volunteer in self.volunteers:

            # Check tuple lengths before looking up values by role or period.
            if not isinstance(volunteer.available, frozenset):
                raise ValueError(
                    f"Volunteer #{volunteer.id}'s available is supposed to be a frozen set."
                )

            expected_tuple_sets = ["skill", "pref_role", "pref_time"]
            for e in expected_tuple_sets:
                if not isinstance(getattr(volunteer, e), tuple):
                    raise ValueError(
                        f"Volunteer #{volunteer.id}'s {e} is supposed to be a tuple."
                    )

            if not len(volunteer.skill) == self.n_roles:
                raise ValueError(
                    f"Volunteer #{volunteer.id}'s skill has the wrong length got {len(volunteer.skill)}, expected {self.n_roles}"
                )
            if not len(volunteer.pref_time) == self.n_periods:
                raise ValueError(
                    f"Volunteer #{volunteer.id}'s preffered time has the wrong length got {len(volunteer.pref_time)}, expected {self.n_periods}"
                )
            if not len(volunteer.pref_role) == self.n_roles:
                raise ValueError(
                    f"Volunteer #{volunteer.id}'s preffered roles has the wrong length got {len(volunteer.pref_role)}, expected {self.n_roles}"
                )

            # Skills 0..3 are used by the novice, mentor and expert rules (exposé 4.2).
            if any(s for s in volunteer.skill if s not in range(self.MAX_SKILL + 1)):
                raise ValueError(
                    f"Volunteer #{volunteer.id}'s skill ({volunteer.skill}) is not in range [0, {self.MAX_SKILL}]"
                )

            # §4.2: a_wp ∈ {0, 1} for p ∈ 𝒫
            if any(a for a in volunteer.available if a not in range(self.n_periods)):
                raise ValueError(
                    f"Volunteer #{volunteer.id}'s availability ({volunteer.available}) is is not in range [0, f{volunteer.available}]"
                )

            # §4.2: σ_w ∈ {0, 1, 2}
            if volunteer.seniority not in range(self.MAX_SENIORITY + 1):
                raise ValueError(
                    f"Volunteer #{volunteer.id}'s seniority ({volunteer.seniority}) is is not in range [0, f{self.MAX_SENIORITY}]"
                )

            # §4.2: workload cap L_w; A1: {10, 11, 12}
            if not volunteer.cap > 0:
                raise ValueError(
                    f"Volunteer #{volunteer.id}'s capacity has to be greater than 0"
                )

            # §4.2 / A1: p ∈ {0, 1, 2}
            if any(r for r in volunteer.pref_role if r not in range(self.MAX_PREF + 1)):
                raise ValueError(
                    f"Volunteer #{volunteer.id} has illegal role preference values for roles"
                )
            if any(r for r in volunteer.pref_time if r not in range(self.MAX_PREF + 1)):
                raise ValueError(
                    f"Volunteer #{volunteer.id} has illegal time preference values for roles"
                )

        # Check the experiment metadata as well as the scheduling data.
        # §4.3: main classes n ∈ {20, 40, 80};
        if self.meta.size_class not in self.KNOWN_SIZE_CLASSES:
            raise ValueError(f"Unknown Size clas: {self.meta.size_class}")

        # The nominal scales are 1, 2 and 4; reject zero or unknown scales (4.4).
        if self.meta.scale not in self.KNOWN_SCALES:
            raise ValueError(f"Unknown scale: {self.meta.scale}")

        # §4.3: the under-supplied class keeps “the nominal 40-volunteer demand”
        if self.meta.under_supplied and self.meta.size_class < 40:
            raise ValueError(
                f"Undersupplied and size class not 40: {self.meta.size_class}"
            )

    @property
    def n_cells(self) -> int:
        """Return J = |W| * |S|, including cells where the volunteer is unavailable.

        Churn and baseline drift use this same denominator. Counting only available
        cells would change the denominator after an availability event.
        """
        return len(self.volunteers) * len(self.shifts)

    def with_volunteer(self, volunteer: Volunteer) -> Instance:
        """Return a new, validated instance with this volunteer replaced by id."""
        volunteers = _replace_at(self.volunteers, volunteer.id, volunteer, "volunteer")
        return replace(self, volunteers=volunteers)

    def with_shift(self, shift: Shift) -> Instance:
        """Return a new, validated instance with this shift replaced by id."""
        shifts = _replace_at(self.shifts, shift.id, shift, "shift")
        return replace(self, shifts=shifts)


# Events


@dataclass(frozen=True)
class DemandChange:
    """P1: add delta to the shift demand; the sampler chooses +k or -k."""

    shift: ShiftId
    delta: int
    fallback: bool = False  # True if the fallback rule chose the target


@dataclass(frozen=True)
class AvailabilityRemoval:
    """P2: remove availability for each listed (volunteer, period) pair."""

    pairs: tuple[tuple[VolunteerId, int], ...]
    fallback: bool = False  # True when the eligible targets needed a top-up (decision 13.9).


@dataclass(frozen=True)
class PreferenceChange:
    """P3: set each listed (volunteer, role) preference to its new value."""

    changes: tuple[tuple[VolunteerId, int, int], ...]
    fallback: bool = False  # True when the eligible targets needed a top-up (decision 13.9).


@dataclass(frozen=True)
class AvailabilityRestoration:
    """Restore the pair removed at RQ2 step 1, normally at step 6."""

    volunteer: VolunteerId
    period: int


Event = Union[
    DemandChange, AvailabilityRemoval, PreferenceChange, AvailabilityRestoration
]


@dataclass(frozen=True)
class Sequence:
    kind: SequenceKind
    profile: SequenceProfile
    seed: int
    events: tuple[Event, ...]  # length T
    selected: tuple[VolunteerId, int] | None = None  # RQ2 forcing pair

    def digest(self) -> str:
        """Hash the sequence fields so the event list can be compared across arms."""
        return _sha256(_stable_json(self))


# Arms and results


class PolicyName(StrEnum):
    QO = "QO"
    SF = "SF"
    QL = "QL"
    WQS = "WQS"
    HWF = "HWF"


class ReferenceMode(StrEnum):
    ANCHORED = "anchored"
    CHAINED = "chained"


class StageStatus(StrEnum):
    OPTIMAL = "optimal"
    TIME_LIMIT = "time_limit"
    NO_SOLUTION = "no_solution"


class Classification(StrEnum):
    BINDING = "binding"
    RELEVANT = "relevant"
    INERT = "inert"
    UNDETERMINED = "undetermined"


class SequenceKind(StrEnum):
    RQ1 = "rq1"
    RQ2 = "rq2"
    INDEPENDENT = "independent"


class SequenceProfile(StrEnum):
    BALANCED = "balanced"
    AVAILABILITY = "availability"
    PREFERENCE = "preference"
    RQ2 = "rq2"


@dataclass(frozen=True)
class Arm:
    policy: PolicyName
    reference: ReferenceMode | None  # None only for QO
    lam: int | None = None  # WQS: utility points per changed cell
    alpha: Fraction | None = None  # HWF: history weight

    @property
    def name(self) -> str:
        """e.g. 'QO', 'SF-chained', 'WQS-chained-l2', 'HWF-chained-a1/2'."""
        if not self.policy:
            raise ValueError("Each Arm needs at least a policy definition")

        return (
            self.policy
            + ("-" + self.reference if self.reference else "")
            + ("-l" + str(self.lam) if self.lam else "")
            + ("-a" + str(self.alpha.numerator) if self.alpha else "")
            + (
                "/" + str(self.alpha.denominator)
                if self.alpha and self.alpha.denominator != 1
                else ""
            )
        )


@dataclass(frozen=True)
class StageRecord:
    name: str  # objective spec name, e.g. "MaxQuality"
    status: StageStatus
    value: int | None  # incumbent objective value (integer)
    bound: float | None  # dual bound at termination
    seconds: float
    retried: bool = False


class StageResult(NamedTuple):
    """The answer from one call to a stage solver.

    A schedule and value describe the best solution found. The status says
    whether it was proven optimal. The bound can differ from the value after
    a time limit. No solution is represented by schedule=None and value=None.
    The runner adds the stage name, elapsed time and retry flag later.
    """

    schedule: Schedule | None  # None if no schedule was found
    value: int | None  # objective value of schedule (the incumbent)
    status: StageStatus = StageStatus.OPTIMAL
    bound: float | None = None  # dual bound; None if the solver has none (brute force)


@dataclass(frozen=True)
class Step:
    t: int
    schedule: Schedule
    stages: tuple[StageRecord, ...]  # policy stages + tie-break (coverage is shared)
    # No classification here: metrics.classifications() derives it from stored data.


@dataclass(frozen=True)
class LexResult:
    """The final schedule and recorded stages of one lexicographic solve."""

    schedule: Schedule | None  # None only if a stage had no solution after the retry
    stages: tuple[StageRecord, ...]

    @property
    def certified(self) -> bool:
        """Return True when a schedule exists and every recorded stage is optimal.

        A feasible schedule alone is not enough: a timed-out stage can still have a
        better solution. A successful retry counts as certified. This property checks
        the recorded statuses, not the mathematical correctness of the solver model.
        """
        return (
            self.schedule is not None
            and len(self.stages) > 0
            and all(record.status == StageStatus.OPTIMAL for record in self.stages)
        )


@dataclass(frozen=True)
class Trajectory:
    arm: Arm
    steps: tuple[Step, ...]
    ended_early: bool = False


@dataclass(frozen=True)
class CoverageStep:
    t: int
    lambda_star: int
    record: StageRecord


@dataclass(frozen=True)
class AcceptanceLog:
    """Record which candidate was accepted and how many attempts were needed."""

    base_seed: int  # seed from the protocol's seed list
    used_seed: int  # Accepted candidate seed, equal to instance.meta.seed.
    draws: int  # candidates drawn for this base seed
    rejected: int  # Rejected main candidates, including calls with no returned schedule.
    replaced: bool  # True if base_seed failed and a reserve seed was used
    lambda_star_0: int


@dataclass(frozen=True)
class SequenceResult:
    """Store the inputs and outcomes of one whole perturbation sequence.

    The base instance is included in full. Later metric calculations can then
    reconstruct the steps without drawing the instance again.
    """

    cell: str  # Experiment cell, e.g. "main"; size is stored in instance.meta.
    instance: Instance  # accepted base instance, before any event
    acceptance: AcceptanceLog
    sequence: Sequence
    s0: Schedule
    s0_stages: tuple[StageRecord, ...]
    coverage: tuple[CoverageStep, ...]
    trajectories: tuple[Trajectory, ...]
    manifest: dict


def _json_default(obj: object) -> dict[str, object]:
    """Convert supported dataclasses to tagged dictionaries for the digest."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            "__type__": type(obj).__name__,
            **{f.name: getattr(obj, f.name) for f in fields(obj)},
        }
    raise TypeError(f"cannot digest {type(obj).__name__}: {obj!r}")


def _stable_json(obj: object) -> str:
    """Use sorted JSON keys so equal data gives the same digest input."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
