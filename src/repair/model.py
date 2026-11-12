"""Plain data model. Immutable dataclasses only; no solver objects, no I/O.

Conventions
-----------
- Volunteers and shifts are indexed 0..n-1 / 0..|S|-1; ids equal positions.
- A Schedule is a frozenset of (volunteer_id, shift_id) cells.
- A shift is cancelled iff it has no assignees (open shifts need >= ceil(d_s/2) >= 1).
- Events never mutate an Instance; perturbations.apply_event returns a new one.
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
    """New tuple with items[index] replaced; index must be a real position (no negatives)."""
    if not 0 <= index < len(items):
        raise ValueError(f"no {label} with id {index} (valid ids: 0..{len(items) - 1})")
    return items[:index] + (item,) + items[index + 1 :]


# --------------------------------------------------------------------------- instance


@dataclass(frozen=True)
class UtilityWeights:
    """Objective weights from Table A1 (all integers)."""

    beta_mentor: int = 1
    beta_buffer: int = 5
    buffer_cap: int = 2  # b_s
    gamma_extra: int = 10


@dataclass(frozen=True)
class Volunteer:
    id: VolunteerId
    available: frozenset[int]  # periods with a_wp = 1
    skill: tuple[int, ...]  # k_wr per role, 0..3
    seniority: int  # sigma_w, 0..2
    cap: int  # L_w
    pref_role: tuple[int, ...]  # p^role_wr per role, 0..2
    pref_time: tuple[int, ...]  # p^time_wp per period, 0..2


@dataclass(frozen=True)
class Shift:
    id: ShiftId
    period: int
    role: int
    demand: int  # current requested crew d_s (>= 1)
    min_skill: int  # kappa_s in {1, 2}
    expert: bool  # e_s


@dataclass(frozen=True)
class InstanceMeta:
    """Experiment bookkeeping; never used by constraints or utility."""

    size_class: int  # demand class (20/40/80); 40 for the under-supplied class
    scale: int  # perturbation scale k
    seed: int  # base seed (also seeds the tie-break ranks)
    under_supplied: bool = False


@dataclass(frozen=True)
class Instance:
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
        """Checks that an Instance is a well-formed member of the problem class defined in exposé §4.2 and Table A1."""
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
            # §4.2: period set 𝒫, |𝒫| = 28 – Availability and one-shift-per-period constraints index by period
            if not 0 <= shift.period < self.n_periods:
                raise ValueError(
                    f"Shift #{shift.id}'s period index ({shift.period}) is out of range (0 - {self.n_periods})"
                )

            # §4.2: roles ℛ, |ℛ| = 5	Skills and role preferences are looked up by role
            if not 0 <= shift.role < self.n_roles:
                raise ValueError(
                    f"Shift #{shift.id}'s role index ({shift.role}) is out of range (0 - {self.n_roles})"
                )

            # §4.2: κ_s ∈ {1, 2}
            if shift.min_skill not in range(1, self.MAX_SKILL):
                raise ValueError(
                    f"Shift #{shift.id}'s minimum required skill ({shift.min_skill}) is out of range (1 - {self.MAX_SKILL-1})"
                )

            # §4.2: requested crew d_s; §4.4: P1 decreases only if d_s − k ≥ 1
            if not shift.demand > 0:
                raise ValueError(
                    f"Shift #{shift.id}'s demand has to be greater than 0."
                )

            combo = (shift.period, shift.role)
            if combo in _seen_slots:
                raise ValueError(f"Duplicate period and role combination: {combo}")

            _seen_slots.add(combo)

        for volunteer in self.volunteers:

            # Structural checks
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

            # §4.2: k_wr ∈ {0, …, 3}	Novice/mentor rules (k ≤ 1, k ≥ 2), κ_s and expert (k = 3) depend on these exact levels
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

        # ---- META DATA CHECKS --------------
        # §4.3: main classes n ∈ {20, 40, 80};
        if self.meta.size_class not in self.KNOWN_SIZE_CLASSES:
            raise ValueError(f"Unknown Size clas: {self.meta.size_class}")

        # §4.4: k = n/20 ∈ {1, 2, 4}, and k = 2 in the under-supplied class	P1 changes demand by ±0, P2 removes 0 pairs, P3 changes 0 preferences.
        if self.meta.scale not in self.KNOWN_SCALES:
            raise ValueError(f"Unknown scale: {self.meta.scale}")

        # §4.3: the under-supplied class keeps “the nominal 40-volunteer demand”
        if self.meta.under_supplied and self.meta.size_class < 40:
            raise ValueError(
                f"Undersupplied and size class not 40: {self.meta.size_class}"
            )

    @property
    def n_cells(self) -> int:
        """J = |W| · |S| over all cells (not the available ones): the normaliser of C_t and B_t."""
        return len(self.volunteers) * len(self.shifts)

    def with_volunteer(self, volunteer: Volunteer) -> Instance:
        """New instance with volunteers[volunteer.id] replaced."""
        volunteers = _replace_at(self.volunteers, volunteer.id, volunteer, "volunteer")
        return replace(self, volunteers=volunteers)

    def with_shift(self, shift: Shift) -> Instance:
        """New instance with shifts[shift.id] replaced."""
        shifts = _replace_at(self.shifts, shift.id, shift, "shift")
        return replace(self, shifts=shifts)


# --------------------------------------------------------------------------- events


@dataclass(frozen=True)
class DemandChange:
    """P1: d_s += delta (delta = +k or -k)."""

    shift: ShiftId
    delta: int
    fallback: bool = False  # True if the fallback rule chose the target


@dataclass(frozen=True)
class AvailabilityRemoval:
    """P2: a_wp := 0 for each (volunteer, period)."""

    pairs: tuple[tuple[VolunteerId, int], ...]


@dataclass(frozen=True)
class PreferenceChange:
    """P3: p^role_wr := new value for each (volunteer, role, new_value)."""

    changes: tuple[tuple[VolunteerId, int, int], ...]


@dataclass(frozen=True)
class AvailabilityRestoration:
    """RQ2 step 6: a_wp := 1 for exactly the pair removed at step 1."""

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
        """Stable hash of kind, profile, seed, events and selected (replay check across arms)."""
        return _sha256(_stable_json(self))


# --------------------------------------------------------------------------- arms and results


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
    """What one solver call returns for one stage (milp or brute force).

    The runner turns it into a StageRecord (adding the name, seconds and retried).
    Examples:
        StageResult(frozenset({(0, 0)}), 6)                       optimal, U = 6
        StageResult(s, 3, StageStatus.TIME_LIMIT, bound=6.0)      best found 3, proven <= 6
        StageResult(None, None, StageStatus.NO_SOLUTION)          nothing found
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
    """Outcome of one lexicographic solve (runner.solve_lexicographic)."""

    schedule: Schedule | None  # None only if a stage had no solution after the retry
    stages: tuple[StageRecord, ...]

    @property
    def certified(self) -> bool:
        """True iff a schedule exists and every stage is 'optimal'.
        One lexicographic solve, for example S₀ or one arm's repair at step t, runs several stages in a row (MinDistance, then MaxQuality, then MaxTieBreak).
        LexResult bundles what came out: the final schedule and one StageRecord per stage, each with a status of "optimal", "time_limit" or "no_solution".

        certified = “This result is proven correct”: a schedule exists and every stage was solved to proven optimality. It's a yes/no summary that other code relies on.
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
    """How the base instance was obtained (reported by size class, exposé 4.3)."""

    base_seed: int  # seed from the protocol's seed list
    used_seed: int  # seed of the accepted candidate (= instance.seed)
    draws: int  # candidates drawn for this base seed
    rejected: int  # candidates with Lambda*_0 > 0 (main classes only)
    replaced: bool  # True if base_seed failed and a reserve seed was used
    lambda_star_0: int


@dataclass(frozen=True)
class SequenceResult:
    """Everything one sequence produces; storage writes exactly this.

    The base instance is stored in full so metrics never depend on regenerating it.
    """

    cell: str  # experiment cell, e.g. "main-n40"
    instance: Instance  # accepted base instance, before any event
    acceptance: AcceptanceLog
    sequence: Sequence
    s0: Schedule
    s0_stages: tuple[StageRecord, ...]
    coverage: tuple[CoverageStep, ...]
    trajectories: tuple[Trajectory, ...]
    manifest: dict


def _json_default(obj: object) -> dict[str, object]:
    """Dataclasses become dicts tagged with their class name; anything else is an error."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            "__type__": type(obj).__name__,
            **{f.name: getattr(obj, f.name) for f in fields(obj)},
        }
    raise TypeError(f"cannot digest {type(obj).__name__}: {obj!r}")


def _stable_json(obj: object) -> str:
    """Helper for digests: canonical JSON with sorted keys (StrEnums encode as their values)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
