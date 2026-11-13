"""runner.py: solve_lexicographic, check_oracles, accept_instance, initial_schedule

All tests run with solve=bruteforce.solve_stage, a spy around it, or a scripted fake:
- spy  = Recorder():            records each call, answers with the brute force
- fake = Recorder([r1, r2]):    records each call, answers r1, then r2, ... (no solving)
The check_oracles tests build a LexResult by hand and let the oracle judge it.
The accept_instance tests use Coverage, a fake that answers scripted Lambda*_0 values
for real generated candidates (no solving).
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from fractions import Fraction
from math import floor
from typing import NamedTuple

import pytest

import oracle
from repair import milp
from repair.generator import generate_instance

from repair import metrics
from repair.expected import REPAIR_SLICE_CHOICE
from repair.model import (
    AcceptanceLog,
    Arm,
    AvailabilityRemoval,
    AvailabilityRestoration,
    DemandChange,
    Instance,
    LexResult,
    PolicyName,
    ReferenceMode,
    Sequence as EventSequence,
    SequenceKind,
    SequenceProfile,
    Schedule,
    StageRecord,
    StageResult,
    StageStatus,
)
from repair.objectives import (
    MAX_QUALITY,
    MAX_TIE_BREAK,
    MIN_SHORTFALL,
    S0_SPECS,
    Objective,
    tie_break_ranks,
)
from repair.perturbations import instances_along
from repair.policies import history_weights
from repair.protocol import GeneratorConfig, SolverSettings, derive_seed, main_arms, rq2_arms
from repair.runner import (
    AcceptanceFailed,
    OracleMismatch,
    accept_instance,
    candidate_seed,
    check_oracles,
    initial_schedule,
    run_arm,
    run_sequence,
    shared_coverage,
    solve_lexicographic,
)
from tests.bruteforce import solve_stage
from tests.builders import Grid

# A test-only third objective (number of cells), so a test can run three stages
# without depending on the tie-break ranks
MIN_CELLS = Objective("MinCells", maximize=False, evaluate=lambda s, _: len(s))


class Call(NamedTuple):
    """One recorded solver call. Objectives are stored by name, so asserts stay readable."""

    objective: str
    fixed: dict[str, int]  # objective name -> value, e.g. {"MinShortfall": 0}
    cap: dict[str, int]
    start: Schedule | None
    time_limit: float


class Recorder:
    """A stage solver (runner.StageSolver) that records every call it receives.

    Recorder()                    -> spy: passes each call on to bruteforce.solve_stage
    Recorder([result1, result2])  -> fake: returns the scripted results in order
    """

    def __init__(self, script: Iterable[StageResult] | None = None) -> None:
        self.script = list(script) if script is not None else None
        self.calls: list[Call] = []

    def __call__(
        self,
        instance: Instance,
        objective: Objective,
        *,
        fixed: Mapping[Objective, int],
        cap: Mapping[Objective, int],
        start: Schedule | None,
        time_limit: float,
    ) -> StageResult:
        self.calls.append(
            Call(
                objective.name,
                {o.name: v for o, v in fixed.items()},
                {o.name: v for o, v in cap.items()},
                start,
                time_limit,
            )
        )
        if self.script is None:
            return solve_stage(
                instance,
                objective,
                fixed=fixed,
                cap=cap,
                start=start,
                time_limit=time_limit,
            )
        return self.script.pop(0)


# Scripted results and schedules on single_shift (cells (volunteer, shift))
NOTHING = StageResult(None, None, StageStatus.NO_SOLUTION)  # a stage that found nothing
A = frozenset({(0, 0)})  # v0 alone: Lambda = 0, U = 6
B = frozenset({(1, 0)})  # v1 alone: Lambda = 0, U = 3
AB = frozenset({(0, 0), (1, 0)})  # both: Lambda = 0, U = 5


def _record(name: str, status: StageStatus, value: int) -> StageRecord:
    """A StageRecord for a hand-built LexResult; bound and seconds do not matter here."""
    return StageRecord(name=name, status=status, value=value, bound=None, seconds=0.0)


# --- solve_lexicographic -------------------------------------------------------


def test_s0_specs_on_single_shift(single_shift):
    """[MinShortfall, MaxQuality, MaxTieBreak] gives {(0, 0)} with values (0, 6, rank of (0, 0))."""
    result = solve_lexicographic(single_shift, S0_SPECS, solve_stage, time_limit=10)

    assert result.schedule == A
    assert [r.name for r in result.stages] == [
        "MinShortfall",
        "MaxQuality",
        "MaxTieBreak",
    ]
    assert tuple(r.value for r in result.stages) == (
        0,
        6,
        tie_break_ranks(single_shift)[(0, 0)],
    )
    assert result.certified


def test_fixed_grows_stage_by_stage(single_shift):
    """Each optimal stage is fixed for every later stage."""
    spy = Recorder()

    solve_lexicographic(
        single_shift, (MIN_SHORTFALL, MAX_QUALITY, MIN_CELLS), spy, time_limit=10
    )

    assert [c.fixed for c in spy.calls] == [
        {},
        {"MinShortfall": 0},
        {"MinShortfall": 0, "MaxQuality": 6},
    ]
    assert all(c.cap == {} for c in spy.calls)


@pytest.mark.ivo
def test_start_is_previous_stage_schedule(single_shift):
    """Stage 1 starts from the caller's start, every later stage from the previous stage's schedule."""
    fake = Recorder([StageResult(AB, 0), StageResult(A, 6), StageResult(A, 1)])

    solve_lexicographic(
        single_shift,
        (MIN_SHORTFALL, MAX_QUALITY, MIN_CELLS),
        fake,
        time_limit=10,
        start=B,
    )

    assert [c.start for c in fake.calls] == [B, AB, A]


def test_retry_once_with_double_time_limit(single_shift):
    """No schedule in time: one retry with retry_factor * time_limit, marked as retried."""
    fake = Recorder([NOTHING, StageResult(A, 6)])

    result = solve_lexicographic(single_shift, (MAX_QUALITY,), fake, time_limit=10)

    assert [c.time_limit for c in fake.calls] == [10, 20]
    assert result.schedule == A
    assert result.stages[0].retried
    assert result.stages[0].value == 6


def test_two_failures_return_lexresult_none(single_shift):
    """If the retry also fails, the solve stops: no schedule, no later stages."""
    fake = Recorder([NOTHING, NOTHING])

    result = solve_lexicographic(
        single_shift, (MIN_SHORTFALL, MAX_QUALITY), fake, time_limit=10
    )

    assert result.schedule is None
    assert len(fake.calls) == 2  # MaxQuality never runs
    assert [r.status for r in result.stages] == [StageStatus.NO_SOLUTION]
    assert result.stages[0].retried
    assert not result.certified


def test_timeout_fixes_incumbent_value(single_shift):
    """A time_limit stage passes its incumbent value on as a cap, not as fixed and not its bound."""
    fake = Recorder(
        [StageResult(B, 3, StageStatus.TIME_LIMIT, bound=6.0), StageResult(B, 0)]
    )

    result = solve_lexicographic(
        single_shift, (MAX_QUALITY, MIN_SHORTFALL), fake, time_limit=10
    )

    assert fake.calls[1].fixed == {}
    assert fake.calls[1].cap == {"MaxQuality": 3}
    assert result.stages[0].status == StageStatus.TIME_LIMIT
    assert result.stages[0].value == 3
    assert result.stages[0].bound == 6.0
    assert not result.certified


def test_base_cap_applies_to_every_stage(expert_shift):
    """base_cap holds from the first stage on: Lambda <= 0 forces both expert_shift
    shifts open (U = 6, not 7). The caller's mapping is never mutated."""
    spy = Recorder()
    base_cap = {MIN_SHORTFALL: 0}

    result = solve_lexicographic(
        expert_shift, [MAX_QUALITY, MIN_CELLS], spy, time_limit=10, base_cap=base_cap
    )

    assert all(c.cap.get("MinShortfall") == 0 for c in spy.calls)
    assert result.stages[0].value == 6
    assert base_cap == {MIN_SHORTFALL: 0}


# --- check_oracles -------------------------------------------------------------


@pytest.mark.ivo
def test_check_oracles_raises_on_infeasible(min_crew_both_available):
    """One person on the d = 3 shift is below the minimum crew."""
    result = LexResult(
        schedule=A,
        stages=(
            _record("MinShortfall", StageStatus.OPTIMAL, 2),
            _record("MaxQuality", StageStatus.OPTIMAL, 5),
        ),
    )

    with pytest.raises(OracleMismatch, match="^infeasible"):
        check_oracles(min_crew_both_available, (MIN_SHORTFALL, MAX_QUALITY), result)


@pytest.mark.ivo
def test_check_oracles_raises_on_cap_exceeded(single_shift):
    """An unproven MaxQuality stage claimed U = 6; the final schedule only has U = 3."""
    result = LexResult(
        schedule=B,
        stages=(
            _record("MaxQuality", StageStatus.TIME_LIMIT, 6),
            _record("MinShortfall", StageStatus.OPTIMAL, 0),
        ),
    )

    with pytest.raises(OracleMismatch, match="^cap"):
        check_oracles(single_shift, (MAX_QUALITY, MIN_SHORTFALL), result)


@pytest.mark.ivo
def test_check_oracles_raises_on_value_mismatch(single_shift):
    """An optimal stage recorded U = 5, but the oracle computes U = 6 for {(0, 0)}."""
    result = LexResult(
        schedule=A,
        stages=(
            _record("MinShortfall", StageStatus.OPTIMAL, 0),
            _record("MaxQuality", StageStatus.OPTIMAL, 5),
        ),
    )

    with pytest.raises(OracleMismatch, match="^value"):
        check_oracles(single_shift, (MIN_SHORTFALL, MAX_QUALITY), result)


@pytest.mark.ivo
def test_check_oracles_accepts_improved_unproven_stage(single_shift):
    """An unproven stage claimed U = 3; a later stage improved it to 6. That is allowed."""
    result = LexResult(
        schedule=A,
        stages=(
            _record("MaxQuality", StageStatus.TIME_LIMIT, 3),
            _record("MinShortfall", StageStatus.OPTIMAL, 0),
        ),
    )

    check_oracles(single_shift, (MAX_QUALITY, MIN_SHORTFALL), result)


def test_check_oracles_raises_on_cap_exceeded_with_recorded_zero():
    """Regression: a recorded value of 0 is still checked (0 is falsy, easy to skip by accident)."""
    grid = Grid(n_periods=1, n_roles=1)
    instance = grid.instance(grid.volunteers(1), grid.shifts(1))
    # Empty schedule: Lambda = 1, but the unproven stage claimed Lambda = 0
    result = LexResult(
        schedule=frozenset(),
        stages=(_record("MinShortfall", StageStatus.TIME_LIMIT, 0),),
    )

    with pytest.raises(OracleMismatch, match="^cap"):
        check_oracles(instance, (MIN_SHORTFALL,), result)


def test_check_oracles_raises_when_stages_do_not_match_specs(single_shift):
    """A missing stage would otherwise be skipped silently by zip()."""
    result = LexResult(
        schedule=A,
        stages=(_record("MinShortfall", StageStatus.OPTIMAL, 0),),
    )

    with pytest.raises(OracleMismatch, match="^stages"):
        check_oracles(single_shift, (MIN_SHORTFALL, MAX_QUALITY), result)


# --- accept_instance ------------------------------------------------------------

CFG = GeneratorConfig()
SETTINGS = SolverSettings(time_limit=11.0, s0_time_limit=7.0, retry_factor=3.0)
BASE = 5  # a base seed; the candidates are candidate_seed(BASE, 0), (BASE, 1), ...
RESERVE = [901, 902]


class Coverage:
    """A fake coverage solver for accept_instance: the next scripted Lambda*_0 per
    candidate (None = nothing found, so the runner retries; the retry gets nothing
    too). Records the candidate seeds it saw and each call's objective and time limit.
    """

    def __init__(
        self, values: Iterable[int | None], status: StageStatus = StageStatus.OPTIMAL
    ) -> None:
        self.values = list(values)
        self.status = status
        self.seeds: list[int] = []  # one entry per candidate (retries not repeated)
        self.calls: list[tuple[str, float]] = []

    def __call__(
        self,
        instance: Instance,
        objective: Objective,
        *,
        fixed: Mapping[Objective, int],
        cap: Mapping[Objective, int],
        start: Schedule | None,
        time_limit: float,
    ) -> StageResult:
        self.calls.append((objective.name, time_limit))
        retry = bool(self.seeds) and self.seeds[-1] == instance.meta.seed and self._last is None
        if not retry:
            self.seeds.append(instance.meta.seed)
            self._last = self.values.pop(0)
        if self._last is None:
            return NOTHING
        return StageResult(frozenset(), self._last, self.status)

    _last: int | None = None


def _accept(coverage, **kwargs):
    kwargs.setdefault("reserve", RESERVE)
    return accept_instance(CFG, 20, BASE, settings=SETTINGS, solve=coverage, **kwargs)


def test_candidate_seed():
    """Draw 0 is the seed itself; the others are distinct and reproducible."""
    seeds = [candidate_seed(BASE, d) for d in range(100)]

    assert seeds[0] == BASE
    assert len(set(seeds)) == 100
    assert seeds == [candidate_seed(BASE, d) for d in range(100)]
    assert candidate_seed(BASE, 1) == derive_seed(BASE, "redraw", 1)


def test_main_accepts_only_zero_shortfall():
    """★ Lambda*_0 = 2, 1, 0: the third candidate is accepted, the log says so, and the
    coverage check is one MinShortfall stage with time_limit tau."""
    coverage = Coverage([2, 1, 0])

    instance, log = _accept(coverage)

    assert coverage.seeds == [candidate_seed(BASE, d) for d in range(3)]
    assert (log.base_seed, log.used_seed) == (BASE, coverage.seeds[-1])
    assert (log.draws, log.rejected, log.replaced, log.lambda_star_0) == (3, 2, False, 0)
    assert instance == generate_instance(CFG, 20, log.used_seed)
    assert instance.meta.seed == log.used_seed
    assert set(coverage.calls) == {("MinShortfall", SETTINGS.time_limit)}


def test_first_candidate_accepted_keeps_base_seed():
    instance, log = _accept(Coverage([0]))

    assert log.used_seed == BASE == instance.meta.seed
    assert (log.draws, log.rejected) == (1, 0)


@pytest.mark.parametrize(
    "values, status, draws",
    [
        pytest.param([None, 0], StageStatus.OPTIMAL, 2, id="nothing_found_rejected"),
        pytest.param([0], StageStatus.TIME_LIMIT, 1, id="unproven_zero_accepted"),
    ],
)
def test_acceptance_edge_cases(values, status, draws):
    """No schedule counts as a rejection; an incumbent of 0 is accepted even after a
    timeout (Lambda >= 0, so 0 is optimal)."""
    _, log = _accept(Coverage(values, status))

    assert (log.draws, log.lambda_star_0) == (draws, 0)


def test_redraw_then_reserve_after_max_draws():
    """3 rejections exhaust the base seed; reserve[0] is drawn next (its first candidate
    is the reserve seed itself) and its second candidate is accepted."""
    reserve = list(RESERVE)
    coverage = Coverage([1, 1, 1, 1, 0])

    instance, log = _accept(coverage, reserve=reserve, max_draws=3)

    assert coverage.seeds == [
        *(candidate_seed(BASE, d) for d in range(3)),
        RESERVE[0],
        candidate_seed(RESERVE[0], 1),
    ]
    assert log.replaced
    assert (log.base_seed, log.used_seed) == (BASE, candidate_seed(RESERVE[0], 1))
    assert (log.draws, log.rejected) == (5, 4)
    assert instance.meta.seed == log.used_seed
    assert reserve == RESERVE  # never mutated


def test_all_reserve_seeds_fail_raises():
    coverage = Coverage([1] * 4)

    with pytest.raises(AcceptanceFailed):
        _accept(coverage, reserve=[901], max_draws=2)
    assert len(coverage.seeds) == 4  # 2 for the base seed, 2 for the reserve seed


def test_under_supplied_never_rejected():
    """★ The first candidate is accepted whatever its Lambda*_0, which is recorded."""
    cfg = replace(CFG, workforce_factor=0.8)

    instance, log = accept_instance(
        cfg, 40, BASE, RESERVE, SETTINGS, Coverage([3]), under_supplied=True
    )

    assert (log.draws, log.rejected, log.replaced) == (1, 0, False)
    assert (log.used_seed, log.lambda_star_0) == (BASE, 3)
    assert instance.meta.under_supplied
    assert len(instance.volunteers) == floor(40 * 0.8)


def test_under_supplied_without_coverage_result_raises():
    cfg = replace(CFG, workforce_factor=0.8)

    with pytest.raises(ValueError):
        accept_instance(cfg, 40, BASE, RESERVE, SETTINGS, Coverage([None]), under_supplied=True)


def test_accept_is_deterministic():
    """Same arguments and the same answers -> the same instance and log."""
    first = _accept(Coverage([1, 0]))
    second = _accept(Coverage([1, 0]))

    assert first == second
    assert first[0] is not second[0]  # really built twice


# --- initial_schedule -----------------------------------------------------------


def test_initial_schedule_uses_s0_specs_and_time_limit(single_shift):
    """The S0_SPECS stages in order, each with s0_time_limit (not tau)."""
    spy = Recorder()

    result = initial_schedule(single_shift, SETTINGS, spy)

    assert [c.objective for c in spy.calls] == [o.name for o in S0_SPECS]
    assert {c.time_limit for c in spy.calls} == {SETTINGS.s0_time_limit}
    direct = solve_lexicographic(
        single_shift, S0_SPECS, solve_stage, time_limit=SETTINGS.s0_time_limit
    )
    # same schedule and stage values as a direct solve (the measured seconds differ)
    assert result.schedule == direct.schedule == A
    assert [(r.name, r.status, r.value) for r in result.stages] == [
        (r.name, r.status, r.value) for r in direct.stages
    ]
    assert result.certified


def test_initial_schedule_retry_uses_retry_factor(single_shift):
    """A stage that finds nothing is retried with retry_factor * s0_time_limit."""
    tie = MAX_TIE_BREAK.evaluate(A, single_shift)
    fake = Recorder(
        [NOTHING, StageResult(A, 0), StageResult(A, 6), StageResult(A, tie)]
    )

    initial_schedule(single_shift, SETTINGS, fake)

    assert [c.time_limit for c in fake.calls] == [7.0, 21.0, 7.0, 7.0]


def test_initial_schedule_runs_the_oracle_check(single_shift):
    """A solver that claims U = 999 for {(0, 0)} (really 6) is caught."""
    tie = MAX_TIE_BREAK.evaluate(A, single_shift)
    liar = Recorder([StageResult(A, 0), StageResult(A, 999), StageResult(A, tie)])

    with pytest.raises(OracleMismatch, match="^value:"):
        initial_schedule(single_shift, SETTINGS, liar)


# --- with SCIP ------------------------------------------------------------------


@pytest.mark.solver
def test_accept_and_s0_generated_n20():
    """Real SCIP on a generated n = 20 instance: accepted with Lambda*_0 = 0, and S0 is
    certified, feasible and fully covered."""
    settings = SolverSettings(time_limit=60.0, s0_time_limit=60.0)
    base = derive_seed(1, "instance", 20, 0)
    reserve = [derive_seed(1, "reserve", 20, 0, j) for j in range(3)]

    instance, log = accept_instance(CFG, 20, base, reserve, settings, milp.solve_stage)
    s0 = initial_schedule(instance, settings, milp.solve_stage)

    assert log.lambda_star_0 == 0
    assert s0.certified
    assert s0.schedule is not None
    assert oracle.is_feasible(s0.schedule, instance)
    assert oracle.shortfall(s0.schedule, instance) == 0
    assert s0.stages[0].value == 0


# --- arms (step 3.3) ---------------------------------------------------------------
#
# repair_slice with three events:
#   t = 1  A loses period 0                    Lambda* = 0, U* = 16
#   t = 2  s1 demand 1 -> 3 (only B, C fit)    Lambda* = 2, U* = 14
#   t = 3  A gets period 0 back                Lambda* = 1, U* = 19
# (Lambda* and U* from the brute force, the U* under the cap Lambda <= Lambda*.)

EVENTS = (
    AvailabilityRemoval(pairs=((0, 0),)),
    DemandChange(shift=1, delta=2),
    AvailabilityRestoration(volunteer=0, period=0),
)
LAMBDA_STAR = (0, 2, 1)
U_STAR = (16, 14, 19)

QO, SF, QL, WQS, HWF = (PolicyName.QO, PolicyName.SF, PolicyName.QL, PolicyName.WQS, PolicyName.HWF)
CHAINED, ANCHORED = ReferenceMode.CHAINED, ReferenceMode.ANCHORED
QO_ARM = Arm(QO, None)
SF_CHAINED = Arm(SF, CHAINED)
SF_ANCHORED = Arm(SF, ANCHORED)
HWF_0 = Arm(HWF, CHAINED, alpha=Fraction(0))
HWF_1 = Arm(HWF, CHAINED, alpha=Fraction(1))
WQS_HUGE = Arm(WQS, CHAINED, lam=10_000)  # far above any U difference on this instance
ALL_ARMS = (*main_arms(2, Fraction(1)), *rq2_arms(), HWF_0, WQS_HUGE)


class RawCall(NamedTuple):
    """One solver call with the real objects (for identity and weight checks)."""

    instance: Instance
    objective: Objective
    cap: dict[Objective, int]
    start: Schedule | None
    time_limit: float


class Spy:
    """Brute force with a record of every call; `transform` may alter the answer."""

    def __init__(
        self,
        transform: Callable[[Instance, Objective, StageResult], StageResult] | None = None,
    ) -> None:
        self.transform = transform
        self.calls: list[RawCall] = []

    def __call__(
        self,
        instance: Instance,
        objective: Objective,
        *,
        fixed: Mapping[Objective, int],
        cap: Mapping[Objective, int],
        start: Schedule | None,
        time_limit: float,
    ) -> StageResult:
        self.calls.append(RawCall(instance, objective, dict(cap), start, time_limit))
        result = solve_stage(
            instance, objective, fixed=fixed, cap=cap, start=start, time_limit=time_limit
        )
        return self.transform(instance, objective, result) if self.transform else result


class Story(NamedTuple):
    instance: Instance
    s0_result: LexResult
    s0: Schedule
    instances_t: list[Instance]


@pytest.fixture
def story(repair_slice):
    """repair_slice, its certified S0 (brute force) and the three step instances."""
    s0_result = solve_lexicographic(repair_slice, S0_SPECS, solve_stage, time_limit=10)
    assert s0_result.certified and s0_result.schedule is not None
    return Story(
        repair_slice, s0_result, s0_result.schedule, instances_along(repair_slice, EVENTS)
    )


def _run(story, arm, solve=solve_stage):
    coverage = shared_coverage(story.instances_t, SETTINGS, solve_stage)
    return run_arm(arm, story.instances_t, story.s0, coverage, SETTINGS, solve)


def test_shared_coverage_values(story):
    """Lambda*_t per step, each solved from the all-cancelled start with tau; the
    coverage schedule of step t has exactly Lambda*_t."""
    spy = Spy()

    coverage = shared_coverage(story.instances_t, SETTINGS, spy)

    assert [s.t for s in coverage.steps] == [1, 2, 3]
    assert tuple(s.lambda_star for s in coverage.steps) == LAMBDA_STAR
    assert all(s.record.status == StageStatus.OPTIMAL for s in coverage.steps)
    for start, instance, lam in zip(coverage.starts, story.instances_t, LAMBDA_STAR):
        assert oracle.shortfall(start, instance) == lam
    assert {(c.objective.name, c.start, c.time_limit) for c in spy.calls} == {
        ("MinShortfall", frozenset(), SETTINGS.time_limit)
    }


def test_shared_coverage_without_schedule_raises(story):
    """The all-cancelled start is always feasible: finding nothing is a solver bug."""
    with pytest.raises(RuntimeError):
        shared_coverage(story.instances_t, SETTINGS, Recorder([NOTHING] * 2))


def test_qo_arm_equals_bruteforce_max_quality(story):
    """QO reaches U*_t (under the cap) at every step."""
    trajectory = _run(story, QO_ARM)

    assert not trajectory.ended_early and trajectory.arm == QO_ARM
    assert [s.t for s in trajectory.steps] == [1, 2, 3]
    assert tuple(
        oracle.quality(s.schedule, i) for s, i in zip(trajectory.steps, story.instances_t)
    ) == U_STAR
    assert all(
        [r.name for r in s.stages] == ["MaxQuality", "MaxTieBreak"] for s in trajectory.steps
    )


@pytest.mark.parametrize("arm", ALL_ARMS, ids=lambda a: a.name)
def test_every_arm_reaches_lambda_star_every_step(story, arm):
    """★ Coverage comes first for every arm: Lambda(S_t) = Lambda*_t, S_t feasible."""
    trajectory = _run(story, arm)

    assert len(trajectory.steps) == 3
    for step, instance, lam in zip(trajectory.steps, story.instances_t, LAMBDA_STAR):
        assert oracle.is_feasible(step.schedule, instance)
        assert oracle.shortfall(step.schedule, instance) == lam


def test_arm_calls_capped_and_started_from_coverage(story):
    """Every stage carries the cap Lambda <= Lambda*_t; the first stage of a step starts
    from the shared coverage schedule, later stages from the previous stage."""
    coverage = shared_coverage(story.instances_t, SETTINGS, solve_stage)
    spy = Spy()

    run_arm(SF_CHAINED, story.instances_t, story.s0, coverage, SETTINGS, spy)

    assert len(spy.calls) == 3 * 3  # MinDistance, MaxQuality, MaxTieBreak per step
    for t in range(3):
        calls = spy.calls[3 * t : 3 * t + 3]
        assert all(c.instance is story.instances_t[t] for c in calls)
        assert all(c.cap[MIN_SHORTFALL] == LAMBDA_STAR[t] for c in calls)
        assert all(c.time_limit == SETTINGS.time_limit for c in calls)
        assert calls[0].start == coverage.starts[t]


@pytest.mark.parametrize("name, expected", list(REPAIR_SLICE_CHOICE.items()))
def test_t1_choices_match_value_sheet(story, name, expected):
    """The value sheet's repair at t = 1 for each arm (REPAIR_SLICE_CHOICE)."""
    arms = {
        a.name: a
        for a in (
            *main_arms(2, Fraction(1)),
            *rq2_arms(),
            Arm(WQS, CHAINED, lam=1),
            Arm(WQS, CHAINED, lam=4),
        )
    }

    assert _run(story, arms[name]).steps[0].schedule == expected


def _schedules(trajectory):
    return [s.schedule for s in trajectory.steps]


def _values(trajectory):
    return [[r.value for r in s.stages] for s in trajectory.steps]


def test_sf_anchored_equals_chained_at_t1(story):
    """At t = 1 the previous schedule is S0, so both references are S0."""
    anchored = _run(story, SF_ANCHORED)
    chained = _run(story, SF_CHAINED)

    # whole Steps differ in the measured seconds, so compare schedule and values
    assert _schedules(anchored)[0] == _schedules(chained)[0]
    assert _values(anchored)[0] == _values(chained)[0]


def test_hwf_alpha0_equals_sf(story):
    """★ alpha = 0 makes every weight 1, so HWF is SF at every step."""
    hwf, sf = _run(story, HWF_0), _run(story, SF_CHAINED)

    assert _schedules(hwf) == _schedules(sf)
    assert _values(hwf) == _values(sf)


def test_wqs_large_lambda_equals_sf(story):
    """★ lam above the U range: one changed cell costs more than any quality gain, so
    WQS minimizes H first and then maximizes U, exactly like SF."""
    assert _schedules(_run(story, WQS_HUGE)) == _schedules(_run(story, SF_CHAINED))


def test_burden_passed_to_hwf_matches_metrics_burden(story):
    """The weights HWF gets at step t are history_weights(alpha, b_{t-1}), with b from
    metrics.burden over the arm's own trajectory; some burden really builds up."""
    coverage = shared_coverage(story.instances_t, SETTINGS, solve_stage)
    spy = Spy()
    ids = [v.id for v in story.instance.volunteers]

    trajectory = run_arm(HWF_1, story.instances_t, story.s0, coverage, SETTINGS, spy)

    weights = [c.objective.weight_map() for c in spy.calls if c.objective.name == "MinDistance"]
    assert len(weights) == 3
    for t, got in enumerate(weights, start=1):
        b = metrics.burden(trajectory, story.s0, ids, until=t - 1)
        assert got == history_weights(Fraction(1), b, ids), t
    last = weights[-1]
    assert last is not None and any(w > 1 for w in last.values())  # non-vacuous


def test_all_arms_receive_identical_instances(story):
    """★ run_sequence builds the step instances once: the coverage stage and every arm
    see the very same three objects, equal to instances_along."""
    spy = Spy()
    sequence = EventSequence(SequenceKind.RQ1, SequenceProfile.BALANCED, 0, EVENTS)
    acceptance = AcceptanceLog(0, 0, 1, 0, False, 0)

    result = run_sequence(
        "test", story.instance, acceptance, story.s0_result, sequence, ALL_ARMS,
        SETTINGS, spy, {"note": "manifest"},
    )

    seen = {id(c.instance): c.instance for c in spy.calls}
    assert len(seen) == 3
    assert sorted(seen.values(), key=story.instances_t.index) == story.instances_t
    assert [tr.arm for tr in result.trajectories] == list(ALL_ARMS)
    assert tuple(s.lambda_star for s in result.coverage) == LAMBDA_STAR
    assert (result.cell, result.instance, result.acceptance) == ("test", story.instance, acceptance)
    assert (result.sequence, result.s0, result.s0_stages) == (
        sequence, story.s0, story.s0_result.stages
    )
    assert result.manifest == {"note": "manifest"}


def test_run_sequence_rejects_uncertified_s0(story):
    unproven = LexResult(
        story.s0,
        tuple(replace(r, status=StageStatus.TIME_LIMIT) for r in story.s0_result.stages),
    )
    sequence = EventSequence(SequenceKind.RQ1, SequenceProfile.BALANCED, 0, EVENTS)

    with pytest.raises(ValueError):
        run_sequence(
            "test", story.instance, AcceptanceLog(0, 0, 1, 0, False, 0), unproven,
            sequence, (QO_ARM,), SETTINGS, solve_stage, {},
        )


def test_ended_early_when_solve_fails(story):
    """MaxQuality finds nothing at t = 2 (also on the retry): QO stops after step 1."""

    def fail_at_step_2(instance, objective, result):
        if instance is story.instances_t[1] and objective == MAX_QUALITY:
            return NOTHING
        return result

    trajectory = _run(story, QO_ARM, Spy(fail_at_step_2))

    assert trajectory.ended_early
    assert [s.t for s in trajectory.steps] == [1]


def test_oracle_mismatch_raises(story):
    """★ A solver that claims a tie-break value one too high is caught by the oracle
    check. (The last stage: an inflated earlier value would become a fixing, the next
    stage would find nothing, and the arm would end early instead.)"""

    def overstate(instance, objective, result):
        if objective == MAX_TIE_BREAK and result.value is not None:
            return result._replace(value=result.value + 1)
        return result

    with pytest.raises(OracleMismatch, match="^value:"):
        _run(story, QO_ARM, Spy(overstate))


def test_coverage_cap_violation_raises(story):
    """A solver that ignores the cap and answers the all-cancelled schedule (with its
    true values, so check_oracles is satisfied) breaks Lambda <= Lambda*_t."""

    def cancel_everything(instance, objective, result):
        empty: Schedule = frozenset()
        return StageResult(empty, objective.evaluate(empty, instance), StageStatus.OPTIMAL)

    with pytest.raises(OracleMismatch, match="^coverage:"):
        _run(story, QO_ARM, Spy(cancel_everything))
