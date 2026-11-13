"""runner.py: solve_lexicographic, check_oracles, accept_instance, initial_schedule

All tests run with solve=bruteforce.solve_stage, a spy around it, or a scripted fake:
- spy  = Recorder():            records each call, answers with the brute force
- fake = Recorder([r1, r2]):    records each call, answers r1, then r2, ... (no solving)
The check_oracles tests build a LexResult by hand and let the oracle judge it.
The accept_instance tests use Coverage, a fake that answers scripted Lambda*_0 values
for real generated candidates (no solving).
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from math import floor
from typing import NamedTuple

import pytest

import oracle
from repair import milp
from repair.generator import generate_instance

from repair.model import (
    Instance,
    LexResult,
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
from repair.protocol import GeneratorConfig, SolverSettings, derive_seed
from repair.runner import (
    AcceptanceFailed,
    OracleMismatch,
    accept_instance,
    candidate_seed,
    check_oracles,
    initial_schedule,
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
