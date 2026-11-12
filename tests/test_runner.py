"""runner.py: solve_lexicographic and check_oracles

All tests run with solve=bruteforce.solve_stage, a spy around it, or a scripted fake:
- spy  = Recorder():            records each call, answers with the brute force
- fake = Recorder([r1, r2]):    records each call, answers r1, then r2, ... (no solving)
The check_oracles tests build a LexResult by hand and let the oracle judge it.
"""

from collections.abc import Iterable, Mapping
from typing import NamedTuple

import pytest

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
    MIN_SHORTFALL,
    S0_SPECS,
    Objective,
    tie_break_ranks,
)
from repair.runner import OracleMismatch, check_oracles, solve_lexicographic
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
