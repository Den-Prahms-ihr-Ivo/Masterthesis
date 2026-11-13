"""milp.py: every result is compared with the brute force (the reference).

Values are compared, not schedules: when two schedules tie, the MILP and the brute
force may pick different ones. Only with the tie-break stage (S0_SPECS) must the
schedules be identical.
"""

from dataclasses import replace

import pytest

import oracle
from repair import milp
from repair.generator import generate_instance
from repair.expected import (
    EXPERT_SHIFT_OPTIMUM,
    EXPERT_SHIFT_OPTIMUM_QUALITY_ONLY,
    REPAIR_SLICE_DISTANCE,
    REPAIR_SLICE_EVENT,
    REPAIR_SLICE_S0,
    REPAIR_SLICE_SF,
    REPAIR_SLICE_X,
)
from repair.model import StageResult, StageStatus
from repair.objectives import (
    MAX_QUALITY,
    MAX_TIE_BREAK,
    MIN_SHORTFALL,
    S0_SPECS,
    max_quality_minus_distance,
    min_distance,
)
from repair.perturbations import apply_event
from repair.protocol import GeneratorConfig
from repair.runner import solve_lexicographic
from tests import bruteforce

pytestmark = pytest.mark.solver

FIXTURES = [
    "single_shift",
    "min_crew",
    "min_crew_both_available",
    "expert_shift",
    "repair_slice",
]


def _milp(instance, objective, *, fixed=None, cap=None, start=None) -> StageResult:
    """milp.solve_stage with test defaults (no constraints, no start, 10 s)."""
    return milp.solve_stage(
        instance,
        objective,
        fixed=fixed or {},
        cap=cap or {},
        start=start,
        time_limit=10,
    )


def _brute(instance, objective, *, fixed=None, cap=None) -> StageResult:
    return bruteforce.solve_stage(instance, objective, fixed=fixed, cap=cap)


def _assert_matches_bruteforce(
    instance, objective, *, fixed=None, cap=None
) -> StageResult:
    """Same value as the brute force, and the MILP's schedule really has that value."""
    got = _milp(instance, objective, fixed=fixed, cap=cap)
    want = _brute(instance, objective, fixed=fixed, cap=cap)

    assert got.status == StageStatus.OPTIMAL
    assert got.value == want.value
    assert got.schedule is not None
    assert oracle.is_feasible(got.schedule, instance)
    assert objective.evaluate(got.schedule, instance) == got.value
    return got


# --- objectives ----------------------------------------------------------------


def test_maxquality_single_shift(single_shift):
    """[MaxQuality] on single_shift: {(0, 0)}, U = 6."""
    got = _milp(single_shift, MAX_QUALITY)

    assert got.schedule == frozenset({(0, 0)})
    assert got.value == 6
    assert got.status == StageStatus.OPTIMAL


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_maxquality_matches_bruteforce_all_fixtures(request, fixture_name):
    """★ Hard constraints and U agree with the oracle on every fixture."""
    _assert_matches_bruteforce(request.getfixturevalue(fixture_name), MAX_QUALITY)


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_minshortfall_matches_bruteforce(request, fixture_name):
    """★ Lambda agrees with the oracle on every fixture."""
    _assert_matches_bruteforce(request.getfixturevalue(fixture_name), MIN_SHORTFALL)


def test_cap_respected(expert_shift):
    """Lambda <= 0 forces both shifts open (U = 6); Lambda <= 1 allows U = 7 again."""
    tight = _assert_matches_bruteforce(
        expert_shift, MAX_QUALITY, cap={MIN_SHORTFALL: 0}
    )
    assert tight.schedule == EXPERT_SHIFT_OPTIMUM

    loose = _assert_matches_bruteforce(
        expert_shift, MAX_QUALITY, cap={MIN_SHORTFALL: 1}
    )
    assert loose.schedule == EXPERT_SHIFT_OPTIMUM_QUALITY_ONLY


def test_fixing_is_exact(repair_slice):
    """★ MaxQuality, then MinDistance to X with U fixed: U stays exactly U* = 18."""
    best = _milp(repair_slice, MAX_QUALITY)
    assert best.value == 18

    objective = min_distance(REPAIR_SLICE_X)
    got = _assert_matches_bruteforce(repair_slice, objective, fixed={MAX_QUALITY: 18})

    assert got.schedule is not None
    assert oracle.quality(got.schedule, repair_slice) == 18
    assert got.schedule == REPAIR_SLICE_S0  # the only schedule with U = 18


# --- weighted MinDistance and WQS (step 3.1) -------------------------------------


def _references(instance):
    """References that pull the solver in different directions: the S0 optimum, the
    empty schedule (H = |S|) and every cell (H = J - |S|, partly infeasible)."""
    s0 = solve_lexicographic(instance, S0_SPECS, bruteforce.solve_stage, time_limit=10)
    every_cell = frozenset(
        (v.id, s.id) for v in instance.volunteers for s in instance.shifts
    )
    assert s0.schedule is not None
    return [s0.schedule, frozenset(), every_cell]


def _weights(instance):
    """Distinct weights 1, 3, 5, ... so a cell counted for the wrong volunteer shows."""
    return {v.id: 1 + 2 * v.id for v in instance.volunteers}


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_weighted_distance_matches_bruteforce(request, fixture_name):
    """★ Weighted H agrees with the oracle, alone and under the coverage cap."""
    instance = request.getfixturevalue(fixture_name)
    lambda_star = _brute(instance, MIN_SHORTFALL).value
    assert lambda_star is not None

    for reference in _references(instance):
        objective = min_distance(reference, weights=_weights(instance))
        _assert_matches_bruteforce(instance, objective)
        _assert_matches_bruteforce(instance, objective, cap={MIN_SHORTFALL: lambda_star})


def test_weighted_distance_counts_unavailable_reference_cells(repair_slice):
    """(A, s0) in S0 is unavailable after the event and has no variable: its weight
    still counts (the constant term), like in the unweighted case."""
    after = apply_event(repair_slice, REPAIR_SLICE_EVENT)
    objective = min_distance(REPAIR_SLICE_S0, weights={0: 7, 1: 1, 2: 2})

    got = _assert_matches_bruteforce(after, objective)
    assert got.value is not None and got.value >= 7


@pytest.mark.parametrize("lam", [1, 2, 4])
@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_wqs_matches_bruteforce_for_lambda_1_2_4(request, fixture_name, lam):
    """★ U - lam * H agrees with the oracle for every reference, also under the cap."""
    instance = request.getfixturevalue(fixture_name)
    lambda_star = _brute(instance, MIN_SHORTFALL).value
    assert lambda_star is not None

    for reference in _references(instance):
        objective = max_quality_minus_distance(reference, lam)
        _assert_matches_bruteforce(instance, objective)
        _assert_matches_bruteforce(instance, objective, cap={MIN_SHORTFALL: lambda_star})


@pytest.mark.parametrize(
    "lam, value, schedule",
    [
        pytest.param(1, 12, REPAIR_SLICE_X, id="lam1_X"),  # 16 - 1 * 4
        pytest.param(2, 9, REPAIR_SLICE_SF, id="lam2_SF"),  # 13 - 2 * 2
        pytest.param(4, 5, REPAIR_SLICE_SF, id="lam4_SF"),  # 13 - 4 * 2
    ],
)
def test_wqs_after_event_value_sheet(repair_slice, lam, value, schedule):
    """After the event, reference S0, under the coverage cap Lambda <= 0 (as in a real
    repair step): the unique WQS optimum is the value sheet's choice. Without the cap,
    lam = 4 would rather cancel s0 (value 6)."""
    after = apply_event(repair_slice, REPAIR_SLICE_EVENT)
    objective = max_quality_minus_distance(REPAIR_SLICE_S0, lam)

    got = _assert_matches_bruteforce(after, objective, cap={MIN_SHORTFALL: 0})
    assert (got.value, got.schedule) == (value, schedule)


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_fixing_wqs_and_weighted_distance_is_exact(request, fixture_name):
    """A later stage with a WQS or weighted-H value fixed: the fixing constraint uses the
    same expression, so the tie-break optimum matches the brute force."""
    instance = request.getfixturevalue(fixture_name)
    reference = _references(instance)[0]

    for objective in (
        max_quality_minus_distance(reference, 2),
        min_distance(reference, weights=_weights(instance)),
    ):
        best = _brute(instance, objective).value
        assert best is not None
        _assert_matches_bruteforce(instance, MAX_TIE_BREAK, fixed={objective: best})


# --- lexicographic -------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_s0_specs_give_identical_schedule_to_bruteforce(request, fixture_name):
    """★ With the tie-break stage the optimum is unique, so the schedules must be identical."""
    instance = request.getfixturevalue(fixture_name)

    got = solve_lexicographic(instance, S0_SPECS, milp.solve_stage, time_limit=10)
    want = solve_lexicographic(
        instance, S0_SPECS, bruteforce.solve_stage, time_limit=10
    )

    assert got.schedule == want.schedule
    assert [r.value for r in got.stages] == [r.value for r in want.stages]
    assert got.certified


def test_min_distance_with_unavailable_reference_cells(repair_slice):
    """★ The reference S0 contains (A, s0), which is now unavailable and has no variable.

    H must still count it: the constant term |reference| does that.
    Alone: drop (A, s0), keep the rest -> H = 1.
    After MinShortfall (Lambda = 0): s0 needs C -> SF*, H = 2.
    """
    a = repair_slice.volunteers[0]
    after = repair_slice.with_volunteer(replace(a, available=frozenset({1})))
    objective = min_distance(REPAIR_SLICE_S0)

    alone = _assert_matches_bruteforce(after, objective)
    assert alone.value == 1

    covered = _assert_matches_bruteforce(after, objective, fixed={MIN_SHORTFALL: 0})
    assert covered.value == REPAIR_SLICE_DISTANCE[REPAIR_SLICE_SF]
    assert covered.schedule == REPAIR_SLICE_SF


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_aux_variables_bound_quality_from_below(request, fixture_name):
    """★ MinDistance with U fixed at U*: the true U of the result still meets U*.

    The mentorship, buffer and surplus variables are only pushed to their true values
    when U is the objective. Here it is a constraint, so a loose aux variable would let
    the MILP claim U* for a schedule whose real U is lower. Every feasible schedule is
    tried as the reference, so each one pulls the solver in another direction.
    """
    instance = request.getfixturevalue(fixture_name)
    u_star = _brute(instance, MAX_QUALITY).value
    assert u_star is not None

    for reference in bruteforce.all_schedules(instance):
        got = _milp(instance, min_distance(reference), fixed={MAX_QUALITY: u_star})

        assert got.schedule is not None
        assert oracle.quality(got.schedule, instance) >= u_star, sorted(reference)


# --- solver behaviour ----------------------------------------------------------


def test_warm_start_feasible_and_infeasible(repair_slice):
    """A feasible start and an infeasible one both give the optimum, and neither raises."""
    best = _brute(repair_slice, MAX_QUALITY)
    every_cell = frozenset(
        (v.id, s.id) for v in repair_slice.volunteers for s in repair_slice.shifts
    )
    assert not oracle.is_feasible(every_cell, repair_slice)

    for start in (best.schedule, every_cell):
        got = _milp(repair_slice, MAX_QUALITY, start=start)
        assert got.value == best.value


def test_three_runs_identical(repair_slice):
    """Same input, same settings -> the same schedule and the same stage values."""
    runs = [
        solve_lexicographic(repair_slice, S0_SPECS, milp.solve_stage, time_limit=10)
        for _ in range(3)
    ]

    assert len({run.schedule for run in runs}) == 1
    assert len({tuple(r.value for r in run.stages) for run in runs}) == 1


def test_milp_timeout_status():
    """A stage stopped by its time limit reports time_limit with a usable incumbent.

    Generated n = 80, tie-break stage with Lambda and U fixed, warm-started from the
    MaxQuality solution. The roadmap says tau = 1 s, but this machine solves every
    n = 80 stage within 1 s (tie-break: ~0.5 s); 0.05 s with a warm start stopped with
    time_limit in every probe. The incumbent must be feasible, keep the fixed values,
    have its claimed value, and lie below the dual bound (maximisation).
    """
    instance = generate_instance(GeneratorConfig(), 80, 0)
    quality = milp.solve_stage(
        instance, MAX_QUALITY, fixed={MIN_SHORTFALL: 0}, cap={}, start=None, time_limit=60
    )
    assert quality.status == StageStatus.OPTIMAL and quality.value is not None
    fixed = {MIN_SHORTFALL: 0, MAX_QUALITY: quality.value}

    got = milp.solve_stage(
        instance, MAX_TIE_BREAK, fixed=fixed, cap={}, start=quality.schedule, time_limit=0.05
    )

    assert got.status == StageStatus.TIME_LIMIT
    assert got.schedule is not None and got.value is not None and got.bound is not None
    assert oracle.is_feasible(got.schedule, instance)
    assert oracle.shortfall(got.schedule, instance) == 0
    assert oracle.quality(got.schedule, instance) == quality.value
    assert MAX_TIE_BREAK.evaluate(got.schedule, instance) == got.value
    assert got.value <= got.bound


def test_negative_cap_no_solution(single_shift):
    """Lambda <= -1 is impossible, so SCIP reports infeasible -> no solution."""
    got = _milp(single_shift, MAX_QUALITY, cap={MIN_SHORTFALL: -1})

    assert got == StageResult(None, None, StageStatus.NO_SOLUTION)


def test_solver_versions_non_empty():
    """Both versions are recorded and non-empty."""
    versions = milp.solver_versions()

    assert set(versions) == {"scip", "pyscipopt"}
    assert all(isinstance(v, str) and v for v in versions.values())
