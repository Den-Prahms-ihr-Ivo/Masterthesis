"""tests/expected.py: internal consistency of the hand-derived values.

These tests use no project code except the data types. They check that the sheet's
numbers agree with each other. Later the Brute Force Solver will prove optimality.
At this point these are derived by hand.
"""

from fractions import Fraction

import pytest

from repair import expected as E

ALL_ROWS = {
    "single_shift": E.SINGLE_SHIFT_ROWS,
    "min_crew": E.MIN_CREW_ROWS,
    "min_crew_both_available": E.MIN_CREW_BOTH_ROWS,
    "expert_shift": E.EXPERT_SHIFT_ROWS,
    "repair_slice_t0": E.REPAIR_SLICE_ROWS_0,
    "repair_slice_t1": E.REPAIR_SLICE_ROWS_1,
}


def _feasible(rows):
    return {s: r for s, r in rows.items() if not r.violations}


def _coverage_first_best(rows):
    """[MinShortfall, MaxQuality] over the listed feasible rows: all best schedules."""
    feasible = _feasible(rows)
    best = min((r.lam, -r.u) for r in feasible.values())
    return {s for s, r in feasible.items() if (r.lam, -r.u) == best}


# --- shape of the data -------------------------------------------------------------


@pytest.mark.parametrize("name", list(ALL_ROWS))
def test_keys_are_schedules(name):
    """Every key is a frozenset of (int, int) cells."""
    for schedule in ALL_ROWS[name]:
        assert type(schedule) is frozenset
        for cell in schedule:
            assert isinstance(cell, tuple) and len(cell) == 2
            assert all(type(x) is int and x >= 0 for x in cell)


@pytest.mark.parametrize("name", list(ALL_ROWS))
def test_rows_use_known_rule_names_and_int_values(name):
    for row in ALL_ROWS[name].values():
        assert row.violations <= E.RULES
        assert type(row.lam) is int and row.lam >= 0
        assert type(row.u) is int


# --- stated optima -----------------------------------------------------------------


@pytest.mark.parametrize(
    "rows, optimum",
    [
        pytest.param(E.SINGLE_SHIFT_ROWS, E.SINGLE_SHIFT_OPTIMUM, id="single_shift"),
        pytest.param(E.MIN_CREW_ROWS, E.MIN_CREW_OPTIMUM, id="min_crew"),
        pytest.param(
            E.MIN_CREW_BOTH_ROWS, E.MIN_CREW_BOTH_OPTIMUM, id="min_crew_both_available"
        ),
        pytest.param(E.EXPERT_SHIFT_ROWS, E.EXPERT_SHIFT_OPTIMUM, id="expert_shift"),
        pytest.param(E.REPAIR_SLICE_ROWS_0, E.REPAIR_SLICE_S0, id="repair_slice_s0"),
    ],
)
def test_coverage_first_optimum_is_unique_best_listed_row(rows, optimum):
    """The stated optimum is listed, feasible, and the unique best row under [MinShortfall, MaxQuality]."""
    assert optimum in rows and not rows[optimum].violations
    assert _coverage_first_best(rows) == {optimum}


def test_expert_quality_only_optimum():
    """Without the coverage stage, the expert's favourite shift wins and s0 stays cancelled."""
    feasible = _feasible(E.EXPERT_SHIFT_ROWS)
    best_u = max(r.u for r in feasible.values())
    assert {s for s, r in feasible.items() if r.u == best_u} == {
        E.EXPERT_SHIFT_OPTIMUM_QUALITY_ONLY
    }
    assert E.EXPERT_SHIFT_ROWS[E.EXPERT_SHIFT_OPTIMUM_QUALITY_ONLY].lam > 0


def test_quality_only_beats_coverage_first_on_u():
    """The point of expert_shift: coverage-first costs quality (7 > 6)."""
    qo = E.EXPERT_SHIFT_ROWS[E.EXPERT_SHIFT_OPTIMUM_QUALITY_ONLY]
    cf = E.EXPERT_SHIFT_ROWS[E.EXPERT_SHIFT_OPTIMUM]
    assert qo.u > cf.u and qo.lam > cf.lam == 0


# --- repair_slice --------------------------------------------------------------------


def test_s0_is_infeasible_after_the_event():
    """The event makes S0 binding: unavailable is its only violation at t = 1."""
    assert E.REPAIR_SLICE_ROWS_1[E.REPAIR_SLICE_S0].violations == {"unavailable"}


def test_event_pair_belongs_to_an_s0_assignment_of_that_volunteer():
    """The removed (volunteer, period) pair's volunteer is assigned in S0 (the period is checked in test_fixtures)."""
    ((volunteer, _period),) = E.REPAIR_SLICE_EVENT.pairs
    assert any(w == volunteer for (w, _s) in E.REPAIR_SLICE_S0)


def test_distances_are_symmetric_differences():
    for schedule, h in E.REPAIR_SLICE_DISTANCE.items():
        assert h == len(schedule ^ E.REPAIR_SLICE_S0)


def test_every_t1_row_has_a_distance():
    assert set(E.REPAIR_SLICE_ROWS_1) == set(E.REPAIR_SLICE_DISTANCE)


def test_t1_rows_respect_the_coverage_cap():
    """Every feasible repair row reaches Lambda*_1, as the shared coverage stage requires."""
    for row in _feasible(E.REPAIR_SLICE_ROWS_1).values():
        assert row.lam == E.REPAIR_SLICE_LAMBDA_STAR_1


def _choose(key):
    """Unique best feasible t = 1 row under `key` (larger is better)."""
    feasible = _feasible(E.REPAIR_SLICE_ROWS_1)
    scored = {s: key(r, E.REPAIR_SLICE_DISTANCE[s]) for s, r in feasible.items()}
    best = max(scored.values())
    winners = [s for s, v in scored.items() if v == best]
    assert len(winners) == 1, f"tie among {winners}"
    return winners[0]


@pytest.mark.parametrize(
    "arm, key",
    [
        pytest.param("SF-chained", lambda r, h: (-h, r.u), id="SF"),
        pytest.param("SF-anchored", lambda r, h: (-h, r.u), id="SF-anchored"),
        pytest.param("HWF-chained-a1", lambda r, h: (-h, r.u), id="HWF_t1_zero_burden"),
        pytest.param("QL-chained", lambda r, h: (r.u, -h), id="QL"),
        pytest.param("QO", lambda r, h: r.u, id="QO"),
        pytest.param("WQS-chained-l1", lambda r, h: r.u - 1 * h, id="WQS_l1"),
        pytest.param("WQS-chained-l2", lambda r, h: r.u - 2 * h, id="WQS_l2"),
        pytest.param("WQS-chained-l4", lambda r, h: r.u - 4 * h, id="WQS_l4"),
    ],
)
def test_policy_choice_follows_from_rows(arm, key):
    """Each stated choice is the unique best listed repair under that arm's stage order."""
    assert E.REPAIR_SLICE_CHOICE[arm] == _choose(key)


def test_sf_and_ql_disagree():
    """The fixture's purpose: minimal change and best quality pick different repairs."""
    assert E.REPAIR_SLICE_CHOICE["SF-chained"] != E.REPAIR_SLICE_CHOICE["QL-chained"]


def test_sf_churn():
    assert E.REPAIR_SLICE_SF_CHURN == Fraction(
        E.REPAIR_SLICE_DISTANCE[E.REPAIR_SLICE_SF], E.REPAIR_SLICE_N_CELLS
    )
