"""Check the shape and inputs of the examples in conftest.py.

The fixtures must match the value sheet and repair.expected. These tests
check the data; test_oracle.py checks quality, shortfall and violations.
"""

import dataclasses

import pytest

from repair import expected as E

ROWS_FOR = {
    "single_shift": [E.SINGLE_SHIFT_ROWS],
    "min_crew": [E.MIN_CREW_ROWS],
    "min_crew_both_available": [E.MIN_CREW_BOTH_ROWS],
    "expert_shift": [E.EXPERT_SHIFT_ROWS],
    "repair_slice": [E.REPAIR_SLICE_ROWS_0, E.REPAIR_SLICE_ROWS_1],
}

FIXTURES = [
    "single_shift",
    "min_crew",
    "min_crew_both_available",
    "expert_shift",
    "repair_slice",
]


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_builds_and_is_hashable(request, name):
    """Building the fixture passed the Instance validator; hashing checks the container types."""
    print("IVO:")
    print(name)
    print(request)
    print(request.getfixturevalue(name))
    hash(request.getfixturevalue(name))


@pytest.mark.parametrize(
    "name, n_volunteers, n_shifts, n_periods, n_roles",
    [
        ("single_shift", 2, 1, 1, 1),
        ("min_crew", 2, 1, 1, 1),
        ("min_crew_both_available", 2, 1, 1, 1),
        ("expert_shift", 3, 2, 1, 2),
        ("repair_slice", 3, 3, 2, 2),
    ],
)
def test_fixture_shapes(request, name, n_volunteers, n_shifts, n_periods, n_roles):
    instance = request.getfixturevalue(name)
    assert (
        len(instance.volunteers),
        len(instance.shifts),
        instance.n_periods,
        instance.n_roles,
    ) == (n_volunteers, n_shifts, n_periods, n_roles)


@pytest.mark.parametrize("name", FIXTURES)
def test_expected_cells_exist_in_fixture(request, name):
    """Every cell in the expected rows names a real volunteer and shift of this fixture."""
    instance = request.getfixturevalue(name)
    for rows in ROWS_FOR[name]:
        for schedule in rows:
            for w, s in schedule:
                assert 0 <= w < len(instance.volunteers) and 0 <= s < len(
                    instance.shifts
                )


# Data spot checks against the sheet


def test_single_shift_data(single_shift):
    v0, v1 = single_shift.volunteers
    assert (v0.skill, v0.seniority, v0.pref_role, v0.pref_time) == ((1,), 0, (2,), (1,))
    assert (v1.skill, v1.seniority, v1.pref_role, v1.pref_time) == ((3,), 2, (0,), (0,))
    (s0,) = single_shift.shifts
    assert (s0.demand, s0.min_skill, s0.expert) == (1, 1, False)


def test_min_crew_data(min_crew):
    v0, v1 = min_crew.volunteers
    assert v0.available == frozenset({0})
    assert v1.available == frozenset()
    assert min_crew.shifts[0].demand == 3


def test_min_crew_both_available_differs_only_in_v1_availability(
    min_crew, min_crew_both_available
):
    assert min_crew_both_available.volunteers[1] == dataclasses.replace(
        min_crew.volunteers[1], available=frozenset({0})
    )
    assert min_crew_both_available.volunteers[0] == min_crew.volunteers[0]
    assert min_crew_both_available.shifts == min_crew.shifts


def test_expert_shift_data(expert_shift):
    s0, s1 = expert_shift.shifts
    assert (s0.period, s0.role, s0.expert) == (0, 0, True)
    assert (s1.period, s1.role, s1.expert) == (0, 1, False)
    skill3_in_role0 = [v.id for v in expert_shift.volunteers if v.skill[0] == 3]
    assert skill3_in_role0 == [0], "v0 must be the only expert for s0"


def test_repair_slice_data(repair_slice):
    a, b, c = repair_slice.volunteers
    assert (a.cap, b.cap, c.cap) == (1, 10, 10)
    assert all(v.available == frozenset({0, 1}) for v in repair_slice.volunteers)
    assert [(s.period, s.role) for s in repair_slice.shifts] == [(0, 0), (0, 1), (1, 0)]
    assert repair_slice.n_cells == E.REPAIR_SLICE_N_CELLS


def test_repair_slice_event_hits_s0(repair_slice):
    """The removed (volunteer, period) is a period in which that volunteer works in S0."""
    ((w, p),) = E.REPAIR_SLICE_EVENT.pairs
    assert any(
        v == w and repair_slice.shifts[s].period == p for (v, s) in E.REPAIR_SLICE_S0
    )
