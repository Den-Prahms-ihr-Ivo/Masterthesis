"""Small scheduling examples shared by the tests.

The tables show the inputs and hand-calculated values. They are kept small
enough to check on paper and enumerate with the brute-force solver. See
tests/fixture_value_sheet.md for the full calculation of the repair example.
"""

import pytest

from tests.builders import Grid, make_shift


@pytest.fixture
def single_shift():
    """n_periods = 1, n_roles = 1

    |     | available | skill | seniority | cap | pref_role | pref_time | u(s0)           |
    | --- | --------- | ----- | --------- | --- | --------- | --------- | --------------- |
    | v0  | {0}       | (1)   | 0         | 10  | (2)       | (1)       | 2·2 + 1 + 1 = 6 |
    | v1  | {0}       | (3)   | 2         | 10  | (0)       | (0)       | 0 + 0 + 3 = 3   |

    s0: period 0, role 0, d = 1, κ = 1, no expert.

    | Schedule          | Violations | Λ   | Σu  | mentor | +5z | −10q | U   |
    | ----------------- | ---------- | --- | --- | ------ | --- | ---- | --- |
    | {}                | none       | 1   | 0   | 0      | 0   | 0    | 0   |
    | {(v0,s0)}         | none       | 0   | 6   | 0      | 0   | 0    | 6   |
    | {(v1,s0)}         | none       | 0   | 3   | 0      | 0   | 0    | 3   |
    | {(v0,s0),(v1,s0)} | none       | 0   | 9   | 1      | 5   | −10  | 5   |

    **Optimum** [MinShortfall, MaxQuality]: {(v0,s0)}, Λ = 0, U = 6.
    """
    grid = Grid(n_periods=1, n_roles=1)
    volunteers = [
        grid.volunteer(0, skill=(1,), pref_role=(2,), pref_time=(1,)),
        grid.volunteer(1, skill=(3,), pref_role=(0,), pref_time=(0,), seniority=2),
    ]
    return grid.instance(volunteers=volunteers, shifts=grid.shifts(1))


@pytest.fixture
def min_crew():
    """n_periods = 1, n_roles = 1. s0: d = 3, so the minimum crew is ⌈3/2⌉ = 2; κ = 1, no expert.

    |     | available | skill | seniority | cap | pref_role | pref_time | u(s0)         |
    | --- | --------- | ----- | --------- | --- | --------- | --------- | ------------- |
    | v0  | {0}       | (2)   | 0         | 10  | (1)       | (1)       | 2 + 1 + 2 = 5 |
    | v1  | { }       | (2)   | 0         | 10  | (1)       | (1)       | 5             |

    | Schedule          | Violations            | Λ   | U   |
    | ----------------- | --------------------- | --- | --- |
    | {}                | none                  | 3   | 0   |
    | {(v0,s0)}         | min_crew              | 2   | 5   |
    | {(v1,s0)}         | unavailable, min_crew | 2   | 5   |
    | {(v0,s0),(v1,s0)} | unavailable           | 1   | 10  |

    **Optimum:** {}, the only feasible schedule; Λ* = 3, U = 0.
    """
    grid = Grid(n_periods=1, n_roles=1)

    volunteers = [
        grid.volunteer(0, skill=(2,), pref_role=(1,), pref_time=(1,), available=[0]),
        grid.volunteer(1, skill=(2,), pref_role=(1,), pref_time=(1,), available=[]),
    ]
    shifts = [make_shift(0, demand=3)]
    return grid.instance(volunteers=volunteers, shifts=shifts)


@pytest.fixture
def min_crew_both_available():
    """### Variant `min_crew_both_available` (v1 available in {0})

    | Schedule          | Violations | Λ   | U   |
    | ----------------- | ---------- | --- | --- |
    | {}                | none       | 3   | 0   |
    | {(v0,s0)}         | min_crew   | 2   | 5   |
    | {(v1,s0)}         | min_crew   | 2   | 5   |
    | {(v0,s0),(v1,s0)} | none       | 1   | 10  |

    **Optimum:** {(v0,s0),(v1,s0)}, Λ = 1, U = 10.

    Only the empty schedule and the pair are feasible. One person is below the
    minimum crew, and the pair has less shortfall than the empty schedule.
    """
    grid = Grid(n_periods=1, n_roles=1)

    volunteers = [
        grid.volunteer(0, skill=(2,), pref_role=(1,), pref_time=(1,), available=[0]),
        grid.volunteer(1, skill=(2,), pref_role=(1,), pref_time=(1,), available=[0]),
    ]
    shifts = [make_shift(0, demand=3)]
    return grid.instance(volunteers=volunteers, shifts=shifts)


@pytest.fixture
def expert_shift():
    """Two shifts in one period; s0 is expert-flagged and only v0 can staff it.

    Value sheet (tests/fixture_value_sheet.md §3):

        u-table       s0 (r0, expert)   s1 (r1)
        v0                 4               7
        v1                 1               2
        v2                 0               1

        [MaxQuality] alone           -> {(v0,s1)}          Lambda = 1, U = 7  (s0 cancelled)
        [MinShortfall, MaxQuality]   -> {(v0,s0),(v1,s1)}  Lambda = 0, U = 6
    """
    grid = Grid(n_periods=1, n_roles=2)
    volunteers = [
        grid.volunteer(0, skill=(3, 2), pref_role={1: 2}, pref_time={0: 1}),
        grid.volunteer(1, skill=(1, 2)),
        grid.volunteer(2, skill=(0, 1)),
    ]
    shifts = [
        make_shift(0, period=0, role=0, expert=True),
        make_shift(1, period=0, role=1),
    ]
    return grid.instance(volunteers, shifts)


@pytest.fixture
def repair_slice():
    """n_periods = 2, n_roles = 2. All volunteers: available {0, 1}, seniority 0. Ids: A = v0, B = v1, C = v2.

    |     | skill (r0, r1) | pref_role (r0, r1) | pref_time (p0, p1) | cap   | u(s0) | u(s1) | u(s2) |
    | --- | -------------- | ------------------ | ------------------ | ----- | ----- | ----- | ----- |
    | A   | (2, 0)         | (2, 0)             | (2, 0)             | **1** | 8     | 2     | 6     |
    | B   | (1, 2)         | (0, 2)             | (1, 0)             | 10    | 2     | 7     | 1     |
    | C   | (1, 1)         | (1, 0)             | (0, 0)             | 10    | 3     | 1     | 3     |

    Shifts (all d = 1, κ = 1, no expert): s0 = (p0, r0), s1 = (p0, r1), s2 = (p1, r0). J = 3 · 3 = 9.

    Return the base instance before the availability-removal event.
    """
    grid = Grid(n_periods=2, n_roles=2)
    volunteers = [
        grid.volunteer(0, skill=(2, 0), pref_role=(2, 0), pref_time=(2, 0), cap=1),  # A
        grid.volunteer(1, skill=(1, 2), pref_role=(0, 2), pref_time=(1, 0)),  # B
        grid.volunteer(2, skill=(1, 1), pref_role=(1, 0), pref_time=(0, 0)),  # C
    ]
    shifts = [
        make_shift(0, period=0, role=0),
        make_shift(1, period=0, role=1),
        make_shift(2, period=1, role=0),
    ]
    return grid.instance(volunteers, shifts)
