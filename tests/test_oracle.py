"""oracle.py

Feasibility
- Empty schedule is feasible for every fixture (all shifts cancelled).
- Each hard constraint violated in isolation gives exactly one violation message:
    unavailable cell; two shifts in one period; cap exceeded; open shift below
    ceil(d/2); no assignee with skill >= kappa; expert shift without skill 3.
- min_crew: 1->1, 2->1, 3->2, 4->2, 5->3, 6->3.
- Below d_s but at or above ceil(d_s/2) is feasible (soft coverage); overstaffing is feasible.
- Property (seeded loop over generated instances and sequences): the all-cancelled
  schedule stays feasible after every event (exposé 4.8).

Quality and coverage
- single_shift fixture: U = 0, 6, 3, 5 and Lambda = 1, 0, 0, 0 (values in conftest).
- assignment_utility = 2 p_role + p_time + k; range 0..9.
- Mentorship: novice alone -> 0; mentor alone -> 0; both -> 1; seniority 2 with skill 1
  is a novice, not a mentor.
- Surplus with d = 1 and 4 assignees: q = 3, z = 2, contribution 5*2 - 10*3 = -20.
- Lambda counts a cancelled shift as d_s and an understaffed open shift as d_s - staffing.
- After a PreferenceChange, U of an unchanged schedule moves by +-2 or +-4 per
  assignment in the changed (w, r) (exposé 4.4).
"""

# Reading order for the calculations:
# feasibility, assignment utility, surplus, mentoring, quality, shortfall.
import pytest

import oracle

from repair.expected import REPAIR_SLICE_S0, MIN_CREW_BOTH_OPTIMUM
from repair.generator import generate_instance
from repair.model import PreferenceChange, SequenceProfile
from repair.perturbations import (
    PROFILES,
    apply_event,
    instances_along,
    rq1_sequence,
    rq2_sequence,
)
from repair.protocol import GeneratorConfig
from dataclasses import replace
from tests.builders import Grid, greedy_schedule, make_shift

# 1.3 oracle.py: independence

# These independence tests are currently commented out; they do not run.
# FORBIDDEN_MODULES = {"milp", "objectives"}


# def _forbidden_imports(source: str) -> set[str]:
#     """Find solver/objective imports; the oracle should only evaluate schedules."""
#     imported: set[str] = set()
#     for node in ast.walk(ast.parse(source)):
#         if isinstance(node, ast.Import):
#             imported.update(alias.name for alias in node.names)
#         elif isinstance(node, ast.ImportFrom):
#             base = node.module or ""
#             imported.add(base)
#             imported.update(f"{base}.{alias.name}" for alias in node.names)
#     return {name for name in imported if set(name.split(".")) & FORBIDDEN_MODULES}


# def test_oracle_does_not_import_milp_or_objectives():
#     """The oracle checks the MILP's results, so it must not share code with the MILP."""
#     assert _forbidden_imports(Path(oracle.__file__).read_text()) == set()


# @pytest.mark.parametrize(
#     "source",
#     [
#         "import repair.milp",
#         "import repair.objectives as o",
#         "from repair import milp",
#         "from repair.objectives import quality",
#         "from . import objectives",
#         "from .milp import build",
#         "def f():\n    from repair import milp",
#     ],
# )
# def test_forbidden_imports_detects_every_import_form(source):
#     """Guards the test above: it would pass trivially if the helper missed an import form."""
#     assert _forbidden_imports(source)


# @pytest.mark.parametrize(
#     "source", ["", "from repair.model import Instance", "import math"]
# )
# def test_forbidden_imports_allows_other_modules(source):
#     assert _forbidden_imports(source) == set()


# 1.3 oracle.py: feasibility


@pytest.mark.parametrize(
    "demand, expected", [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3)]
)
def test_min_crew_values(demand, expected):
    """min_crew(d) == ceil(d / 2)."""
    assert oracle.min_crew(demand) == expected


@pytest.mark.parametrize(
    "fixture_name", ["single_shift", "expert_shift", "min_crew", "repair_slice"]
)
def test_empty_schedule_feasible_for_all_fixtures(request, fixture_name):
    """frozenset() has no violations on every fixture (all shifts cancelled)."""
    instance = request.getfixturevalue(fixture_name)

    assert not oracle.violations(frozenset(), instance)
    assert oracle.is_feasible(frozenset(), instance)


def test_violation_unavailable_cell(repair_slice):
    """Assigning in an unavailable period gives exactly one violation."""
    schedule = REPAIR_SLICE_S0
    assert not oracle.violations(schedule, repair_slice)

    a = repair_slice.volunteers[0]
    after_event = repair_slice.with_volunteer(replace(a, available=frozenset({1})))

    found = oracle.violations(schedule, after_event)
    assert len(found) == 1
    assert found[0].startswith("unavailable:")


def test_violation_two_shifts_same_period(repair_slice):
    """Two shifts in one period for one volunteer gives exactly one violation."""
    schedule = REPAIR_SLICE_S0
    assert not oracle.violations(schedule, repair_slice)

    # Volunteer 1 already works shift 1; also assign shift 0, both in period 0
    broken_schedule = schedule | {(1, 0)}

    found = oracle.violations(broken_schedule, repair_slice)
    assert len(found) == 1
    assert found[0].startswith("two_shifts_same_period:")


def test_violation_cap_exceeded(repair_slice):
    """More than L_w shifts gives exactly one violation."""
    schedule = REPAIR_SLICE_S0
    assert not oracle.violations(schedule, repair_slice)

    # Volunteer 0 now also works shift 2 in period 1; v0 has a capacity of 1.
    broken_schedule = schedule | {(0, 2)}

    found = oracle.violations(broken_schedule, repair_slice)
    assert len(found) == 1
    assert found[0].startswith("cap:")


def test_violation_open_below_min_crew(min_crew):
    """An open shift below ceil(d/2) gives exactly one violation."""

    # 1 person on the d = 3 shift: open, but below 2
    broken_schedule = frozenset({(0, 0)})

    found = oracle.violations(broken_schedule, min_crew)
    assert len(found) == 1
    assert found[0].startswith("min_crew:")


def test_violation_no_skilled_assignee(min_crew_both_available):
    """An open shift without skill >= kappa gives exactly one violation."""
    schedule = MIN_CREW_BOTH_OPTIMUM

    s = min_crew_both_available.shifts[0]
    both_qualified = min_crew_both_available.with_shift(replace(s, min_skill=2))

    assert not oracle.violations(schedule, both_qualified)

    # Volunteer 0 now gets a skill of 1; volunteer 1 still meets min_skill = 2
    v0 = both_qualified.volunteers[0]
    only_v1_qualified = both_qualified.with_volunteer(replace(v0, skill=(1,)))

    assert not oracle.violations(schedule, only_v1_qualified)

    # Volunteer 1 now gets a skill of 1; nobody meets min_skill = 2, so the schedule is infeasible
    v1 = only_v1_qualified.volunteers[1]
    nobody_qualified = only_v1_qualified.with_volunteer(replace(v1, skill=(1,)))

    found = oracle.violations(schedule, nobody_qualified)

    assert len(found) == 1
    assert found[0].startswith("skill:")


def test_violation_no_skilled_assignee_checks_the_shifts_role():
    """Skill in another role does not count."""
    grid = Grid(n_periods=1, n_roles=2)
    instance = grid.instance([grid.volunteer(0, skill=(2, 0))], [make_shift(0, role=1)])

    found = oracle.violations(frozenset({(0, 0)}), instance)

    assert len(found) == 1
    assert found[0].startswith("skill")


def test_violation_expert_missing(expert_shift):
    """An open expert shift without a skill-3 assignee gives exactly one violation."""

    # Volunteer 1 works alone on shift 0 without the required expert skill.
    broken_schedule = frozenset({(1, 0)})

    found = oracle.violations(broken_schedule, expert_shift)
    assert len(found) == 1
    assert found[0].startswith("expert")


def test_understaffed_above_min_crew_is_feasible(min_crew_both_available):
    """Staffing between ceil(d/2) and d - 1 is feasible (soft coverage)."""
    # Both volunteers on the d = 3 shift: 2 of 3, at the minimum crew but short of demand
    schedule = MIN_CREW_BOTH_OPTIMUM
    demand = min_crew_both_available.shifts[0].demand
    assert oracle.min_crew(demand) <= len(schedule) < demand

    assert not oracle.violations(schedule, min_crew_both_available)


def test_overstaffing_is_feasible(single_shift):
    """Staffing above d is feasible."""
    # Both volunteers on the d = 1 shift: 2 of 1
    schedule = frozenset({(0, 0), (1, 0)})
    assert len(schedule) > single_shift.shifts[0].demand

    assert not oracle.violations(schedule, single_shift)


# 1.3 oracle.py: quality and shortfall


@pytest.mark.parametrize(
    "w, s, expected",
    [(0, 0, 4), (0, 1, 7), (1, 0, 1), (1, 1, 2), (2, 0, 0), (2, 1, 1)],
)
def test_assignment_utility_formula(expert_shift, w, s, expected):
    """u_ws == 2 * p_role + p_time + k for a hand-picked volunteer and shift."""
    # The u-table from the expert_shift docstring
    assert oracle.assignment_utility(expert_shift, w, s) == expected


def test_assignment_utility_range():
    """u_ws is 0 for all-zero inputs and 9 for all-maximum inputs."""
    grid = Grid(n_periods=1, n_roles=1)
    low = grid.volunteer(0, skill=(0,), pref_role=(0,), pref_time=(0,))
    high = grid.volunteer(1, skill=(3,), pref_role=(2,), pref_time=(2,))
    instance = grid.instance([low, high], grid.shifts(1))

    assert oracle.assignment_utility(instance, 0, 0) == 0
    assert oracle.assignment_utility(instance, 1, 0) == 2 * 2 + 2 + 3


@pytest.mark.parametrize(
    "schedule, expected",
    [
        (frozenset(), 0),
        (frozenset({(0, 0)}), 6),
        (frozenset({(1, 0)}), 3),
        (frozenset({(0, 0), (1, 0)}), 5),
    ],
)
def test_single_shift_quality_values(single_shift, schedule, expected):
    """U on the four single_shift schedules (values from conftest)."""
    assert oracle.quality(schedule, single_shift) == expected


@pytest.mark.parametrize(
    "schedule, expected",
    [
        (frozenset(), 1),
        (frozenset({(0, 0)}), 0),
        (frozenset({(1, 0)}), 0),
        (frozenset({(0, 0), (1, 0)}), 0),
    ],
)
def test_single_shift_shortfall_values(single_shift, schedule, expected):
    """Lambda on the four single_shift schedules."""
    assert oracle.shortfall(schedule, single_shift) == expected


def test_shortfall_cancelled_and_understaffed(min_crew_both_available):
    """A cancelled shift counts d_s; an understaffed open shift counts d_s - staffing."""
    # d = 3: cancelled -> 3; both volunteers -> 3 - 2 = 1
    assert oracle.shortfall(frozenset(), min_crew_both_available) == 3
    assert oracle.shortfall(MIN_CREW_BOTH_OPTIMUM, min_crew_both_available) == 1


def _mentorship_instance():
    """v0 novice, v1 mentor, v2 seniority 2 but skill 1 (a novice), v3 skill 2 but seniority 0; d = 3."""
    grid = Grid(n_periods=1, n_roles=1)
    volunteers = [
        grid.volunteer(0, skill=(1,), seniority=0),
        grid.volunteer(1, skill=(2,), seniority=2),
        grid.volunteer(2, skill=(1,), seniority=2),
        grid.volunteer(3, skill=(2,), seniority=0),
    ]
    return grid.instance(volunteers, [make_shift(0, demand=3)])


@pytest.mark.parametrize(
    "workers, expected",
    [
        ({0}, frozenset()),  # novice alone
        ({1}, frozenset()),  # mentor alone
        ({0, 1}, frozenset({0})),  # novice and mentor
        ({0, 2}, frozenset()),  # v2 is no mentor: two novices
        ({1, 2}, frozenset({0})),  # v2 counts as the novice
        ({0, 3}, frozenset()),  # v3 is skilled but not senior: no mentor
    ],
)
def test_mentorship_cases(workers, expected):
    """Novice alone -> 0; mentor alone -> 0; both -> 1; seniority 2 with skill 1 is a novice."""
    schedule = frozenset((w, 0) for w in workers)

    assert oracle.mentored(schedule, _mentorship_instance()) == expected


def test_surplus_contribution():
    """d = 1 with 4 assignees: q = 3, z = 2, contribution 5*2 - 10*3 = -20."""
    grid = Grid(n_periods=1, n_roles=1)
    instance = grid.instance(grid.volunteers(4), grid.shifts(1))
    schedule = frozenset((w, 0) for w in range(4))

    assert oracle.surplus(schedule, instance) == {0: 3}

    # Default volunteers have seniority 0, so there is no mentor term
    utility = sum(oracle.assignment_utility(instance, w, s) for w, s in schedule)
    assert oracle.quality(schedule, instance) - utility == 5 * 2 - 10 * 3


# 2.3 needs apply_event


# C works two role-0 shifts (s0 and s2) here, so a change in C's role-0 preference counts twice
_C_TWICE = frozenset({(0, 0), (1, 1), (2, 0), (2, 2)})


@pytest.mark.parametrize(
    "schedule, volunteer, role, new, expected_delta",
    [
        (
            REPAIR_SLICE_S0,
            1,
            1,
            0,
            -4,
        ),  # B on s1 (role 1): p_role 2 -> 0, one assignment
        (REPAIR_SLICE_S0, 1, 1, 1, -2),  # B: 2 -> 1
        (REPAIR_SLICE_S0, 2, 0, 2, +2),  # C on s2 (role 0): 1 -> 2
        (REPAIR_SLICE_S0, 0, 0, 0, -4),  # A on s0 (role 0): 2 -> 0
        (REPAIR_SLICE_S0, 0, 1, 2, 0),  # A works no role-1 shift: U unchanged
        (_C_TWICE, 2, 0, 2, +4),  # C on s0 and s2: +2 per assignment
    ],
)
def test_preference_change_moves_quality_by_2_or_4(
    repair_slice, schedule, volunteer, role, new, expected_delta
):
    """After a PreferenceChange, U of an unchanged schedule moves by +-2 or +-4 per affected assignment.

    u_ws = 2 * p_role + p_time + k, so a change of p_role by +-1 or +-2 moves u by
    +-2 or +-4 for every shift of that volunteer in that role; nothing else in U changes.
    """
    after = apply_event(
        repair_slice, PreferenceChange(changes=((volunteer, role, new),))
    )

    delta = oracle.quality(schedule, after) - oracle.quality(schedule, repair_slice)
    assert delta == expected_delta


# 2.4 property test


def test_all_cancelled_feasible_after_every_event():
    """For 200 generated n=20 sequences, frozenset() is feasible for every step instance.

    Exposé 4.8: cancelling every shift is always a feasible repair, so a repair stage can
    never be infeasible. 40 instances x 5 sequence types (the three RQ1 profiles,
    independent targeting, RQ2) cover every event type, both P1 fallbacks and the RQ2
    restoration. S0 is a solver-free stand-in: it only steers which cells are targeted.
    """
    checked = 0

    for i in range(40):
        instance = generate_instance(GeneratorConfig(), 20, i)
        s0 = greedy_schedule(instance, seed=i)
        sequences = [
            *(rq1_sequence(instance, s0, profile, i) for profile in PROFILES),
            rq1_sequence(instance, s0, SequenceProfile.BALANCED, i, targeting="independent"),
            rq2_sequence(instance, s0, i),
        ]

        for seq in sequences:
            for t, after in enumerate(instances_along(instance, seq.events), start=1):
                assert oracle.violations(frozenset(), after) == [], (i, seq.kind, seq.profile, t)
            checked += 1

    assert checked == 200
