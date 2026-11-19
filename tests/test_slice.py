"""Follow one complete repair step on the hand-calculated repair_slice.

Solve S0, remove A's availability, run the SF stages and calculate C_1.
Run this once with brute force and once with SCIP. Both should agree with
repair.expected, so the test also checks how the parts work together.
"""

from fractions import Fraction

import pytest

import oracle
from repair import milp
from repair.expected import (
    REPAIR_SLICE_EVENT,
    REPAIR_SLICE_LAMBDA_STAR_1,
    REPAIR_SLICE_ROWS_0,
    REPAIR_SLICE_ROWS_1,
    REPAIR_SLICE_S0,
    REPAIR_SLICE_SF,
    REPAIR_SLICE_SF_CHURN,
)
from repair.metrics import distance
from repair.objectives import (
    MAX_QUALITY,
    MAX_TIE_BREAK,
    MIN_SHORTFALL,
    S0_SPECS,
    min_distance,
)
from repair.perturbations import apply_event
from repair.runner import check_oracles, solve_lexicographic
from tests import bruteforce

SOLVERS = [
    pytest.param(bruteforce.solve_stage, id="bruteforce"),
    pytest.param(milp.solve_stage, id="scip", marks=pytest.mark.solver),
]


@pytest.mark.parametrize("solve", SOLVERS)
def test_vertical_slice(repair_slice, solve):
    # t = 0: the starting schedule
    s0 = solve_lexicographic(repair_slice, S0_SPECS, solve, time_limit=10)
    check_oracles(repair_slice, S0_SPECS, s0)

    assert s0.schedule == REPAIR_SLICE_S0
    assert [r.value for r in s0.stages][:2] == [
        0,
        REPAIR_SLICE_ROWS_0[REPAIR_SLICE_S0].u,
    ]

    # t = 1: A loses period 0, so S0 breaks (the step is binding)
    after = apply_event(repair_slice, REPAIR_SLICE_EVENT)
    assert not oracle.is_feasible(REPAIR_SLICE_S0, after)

    # SF repair: coverage first, then as few changes as possible, then quality, then tie-break
    sf_specs = (
        MIN_SHORTFALL,
        min_distance(REPAIR_SLICE_S0),
        MAX_QUALITY,
        MAX_TIE_BREAK,
    )
    repair = solve_lexicographic(
        after, sf_specs, solve, time_limit=10, start=REPAIR_SLICE_S0
    )
    check_oracles(after, sf_specs, repair)

    assert repair.schedule == REPAIR_SLICE_SF
    assert [r.value for r in repair.stages][:3] == [
        REPAIR_SLICE_LAMBDA_STAR_1,  # Lambda*_1 = 0
        2,  # H(SF*, S0)
        REPAIR_SLICE_ROWS_1[REPAIR_SLICE_SF].u,  # U = 13
    ]

    # Churn C_1 = H / J
    churn = Fraction(distance(repair.schedule, REPAIR_SLICE_S0), after.n_cells)
    assert churn == REPAIR_SLICE_SF_CHURN
