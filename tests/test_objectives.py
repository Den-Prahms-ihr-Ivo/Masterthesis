"""objectives.py: the weighted MinDistance (HWF), MaxQualityMinusDistance (WQS) and the
tie-break ranks, evaluated through the oracle (hand values from the value sheet)."""

from collections import Counter

import pytest

from repair.expected import (
    REPAIR_SLICE_EVENT,
    REPAIR_SLICE_S0,
    REPAIR_SLICE_SF,
    REPAIR_SLICE_X,
)
from repair.objectives import (
    max_quality_minus_distance,
    min_distance,
    tie_break_ranks,
)
from repair.perturbations import apply_event
from tests.builders import Grid

# --- weighted MinDistance (HWF) --------------------------------------------------


def test_weighted_distance_hand_value(repair_slice):
    """weights {0: 3, 1: 1}: S = {(0, 0), (1, 0)} vs reference {(1, 1)} changes the
    cells (0, 0), (1, 0), (1, 1) -> 3 + 1 + 1 = 5 (unweighted: 3)."""
    schedule = frozenset({(0, 0), (1, 0)})
    reference = frozenset({(1, 1)})

    weighted = min_distance(reference, weights={0: 3, 1: 1})

    assert weighted.evaluate(schedule, repair_slice) == 5
    assert min_distance(reference).evaluate(schedule, repair_slice) == 3
    assert weighted.name == "MinDistance" and not weighted.maximize


def test_all_ones_weights_equal_unweighted(repair_slice):
    """Weight 1 for everyone is exactly H (HWF with alpha = 0 is SF)."""
    ones = min_distance(REPAIR_SLICE_S0, weights={0: 1, 1: 1, 2: 1})

    for schedule in (REPAIR_SLICE_S0, REPAIR_SLICE_SF, REPAIR_SLICE_X, frozenset()):
        assert ones.evaluate(schedule, repair_slice) == min_distance(
            REPAIR_SLICE_S0
        ).evaluate(schedule, repair_slice)


def test_weighted_distance_missing_weight_raises(repair_slice):
    """Volunteer 2 has a changed cell but no weight: an error, not a silent 0."""
    objective = min_distance(frozenset(), weights={0: 1})

    with pytest.raises(ValueError):
        objective.evaluate(frozenset({(2, 0)}), repair_slice)


@pytest.mark.parametrize("weights", [{0: -1}, {0: 1.5}], ids=["negative", "float"])
def test_weighted_distance_rejects_bad_weights(weights):
    with pytest.raises(ValueError):
        min_distance(frozenset(), weights=weights)


def test_objectives_hashable_and_compared_by_parameters():
    """Objectives are dict keys in fixed / cap: equal parameters -> equal and same hash;
    different weights -> different objectives."""
    a = min_distance(REPAIR_SLICE_S0, weights={0: 2, 1: 5})
    b = min_distance(REPAIR_SLICE_S0, weights={1: 5, 0: 2})  # insertion order irrelevant
    c = min_distance(REPAIR_SLICE_S0, weights={0: 2, 1: 4})

    assert a == b and hash(a) == hash(b)
    assert a != c
    assert a != min_distance(REPAIR_SLICE_S0)
    assert {a: 1}[b] == 1
    assert max_quality_minus_distance(REPAIR_SLICE_S0, 2) == max_quality_minus_distance(
        REPAIR_SLICE_S0, 2
    )
    assert max_quality_minus_distance(REPAIR_SLICE_S0, 2) != max_quality_minus_distance(
        REPAIR_SLICE_S0, 4
    )


# --- MaxQualityMinusDistance (WQS) --------------------------------------------------


@pytest.mark.parametrize(
    "schedule, u, h",
    [
        pytest.param(REPAIR_SLICE_SF, 13, 2, id="SF"),
        pytest.param(REPAIR_SLICE_X, 16, 4, id="X"),
    ],
)
def test_wqs_value(repair_slice, schedule, u, h):
    """After the event (value sheet): U - 2 * H(S, S0); SF* gives 13 - 4 = 9, X gives
    16 - 8 = 8, so WQS(lam = 2) prefers SF* (REPAIR_SLICE_CHOICE)."""
    after = apply_event(repair_slice, REPAIR_SLICE_EVENT)
    objective = max_quality_minus_distance(REPAIR_SLICE_S0, 2)

    assert objective.evaluate(schedule, after) == u - 2 * h
    assert objective.name == "MaxQualityMinusDistance" and objective.maximize
    assert (objective.reference, objective.lam) == (REPAIR_SLICE_S0, 2)


def test_wqs_lambda_0_is_quality(repair_slice):
    after = apply_event(repair_slice, REPAIR_SLICE_EVENT)
    objective = max_quality_minus_distance(REPAIR_SLICE_S0, 0)

    assert objective.evaluate(REPAIR_SLICE_X, after) == 16


@pytest.mark.parametrize("lam", [-1, 1.5, True], ids=["negative", "float", "bool"])
def test_wqs_rejects_bad_lambda(lam):
    with pytest.raises(ValueError):
        max_quality_minus_distance(REPAIR_SLICE_S0, lam)


# --- tie-break ranks (3.2) -----------------------------------------------------------


def _instance(seed):
    grid = Grid(n_periods=2, n_roles=2)
    return grid.instance(grid.volunteers(4), grid.shifts(4), seed=seed)


def test_tiebreak_ranks_permutation_deterministic_seed_dependent():
    """Ranks 1..J, each once, for every cell; same seed -> same ranks; another seed ->
    another permutation (so no volunteer id is systematically favoured)."""
    instance = _instance(seed=3)
    ranks = tie_break_ranks(instance)
    cells = {(v.id, s.id) for v in instance.volunteers for s in instance.shifts}

    assert set(ranks) == cells
    assert sorted(ranks.values()) == list(range(1, len(cells) + 1))
    assert tie_break_ranks(_instance(seed=3)) == ranks
    assert tie_break_ranks(_instance(seed=4)) != ranks


def test_tiebreak_ranks_do_not_favour_low_ids():
    """Over many seeds, the top rank lands on every volunteer."""
    winners = Counter(
        max(ranks, key=lambda cell: ranks[cell])[0]
        for ranks in (tie_break_ranks(_instance(seed)) for seed in range(200))
    )

    assert set(winners) == {0, 1, 2, 3}
