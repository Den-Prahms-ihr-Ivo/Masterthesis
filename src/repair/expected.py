"""Hand-derived expected values: tests/fixtures_sheet.md in code form.

The only expected values in the project that are not derived from the code.
If a value here and the sheet disagree, the sheet wins: fix this file, then find out
how they diverged.

Cells are (volunteer_id, shift_id). In repair_slice: A = 0, B = 1, C = 2.
"""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

from repair.model import AvailabilityRemoval

RULES = frozenset({"unavailable", "two_shifts_same_period", "cap", "min_crew", "skill", "expert"})


class Row(NamedTuple):
    violations: frozenset[str]   # rule keys; empty = feasible
    lam: int                     # Lambda(S)
    u: int                       # U(S)


def _s(*cells: tuple[int, int]) -> frozenset[tuple[int, int]]:
    return frozenset(cells)


NONE: frozenset[str] = frozenset()

# --------------------------------------------------------------------- single_shift
SINGLE_SHIFT_ROWS = {
    _s(): Row(NONE, 1, 0),
    _s((0, 0)): Row(NONE, 0, 6),
    _s((1, 0)): Row(NONE, 0, 3),
    _s((0, 0), (1, 0)): Row(NONE, 0, 5),
}
SINGLE_SHIFT_OPTIMUM = _s((0, 0))

# --------------------------------------------------------------------- min_crew
MIN_CREW_ROWS = {
    _s(): Row(NONE, 3, 0),
    _s((0, 0)): Row(frozenset({"min_crew"}), 2, 5),
    _s((1, 0)): Row(frozenset({"unavailable", "min_crew"}), 2, 5),
    _s((0, 0), (1, 0)): Row(frozenset({"unavailable"}), 1, 10),
}
MIN_CREW_OPTIMUM = _s()

MIN_CREW_BOTH_ROWS = {
    _s(): Row(NONE, 3, 0),
    _s((0, 0)): Row(frozenset({"min_crew"}), 2, 5),
    _s((1, 0)): Row(frozenset({"min_crew"}), 2, 5),
    _s((0, 0), (1, 0)): Row(NONE, 1, 10),
}
MIN_CREW_BOTH_OPTIMUM = _s((0, 0), (1, 0))

# --------------------------------------------------------------------- expert_shift
EXPERT_SHIFT_ROWS = {
    _s(): Row(NONE, 2, 0),
    _s((1, 0)): Row(frozenset({"expert"}), 1, 1),
    _s((2, 0)): Row(frozenset({"skill", "expert"}), 1, 0),
    _s((0, 0), (0, 1)): Row(frozenset({"two_shifts_same_period"}), 0, 11),
    _s((0, 1), (1, 0)): Row(frozenset({"expert"}), 0, 8),
    _s((0, 0)): Row(NONE, 1, 4),
    _s((0, 1)): Row(NONE, 1, 7),
    _s((0, 0), (2, 1)): Row(NONE, 0, 5),
    _s((0, 0), (1, 1)): Row(NONE, 0, 6),
}
EXPERT_SHIFT_OPTIMUM_QUALITY_ONLY = _s((0, 1))           # [MaxQuality]
EXPERT_SHIFT_OPTIMUM = _s((0, 0), (1, 1))                # [MinShortfall, MaxQuality]

# --------------------------------------------------------------------- repair_slice
REPAIR_SLICE_ROWS_0 = {                                   # under the initial instance
    _s((0, 0), (1, 1), (2, 2)): Row(NONE, 0, 18),
    _s((0, 2), (1, 1), (2, 0)): Row(NONE, 0, 16),
    _s((0, 0), (1, 1), (1, 2)): Row(NONE, 0, 16),
    _s((0, 0), (1, 1), (2, 0), (2, 2)): Row(NONE, 0, 16),
    _s((0, 0), (0, 2), (1, 1)): Row(frozenset({"cap"}), 0, 21),
}
REPAIR_SLICE_S0 = _s((0, 0), (1, 1), (2, 2))
REPAIR_SLICE_EVENT = AvailabilityRemoval(pairs=((0, 0),))   # A unavailable in period 0
REPAIR_SLICE_LAMBDA_STAR_1 = 0

REPAIR_SLICE_SF = _s((1, 1), (2, 0), (2, 2))
REPAIR_SLICE_X = _s((0, 2), (1, 1), (2, 0))
REPAIR_SLICE_ROWS_1 = {                                   # under the instance after the event
    REPAIR_SLICE_S0: Row(frozenset({"unavailable"}), 0, 18),
    REPAIR_SLICE_SF: Row(NONE, 0, 13),
    REPAIR_SLICE_X: Row(NONE, 0, 16),
    _s((0, 2), (1, 1), (2, 0), (2, 2)): Row(NONE, 0, 14),
}
REPAIR_SLICE_DISTANCE = {                                 # H(S, S0)
    REPAIR_SLICE_S0: 0,
    REPAIR_SLICE_SF: 2,
    REPAIR_SLICE_X: 4,
    _s((0, 2), (1, 1), (2, 0), (2, 2)): 3,
}
REPAIR_SLICE_CHOICE = {                                   # arm name -> repair at t = 1
    "SF-chained": REPAIR_SLICE_SF,
    "SF-anchored": REPAIR_SLICE_SF,
    "QL-chained": REPAIR_SLICE_X,
    "QO": REPAIR_SLICE_X,
    "WQS-chained-l1": REPAIR_SLICE_X,
    "WQS-chained-l2": REPAIR_SLICE_SF,
    "WQS-chained-l4": REPAIR_SLICE_SF,
    "HWF-chained-a1": REPAIR_SLICE_SF,
}
REPAIR_SLICE_N_CELLS = 9
REPAIR_SLICE_SF_CHURN = Fraction(2, 9)
