"""Brute force: the reference solver for small instances.

Enumerates every subset of cells, keeps the feasible ones (oracle.is_feasible) and
picks the best by plain comparison. Shares no code with milp: it is the reference
the MILP is checked against, so it is checked against the hand values first.

One stage optimizes one objective. Earlier stages enter as constraints on other
objectives' values:
- fixed: value == v (an earlier stage held at its optimum);
- cap:   value <= v (an upper bound, e.g. Lambda <= Lambda*).
Ties go to the smallest schedule as a sorted tuple of cells, so results are deterministic.
"""

import itertools
import oracle

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import NamedTuple
from repair.model import Instance, Schedule

MAX_CELLS = 20  # 2^20 subsets; beyond that the brute force is too slow to be useful


@dataclass(frozen=True)
class Objective:
    name: str  # matches StageRecord.name, e.g. "MaxQuality"
    maximize: bool
    evaluate: Callable[[Schedule, Instance], int]


MIN_SHORTFALL = Objective("MinShortfall", maximize=False, evaluate=oracle.shortfall)
MAX_QUALITY = Objective("MaxQuality", maximize=True, evaluate=oracle.quality)


class Solution(NamedTuple):
    schedule: Schedule | None  # None if no feasible schedule meets fixed and cap
    value: int | None


def all_schedules(instance: Instance) -> Iterator[Schedule]:
    """Every feasible schedule of the instance, each exactly once.

    Raises ValueError if the instance has more than MAX_CELLS cells.
    """
    volunteer_ids = [v.id for v in instance.volunteers]
    shift_ids = [s.id for s in instance.shifts]

    # 1. Get the full universe of individual pairs
    universe_pairs = list(itertools.product(volunteer_ids, shift_ids))
    # For 3 volunteers and 1 shift: [(0, 0), (1, 0), (2, 0)]

    if len(universe_pairs) > MAX_CELLS:
        raise ValueError("Instance has more than MAX_CELLS")

    # 2. Generate all possible subsets (empty set, partial sets, full set), one at a time
    power_set = itertools.chain.from_iterable(
        itertools.combinations(universe_pairs, r)
        for r in range(len(universe_pairs) + 1)
    )

    for combination in power_set:
        schedule = frozenset(combination)

        if oracle.is_feasible(schedule, instance):
            yield schedule


def solve_stage(
    instance: Instance,
    objective: Objective,
    *,
    fixed: Mapping[Objective, int] | None = None,
    cap: Mapping[Objective, int] | None = None,
) -> Solution:
    """Best feasible schedule for objective, subject to fixed (==) and cap (<=).

    Ties go to the smallest tuple(sorted(schedule)).
    """
    fixed = fixed or {}
    cap = cap or {}

    candidates: list[tuple[Schedule, int]] = []
    for s in all_schedules(instance):
        meets_fixed = all(o.evaluate(s, instance) == v for o, v in fixed.items())
        meets_cap = all(o.evaluate(s, instance) <= v for o, v in cap.items())

        if meets_fixed and meets_cap:
            candidates.append((s, objective.evaluate(s, instance)))

    if not candidates:
        return Solution(None, None)

    sign = -1 if objective.maximize else 1
    best, value = min(
        candidates,
        key=lambda c: (sign * c[1], tuple(sorted(c[0]))),
    )
    return Solution(best, value)
