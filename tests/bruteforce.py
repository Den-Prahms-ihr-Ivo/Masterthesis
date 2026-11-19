"""Solve very small test instances by trying every possible schedule.

Check each subset of assignment cells with the oracle, then compare the
feasible schedules directly. I use this as a reference for the MILP, after
checking it against the hand-calculated fixture values.

A stage optimises one objective. Earlier optimal values are fixed with ==;
unproven values enter as caps (<= when minimising, >= when maximising).
Equal objective values are resolved by the smallest sorted schedule tuple.

The function signature matches the runner's StageSolver, so tests can pass
this solver into the same pipeline. Enumeration grows as 2^J and is only
practical for the small fixtures.
"""

import itertools
import oracle

from collections.abc import Iterator, Mapping
from repair.model import Instance, Schedule, StageResult, StageStatus
from repair.objectives import Objective

MAX_CELLS = 20  # 2^20 subsets; beyond that the brute force is too slow to be useful


def all_schedules(instance: Instance) -> Iterator[Schedule]:
    """Every feasible schedule of the instance, each exactly once.

    Order: by size (empty schedule first), then itertools.combinations order.
    Raises ValueError if the instance has more than MAX_CELLS cells. Because this is
    a generator, the error appears at the first next(), not at the call.
    """
    volunteer_ids = [v.id for v in instance.volunteers]
    shift_ids = [s.id for s in instance.shifts]

    # 1. Every possible cell (w, s): J = volunteers x shifts
    universe_pairs = list(itertools.product(volunteer_ids, shift_ids))
    # For 3 volunteers and 1 shift: [(0, 0), (1, 0), (2, 0)]

    if len(universe_pairs) > MAX_CELLS:
        raise ValueError("Instance has more than MAX_CELLS")

    # 2. All 2^J subsets (empty set, partial sets, full set), generated one at a time
    #    instead of building a list first, so memory stays small
    power_set = itertools.chain.from_iterable(
        itertools.combinations(universe_pairs, r)
        for r in range(len(universe_pairs) + 1)
    )

    # 3. Keep only the subsets the oracle accepts
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
    start: Schedule | None = None,
    time_limit: float | None = None,
) -> StageResult:
    """Best feasible schedule for objective, subject to fixed (==) and cap (not worse).

    Ties go to the smallest tuple(sorted(schedule)). start and time_limit are accepted
    so the runner can call this like milp.solve_stage; the brute force ignores both.

    Returns StageResult(schedule, value) with status "optimal" (the brute force always
    proves optimality), or StageResult(None, None, "no_solution") if no feasible
    schedule meets fixed and cap.
    """
    fixed = fixed or {}
    cap = cap or {}

    # 1. Filter: feasible schedules that keep every fixed value and respect every cap
    candidates: list[tuple[Schedule, int]] = []  # (schedule, its objective value)
    for s in all_schedules(instance):
        meets_fixed = all(o.evaluate(s, instance) == v for o, v in fixed.items())
        meets_cap = all(o.not_worse(o.evaluate(s, instance), v) for o, v in cap.items())

        if meets_fixed and meets_cap:
            candidates.append((s, objective.evaluate(s, instance)))

    if not candidates:  # min() of an empty list would raise
        return StageResult(None, None, StageStatus.NO_SOLUTION)

    # 2. Pick the best. One key does both jobs:
    #    - sign * value: min() finds the smallest; sign = -1 turns "largest U" into "smallest -U"
    #    - tuple(sorted(schedule)): breaks ties by the smallest sorted schedule
    #    (reverse=True or max() would also flip the tie-break, so neither is used)
    sign = -1 if objective.maximize else 1
    best, value = min(
        candidates,
        key=lambda c: (sign * c[1], tuple(sorted(c[0]))),
    )
    return StageResult(best, value)
