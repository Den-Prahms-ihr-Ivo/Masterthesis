"""Objective specs: what one lexicographic stage optimizes.

An Objective has three parts:
- name:     shows up in StageRecord.name, e.g. "MaxQuality"
- maximize: the direction (True = larger is better, False = smaller is better)
- evaluate: computes the objective's value for (schedule, instance), via the oracle
- reference: MinDistance and MaxQualityMinusDistance, the schedule the distance is
             measured to (S_{t-1} or S_0)
- weights:   weighted MinDistance only (HWF), volunteer -> integer weight
- lam:       MaxQualityMinusDistance only (WQS), utility points per changed cell

It is solver-independent. The brute force calls evaluate on every candidate schedule;
runner.check_oracles calls it to re-check every stage of a solver result.

Objectives are frozen dataclasses, so they are hashable and can be dict keys:
fixed = {MIN_SHORTFALL: 0} means "Lambda must stay 0". Equality ignores evaluate, so
min_distance(S0) == min_distance(S0) although each call builds a new function.

milp.solve_stage builds the linear objective from name (and reference); it never calls
evaluate. That keeps the MILP independent of the oracle.
"""

import random

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import oracle
from repair import metrics
from repair.model import Cell, Instance, Schedule, VolunteerId


@dataclass(frozen=True)
class Objective:
    name: str  # matches StageRecord.name, e.g. "MaxQuality"
    maximize: bool  # True: larger is better (U); False: smaller is better (Lambda)
    evaluate: Callable[[Schedule, Instance], int] = field(
        compare=False
    )  # e.g. oracle.quality
    reference: Schedule | None = None  # MinDistance, MaxQualityMinusDistance
    # weighted MinDistance: sorted (volunteer, weight) pairs; a tuple, not a dict, so
    # the Objective stays hashable (it is a dict key in fixed / cap)
    weights: tuple[tuple[VolunteerId, int], ...] | None = None
    lam: int | None = None  # MaxQualityMinusDistance

    def weight_map(self) -> dict[VolunteerId, int] | None:
        """weights as a dict (None for an unweighted objective)."""
        return None if self.weights is None else dict(self.weights)

    def not_worse(self, value: int, bound: int) -> bool:
        """value is at least as good as bound: >= when maximizing, <= when minimizing.

        Used for caps: MAX_QUALITY.not_worse(7, 6) is True, MIN_SHORTFALL.not_worse(1, 0)
        is False.
        """
        return value >= bound if self.maximize else value <= bound


# ------------------------------------------------------------------ tie-break


def tie_break_ranks(instance: Instance) -> dict[Cell, int]:
    """Rank 1..J for every cell (J = volunteers x shifts), a permutation seeded by instance.meta.seed.

    Same instance -> same ranks, so the final schedule is reproducible.
    """
    cells = [(v.id, s.id) for v in instance.volunteers for s in instance.shifts]
    ranks = list(range(1, len(cells) + 1))  # 1, 2, ..., J
    random.Random(instance.meta.seed).shuffle(ranks)  # same seed -> same order
    return dict(zip(cells, ranks))


def tie_break(schedule: Schedule, instance: Instance) -> int:
    """Sum of the tie-break ranks of the schedule's cells.

    The last stage maximizes this. It picks one schedule among those that tie on
    every earlier objective, deterministically and without favouring low ids.
    """
    ranks = tie_break_ranks(instance)
    return sum(ranks[cell] for cell in schedule)


# ------------------------------------------------------------------ the objectives

MIN_SHORTFALL = Objective(
    "MinShortfall", maximize=False, evaluate=oracle.shortfall
)  # Lambda
MAX_QUALITY = Objective("MaxQuality", maximize=True, evaluate=oracle.quality)  # U
MAX_TIE_BREAK = Objective("MaxTieBreak", maximize=True, evaluate=tie_break)

# the starting schedule S0: best coverage, then best quality, then the tie-break
S0_SPECS = (MIN_SHORTFALL, MAX_QUALITY, MAX_TIE_BREAK)


def min_distance(
    reference: Schedule,
    weights: Mapping[VolunteerId, int] | None = None,
) -> Objective:
    """Minimize ordinary or weighted assignment distance."""

    if weights is None:

        def evaluate(schedule: Schedule, instance: Instance) -> int:
            return metrics.distance(schedule, reference)

        return Objective(
            "MinDistance",
            maximize=False,
            evaluate=evaluate,
            reference=reference,
        )

    # Copy the caller's mapping so later changes cannot alter this objective.
    weight_map = dict(weights)

    for w, weight in weight_map.items():
        if not isinstance(weight, int) or isinstance(weight, bool) or weight < 0:
            raise ValueError(
                f"weight for volunteer {w!r} must be "
                "a non-negative integer (not bool)"
            )

    # Immutable representation keeps Objective hashable for fixed/cap dictionaries.
    frozen = tuple(sorted(weight_map.items()))

    def evaluate_weighted(schedule: Schedule, instance: Instance) -> int:
        changed = schedule ^ reference
        missing = {w for w, _ in changed} - weight_map.keys()

        if missing:
            raise ValueError(
                f"missing distance weights for volunteers {sorted(missing)}"
            )

        return sum(weight_map[w] for w, _ in changed)

    return Objective(
        "MinDistance",
        maximize=False,
        evaluate=evaluate_weighted,
        reference=reference,
        weights=frozen,
    )


def max_quality_minus_distance(
    reference: Schedule,
    lam: int,
) -> Objective:
    """Maximize quality minus lam times assignment distance."""

    if not isinstance(lam, int) or isinstance(lam, bool) or lam < 0:
        raise ValueError("lam must be a non-negative integer (not bool)")

    def evaluate(schedule: Schedule, instance: Instance) -> int:
        return oracle.quality(schedule, instance) - lam * metrics.distance(
            schedule, reference
        )

    return Objective(
        "MaxQualityMinusDistance",
        maximize=True,
        evaluate=evaluate,
        reference=reference,
        lam=lam,
    )
