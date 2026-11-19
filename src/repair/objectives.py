"""Describe what each solver stage should optimise.

An Objective stores its name, direction and a function for evaluating a
schedule. Distance objectives also store the reference schedule and, where
needed, volunteer weights or lambda.

The brute-force solver uses evaluate() directly. The runner uses it to check
the MILP answer. The MILP builds its own linear expression from the name and
parameters, without calling evaluate(). This keeps the calculations separate.

Objectives are frozen and can be dictionary keys. For example,
fixed = {MIN_SHORTFALL: 0} tells later stages to keep shortfall at zero.
Equality ignores the function object and compares the stored parameters.
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


# Tie-break


def tie_break_ranks(instance: Instance) -> dict[Cell, int]:
    """Assign the ranks 1..J to all cells using the instance seed.

    Equal instances give equal ranks. Permuting ranks avoids always giving the
    largest weight to the same volunteer ids. Different schedules can still
    have the same rank sum, so this is not a proof of a unique optimum.
    """
    cells = [(v.id, s.id) for v in instance.volunteers for s in instance.shifts]
    ranks = list(range(1, len(cells) + 1))  # 1, 2, ..., J
    random.Random(instance.meta.seed).shuffle(ranks)  # same seed -> same order
    return dict(zip(cells, ranks))


def tie_break(schedule: Schedule, instance: Instance) -> int:
    """Sum the ranks of assigned cells for the last optimisation stage.

    Earlier objective values remain fixed or capped. The rank sum helps choose
    between schedules with equal earlier values, but equal sums are possible.
    """
    ranks = tie_break_ranks(instance)
    return sum(ranks[cell] for cell in schedule)


# The objectives

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
    """Create the ordinary or history-weighted distance objective.

    Both added and removed cells count. With weights, each changed cell gets
    its volunteer's weight. Copy the supplied weights so later changes to the
    caller's dictionary cannot change this objective.
    """

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
    """Create the WQS objective U(S) - lambda * H(S, reference).

    Lambda is the quality cost of one changed cell. Only non-negative integers
    are accepted here so the objective remains integer-valued.
    """

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
