"""Objective specs: what one lexicographic stage optimizes.

An Objective has three parts:
- name:     shows up in StageRecord.name, e.g. "MaxQuality"
- maximize: the direction (True = larger is better, False = smaller is better)
- evaluate: computes the objective's value for (schedule, instance), via the oracle

It is solver-independent. The brute force calls evaluate on every candidate schedule;
runner.check_oracles calls it to re-check every stage of a solver result.

Objectives are frozen dataclasses, so they are hashable and can be dict keys:
fixed = {MIN_SHORTFALL: 0} means "Lambda must stay 0".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import oracle
from repair.model import Cell, Instance, Schedule


@dataclass(frozen=True)
class Objective:
    name: str  # matches StageRecord.name, e.g. "MaxQuality"
    maximize: bool  # True: larger is better (U); False: smaller is better (Lambda)
    evaluate: Callable[[Schedule, Instance], int]  # e.g. oracle.quality

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
    raise NotImplementedError


def tie_break(schedule: Schedule, instance: Instance) -> int:
    """Sum of the tie-break ranks of the schedule's cells.

    The last stage maximizes this. It picks one schedule among those that tie on
    every earlier objective, deterministically and without favouring low ids.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ the objectives

MIN_SHORTFALL = Objective("MinShortfall", maximize=False, evaluate=oracle.shortfall)  # Lambda
MAX_QUALITY = Objective("MaxQuality", maximize=True, evaluate=oracle.quality)  # U
MAX_TIE_BREAK = Objective("MaxTieBreak", maximize=True, evaluate=tie_break)

# the starting schedule S0: best coverage, then best quality, then the tie-break
S0_SPECS = (MIN_SHORTFALL, MAX_QUALITY, MAX_TIE_BREAK)
