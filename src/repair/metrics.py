"""Calculate the outcome measures from stored schedules and events.

I use Fraction for ratios to avoid rounding during the calculations. None
means a value is undefined, which is different from a measured zero.

The main burden measure counts repairs that affected each volunteer. The
sensitivity measure counts changed assignment cells. Both compare successive
schedules, even when the policy itself uses S0 as its repair reference.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import replace
from fractions import Fraction
from typing import Any, Literal

import oracle

from repair.model import (
    Classification,
    Instance,
    ReferenceMode,
    Schedule,
    SequenceResult,
    StageRecord,
    StageStatus,
    Trajectory,
    VolunteerId,
)
from repair.perturbations import instances_along


def distance(a: Schedule, b: Schedule) -> int:
    """Count added and removed assignment cells: H(a, b) = |a xor b|.

    Moving one volunteer from one shift to another changes two cells, not one.
    """
    return len(a ^ b)


def affected(previous: Schedule, current: Schedule) -> frozenset[VolunteerId]:
    """Return each volunteer with a changed cell once, even if several cells changed."""
    return frozenset(w for w, _ in previous ^ current)


def changed_cells_by_volunteer(
    previous: Schedule, current: Schedule
) -> dict[VolunteerId, int]:
    """Count changed cells for each affected volunteer; omit unchanged people."""
    return dict(Counter(w for w, _ in previous ^ current))


def _require_steps(trajectory: Trajectory, until: int) -> None:
    """Check that the requested prefix exists and its step numbers are consecutive."""
    if until < 0 or until > len(trajectory.steps):
        raise ValueError(
            f"until must be between 0 and {len(trajectory.steps)}, got {until}"
        )
    for t, step in enumerate(trajectory.steps[:until], start=1):
        if step.t != t:
            raise ValueError(f"expected step {t}, got {step.t}")


def burden(
    trajectory: Trajectory,
    s0: Schedule,
    volunteers: Iterable[VolunteerId],
    until: int,
    variant: Literal["repairs", "cells"] = "repairs",
) -> dict[VolunteerId, int]:
    """Accumulate disruption from S0 through the requested repair step.

    With variant="repairs", a volunteer receives at most one count per repair.
    With variant="cells", every added or removed assignment counts. Include
    unaffected volunteers with zero burden so inequality uses the whole workforce.
    """
    if variant not in ("repairs", "cells"):
        raise ValueError(f"unknown burden variant {variant!r}")
    _require_steps(trajectory, until)

    counts = {w: 0 for w in volunteers}
    previous = s0
    for step in trajectory.steps[:until]:
        changes = changed_cells_by_volunteer(previous, step.schedule)
        for w, count in changes.items():
            if w not in counts:
                raise ValueError(f"volunteers is missing changed volunteer {w!r}")
            counts[w] += 1 if variant == "repairs" else count
        previous = step.schedule
    return counts


def _validate_vector(values: Sequence[int]) -> None:
    if len(values) == 0:
        raise ValueError("burden vector must not be empty")
    if any(value < 0 for value in values):
        raise ValueError("burden vector must not contain negative values")


def gini(values: Sequence[int]) -> Fraction:
    """Calculate sum_ij |x_i - x_j| / (2 * n * sum(x)) exactly.

    Keep zero-burden volunteers in the input. If all burdens are zero, return
    zero by the study's convention. Empty or negative inputs are rejected.
    """
    _validate_vector(values)
    total = sum(values)
    if total == 0:
        return Fraction(0)
    differences = sum(abs(a - b) for a in values for b in values)
    return Fraction(differences, 2 * len(values) * total)


def max_share(values: Sequence[int]) -> Fraction | None:
    """Return the largest burden as a share of total burden.

    When total burden is zero, return None because there is no share to compare.
    """
    _validate_vector(values)
    total = sum(values)
    return None if total == 0 else Fraction(max(values), total)


def is_binding(previous: Schedule, instance_t: Instance, lambda_star: int) -> bool:
    """Check whether the previous schedule can still be kept after this event.

    It is binding if a hard constraint is broken or its shortfall exceeds the
    shared coverage cap. Being feasible alone is therefore not enough.
    """
    return bool(oracle.violations(previous, instance_t)) or (
        oracle.shortfall(previous, instance_t) > lambda_star
    )


def classify(
    specs: Sequence[Any],
    previous: Schedule,
    current: Schedule,
    instance_t: Instance,
    lambda_star: int,
    stages: Sequence[StageRecord],
) -> Classification:
    """Classify a repair as binding, relevant, undetermined or inert.

    First check whether the previous schedule is still allowed. Otherwise compare
    the policy objectives in their priority order. The first different value
    decides whether the new schedule is better. Leave the tie-break stage out:
    a different schedule with the same policy values is not a relevant change.
    Without an improvement, an unproven policy stage leaves the result undetermined.
    """
    if len(specs) != len(stages):
        raise ValueError("specs and stages must have the same length")
    if is_binding(previous, instance_t, lambda_star):
        return Classification("binding")

    for spec in specs:
        before = spec.evaluate(previous, instance_t)
        after = spec.evaluate(current, instance_t)
        if after != before:
            if spec.not_worse(after, before):
                return Classification("relevant")
            # A worse higher-priority value cannot be offset by a later objective.
            break

    if any(stage.status != StageStatus.OPTIMAL for stage in stages):
        return Classification("undetermined")
    return Classification("inert")


def classifications(
    result: SequenceResult, trajectory: Trajectory, instances_t: Sequence[Instance]
) -> list[Classification]:
    """Rebuild the policy used at each step before classifying it.

    Each arm has its own previous schedule and burden history. For HWF, only
    burden up to t - 1 enters the weights; including the current repair would
    change the objective that is being checked.
    """
    from repair import policies

    horizon = len(trajectory.steps)
    _require_steps(trajectory, horizon)
    if len(instances_t) < horizon or len(result.coverage) < horizon:
        raise ValueError("missing step instances or coverage records")

    volunteers = tuple(v.id for v in result.instance.volunteers)
    previous = result.s0
    output = []
    for step, instance_t in zip(trajectory.steps, instances_t):
        t = step.t
        coverage = result.coverage[t - 1]
        if coverage.t != t:
            raise ValueError(f"expected coverage step {t}, got {coverage.t}")
        reference = policies.reference_for(trajectory.arm, result.s0, previous)
        history = burden(trajectory, result.s0, volunteers, until=t - 1)
        specs = policies.policy_stages(trajectory.arm, reference, history, instance_t)
        output.append(
            classify(
                specs,
                previous,
                step.schedule,
                instance_t,
                coverage.lambda_star,
                step.stages[: len(specs)],
            )
        )
        previous = step.schedule
    return output


STEP_KEYS = ("arm", "t", "U", "g", "C", "B", "A", "certified", "classification")
TERMINAL_KEYS = (
    "arm",
    "until",
    "sum_g",
    "sum_C",
    "B",
    "total_burden",
    "gini",
    "max_share",
    "gini_cells",
    "binding_rate",
    "burden_repairs",
    "burden_cells",
)


def _qo_optima(result: SequenceResult, until: int | None = None) -> dict[int, int]:
    """Read proven QO quality values for the quality-gap comparison.

    A timed-out MaxQuality value is only an incumbent, so it cannot define U*_t.
    Missing QO values remain missing instead of being replaced with zero.
    """
    qo = [
        trajectory
        for trajectory in result.trajectories
        if trajectory.arm.policy == "QO"
    ]
    if len(qo) > 1:
        raise ValueError("expected at most one QO trajectory")
    if not qo:
        return {}
    optima = {}
    for step in qo[0].steps[:until]:
        for stage in step.stages:
            if stage.name == "MaxQuality" and stage.status == StageStatus.OPTIMAL:
                if stage.value is None:
                    raise ValueError("optimal MaxQuality record has no value")
                optima[step.t] = stage.value
                break
    return optima


def _rows_for_trajectory(
    result: SequenceResult,
    trajectory: Trajectory,
    instances_t: Sequence[Instance],
    qo_optima: dict[int, int],
) -> list[dict[str, Any]]:
    labels = classifications(result, trajectory, instances_t)
    previous = result.s0
    rows = []
    for step, instance_t, label in zip(trajectory.steps, instances_t, labels):
        j = instance_t.n_cells
        if j <= 0:
            raise ValueError("churn and drift require instance.n_cells > 0")
        quality = oracle.quality(step.schedule, instance_t)
        optimum = qo_optima.get(step.t)
        rows.append(
            {
                "arm": trajectory.arm.name,
                "t": step.t,
                "U": quality,
                "g": None if optimum is None else optimum - quality,
                "C": Fraction(distance(step.schedule, previous), j),
                "B": Fraction(distance(step.schedule, result.s0), j),
                "A": len(affected(previous, step.schedule)),
                "certified": all(
                    stage.status == StageStatus.OPTIMAL for stage in step.stages
                ),
                "classification": label.value,
            }
        )
        previous = step.schedule
    return rows


def step_rows(result: SequenceResult) -> list[dict[str, Any]]:
    """Build one row per arm and repair step, keeping ratios as Fractions."""
    instances_t = instances_along(result.instance, result.sequence.events)
    optima = _qo_optima(result)
    return [
        row
        for trajectory in result.trajectories
        for row in _rows_for_trajectory(result, trajectory, instances_t, optima)
    ]


def terminal_rows(result: SequenceResult, until: int) -> list[dict[str, Any]]:
    """Summarise each arm over the same number of completed repair steps.

    Later events and schedules are excluded before the metrics are calculated.
    This matters for the secondary analysis, which stops at the shared certified
    prefix. If any quality gap in that prefix is unknown, sum_g is also unknown.
    """
    if until < 1:
        raise ValueError("until must be at least 1")
    if len(result.sequence.events) < until:
        raise ValueError("sequence has fewer events than until")
    for trajectory in result.trajectories:
        _require_steps(trajectory, until)

    # Later events and repairs must not affect a prefix summary.
    instances_t = instances_along(result.instance, result.sequence.events[:until])
    optima = _qo_optima(result, until)
    volunteers = tuple(sorted(v.id for v in result.instance.volunteers))
    output = []
    for trajectory in result.trajectories:
        prefix = replace(trajectory, steps=trajectory.steps[:until])
        rows = _rows_for_trajectory(result, prefix, instances_t, optima)
        repairs = burden(trajectory, result.s0, volunteers, until, "repairs")
        cells = burden(trajectory, result.s0, volunteers, until, "cells")
        repair_values = [repairs[w] for w in volunteers]
        cell_values = [cells[w] for w in volunteers]
        gaps = [row["g"] for row in rows]
        output.append(
            {
                "arm": trajectory.arm.name,
                "until": until,
                "sum_g": None if any(gap is None for gap in gaps) else sum(gaps),
                "sum_C": sum((row["C"] for row in rows), Fraction(0)),
                "B": rows[-1]["B"],
                "total_burden": sum(repair_values),
                "gini": gini(repair_values),
                "max_share": max_share(repair_values),
                "gini_cells": gini(cell_values),
                "binding_rate": Fraction(
                    sum(row["classification"] == "binding" for row in rows), until
                ),
                "burden_repairs": repair_values,
                "burden_cells": cell_values,
            }
        )
    return output


def persistence(
    s0: Schedule,
    anchored: Trajectory,
    chained: Trajectory,
    forcing: int = 1,
    reconsider: int = 6,
) -> Fraction | None:
    """Measure which forcing changes persist until reconsideration.

    Start with cells changed by the chained arm relative to S0 at `forcing`.
    A cell counts only if its value stays unchanged at every step up to
    `reconsider`, and the anchored arm differs there at reconsideration.
    Changing back temporarily and later returning does not count. Return None
    when the forcing repair changed no cells, because the denominator is zero.
    """
    if not 1 <= forcing <= reconsider:
        raise ValueError("require 1 <= forcing <= reconsider")
    _require_steps(anchored, reconsider)
    _require_steps(chained, reconsider)

    forced = chained.steps[forcing - 1].schedule
    changed = forced ^ s0
    if not changed:
        return None

    kept = set(changed)
    for step in chained.steps[forcing - 1 : reconsider]:
        kept.difference_update(step.schedule ^ forced)
    kept.intersection_update(
        anchored.steps[reconsider - 1].schedule ^ chained.steps[reconsider - 1].schedule
    )
    return Fraction(len(kept), len(changed))


RQ2_KEYS = ("D", "D_pre", "P", "lock_in", "delta_u_6", "certified")


def rq2_row(result: SequenceResult) -> dict[str, Any]:
    """Compare anchored and chained outcomes over the 15 RQ2 steps.

    D sums quality differences (anchored minus chained) over steps 6..15;
    D_pre uses steps 2..5. Lock-in needs both positive persistence and a better
    anchored schedule at step 6. A persistent difference alone is not enough.

    The returned certification flag checks the arm stages. analysis.certified_prefix
    also checks the shared coverage stages before including a result as primary.
    """
    anchored_arms = [
        trajectory
        for trajectory in result.trajectories
        if trajectory.arm.reference == ReferenceMode.ANCHORED
    ]
    chained_arms = [
        trajectory
        for trajectory in result.trajectories
        if trajectory.arm.reference == ReferenceMode.CHAINED
    ]
    if len(anchored_arms) != 1 or len(chained_arms) != 1:
        raise ValueError("RQ2 requires exactly one anchored and one chained trajectory")
    anchored, chained = anchored_arms[0], chained_arms[0]
    _require_steps(anchored, 15)
    _require_steps(chained, 15)
    if len(result.sequence.events) < 15:
        raise ValueError("RQ2 requires at least 15 events")

    instances_t = instances_along(result.instance, result.sequence.events[:15])
    deltas = [
        oracle.quality(anchored.steps[t].schedule, instance_t)
        - oracle.quality(chained.steps[t].schedule, instance_t)
        for t, instance_t in enumerate(instances_t)
    ]
    p = persistence(result.s0, anchored, chained)
    return {
        "D": sum(deltas[5:15]),
        "D_pre": sum(deltas[1:5]),
        "P": p,
        "lock_in": p is not None and p > 0 and deltas[5] > 0,
        "delta_u_6": deltas[5],
        "certified": all(
            stage.status == StageStatus.OPTIMAL
            for trajectory in (anchored, chained)
            for step in trajectory.steps[:15]
            for stage in step.stages
        ),
    }
