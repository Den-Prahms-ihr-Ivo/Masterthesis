"""Turn each repair policy into an ordered list of objectives (Table 1).

All policies receive the same coverage cap from the runner. Their own stages are:
    QO:  MaxQuality
    SF:  MinDistance, then MaxQuality
    QL:  MaxQuality, then MinDistance
    WQS: MaxQualityMinusDistance
    HWF: history-weighted MinDistance, then MaxQuality

The runner adds the final tie-break. HWF uses burden from earlier repairs
only. This module builds the objectives; it does not solve them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction

from repair.model import Arm, Instance, ReferenceMode, Schedule, VolunteerId
from repair.objectives import (
    MAX_QUALITY,
    Objective,
    max_quality_minus_distance,
    min_distance,
)


def validate_arm(arm: Arm) -> None:
    """Reject unknown policies and invalid combinations of policy parameters.

    QO:     reference=None, lam=None, alpha=None.
    SF/QL:  ReferenceMode, lam=None, alpha=None.
    WQS:    ReferenceMode, integer lam >= 0 (not bool), alpha=None.
    HWF:    ReferenceMode, Fraction alpha >= 0, lam=None.
    """
    if arm.policy not in ("QO", "SF", "QL", "WQS", "HWF"):
        raise ValueError(f"unknown policy {arm.policy!r}")

    if arm.policy == "QO":
        if arm.reference is not None:
            raise ValueError("QO requires reference=None")
    elif not isinstance(arm.reference, ReferenceMode):
        raise ValueError(f"{arm.policy} requires a ReferenceMode")

    if arm.policy == "WQS":
        if (
            not isinstance(arm.lam, int)
            or isinstance(arm.lam, bool)
            or arm.lam < 0
        ):
            raise ValueError("WQS requires an integer lam >= 0 (not bool)")
    elif arm.lam is not None:
        raise ValueError(f"{arm.policy} requires lam=None")

    if arm.policy == "HWF":
        if not isinstance(arm.alpha, Fraction) or arm.alpha < 0:
            raise ValueError("HWF requires a Fraction alpha >= 0")
    elif arm.alpha is not None:
        raise ValueError(f"{arm.policy} requires alpha=None")


def reference_for(arm: Arm, s0: Schedule, previous: Schedule) -> Schedule | None:
    """Choose the schedule against which repair distance is measured.

    Anchored uses S0 throughout. Chained uses this arm's previous schedule.
    QO has no distance objective and therefore returns None.
    """
    validate_arm(arm)

    if arm.policy == "QO":
        return None
    if arm.reference == ReferenceMode.ANCHORED:
        return s0
    return previous  # validation guarantees CHAINED here


def history_weights(
    alpha: Fraction,
    burden: Mapping[VolunteerId, int],
    volunteers: Iterable[VolunteerId],
) -> dict[VolunteerId, int]:
    """Calculate integer weights for HWF from the earlier burden.

    For alpha = a/b, use b + a * burden[w]. This is the same as multiplying
    1 + alpha * burden[w] by b, so the minimiser stays the same and no rounding
    is needed. Missing volunteers get burden zero. Invalid or negative values
    and unknown volunteers are rejected; the input mapping is left unchanged.
    """
    if not isinstance(alpha, Fraction) or alpha < 0:
        raise ValueError("alpha must be a Fraction >= 0")

    # Materialize once: volunteers may be a generator, consumed by membership checks.
    volunteer_ids = tuple(volunteers)
    known = set(volunteer_ids)

    for w, count in burden.items():
        if w not in known:
            raise ValueError(f"burden contains unknown volunteer {w!r}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(
                f"burden for volunteer {w!r} must be an integer >= 0 (not bool)"
            )

    return {
        w: alpha.denominator + alpha.numerator * burden.get(w, 0)
        for w in volunteer_ids
    }


def policy_stages(
    arm: Arm,
    reference: Schedule | None,
    burden: Mapping[VolunteerId, int],
    instance: Instance,
) -> tuple[Objective, ...]:
    """Build the policy stages for this repair, in priority order.

    Coverage and tie-breaking are added by the runner. QO needs reference=None;
    all other policies need a reference schedule, which may be empty. Only HWF
    uses the burden mapping.
    """
    validate_arm(arm)

    if arm.policy == "QO":
        if reference is not None:
            raise ValueError("QO requires reference=None")
        return (MAX_QUALITY,)

    if reference is None:
        raise ValueError(f"{arm.policy} requires a reference schedule")

    if arm.policy == "SF":
        return (min_distance(reference), MAX_QUALITY)

    if arm.policy == "QL":
        return (MAX_QUALITY, min_distance(reference))

    if arm.policy == "WQS":
        assert arm.lam is not None  # established by validate_arm
        return (max_quality_minus_distance(reference, arm.lam),)

    # HWF is the only remaining policy after validation.
    assert arm.alpha is not None
    weights = history_weights(
        arm.alpha, burden, (v.id for v in instance.volunteers)
    )
    return (min_distance(reference, weights=weights), MAX_QUALITY)
