"""Policies as lists of objectives (exposé Table 1). Pure: no solver, no I/O.

Per arm and step, after the shared coverage cap Lambda <= Lambda*_t:

    QO : [MaxQuality]
    SF : [MinDistance(ref), MaxQuality]
    QL : [MaxQuality, MinDistance(ref)]
    WQS: [MaxQualityMinusDistance(ref, lam)]
    HWF: [MinDistance(ref, weights=history_weights(alpha, b)), MaxQuality]

The runner adds the coverage cap and appends MAX_TIE_BREAK. Neither belongs
in policy_stages. Burden is the history BEFORE the current repair.
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
    """Return S0 for anchored, the previous schedule for chained, or None for QO."""
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
    """Return alpha.denominator + alpha.numerator * burden[w] for every volunteer.

    This equals alpha.denominator * (1 + alpha * burden[w]), using exact
    integer arithmetic. Missing burdens default to zero. Inputs are not mutated.

    Reject a non-Fraction or negative alpha, a non-integer or negative burden
    (including bool), or a burden entry for an unknown volunteer.
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
    """Return the policy's ordered objectives, without coverage or tie-breaking.

    QO requires reference=None; every other policy requires a schedule. An empty
    schedule is valid. Only HWF uses burden, covering every instance volunteer.
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
