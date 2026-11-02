"""Test-only builders, so test bodies stay short. Implement each the first time a test needs it.
"""

from __future__ import annotations

from typing import Callable, Optional

from repair.model import (
    Arm,
    Instance,
    Schedule,
    SequenceResult,
    InstanceMeta,
    Trajectory,
    Shift,
    Volunteer,
)


def make_volunteer(
    volunteer_id,
    *,
    available: Optional[list[int]] = None,
    skill: Optional[list[int]] = None,
    seniority=0,
    cap=10,
    pref_role=None,
    pref_time=None,
    n_periods=2,
    n_roles=2,
) -> Volunteer:
    """Defaults: available in every period, skill 1 in every role, all preferences 0."""
    return Volunteer(
        id=volunteer_id,
        available=frozenset(available if available is not None else (1,) * n_periods),
        skill=(tuple(skill) if skill is not None else (1,) * n_roles),
        seniority=seniority,
        cap=cap,
        pref_role=(tuple(pref_role) if pref_role is not None else (0,) * n_periods),
        pref_time=(tuple(pref_time) if pref_time is not None else (0,) * n_roles),
    )


def make_shift(
    shift_id, period=0, role=0, demand=1, min_skill=1, expert=False
) -> Shift:
    if not demand >= 1:
        raise ValueError(f"demand must be greater or equal to 1, got {demand}")

    if min_skill not in {1, 2}:
        raise ValueError(f"min_skill must be 1 or 2, got {min_skill}")

    return Shift(
        id=shift_id,
        period=period,
        role=role,
        demand=demand,
        min_skill=min_skill,
        expert=expert,
    )


def make_instance(
    volunteers, shifts, n_periods, n_roles, scale=1, size_class=20, seed=0
) -> Instance:
    meta = InstanceMeta(
        size_class=size_class, scale=scale, seed=seed, under_supplied=False
    )
    return Instance(
        volunteers=volunteers,
        shifts=shifts,
        n_periods=n_periods,
        n_roles=n_roles,
        meta=meta,
    )

