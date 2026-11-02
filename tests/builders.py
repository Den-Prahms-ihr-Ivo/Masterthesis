"""Test-only builders, so test bodies stay short. Implement each the first time a test needs it."""

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

from collections.abc import Iterable, Mapping, Sequence


def _dense(
    values: Mapping[int, int] | Sequence[int] | None,
    n: int,
    default: int,
    what: str,
    axis: str,
) -> tuple[int, ...]:
    """None -> all default; Mapping -> sparse overrides; Sequence -> taken as-is."""
    if values is None:
        return (default,) * n
    if isinstance(values, Mapping):
        bad = sorted(k for k in values if k not in range(n))
        if bad:
            raise ValueError(f"{what}: keys {bad} outside range({n}) ({axis}s)")
        return tuple(values.get(i, default) for i in range(n))
    if len(values) != n:
        raise ValueError(
            f"{what}: expected {n} values (one per {axis}), got {len(values)}"
        )
    return tuple(values)


def make_volunteer(
    volunteer_id: int,
    *,
    n_periods: int,
    n_roles: int,
    seniority: int = 0,
    cap: int = 10,
    available: Iterable[int] | None = None,
    skill: Mapping[int, int] | Sequence[int] | None = None,
    pref_role: Mapping[int, int] | Sequence[int] | None = None,
    pref_time: Mapping[int, int] | Sequence[int] | None = None,
) -> Volunteer:
    """Defaults: available in every period, skill 1 in every role, all preferences 0.

    skill, pref_role, pref_time: a dict of overrides by index, or the full
    sequence (one value per role / period). available: the full set of periods.
    """
    if available is None:
        periods = frozenset(range(n_periods))
    else:
        periods = frozenset(available)
        bad = sorted(p for p in periods if p not in range(n_periods))
        if bad:
            raise ValueError(f"available: periods {bad} outside range({n_periods})")

    return Volunteer(
        id=volunteer_id,
        available=periods,
        skill=_dense(skill, n_roles, 1, "skill", "role"),
        seniority=seniority,
        cap=cap,
        pref_role=_dense(pref_role, n_roles, 0, "pref_role", "role"),
        pref_time=_dense(pref_time, n_periods, 0, "pref_time", "period"),
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
    volunteers: Iterable[Volunteer],
    shifts: Iterable[Shift],
    *,
    n_periods: int,
    n_roles: int,
    scale: int = 1,
    size_class: int = 20,
    seed: int = 0,
    under_supplied: bool = False,
) -> Instance:
    """Test builder. Volunteer and shift ids must equal their positions."""
    volunteers = tuple(volunteers)
    shifts = tuple(shifts)

    for kind, items in (("volunteer", volunteers), ("shift", shifts)):
        ids = [id for id, _ in enumerate(items)]
        if ids != list(range(len(items))):
            raise ValueError(
                f"{kind} ids must be 0..{len(items) - 1} in order, got {ids}"
            )

    for s in shifts:
        if s.period not in range(n_periods) or s.role not in range(n_roles):
            raise ValueError(
                f"shift {s.id}: period {s.period} / role {s.role} out of range"
            )

    meta = InstanceMeta(
        size_class=size_class, scale=scale, seed=seed, under_supplied=under_supplied
    )
    return Instance(
        volunteers=volunteers,
        shifts=shifts,
        n_periods=n_periods,
        n_roles=n_roles,
        meta=meta,
    )
