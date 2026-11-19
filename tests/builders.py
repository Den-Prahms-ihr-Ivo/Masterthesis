"""Build small instances with readable defaults for the tests.

Most tests change only the fields involved in their rule. These helpers
fill the rest, but still construct real Instance objects and run their
validation. They should not bypass checks just to make a fixture pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Optional

import numpy as np

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
    """Fill defaults, apply sparse index overrides, or keep a supplied full sequence."""
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
    """Build a test instance and let Instance validate its ids, periods and roles."""
    meta = InstanceMeta(
        size_class=size_class, scale=scale, seed=seed, under_supplied=under_supplied
    )
    return Instance(
        volunteers=tuple(volunteers),
        shifts=tuple(shifts),
        n_periods=n_periods,
        n_roles=n_roles,
        meta=meta,
    )


@dataclass(frozen=True)
class Grid:
    """Keep test dimensions together so volunteers and shifts use the same shape."""

    n_periods: int = 1
    n_roles: int = 1

    def volunteer(self, volunteer_id: int, **fields: Any) -> Volunteer:
        return make_volunteer(
            volunteer_id, n_periods=self.n_periods, n_roles=self.n_roles, **fields
        )

    def volunteers(self, n: int) -> list[Volunteer]:
        """n default volunteers with ids 0..n-1."""
        return [self.volunteer(i) for i in range(n)]

    def shifts(self, n: int) -> list[Shift]:
        """n default shifts with ids 0..n-1, each in its own (period, role) slot."""
        slots = list(product(range(self.n_periods), range(self.n_roles)))
        if n > len(slots):
            raise ValueError(f"{n} shifts do not fit into {len(slots)} slots")
        return [
            make_shift(i, period=period, role=role)
            for i, (period, role) in enumerate(slots[:n])
        ]

    def instance(
        self, volunteers: Iterable[Volunteer], shifts: Iterable[Shift], **meta: Any
    ) -> Instance:
        return make_instance(
            volunteers, shifts, n_periods=self.n_periods, n_roles=self.n_roles, **meta
        )


def greedy_schedule(instance: Instance, seed: int = 0) -> Schedule:
    """Build an S0-like set of cells for sampler tests without using a solver.

    Pick available volunteers in random order, up to each shift's demand,
    without assigning someone twice in a period. Caps, skills and minimum crew
    are ignored, so the result is not necessarily feasible or optimal. Use it
    only in tests that need assignment targets, not a valid solved schedule.
    """
    rng = np.random.default_rng(seed)
    busy: set[tuple[int, int]] = set()
    cells: set[tuple[int, int]] = set()
    for shift in instance.shifts:
        crew = [
            int(w)
            for w in rng.permutation(len(instance.volunteers))
            if shift.period in instance.volunteers[w].available
            and (int(w), shift.period) not in busy
        ][: shift.demand]
        for w in crew:
            busy.add((w, shift.period))
            cells.add((w, shift.id))
    return frozenset(cells)
