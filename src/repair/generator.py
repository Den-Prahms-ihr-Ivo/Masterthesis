"""Draw synthetic instances using the settings in exposé 4.3 and Table A1.

The same configuration and seed give the same candidate instance. There is
no solver or file access here; runner.accept_instance handles acceptance.

Each random component has its own NumPy stream. For example, changing the
skill distribution should not shift the availability draws. This separation
also requires a fixed draw count where one component depends on another.
assign_experts therefore draws once per shift, even for ineligible shifts.

Shifts are ordered by period and then role, with ids from 0 to |S| - 1.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace
from math import floor

from repair.model import Instance, InstanceMeta, Shift, Volunteer
from repair.protocol import GeneratorConfig

# Each component has a fixed position in the spawned random streams.
# Appending a name keeps the old streams, but moving or deleting a name
# changes which stream the later components receive.
SUBSTREAMS = (
    "shifts",  # which (period, role) combinations are active (rho)
    "demand",  # Requested staff per shift, d_s.
    "min_skill",  # Minimum skill requirement, kappa_s.
    "expert",  # Expert requirement, e_s.
    "commitment",  # commitment class per volunteer
    "availability",  # availability count and placement
    "skill",  # Skill per volunteer and role, k_wr.
    "seniority",  # Volunteer seniority, sigma_w.
    "cap",  # Volunteer workload cap, L_w.
    "pref_role",  # Preference for each role.
    "pref_time",  # Preference for each period.
)

UNDER_SUPPLIED_CLASS = 40  # Fewer volunteers, but keep the staffing-demand distribution of n=40.


def make_streams(seed: int) -> dict[str, np.random.Generator]:
    """Create one random generator per component, all derived from the given seed."""
    children = np.random.SeedSequence(seed).spawn(len(SUBSTREAMS))

    return {
        name: np.random.default_rng(child)
        for name, child in zip(SUBSTREAMS, children, strict=True)
    }


def demand_draw(size_class: int, rng: np.random.Generator) -> int:
    """Requested crew d_s for one shift of a demand class:

    n = 20: 1 + Bernoulli(0.25)    -> {1, 2}
    n = 40: 2 + Bernoulli(0.5)     -> {2, 3}
    n = 80: 4 + Binomial(2, 0.5)   -> {4, 5, 6}

    ValueError for any other size class.
    """
    if size_class == 20:
        return int(1 + rng.binomial(1, 0.25))

    if size_class == 40:
        return int(2 + rng.binomial(1, 0.5))

    if size_class == 80:
        return int(4 + rng.binomial(2, 0.5))

    raise ValueError(f"unknown size class {size_class}")


def generate_shifts(
    cfg: GeneratorConfig, size_class: int, streams: dict[str, np.random.Generator]
) -> tuple[Shift, ...]:
    """Period x role grid (n_days * periods_per_day x n_roles); each combination is
    active with probability cfg.role_activation (rho).

    Per active shift: demand from demand_draw ("demand" stream), min_skill = 2 with
    probability cfg.min_skill_two_share, else 1 ("min_skill" stream). expert = False
    here; generate_instance sets it once the volunteers are known.
    Period-major order, ids 0..|S|-1.
    """
    if size_class not in (20, 40, 80):
        raise ValueError(f"unknown size class {size_class}")

    n_periods = cfg.n_days * cfg.periods_per_day

    active = streams["shifts"].random((n_periods, cfg.n_roles)) < cfg.role_activation

    out: list[Shift] = []

    for period in range(n_periods):
        for role in range(cfg.n_roles):
            if not active[period, role]:
                continue

            demand = demand_draw(size_class, streams["demand"])

            min_skill = (
                2 if streams["min_skill"].random() < cfg.min_skill_two_share else 1
            )

            out.append(
                Shift(
                    id=len(out),
                    period=period,
                    role=role,
                    demand=demand,
                    min_skill=min_skill,
                    expert=False,
                )
            )

    return tuple(out)


def generate_volunteers(
    cfg: GeneratorConfig,
    n_volunteers: int,
    n_periods: int,
    streams: dict[str, np.random.Generator],
) -> tuple[Volunteer, ...]:
    """Per volunteer, each from its own stream:

    commitment class  ~ cfg.commitment_shares (low, medium, high)
    |available|       ~ uniform integer in cfg.availability_ranges[class], bounds inclusive
    available periods ~ that many periods, uniformly without replacement
    skill (per role)  ~ cfg.skill_shares over 0..3, independently per role
    seniority         ~ cfg.seniority_shares over 0..2
    cap               ~ uniform over cfg.caps
    pref_role, pref_time ~ uniform over {0, 1, 2}, independently per role / per period

    Table A1: "All attributes are drawn independently unless stated otherwise." The only
    dependences are commitment class -> availability count, and (in assign_experts)
    skills + availability -> expert eligibility.
    """
    out: list[Volunteer] = []

    for volunteer_id in range(n_volunteers):
        commitment = int(
            streams["commitment"].choice(
                3,
                p=cfg.commitment_shares,
            )
        )

        lo, hi = cfg.availability_ranges[commitment]

        if not 0 <= lo <= hi <= n_periods:
            raise ValueError(
                f"invalid availability range {(lo, hi)} " f"for {n_periods} periods"
            )

        count = int(streams["availability"].integers(lo, hi + 1))

        available = frozenset(
            int(period)
            for period in streams["availability"].choice(
                n_periods,
                size=count,
                replace=False,
            )
        )

        skill = tuple(
            int(value)
            for value in streams["skill"].choice(
                4,
                size=cfg.n_roles,
                p=cfg.skill_shares,
            )
        )

        seniority = int(
            streams["seniority"].choice(
                3,
                p=cfg.seniority_shares,
            )
        )

        cap = int(streams["cap"].choice(cfg.caps))

        # Table A1: p^role, p^time in {0, 1, 2}, uniform (synthetic)
        pref_role = tuple(
            int(value)
            for value in streams["pref_role"].integers(0, 3, size=cfg.n_roles)
        )

        pref_time = tuple(
            int(value) for value in streams["pref_time"].integers(0, 3, size=n_periods)
        )

        out.append(
            Volunteer(
                id=volunteer_id,
                available=available,
                skill=skill,
                seniority=seniority,
                cap=cap,
                pref_role=pref_role,
                pref_time=pref_time,
            )
        )

    return tuple(out)


def assign_experts(
    cfg: GeneratorConfig,
    shifts: tuple[Shift, ...],
    volunteers: tuple[Volunteer, ...],
    rng: np.random.Generator,
) -> tuple[Shift, ...]:
    """Set the expert flag only where an available skill-3 volunteer exists.

    For eligible shifts, the probability is cfg.expert_share. I draw one number
    for every shift, including ineligible ones, so eligibility changes do not
    shift the later random draws. This check does not guarantee full coverage;
    the same expert may be needed on competing shifts.
    """
    draws = rng.random(len(shifts))
    out: list[Shift] = []

    for shift in shifts:
        eligible = any(
            volunteer.skill[shift.role] == 3 and shift.period in volunteer.available
            for volunteer in volunteers
        )

        expert = eligible and bool(draws[shift.id] < cfg.expert_share)

        out.append(replace(shift, expert=expert))

    return tuple(out)


def n_volunteers_for(
    size_class: int, cfg: GeneratorConfig, under_supplied: bool = False
) -> int:
    """|W|: size_class for the main classes (cfg.workforce_factor is ignored there);
    floor(40 * cfg.workforce_factor) for the under-supplied class.

    ValueError if under_supplied and size_class != 40, if phi is not in (0, 1)
    (Table A1: "fewer volunteers"; phi >= 1 is not under-supplied), or if phi is so
    small that no volunteer is left.
    """
    if not under_supplied:
        return size_class

    if size_class != UNDER_SUPPLIED_CLASS:
        raise ValueError("under-supplied instances require size class 40")

    phi = cfg.workforce_factor
    if not 0 < phi < 1:
        raise ValueError(f"workforce_factor must be in (0, 1), got {phi}")

    n_volunteers = floor(UNDER_SUPPLIED_CLASS * phi)
    if n_volunteers < 1:
        raise ValueError(f"workforce_factor {phi} leaves no volunteers")

    return n_volunteers


def scale_for(size_class: int) -> int:
    """Perturbation scale k = size_class / 20 -> 1, 2, 4 (exposé 4.4).
    Uses the DEMAND class, not the actual workforce: the under-supplied class (fewer
    volunteers, n = 40 demand) also gets k = 2. ValueError for an unknown size class.
    """
    if size_class not in (20, 40, 80):
        raise ValueError(f"unknown size class {size_class}")

    return size_class // 20


def generate_instance(
    cfg: GeneratorConfig, size_class: int, seed: int, under_supplied: bool = False
) -> Instance:
    """Build one candidate; acceptance is a separate step.

    First create the random streams, then the shifts and volunteers, and finally
    the expert flags. The metadata records the demand class, scale and seed.
    The period count is cfg.n_days * cfg.periods_per_day. Equal inputs give an
    equal Instance, but do not guarantee that all demand can be covered.
    """
    scale = scale_for(size_class)
    n_volunteers = n_volunteers_for(
        size_class,
        cfg,
        under_supplied=under_supplied,
    )
    n_periods = cfg.n_days * cfg.periods_per_day

    streams = make_streams(seed)

    shifts = generate_shifts(cfg, size_class, streams)

    volunteers = generate_volunteers(
        cfg,
        n_volunteers,
        n_periods,
        streams,
    )

    shifts = assign_experts(
        cfg,
        shifts,
        volunteers,
        streams["expert"],
    )

    meta = InstanceMeta(
        size_class,
        scale,
        seed,
        under_supplied,
    )

    return Instance(
        volunteers,
        shifts,
        n_periods,
        cfg.n_roles,
        meta,
        cfg.weights,
    )


def capacity_ratio(instance: Instance) -> float:
    """Compare total workload capacity with total requested staffing.

    The ratio is sum_w L_w / sum_s d_s (exposé 4.3). It is a pilot diagnostic,
    not a feasibility test: availability and skill can still prevent coverage.
    An instance without shifts has no defined ratio and raises ValueError.
    """
    if not instance.shifts:
        raise ValueError("capacity ratio is undefined for an instance with no shifts")

    total_capacity = sum(volunteer.cap for volunteer in instance.volunteers)
    total_demand = sum(shift.demand for shift in instance.shifts)

    return total_capacity / total_demand
