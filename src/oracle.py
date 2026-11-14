"""Check a schedule directly with the rules from exposé 4.2.

I keep these calculations separate from the MILP. Otherwise, the solver and
its check could agree because they use the same wrong formula. This module
therefore does not import milp or objectives.

A volunteer can only work in an available period, at most once per period,
and at most L_w times in total. An open shift needs ceil(d_s / 2) people,
someone with skill >= kappa_s in its role, and skill 3 if it requires an
expert. A shift with nobody assigned is cancelled. This is allowed, but its
full demand still counts towards shortfall.

The quality and coverage formulas are:
    u_ws = 2 * p_role[w, r(s)] + p_time[w, p(s)] + k[w, r(s)]
    U(S) = sum(u_ws * x_ws) + beta_mentor * sum(y_s)
           + beta_buffer * sum(z_s) - gamma * sum(q_s)
    q_s = max(0, staffing_s - d_s); z_s = min(b_s, q_s)
    Lambda(S) = sum(max(0, d_s - staffing_s))

y_s is 1 when a novice (skill <= 1) and a mentor (seniority 2, skill >= 2)
work together in that shift. All returned scores are integers.
"""

from math import ceil
from repair.model import Instance, Schedule, Shift, ShiftId, Volunteer, VolunteerId

# Feasibility


def min_crew(demand: int) -> int:
    """The minimum crew for an open shift: ceil(d_s / 2)."""
    return ceil(demand / 2)


def violations(schedule: Schedule, instance: Instance) -> list[str]:
    """List the hard constraints that the schedule breaks.

    An empty list means the schedule is feasible. Missing coverage is handled
    by shortfall(), so it is not a violation by itself. The cells are expected
    to refer to existing volunteers and shifts.
    """
    found: list[str] = []
    crew: dict[ShiftId, list[Volunteer]] = {s.id: [] for s in instance.shifts}
    worked: dict[VolunteerId, list[Shift]] = {v.id: [] for v in instance.volunteers}
    for w, s in sorted(schedule):
        crew[s].append(instance.volunteers[w])
        worked[w].append(instance.shifts[s])

    for volunteer in instance.volunteers:
        shifts = worked[volunteer.id]
        for shift in shifts:
            if shift.period not in volunteer.available:
                found.append(f"unavailable: volunteer {volunteer.id}, shift {shift.id}")
        periods = [shift.period for shift in shifts]
        for period in sorted(set(periods)):
            if periods.count(period) > 1:
                found.append(
                    f"two_shifts_same_period: volunteer {volunteer.id}, period {period}"
                )
        if len(shifts) > volunteer.cap:
            found.append(
                f"cap: volunteer {volunteer.id}, {len(shifts)} > {volunteer.cap}"
            )

    for shift in instance.shifts:
        people = crew[shift.id]
        if not people:
            continue  # cancelled: satisfies its constraints
        if len(people) < min_crew(shift.demand):
            found.append(f"min_crew: shift {shift.id}")
        if not any(v.skill[shift.role] >= shift.min_skill for v in people):
            found.append(f"skill: shift {shift.id}")
        if shift.expert and not any(v.skill[shift.role] == 3 for v in people):
            found.append(f"expert: shift {shift.id}")
    return found


def is_feasible(schedule: Schedule, instance: Instance) -> bool:
    """Return True when the schedule breaks no hard constraint."""

    return not violations(schedule, instance)


def cancelled_shifts(schedule: Schedule, instance: Instance) -> frozenset[ShiftId]:
    """Return the ids of shifts with nobody assigned."""
    return frozenset(s for s, n in staffing(schedule, instance).items() if n == 0)


# Quality and coverage


def assignment_utility(instance: Instance, w: VolunteerId, s: ShiftId) -> int:
    """Calculate the contribution of one volunteer on one shift.

    u_ws = 2 * p_role[w, r(s)] + p_time[w, p(s)] + k[w, r(s)].
    Role preference has twice the weight of time preference. Skill is looked
    up for this shift's role, not for the volunteer in general.
    """
    volunteer = instance.volunteers[w]
    shift = instance.shifts[s]
    return (
        2 * volunteer.pref_role[shift.role]
        + volunteer.pref_time[shift.period]
        + volunteer.skill[shift.role]
    )


def staffing(schedule: Schedule, instance: Instance) -> dict[ShiftId, int]:
    """Count people on every shift, including zero for cancelled shifts."""
    count: dict[ShiftId, int] = {s.id: 0 for s in instance.shifts}
    for _, s in schedule:
        count[s] += 1
    return count


def surplus(schedule: Schedule, instance: Instance) -> dict[ShiftId, int]:
    """Count assignments above demand for each shift: q_s = max(0, staffing - d_s).

    Understaffed and cancelled shifts both have zero surplus.
    """
    count = staffing(schedule, instance)
    return {s.id: max(0, count[s.id] - s.demand) for s in instance.shifts}


def mentored(schedule: Schedule, instance: Instance) -> frozenset[ShiftId]:
    """Return shifts where a novice and a mentor work together (y_s = 1).

    Both skill checks use the shift's role. Seniority 2 alone does not make
    someone a mentor; that person also needs skill >= 2 in this role.
    """
    crew: dict[ShiftId, list[Volunteer]] = {s.id: [] for s in instance.shifts}
    for w, s in schedule:
        crew[s].append(instance.volunteers[w])

    result: set[ShiftId] = set()
    for shift in instance.shifts:
        skills = [(v.skill[shift.role], v.seniority) for v in crew[shift.id]]
        has_novice = any(k <= 1 for k, _ in skills)
        has_mentor = any(k >= 2 and sigma == 2 for k, sigma in skills)
        if has_novice and has_mentor:
            result.add(shift.id)
    return frozenset(result)


def quality(schedule: Schedule, instance: Instance) -> int:
    """Calculate U(S) from the assignments, mentoring, buffer and surplus.

    The buffer reward stops at buffer_cap, while every extra assignment still
    receives the surplus penalty. This function evaluates the given schedule;
    it does not check feasibility or search for a better one.
    """
    weights = instance.weights
    utility = sum(assignment_utility(instance, w, s) for w, s in schedule)
    mentor = weights.beta_mentor * len(mentored(schedule, instance))
    q = surplus(schedule, instance)
    buffer = weights.beta_buffer * sum(
        min(weights.buffer_cap, q_s) for q_s in q.values()
    )
    extra = weights.gamma_extra * sum(q.values())
    return utility + mentor + buffer - extra


def shortfall(schedule: Schedule, instance: Instance) -> int:
    """Add the missing people over all shifts to obtain Lambda(S).

    A cancelled shift contributes its whole demand. Extra people on one shift
    do not compensate for missing people on another shift.
    """
    count = staffing(schedule, instance)
    return sum(max(0, s.demand - count[s.id]) for s in instance.shifts)
