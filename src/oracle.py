"""Oracle: hard constraints, U(S) and Lambda(S), re-implemented in plain Python.

Must never import milp or objectives: it is the independent check of both.

Hard constraints (exposé 4.2):
- assignment only in available periods;
- at most one shift per volunteer and period;
- at most L_w shifts per volunteer;
- open shift (>= 1 assignee): >= ceil(d_s/2) assignees, >= 1 with skill >= kappa_s,
  and if expert-flagged >= 1 with skill 3.
A shift with no assignees is cancelled and satisfies its constraints.

Quality:
U(S) = sum u_ws x_ws + beta_mentor * sum y_s + beta_buffer * sum z_s - gamma * sum q_s
  u_ws = 2 p^role_{w,r(s)} + p^time_{w,p(s)} + k_{w,r(s)}
  q_s  = max(0, sum_w x_ws - d_s);  z_s = min(b_s, q_s)
  y_s  = 1 iff s has a novice (k <= 1) and a mentor-eligible volunteer (sigma = 2, k >= 2)
Lambda(S) = sum_s max(0, d_s - sum_w x_ws)   (cancelled shift -> full d_s)
All values are integers.
"""

from math import ceil
from repair.model import Instance, Schedule, Shift, ShiftId, Volunteer, VolunteerId

# ------------------------------------------------------------------ feasibility


def min_crew(demand: int) -> int:
    """ceil(d_s / 2)."""
    return ceil(demand / 2)


def violations(schedule: Schedule, instance: Instance) -> list[str]:
    """Human-readable list of every violated hard constraint; empty iff feasible."""
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
    return not violations(schedule, instance)


def cancelled_shifts(schedule: Schedule, instance: Instance) -> frozenset[ShiftId]:
    """Shifts with no assignee."""
    return frozenset(s for s, n in staffing(schedule, instance).items() if n == 0)


# ------------------------------------------------------------------ quality and coverage


def assignment_utility(instance: Instance, w: VolunteerId, s: ShiftId) -> int:
    """u_ws = 2 p^role_{w,r(s)} + p^time_{w,p(s)} + k_{w,r(s)}."""
    volunteer = instance.volunteers[w]
    shift = instance.shifts[s]
    return (
        2 * volunteer.pref_role[shift.role]
        + volunteer.pref_time[shift.period]
        + volunteer.skill[shift.role]
    )


def staffing(schedule: Schedule, instance: Instance) -> dict[ShiftId, int]:
    """Assignees per shift, 0 for cancelled shifts; every shift appears."""
    count: dict[ShiftId, int] = {s.id: 0 for s in instance.shifts}
    for _, s in schedule:
        count[s] += 1
    return count


def surplus(schedule: Schedule, instance: Instance) -> dict[ShiftId, int]:
    """q_s."""
    count = staffing(schedule, instance)
    return {s.id: max(0, count[s.id] - s.demand) for s in instance.shifts}


def mentored(schedule: Schedule, instance: Instance) -> frozenset[ShiftId]:
    """Shifts with y_s = 1."""
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
    """U(S)."""
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
    """Lambda(S)."""
    count = staffing(schedule, instance)
    return sum(max(0, s.demand - count[s.id]) for s in instance.shifts)
