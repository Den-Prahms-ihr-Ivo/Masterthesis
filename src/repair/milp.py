"""Solve one optimisation stage with SCIP through PySCIPOpt (exposé 4.2).

solve_stage follows the same interface as the small brute-force solver. The
runner can therefore use either one. Keep the constraints and utility formulas
separate from oracle.py, which independently checks the returned schedules.
The seeded rank mapping is shared through objectives.tie_break_ranks.

Model (one binary x[w, s] per available cell; unavailable cells get no variable):
- hard constraints:
    sum_s x[w, s] <= 1 per volunteer and period;  sum_s x[w, s] <= L_w per volunteer
    open[s] binary, x[w, s] <= open[s], sum_w x[w, s] >= open[s]  (open if and only if staffed)
    sum_w x[w, s] >= ceil(d_s / 2) * open[s]
    sum_{w: k_w,r(s) >= kappa_s} x[w, s] >= open[s];  expert: sum_{w: k = 3} x[w, s] >= open[s]
- aux variables for U (must never overstate the true U, see below):
    q[s] >= sum_w x[w, s] - d_s, q[s] >= 0     surplus (minimized through -gamma * q)
    z[s] <= q[s], z[s] <= b                   buffer reward
    y[s] <= sum of novices, y[s] <= sum of mentors   mentorship
  A maximizing stage pushes q down and z, y up to their true values. A stage that only
  fixes U (e.g. MinDistance with U == U*) does not, so every aux variable must be
  bounded by its true value from the side that would inflate U:
  model U <= true U for every x. Then "model U == U*" implies true U >= U*.
- r[s] >= d_s - sum_w x[w, s], r[s] >= 0     shortfall per shift (Lambda = sum r)

Objectives (dispatched on Objective.name):
    MaxQuality   U = sum u_ws x + beta_mentor sum y + beta_buffer sum z - gamma sum q
    MinShortfall Lambda = sum r
    MaxTieBreak  sum rank_ws x  (ranks from objectives.tie_break_ranks)
    MinDistance  H = |reference| + sum_{c not in ref} x_c - sum_{c in ref, available} x_c
                 The constant |reference| also counts reference cells that are now
                 unavailable (no variable): they always count as changed.
    MinDistance with weights (HWF): the same with weight w_w per cell of volunteer w,
                 constant sum_{(w, s) in ref} w_w
    MaxQualityMinusDistance (WQS): U - lam * H

fixed: objective expression == v.  cap: expression not worse than v (<= or >=).
start: handed to SCIP with addSol; SCIP checks it and drops it if infeasible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pyscipopt
from pyscipopt import quicksum

from repair.model import Cell, Instance, Schedule, StageResult, StageStatus
from repair.objectives import Objective, tie_break_ranks


@dataclass
class _Model:
    """Keep the SCIP model and variable dictionaries together for one stage solve."""

    scip: pyscipopt.Model
    x: dict[Cell, pyscipopt.Variable]  # available cells only
    open: dict[int, pyscipopt.Variable]  # shift id -> open[s]
    q: dict[int, pyscipopt.Variable]  # surplus
    z: dict[int, pyscipopt.Variable]  # buffer
    y: dict[int, pyscipopt.Variable]  # mentorship
    r: dict[int, pyscipopt.Variable]  # shortfall


# 1. variables and hard constraints


def _build(instance: Instance) -> _Model:
    """Build assignment variables, hard constraints and auxiliary variables.

    Unavailable cells have no x variable. Shift-opening variables allow a shift
    to be cancelled, while an open shift must meet its minimum crew and skill
    rules. The module docstring gives the mathematical expressions.
    """
    scip = pyscipopt.Model()
    scip.hideOutput()
    volunteers = instance.volunteers
    weights = instance.weights

    # Create variables only for available assignments. Unavailable cells
    # cannot be chosen, so they need no separate availability constraint.
    x: dict[Cell, pyscipopt.Variable] = {}
    for v in volunteers:
        for sh in instance.shifts:
            if sh.period in v.available:
                x[(v.id, sh.id)] = scip.addVar(f"x_{v.id}_{sh.id}", vtype="B")

    # Group cells once so later constraints can find the relevant variables.
    by_shift: dict[int, list[Cell]] = {sh.id: [] for sh in instance.shifts}
    by_volunteer: dict[int, list[Cell]] = {v.id: [] for v in volunteers}
    for w, s in x:
        by_shift[s].append((w, s))
        by_volunteer[w].append((w, s))

    # Limit the workload of each volunteer.
    for v in volunteers:
        mine = by_volunteer[v.id]

        # Total assignments must stay within this volunteer's cap.
        scip.addCons(quicksum(x[c] for c in mine) <= v.cap)

        # A volunteer cannot work two roles in the same period.
        for p in range(instance.n_periods):
            in_period = [c for c in mine if instance.shifts[c[1]].period == p]
            if len(in_period) > 1:  # This constraint only matters with several possible assignments.
                scip.addCons(quicksum(x[c] for c in in_period) <= 1)

    # Link the open flag to staffing and apply the rules for an open shift.
    open_: dict[int, pyscipopt.Variable] = {}  # Avoid shadowing Python's open().
    q: dict[int, pyscipopt.Variable] = {}
    z: dict[int, pyscipopt.Variable] = {}
    y: dict[int, pyscipopt.Variable] = {}
    r: dict[int, pyscipopt.Variable] = {}
    for sh in instance.shifts:
        cells = by_shift[sh.id]
        staff = quicksum(x[c] for c in cells)  # An empty sum is zero when nobody is available.
        is_open = open_[sh.id] = scip.addVar(f"open_{sh.id}", vtype="B")

        def skill(c: Cell) -> int:
            return volunteers[c[0]].skill[sh.role]  # Use skill in this shift's role.

        # Any assignment forces the shift to be open.
        for c in cells:
            scip.addCons(x[c] <= is_open)

        # An open shift needs ceil(d/2) people. Keep this separate from the oracle.
        scip.addCons(staff >= ((sh.demand + 1) // 2) * is_open)

        # At least one assigned person must meet the shift's skill requirement.
        scip.addCons(
            quicksum(x[c] for c in cells if skill(c) >= sh.min_skill) >= is_open
        )

        # An expert shift also needs one person with skill 3.
        if sh.expert:
            scip.addCons(quicksum(x[c] for c in cells if skill(c) == 3) >= is_open)

        # Bound the auxiliary variables so they cannot overstate actual quality
        # or understate shortfall. Later stages may fix U without maximising it.

        # Surplus has a penalty, so give q its lower bound.
        q[sh.id] = scip.addVar(f"q_{sh.id}", vtype="I", lb=0)
        scip.addCons(q[sh.id] >= staff - sh.demand)

        # Reward at most buffer_cap surplus places.
        z[sh.id] = scip.addVar(f"z_{sh.id}", vtype="I", lb=0, ub=weights.buffer_cap)
        scip.addCons(z[sh.id] <= q[sh.id])

        # The mentorship reward needs both a novice and an eligible mentor.
        novices = [c for c in cells if skill(c) <= 1]
        mentors = [
            c for c in cells if skill(c) >= 2 and volunteers[c[0]].seniority == 2
        ]
        y[sh.id] = scip.addVar(f"y_{sh.id}", vtype="B")
        scip.addCons(y[sh.id] <= quicksum(x[c] for c in novices))
        scip.addCons(y[sh.id] <= quicksum(x[c] for c in mentors))

        # Missing staff contributes to the shortfall objective.
        r[sh.id] = scip.addVar(f"r_{sh.id}", vtype="I", lb=0)
        scip.addCons(r[sh.id] >= sh.demand - staff)

    return _Model(scip=scip, x=x, open=open_, q=q, z=z, y=y, r=r)


# 2.-6. objective expressions


def _quality(m: _Model, instance: Instance) -> pyscipopt.Expr:
    """Build the linear expression for schedule quality U.

    U = sum(u_ws * x_ws) + beta_mentor * sum(y_s)
        + beta_buffer * sum(z_s) - gamma * sum(q_s).
    The assignment contribution is calculated here independently of the oracle.
    """

    def utility(w: int, s: int) -> int:
        volunteer = instance.volunteers[w]
        shift = instance.shifts[s]

        return (
            2 * volunteer.pref_role[shift.role]
            + volunteer.pref_time[shift.period]
            + volunteer.skill[shift.role]
        )

    weights = instance.weights

    assignment_utility = quicksum(
        utility(w, s) * variable for (w, s), variable in m.x.items()
    )

    mentoring_reward = weights.beta_mentor * quicksum(m.y.values())

    buffer_reward = weights.beta_buffer * quicksum(m.z.values())

    surplus_penalty = weights.gamma_extra * quicksum(m.q.values())

    return assignment_utility + mentoring_reward + buffer_reward - surplus_penalty


def _shortfall(m: _Model, instance: Instance) -> pyscipopt.Expr:
    """Add the missing staffing over all shifts: Lambda = sum(r_s)."""
    return quicksum(m.r.values())


def _tie_break(m: _Model, instance: Instance) -> pyscipopt.Expr:
    """Build the sum of seeded assignment ranks for the final stage."""

    ranks = tie_break_ranks(instance)

    return quicksum(ranks[cell] * variable for cell, variable in m.x.items())


def _distance(m: _Model, reference: Schedule) -> pyscipopt.Expr:
    """Build H(S, reference) from removed and added cells.

    The constant len(reference) includes assignments that are now unavailable.
    These have no variable, but still count as removals. Omitting the constant
    would undercount the disruption caused by availability changes.
    """
    removed = len(reference) - quicksum(
        variable for cell, variable in m.x.items() if cell in reference
    )

    added = quicksum(
        variable for cell, variable in m.x.items() if cell not in reference
    )

    return removed + added


def _weighted_distance(
    m: _Model,
    reference: Schedule,
    weights: Mapping[int, int],
) -> pyscipopt.Expr:
    """Build the same assignment distance with a weight per volunteer.

    Unavailable reference cells still contribute their weight as removals.
    This is the MILP expression used for HWF.
    """
    required = {w for w, _ in reference} | {w for w, _ in m.x}
    missing = required - weights.keys()
    if missing:
        raise ValueError(f"missing distance weights for volunteers {sorted(missing)}")

    constant = sum(weights[w] for w, _ in reference)

    retained = quicksum(
        weights[w] * variable for (w, s), variable in m.x.items() if (w, s) in reference
    )

    added = quicksum(
        weights[w] * variable
        for (w, s), variable in m.x.items()
        if (w, s) not in reference
    )

    return constant - retained + added


def _quality_minus_distance(
    m: _Model,
    instance: Instance,
    reference: Schedule,
    lam: int,
) -> pyscipopt.Expr:
    """Build the WQS expression: quality minus lambda times distance."""
    return _quality(m, instance) - lam * _distance(m, reference)


def _expression(m, instance, objective):
    """Choose the linear expression that matches the objective name and parameters."""
    if objective.name == "MaxQuality":
        return _quality(m, instance)

    if objective.name == "MinShortfall":
        return _shortfall(m, instance)

    if objective.name == "MaxTieBreak":
        return _tie_break(m, instance)

    if objective.name == "MinDistance":
        assert objective.reference is not None
        weights = objective.weight_map()
        if weights is not None:
            return _weighted_distance(m, objective.reference, weights)
        return _distance(m, objective.reference)

    if objective.name == "MaxQualityMinusDistance":
        assert objective.reference is not None and objective.lam is not None
        return _quality_minus_distance(m, instance, objective.reference, objective.lam)

    raise ValueError(f"unknown objective {objective.name!r}")


# 3.-4. fixed and cap


def _constrain(
    m: _Model,
    instance: Instance,
    fixed: Mapping[Objective, int],
    cap: Mapping[Objective, int],
) -> None:
    """Keep earlier stage results as equalities or bounds.

    A proven optimum is fixed with ==. An unproven incumbent becomes a cap:
    >= for a maximisation objective, <= for a minimisation objective. Later
    stages may improve an incumbent but cannot make it worse.
    """
    for objective, value in fixed.items():
        expression = _expression(m, instance, objective)
        m.scip.addCons(expression == value)

    for objective, value in cap.items():
        expression = _expression(m, instance, objective)

        if objective.maximize:
            m.scip.addCons(expression >= value)
        else:
            m.scip.addCons(expression <= value)


# 7. warm start


def _add_start(m: _Model, start: Schedule) -> None:
    """Offer the starting schedule to SCIP as a candidate solution.

    Cells without a variable are skipped. SCIP checks the resulting candidate
    and can reject it if it is infeasible; this should not stop the solve.
    """
    sol = m.scip.createPartialSol()

    for cell, variable in m.x.items():
        m.scip.setSolVal(
            sol,
            variable,
            1.0 if cell in start else 0.0,
        )

    m.scip.addSol(sol)


# 8. solve and status


def solve_stage(
    instance: Instance,
    objective: Objective,
    *,
    fixed: Mapping[Objective, int],
    cap: Mapping[Objective, int],
    start: Schedule | None,
    time_limit: float,
    seed: int = 0,
    threads: int = 1,
) -> StageResult:
    """Build, constrain, optimize; map SCIP's status to a StageResult.

    SCIP status -> StageResult:
        "optimal"                     -> (schedule, value, OPTIMAL, dual bound)
        "timelimit" with a solution   -> (incumbent, value, TIME_LIMIT, dual bound)
        "timelimit" without, "infeasible" -> (None, None, NO_SOLUTION)
    value is the objective value rounded to int (all coefficients are integers).
    seed sets randomization/randomseedshift; threads = 1 keeps runs reproducible.
    """
    m = _build(instance)  # Rebuild the model so no constraints from an earlier call remain.

    _constrain(m, instance, fixed, cap)

    # Use an auxiliary objective variable because setObjective has no constant term.
    # This keeps the constant part of MinDistance in the reported value.
    obj = m.scip.addVar("obj", vtype="C", lb=None, ub=None)  # Allow negative quality values.
    m.scip.addCons(obj == _expression(m, instance, objective))
    m.scip.setObjective(obj, "maximize" if objective.maximize else "minimize")

    # Settings. SCIP's optimize() is single-threaded, so threads needs no parameter yet
    m.scip.setParam("limits/time", time_limit)
    m.scip.setParam("randomization/randomseedshift", seed)

    if start is not None:
        _add_start(m, start)

    m.scip.optimize()

    status = _status(m.scip)
    if status == StageStatus.NO_SOLUTION or m.scip.getNSols() == 0:
        return StageResult(None, None, StageStatus.NO_SOLUTION)

    best = m.scip.getBestSol()
    raw = m.scip.getSolObjVal(best)
    value = round(raw)
    if (
        abs(raw - value) > 1e-6
    ):  # A non-integer value is unexpected with these integer objectives.
        raise ValueError(f"{objective.name}: non-integer objective value {raw}")

    return StageResult(_schedule(m), value, status, m.scip.getDualbound())


def _schedule(m: _Model) -> Schedule:
    """Read assigned cells from SCIP using x > 0.5 to allow numerical rounding."""
    best = m.scip.getBestSol()
    return frozenset(c for c, var in m.x.items() if m.scip.getSolVal(best, var) > 0.5)


_STATUS = {
    "optimal": StageStatus.OPTIMAL,
    "timelimit": StageStatus.TIME_LIMIT,  # solve_stage checks whether a solution exists
    "infeasible": StageStatus.NO_SOLUTION,
}


def _status(scip: pyscipopt.Model) -> StageStatus:
    """Translate the solver status into the statuses stored by this project.

    SCIP handles Ctrl-C itself and reports "userinterrupt". Raise
    KeyboardInterrupt here so an interrupted run stops without writing a false
    job failure. Unexpected statuses raise ValueError instead of being treated
    as ordinary time limits.
    """
    status = scip.getStatus()
    if status == "userinterrupt":
        raise KeyboardInterrupt("SCIP was interrupted (SIGINT)")
    if status not in _STATUS:
        raise ValueError(f"unexpected SCIP status {status!r}")
    return _STATUS[status]


# 9. versions


def solver_versions() -> dict[str, str]:
    """Return the SCIP and PySCIPOpt versions for the run manifest."""

    scip = pyscipopt.Model()

    scip_version = (
        f"{scip.getMajorVersion()}."
        f"{scip.getMinorVersion()}."
        f"{scip.getTechVersion()}"
    )

    return {
        "scip": scip_version,
        "pyscipopt": pyscipopt.__version__,
    }
