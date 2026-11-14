"""Run the optimisation stages and the sequence of schedule repairs.

A lexicographic policy solves its objectives one after another. For S0,
these are MinShortfall, MaxQuality and MaxTieBreak. Once a stage is optimal,
its value is fixed in all later stages. After a time limit, its incumbent
becomes a bound instead: later stages may improve it, but not make it worse.
Each stage also passes its schedule to the next stage as a warm start.

If no schedule is found, the stage gets one retry with more time. A second
failure stops that solve. A result with a schedule can still be uncertified
when one of its stages reached the time limit.

accept_instance checks coverage of candidate base instances, and
initial_schedule computes S0. For a perturbation sequence, coverage is solved
once per step and shared by all arms. Each arm then keeps its own schedule
and burden history. Oracle checks validate S0, shared coverage and repairs.
The acceptance solve itself currently uses the solver's recorded value.

The stage solver is passed in as an argument. This lets the tests use brute
force or a small fake without changing the experiment logic.
"""

import time

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import oracle

from repair import metrics
from repair.generator import generate_instance
from repair.model import (
    AcceptanceLog,
    Arm,
    CoverageStep,
    Instance,
    LexResult,
    Schedule,
    StageResult,
    StageRecord,
    StageStatus,
    Sequence as EventSequence,
    SystematicError,
    SequenceResult,
    Step,
    Trajectory,
)
from repair.objectives import MAX_TIE_BREAK, MIN_SHORTFALL, S0_SPECS, Objective
from repair.perturbations import instances_along
from repair.policies import policy_stages, reference_for
from repair.protocol import GeneratorConfig, SolverSettings, derive_seed


class StageSolver(Protocol):
    """The signature every stage solver must have, so the runner can call any of them.

    Arguments:
        instance:   the problem to solve
        objective:  what this stage optimizes
        fixed:      earlier optimal stages, objective -> value that must be kept exactly
        cap:        earlier unproven stages, objective -> value that must not get worse
        start:      a schedule to start from (warm start); the solver may ignore it
        time_limit: seconds for this one call

    Returns a StageResult; StageResult.schedule is None if nothing was found in time.
    """

    def __call__(
        self,
        instance: Instance,
        objective: Objective,
        *,
        fixed: Mapping[Objective, int],
        cap: Mapping[Objective, int],
        start: Schedule | None,
        time_limit: float,
    ) -> StageResult: ...


class OracleMismatch(SystematicError):
    """A solver result disagrees with the oracle.

    Each message starts with a rule key, so tests can check why it was raised:
        "stages:"      the stages do not line up with the specs (count or names)
        "infeasible:"  the final schedule breaks a hard constraint
        "value:"       an optimal stage recorded a value the oracle does not compute
        "cap:"         an unproven (time_limit) stage got worse than its recorded incumbent
        "coverage:"    a repair schedule exceeds the shared coverage cap Lambda*_t
                       (raised by run_arm, not by check_oracles)
    """


def solve_lexicographic(
    instance: Instance,
    specs: Sequence[Objective],
    solve: StageSolver,
    *,
    time_limit: float,
    retry_factor: float = 2.0,
    start: Schedule | None = None,
    base_cap: Mapping[Objective, int] | None = None,
) -> LexResult:
    """Solve the stages in specs one after another (see the module docstring).

    Arguments:
        specs:        objectives in priority order
        solve:        the stage solver to call
        time_limit:   seconds per stage (a retry gets time_limit * retry_factor)
        retry_factor: SolverSettings.retry_factor, 2.0 in the exposé
        start:        start schedule for the first stage, e.g. S_{t-1}
        base_cap:     caps that hold for every stage from the start, e.g. the shared
                      coverage cap {MIN_SHORTFALL: Lambda*_t} of a repair step (3.3).
                      Copied, never mutated; later time_limit stages add to the copy.

    Returns a LexResult with one StageRecord per stage that ran (seconds measured
    around the solver call) and the last stage's schedule; schedule is None if a
    stage failed twice.
    """
    fixed: dict[Objective, int] = {}  # Later stages must keep these proven values exactly.
    cap: dict[Objective, int] = dict(base_cap) if base_cap is not None else {}
    records: list[StageRecord] = []

    for objective in specs:
        start_time = time.perf_counter()  # Include retry time in the recorded stage time.
        # Pass copies so later updates do not alter an earlier solver call.
        s = solve(
            instance,
            objective,
            fixed=dict(fixed),
            cap=dict(cap),
            start=start,
            time_limit=time_limit,
        )

        # Retry once when the first call returned no schedule.
        retried = s.schedule is None
        if retried:
            s = solve(
                instance,
                objective,
                fixed=dict(fixed),
                cap=dict(cap),
                start=start,
                time_limit=time_limit * retry_factor,
            )

        records.append(
            StageRecord(
                name=objective.name,
                status=s.status,
                value=s.value,
                bound=s.bound,
                seconds=time.perf_counter() - start_time,
                retried=retried,
            )
        )

        # There is still no schedule, so later stages cannot run.
        if s.schedule is None:
            return LexResult(schedule=None, stages=tuple(records))

        assert s.value is not None  # A returned schedule must also have an objective value.
        if s.status == StageStatus.OPTIMAL:
            fixed[objective] = s.value
        else:  # This value is unproven. Keep it as a bound, allowing later improvement.
            cap[objective] = s.value

        start = s.schedule  # Use this solution as the starting point for the next stage.

    return LexResult(schedule=start, stages=tuple(records))


def check_oracles(
    instance: Instance, specs: Sequence[Objective], result: LexResult
) -> None:
    """Re-check a solver result with the oracle; raise OracleMismatch on any disagreement.

    specs[i] belongs to result.stages[i]. Checks, in order:
    1. one stage per spec, with matching names                else "stages:"
    2. oracle.violations(result.schedule) is empty           else "infeasible:"
    3. for every stage, actual = spec.evaluate(result.schedule):
       - optimal stage:    actual == recorded value           else "value:"
       - time_limit stage: actual not worse than recorded     else "cap:"
         (a later stage may have improved it; that is fine)
    A result without a schedule has nothing to check and passes.
    """
    if result.schedule is None:
        return
    schedule = result.schedule

    # Check the lengths and names before zip() can hide a missing stage.
    names = [spec.name for spec in specs]
    recorded = [stage.name for stage in result.stages]
    if names != recorded:
        raise OracleMismatch(f"stages: specs {names}, result has {recorded}")

    # Check feasibility of the final schedule first.
    found = oracle.violations(schedule, instance)
    if found:
        raise OracleMismatch(f"infeasible: {found}")

    # Recalculate the stage values from that same final schedule.
    for spec, stage in zip(specs, result.stages):
        actual = spec.evaluate(schedule, instance)
        claimed = stage.value
        detail = f"{spec.name} recorded {claimed}, oracle computes {actual}"

        if claimed is None:  # a stage with a schedule always records its value
            raise OracleMismatch(f"value: {detail}")

        if stage.status == StageStatus.OPTIMAL and actual != claimed:
            raise OracleMismatch(f"value: {detail}")

        if stage.status == StageStatus.TIME_LIMIT and not spec.not_worse(
            actual, claimed
        ):
            raise OracleMismatch(f"cap: {detail} (worse than the unproven incumbent)")


# Base instances (exposé 4.3)


class AcceptanceFailed(Exception):
    """No candidate was accepted after trying the base and all reserve seeds."""


def candidate_seed(seed: int, draw: int) -> int:
    """Get the seed for one candidate attempt, counting draws from zero.

    The first attempt uses the given seed itself. Later attempts derive a new
    seed with the "redraw" key, so a rejected candidate can be reproduced.
    """
    if draw < 0:
        raise ValueError("draw must be nonnegative")

    if draw == 0:
        return seed

    return derive_seed(seed, "redraw", draw)


def accept_instance(
    cfg: GeneratorConfig,
    size_class: int,
    base_seed: int,
    reserve: Sequence[int],
    settings: SolverSettings,
    solve: StageSolver,
    *,
    max_draws: int = 100,
    under_supplied: bool = False,
) -> tuple[Instance, AcceptanceLog]:
    """Draw candidates until one passes the base-instance acceptance rule.

    For the main classes, run MinShortfall and accept a recorded value of zero.
    Shortfall cannot be negative, so a feasible incumbent with value zero is
    sufficient even after a time limit. Try max_draws candidates per seed,
    then move to the next reserve seed. No schedule counts as a rejection.

    For the under-supplied class, keep the first candidate with a returned
    schedule regardless of shortfall. No returned schedule raises ValueError.
    A positive value after a time limit is an incumbent, not a proven optimum.

    The log keeps the original seed, accepted candidate seed, draw and rejection
    counts, reserve use and recorded shortfall. The reserve list is not changed.

    Current limitation: this function does not call check_oracles on the
    acceptance result. initial_schedule checks its own later solve, but that
    does not independently validate this earlier acceptance decision.
    """
    if max_draws <= 0:
        raise ValueError("max_draws must be positive")

    draws = 0
    rejected = 0
    coverage_specs = (MIN_SHORTFALL,)

    seeds = (base_seed, *reserve)

    for seed_index, seed in enumerate(seeds):
        for draw in range(max_draws):
            used_seed = candidate_seed(seed, draw)

            candidate = generate_instance(
                cfg,
                size_class,
                used_seed,
                under_supplied=under_supplied,
            )
            draws += 1

            result = solve_lexicographic(
                candidate,
                coverage_specs,
                solve,
                time_limit=settings.time_limit,
                retry_factor=settings.retry_factor,
            )

            if result.schedule is None:
                if under_supplied:
                    raise ValueError(
                        "under-supplied coverage solve returned no schedule "
                        f"after retry; candidate seed {used_seed}"
                    )

                rejected += 1
                continue

            # Read the recorded acceptance value; this solve has no oracle check here.
            # S0 is solved and checked later, but that is a different solver call.
            # It cannot independently confirm this earlier acceptance decision.
            shortfall = result.stages[-1].value
            assert shortfall is not None  # A stage returning a schedule also needs its value.

            if under_supplied or shortfall == 0:
                log = AcceptanceLog(
                    base_seed=base_seed,
                    used_seed=used_seed,
                    draws=draws,
                    rejected=rejected,
                    replaced=seed_index > 0,
                    lambda_star_0=shortfall,
                )

                return candidate, log

            rejected += 1

    raise AcceptanceFailed(
        f"base seed {base_seed}: no acceptable instance after "
        f"{draws} candidates across {len(seeds)} seed families "
        f"({max_draws} draws per family)"
    )


def initial_schedule(
    instance: Instance, settings: SolverSettings, solve: StageSolver
) -> LexResult:
    """Find S0 by best coverage, then best quality, then tie-break.

    Use s0_time_limit for each stage and check the result with the oracle.
    make_instances replaces the instance when this result is not certified,
    because the experiment requires a proven starting schedule.
    """
    result = solve_lexicographic(
        instance,
        S0_SPECS,
        solve,
        time_limit=settings.s0_time_limit,
        retry_factor=settings.retry_factor,
    )

    check_oracles(instance, S0_SPECS, result)

    return result


# Arms (step 3.3)


@dataclass(frozen=True)
class SharedCoverage:
    """Keep each step's shared coverage value, solver record and starting schedule."""

    steps: tuple[CoverageStep, ...]  # One coverage value and record for each step t = 1..T.
    starts: tuple[
        Schedule, ...
    ]  # Every arm starts this step from the same coverage solution.


def shared_coverage(
    instances_t: list[Instance], settings: SolverSettings, solve: StageSolver
) -> SharedCoverage:
    """Solve minimum shortfall once per step for all repair arms.

    Start from the empty schedule and check the answer with the oracle. When
    the solve reaches its time limit, keep the incumbent as the shared cap and
    retain the unproven status. With no schedule after retry, raise RuntimeError
    because the repair arms have no coverage result to use.
    """
    steps: list[CoverageStep] = []
    starts: list[Schedule] = []
    specs = (MIN_SHORTFALL,)

    for t, instance in enumerate(instances_t, start=1):
        result = solve_lexicographic(
            instance,
            specs,
            solve,
            time_limit=settings.time_limit,
            retry_factor=settings.retry_factor,
            start=frozenset(),
        )
        if result.schedule is None:
            raise RuntimeError(f"shared coverage at step {t}: no schedule after retry")

        check_oracles(instance, specs, result)
        record = result.stages[0]
        assert record.value is not None
        steps.append(CoverageStep(t=t, lambda_star=record.value, record=record))
        starts.append(result.schedule)

    return SharedCoverage(steps=tuple(steps), starts=tuple(starts))


def run_arm(
    arm: Arm,
    instances_t: list[Instance],
    s0: Schedule,
    coverage: SharedCoverage,
    settings: SolverSettings,
    solve: StageSolver,
) -> Trajectory:
    """Steps 1..T for one arm. The arm keeps its own previous schedule (S_0 = s0) and
    burden b_w (all 0 at the start).

    Per step t (1-based; instance_t = instances_t[t - 1]):
    1. reference = reference_for(arm, s0, previous)
    2. specs = policy_stages(arm, reference, burden, instance_t) + (MAX_TIE_BREAK,)
    3. solve_lexicographic(instance_t, specs, solve, time_limit=settings.time_limit,
       retry_factor=settings.retry_factor, start=coverage.starts[t - 1],
       base_cap={MIN_SHORTFALL: Lambda*_t})
    4. no schedule -> Trajectory(arm, steps so far, ended_early=True), stop
    5. check_oracles(instance_t, specs, result); then oracle.shortfall(S_t) <= Lambda*_t,
       else OracleMismatch("coverage: ...")
    6. Step(t, S_t, result.stages); burden[w] += 1 for w in metrics.affected(previous, S_t)
    """
    if len(coverage.steps) != len(instances_t) or len(coverage.starts) != len(
        instances_t
    ):
        raise ValueError("coverage must have one step and start per instance")

    if not instances_t:
        return Trajectory(arm=arm, steps=(), ended_early=False)

    previous = s0
    burden = {
        v.id: 0 for v in instances_t[0].volunteers
    }  # Burden before this repair, b_{w,t-1}, for each volunteer.
    steps: list[Step] = []

    for t, instance in enumerate(instances_t, start=1):
        coverage_step = coverage.steps[t - 1]
        if coverage_step.t != t:
            raise ValueError(
                f"coverage step {coverage_step.t} is out of order; expected {t}"
            )

        reference = reference_for(arm, s0, previous)
        # Keep the old burden for the objective check; update it after the repair.
        specs = tuple(policy_stages(arm, reference, dict(burden), instance)) + (
            MAX_TIE_BREAK,
        )
        result = solve_lexicographic(
            instance,
            specs,
            solve,
            time_limit=settings.time_limit,
            retry_factor=settings.retry_factor,
            start=coverage.starts[t - 1],
            base_cap={MIN_SHORTFALL: coverage_step.lambda_star},
        )

        if result.schedule is None:
            return Trajectory(arm=arm, steps=tuple(steps), ended_early=True)

        check_oracles(instance, specs, result)
        shortfall = MIN_SHORTFALL.evaluate(result.schedule, instance)
        if shortfall > coverage_step.lambda_star:
            raise OracleMismatch(
                f"coverage: arm {arm.name}, step {t}: shortfall {shortfall} "
                f"exceeds shared cap {coverage_step.lambda_star}"
            )

        steps.append(Step(t=t, schedule=result.schedule, stages=result.stages))
        for w in metrics.affected(previous, result.schedule):
            burden[w] += 1
        previous = result.schedule

    return Trajectory(arm=arm, steps=tuple(steps), ended_early=False)


def run_sequence(
    cell: str,
    instance: Instance,
    acceptance: AcceptanceLog,
    s0_result: LexResult,
    sequence: EventSequence,
    arms: Sequence[Arm],
    settings: SolverSettings,
    solve: StageSolver,
    manifest: dict,
) -> SequenceResult:
    """Run all arms on the same S0 and perturbation sequence.

    Build the step instances once, solve their shared coverage, then run each
    arm in the given order. All arms receive the same instance objects; only
    their repair choices and burden histories differ. S0 must be certified.
    """
    if s0_result.schedule is None or not s0_result.certified:
        raise ValueError("S0 must have a schedule and be certified")

    instances_t = instances_along(instance, sequence.events)
    coverage = shared_coverage(instances_t, settings, solve)
    trajectories = tuple(
        run_arm(arm, instances_t, s0_result.schedule, coverage, settings, solve)
        for arm in arms
    )

    return SequenceResult(
        cell=cell,
        instance=instance,
        acceptance=acceptance,
        sequence=sequence,
        s0=s0_result.schedule,
        s0_stages=s0_result.stages,
        coverage=coverage.steps,
        trajectories=trajectories,
        manifest=dict(manifest),
    )
