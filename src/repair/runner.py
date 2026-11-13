"""Lexicographic solving and the run-time oracle check.

Lexicographic solve
-------------------
A policy is a list of objectives in priority order, e.g. S0_SPECS =
[MinShortfall, MaxQuality, MaxTieBreak]. solve_lexicographic optimizes them one at a
time ("stages"); each stage may not undo what the earlier stages achieved.

After each stage, its result is handed on to every later stage:
- status "optimal"    -> fixed[objective] = value     (later stages keep value == v)
- status "time_limit" -> cap[objective]   = incumbent (later stages keep value not worse
                         than v; they may still improve it, because v was not proven optimal)
- its schedule        -> start of the next stage      (a warm start for the MILP)

Example on single_shift with S0_SPECS:
    stage 1 MinShortfall  fixed {}                      -> Lambda = 0
    stage 2 MaxQuality    fixed {MinShortfall: 0}       -> U = 6
    stage 3 MaxTieBreak   fixed {MinShortfall: 0, MaxQuality: 6}

Retry: a stage that returns no schedule is retried once with retry_factor * time_limit
(StageRecord.retried = True). If the retry also finds nothing, the solve stops: the
later stages never run and LexResult.schedule is None.

Base instances (step 2.5)
-------------------------
accept_instance draws candidates until one passes the coverage check (exposé 4.3);
initial_schedule computes its published schedule S0 = the S0_SPECS solve.

Oracle check
------------
check_oracles re-computes every recorded stage value with the oracle (via
Objective.evaluate) and raises OracleMismatch if the solver and the oracle disagree.
The runner calls it after every solve, so a MILP bug cannot slip into the results.

The solver is a parameter (milp.solve_stage, bruteforce.solve_stage or a test fake),
so this module never imports a specific solver.
"""

import time

from collections.abc import Mapping, Sequence
from typing import Protocol

import oracle

from repair.generator import generate_instance
from repair.model import (
    AcceptanceLog,
    Instance,
    LexResult,
    Schedule,
    StageResult,
    StageRecord,
    StageStatus,
)
from repair.objectives import MIN_SHORTFALL, S0_SPECS, Objective
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


class OracleMismatch(Exception):
    """A solver result disagrees with the oracle.

    Each message starts with a rule key, so tests can check why it was raised:
        "stages:"      the stages do not line up with the specs (count or names)
        "infeasible:"  the final schedule breaks a hard constraint
        "value:"       an optimal stage recorded a value the oracle does not compute
        "cap:"         an unproven (time_limit) stage got worse than its recorded incumbent
    """


def solve_lexicographic(
    instance: Instance,
    specs: Sequence[Objective],
    solve: StageSolver,
    *,
    time_limit: float,
    retry_factor: float = 2.0,
    start: Schedule | None = None,
) -> LexResult:
    """Solve the stages in specs one after another (see the module docstring).

    Arguments:
        specs:        objectives in priority order
        solve:        the stage solver to call
        time_limit:   seconds per stage (a retry gets time_limit * retry_factor)
        retry_factor: SolverSettings.retry_factor, 2.0 in the exposé
        start:        start schedule for the first stage, e.g. S_{t-1}

    Returns a LexResult with one StageRecord per stage that ran (seconds measured
    around the solver call) and the last stage's schedule; schedule is None if a
    stage failed twice.
    """
    fixed: dict[Objective, int] = {}  # earlier optimal stages: value must stay ==
    cap: dict[Objective, int] = (
        {}
    )  # earlier time_limit stages: value must not get worse
    records: list[StageRecord] = []

    for objective in specs:
        start_time = time.perf_counter()  # seconds cover the retry too
        # dict(...) copies, so a solver that keeps the mapping does not see later changes
        s = solve(
            instance,
            objective,
            fixed=dict(fixed),
            cap=dict(cap),
            start=start,
            time_limit=time_limit,
        )

        # No schedule in time: exactly one retry, with more time
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

        # Failed twice: stop here, later stages never run
        if s.schedule is None:
            return LexResult(schedule=None, stages=tuple(records))

        assert s.value is not None  # a schedule always comes with its value
        if s.status == StageStatus.OPTIMAL:
            fixed[objective] = s.value
        else:  # time_limit: the incumbent is not proven optimal, so later stages may improve it
            cap[objective] = s.value

        start = s.schedule  # hand-over: the next stage starts from this schedule

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

    # 1. zip() would silently drop extra stages or pair the wrong ones, so check first
    names = [spec.name for spec in specs]
    recorded = [stage.name for stage in result.stages]
    if names != recorded:
        raise OracleMismatch(f"stages: specs {names}, result has {recorded}")

    # 2. Hard constraints
    found = oracle.violations(schedule, instance)
    if found:
        raise OracleMismatch(f"infeasible: {found}")

    # 3. Each recorded value against the oracle's value
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


# --- base instances (exposé 4.3) -------------------------------------------------


class AcceptanceFailed(Exception):
    """The base seed and every reserve seed failed max_draws candidates each."""


def candidate_seed(seed: int, draw: int) -> int:
    """Seed of candidate `draw` (0-based) for `seed`: draw 0 is the seed itself, so
    most accepted instances have used_seed == base_seed; later draws are
    derive_seed(seed, "redraw", draw)."""
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
    """Draw candidates until one is accepted; return it and how it was found.

    Coverage check per candidate: solve_lexicographic(candidate, [MIN_SHORTFALL], solve,
    time_limit=settings.time_limit, retry_factor=settings.retry_factor), giving Lambda*_0.

    Main classes: accept iff Lambda*_0 == 0. The status does not matter here: Lambda >= 0,
        so an incumbent of 0 is optimal even after a timeout. No schedule -> rejected.
        Candidates: candidate_seed(base_seed, 0..max_draws-1); then the same for
        reserve[0], reserve[1], ... (replaced=True). All failed -> AcceptanceFailed.
    Under-supplied (under_supplied=True): accept the first candidate and record its
        Lambda*_0 (exposé 4.3: never rejected for coverage). No schedule -> ValueError.

    reserve: the reserve seeds of THIS base seed only (the caller derives them, e.g.
    derive_seed(root, "reserve", size_class, index, j)), so two base instances can never
    end up with the same reserve seed. Never mutated.

    AcceptanceLog: base_seed as given; used_seed = the accepted candidate's seed
    (= instance.meta.seed); draws and rejected counted over all seeds tried;
    lambda_star_0 of the accepted candidate.

    No oracle check here: initial_schedule re-solves MinShortfall as its first stage and
    check_oracles verifies that value, so a wrong Lambda*_0 = 0 cannot get through.
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

            # Lambda*_0 is the solver's recorded value. No oracle check here: S0 re-solves
            # MinShortfall as its first stage, and initial_schedule's check_oracles
            # verifies that value, so a wrong 0 is caught there (see the docstring).
            shortfall = result.stages[-1].value
            assert shortfall is not None  # a stage with a schedule records its value

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
    """S0: solve_lexicographic(instance, S0_SPECS, solve, time_limit=settings.s0_time_limit,
    retry_factor=settings.retry_factor), then check_oracles (raises OracleMismatch).

    The result may be uncertified (a stage hit its time limit): the caller
    (make_instances) must then replace the instance, because S0 has to be optimal.
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
