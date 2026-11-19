"""Estimate sample sizes and runtime from pilot results.

The calculation uses the spread of the paired contrasts (H1, H2 and H3; DESIGN.md
13.8) to choose N_I per size class, then combines stage times and planned solve counts
to estimate core hours against the 1,000-hour budget. It never selects the sample size
from observed effect directions.

    PYTHONPATH=src .venv/bin/python scripts/plan_precision.py results/pilot \
        --protocol protocol/pilot.toml

Exit status 0 if the projection fits BUDGET_CORE_HOURS, 1 if not.
"""

import dataclasses
from collections import Counter, defaultdict
from pathlib import Path
import argparse

from repair import analysis, protocol as protocol_module, storage

BUDGET_CORE_HOURS = 1000.0
StageKey = tuple[int, str]  # (size class, stage name, e.g. "MaxQuality")
JobType = tuple[str, int, int]  # (cell, size class, number of arms)


def projected_core_hours(
    stage_seconds: dict[StageKey, float], matrix: dict[StageKey, float]
) -> float:
    """Multiply mean stage times by planned solve counts and convert to hours.

    Each solve uses one thread, so solver seconds count as core seconds here.
    Missing timing data raises an error rather than underestimating the budget.
    Instance generation and other overhead are outside this calculation.
    """
    missing = sorted(
        key for key, count in matrix.items() if count and key not in stage_seconds
    )
    if missing:
        raise ValueError(f"no pilot timing for {missing}")
    return sum(stage_seconds[key] * count for key, count in matrix.items() if count) / 3600.0


def _stages(result):
    """Yield the shared coverage records, then every arm's stage records."""
    for step in result.coverage:  # a CoverageStep holds exactly one record
        yield step.record
    for trajectory in result.trajectories:
        for step in trajectory.steps:
            yield from step.stages


def _job_type(result) -> JobType:
    return (result.cell, result.instance.meta.size_class, len(result.trajectories))


def pilot_timing(
    results,
) -> tuple[dict[StageKey, float], dict[JobType, dict[str, float]]]:
    """Summarise measured stage times and stage counts per job type.

    Group times by size and stage name. Group solve counts by cell, size and
    number of arms, then divide by the number of jobs in that group.
    """
    seconds: dict[StageKey, list[float]] = defaultdict(list)
    solves: dict[JobType, Counter] = defaultdict(Counter)
    n_jobs: Counter = Counter()
    for result in results:
        job_type = _job_type(result)
        n_jobs[job_type] += 1
        for stage in _stages(result):
            seconds[(job_type[1], stage.name)].append(stage.seconds)
            solves[job_type][stage.name] += 1
    mean_seconds = {key: sum(values) / len(values) for key, values in seconds.items()}
    mean_solves = {
        job_type: {stage: count / n_jobs[job_type] for stage, count in counter.items()}
        for job_type, counter in solves.items()
    }
    return mean_seconds, mean_solves


def planned_matrix(planned_jobs, mean_solves) -> dict[StageKey, float]:
    """Estimate stage counts for the planned job list using the pilot counts.

    If a cell was not measured, try a main-cell job with the same size and
    number of arms. Raise an error if no matching pilot group exists.
    """
    matrix: dict[StageKey, float] = defaultdict(float)
    for job in planned_jobs:
        size = job.instance.size_class
        job_type = (job.cell, size, len(job.arms))
        if job_type not in mean_solves:  # cell not in the pilot: same arms,
            job_type = ("main", size, len(job.arms))  # so use the main cell's counts
        if job_type not in mean_solves:
            raise ValueError(f"the pilot has no job like {job_type}")
        for stage, count in mean_solves[job_type].items():
            matrix[(size, stage)] += count
    return dict(matrix)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pilot_results", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path("protocol/pilot.toml"))
    args = parser.parse_args(argv)

    results = list(storage.read_results(args.pilot_results))
    pilot = protocol_module.load_protocol(args.protocol)

    # Every size class of the protocol needs pilot results: without them neither N_I
    # nor the certification share nor the timing exists (e.g. every n = 40 job failed).
    measured = {result.instance.meta.size_class for result in results}
    missing = [size for size in pilot.size_classes if size not in measured]
    if missing:
        raise ValueError(
            "no pilot results for " + ", ".join(f"n={size}" for size in missing)
        )

    # 1. Precision: required N_I per size class and contrast (H1, H2, H3), from
    # dispersion only. A DataFrame: size_class, contrast, sd_diff, pooled_sd, n_required
    plan = analysis.precision_plan(
        analysis.instance_table(analysis.sequence_table(results)),
        rq2=analysis.rq2_table(results),
    )
    n_instances: dict[int, int] = {}
    for size in pilot.size_classes:
        rows = plan[plan["size_class"] == size]
        contrasts = rows[rows["contrast"] != "max"]
        for row in contrasts.itertuples():
            needed = "undefined" if row.n_required != row.n_required else int(row.n_required)
            print(f"n={size}  {row.contrast:<28} N_I >= {needed}")
        maximum = rows.loc[rows["contrast"] == "max", "n_required"]
        if len(maximum) != 1 or maximum.isna().any():
            # precision_plan leaves "max" undefined when any contrast is (fewer than two
            # instances, or a contrast with zero spread): never plan with NaN
            undefined = contrasts.loc[contrasts["n_required"].isna(), "contrast"]
            raise ValueError(
                f"n={size}: N_I is undefined; no usable SD for "
                f"{', '.join(undefined) or 'any contrast (no rows)'}"
            )
        n_instances[size] = int(maximum.item())
        print(f"n={size}  N_I = {n_instances[size]}")

    # 2. Adequacy input: share of sequences certified through T, per size class.
    for size in pilot.size_classes:
        prefixes = [
            analysis.certified_prefix(result)
            for result in results
            if result.instance.meta.size_class == size
        ]
        share = sum(prefix >= pilot.horizon for prefix in prefixes) / len(prefixes)
        print(
            f"n={size}  certified through T={pilot.horizon}: {share:.0%} of {len(prefixes)}"
        )

    # 3. Budget: the main matrix at the planned N_I, costed with the pilot's timings.
    mean_seconds, mean_solves = pilot_timing(results)
    for (size, stage), value in sorted(mean_seconds.items()):
        print(f"n={size}  {stage:<26} {value:8.1f} s per solve")
    planned = dataclasses.replace(
        pilot,
        cells=protocol_module.CELLS,
        n_instances=n_instances,
        # placeholders, only so that jobs() can count the matrix; never run settings
        robustness_scale=pilot.robustness_scale or 1,
        workforce_factor=pilot.workforce_factor or 1.0,
    )
    hours = projected_core_hours(
        mean_seconds, planned_matrix(protocol_module.jobs(planned), mean_solves)
    )
    print(f"projected {hours:.0f} core-hours of {BUDGET_CORE_HOURS:.0f}")
    return 0 if hours <= BUDGET_CORE_HOURS else 1


if __name__ == "__main__":
    raise SystemExit(main())
