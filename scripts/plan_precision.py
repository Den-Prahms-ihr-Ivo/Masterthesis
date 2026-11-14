"""Work in progress: estimate sample sizes and runtime from pilot results.

The intended calculation uses the spread of paired contrasts to choose
N_I, then combines stage times and planned solve counts to estimate core
hours against the 1,000-hour budget. It should not select the sample size
from observed effect directions.

This script still needs to be connected to the current analysis output.
In particular, precision_plan() returns a DataFrame, but main() below still
treats it as iterable row objects. The H3 precision calculation is also
missing. Do not use this command as a completed pilot decision step yet.
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
    return sum(stage_seconds[key] * count for key, count in matrix.items()) / 3600.0


def _stages(result):
    """Yield the shared coverage records, then every arm's stage records."""
    for step in result.coverage:  # Coverage entries store their own stage records.
        yield from step.stages
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

    # 1. Precision: required N_I per size class and contrast, from dispersion only.
    instance_table = analysis.instance_table(
        analysis.sequence_table(results)
    )  # sequence_table and instance_table currently return DataFrames.
    plan = analysis.precision_plan(
        instance_table
    )  # TODO: main() below must iterate DataFrame rows, not column names.
    n_instances = {}
    for size in pilot.size_classes:
        rows = [row for row in plan if row.size_class == size]
        for row in rows:
            print(f"n={size}  {row.contrast:<28} N_I >= {row.n_required}")
        n_instances[size] = max(row.n_required for row in rows)
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
        robustness_scale=pilot.robustness_scale
        or 1,  # Temporary values for counting jobs only;
        workforce_factor=pilot.workforce_factor or 1.0,  # these are not calibrated run settings.
    )
    hours = projected_core_hours(
        mean_seconds, planned_matrix(protocol_module.jobs(planned), mean_solves)
    )
    print(f"projected {hours:.0f} core-hours of {BUDGET_CORE_HOURS:.0f}")
    return 0 if hours <= BUDGET_CORE_HOURS else 1


if __name__ == "__main__":
    raise SystemExit(main())
