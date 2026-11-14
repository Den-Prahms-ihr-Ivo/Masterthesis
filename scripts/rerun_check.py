"""Repeat completed jobs in a fresh directory and compare their results.

Example with src on the import path:
    python scripts/rerun_check.py protocol/dry-run.toml results/dry-run --jobs 2

This is the reproducibility check for Gate M3. Select finished jobs spread
over the run order, copy their base-instance files into a new neighbouring
directory, and run them again. Keep that directory for inspection.

Compare inputs, coverage values, schedules, and stage names, statuses and
values. Timings, dual bounds, retry flags and manifests are left out. A
time-limited stage can give a different outcome on a repeat run, so affected
steps and later steps of that arm are reported as not compared.

Exit with 0 when all compared values match, 1 for differences, or 2 for a
usage error. A successful exit does not mean skipped steps were verified.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from repair import milp, storage
from repair import protocol as protocol_module
from repair.model import SequenceResult, StageRecord, StageStatus

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # run as a script, only scripts/ itself is on the path
    sys.path.insert(0, str(ROOT))
from scripts import run_experiment  # noqa: E402


class Comparison(NamedTuple):
    problems: list[str]  # proven values that differ: the check fails
    notes: list[str]  # what was not compared because a time limit made it run-dependent


def _values(stages: Iterable[StageRecord]) -> tuple[tuple[str, str, int | None], ...]:
    """Keep the stage name, status and value; leave timing-dependent fields out."""
    return tuple((r.name, r.status.value, r.value) for r in stages)


def _proven(stages: Iterable[StageRecord]) -> bool:
    return all(r.status == StageStatus.OPTIMAL for r in stages)


def compare_results(a: SequenceResult, b: SequenceResult) -> Comparison:
    """Compare results for the same job, returning differences and skipped checks.

    Report at most the first difference per arm. Later schedules may already
    differ because of that earlier repair, especially with chained references.
    """
    problems: list[str] = []
    notes: list[str] = []

    for field in ("cell", "instance", "acceptance", "sequence"):
        if getattr(a, field) != getattr(b, field):
            problems.append(f"input differs: {field}")
    if problems:
        return Comparison(problems, notes)  # different inputs: nothing else is comparable

    if a.s0 != b.s0 or _values(a.s0_stages) != _values(b.s0_stages):
        problems.append("S0 differs")

    unproven_t: set[int] = set()
    if len(a.coverage) != len(b.coverage):
        problems.append(f"coverage: {len(a.coverage)} vs {len(b.coverage)} steps")
    for ca, cb in zip(a.coverage, b.coverage):
        if not (_proven([ca.record]) and _proven([cb.record])):
            unproven_t.add(ca.t)
            notes.append(f"coverage t={ca.t}: not compared (time limit)")
        elif (ca.t, ca.lambda_star, _values([ca.record])) != (
            cb.t, cb.lambda_star, _values([cb.record])
        ):
            problems.append(
                f"coverage t={ca.t}: Lambda* {ca.lambda_star} vs {cb.lambda_star}"
            )

    arms_a = [t.arm for t in a.trajectories]
    arms_b = [t.arm for t in b.trajectories]
    if arms_a != arms_b:
        problems.append(f"arms differ: {[x.name for x in arms_a]} vs {[x.name for x in arms_b]}")
        return Comparison(problems, notes)

    for ta, tb in zip(a.trajectories, b.trajectories):
        name = ta.arm.name
        for sa, sb in zip(ta.steps, tb.steps):
            if sa.t in unproven_t or not (_proven(sa.stages) and _proven(sb.stages)):
                notes.append(f"{name}: steps from t={sa.t} on not compared (time limit)")
                break
            if sa.t != sb.t or sa.schedule != sb.schedule:
                problems.append(f"{name} t={sa.t}: schedules differ")
                break
            if _values(sa.stages) != _values(sb.stages):
                problems.append(
                    f"{name} t={sa.t}: stage values differ: "
                    f"{_values(sa.stages)} vs {_values(sb.stages)}"
                )
                break
        else:
            if len(ta.steps) != len(tb.steps) or ta.ended_early != tb.ended_early:
                problems.append(
                    f"{name}: {len(ta.steps)} vs {len(tb.steps)} steps "
                    f"(ended early: {ta.ended_early} vs {tb.ended_early})"
                )

    return Comparison(problems, notes)


def _spread(items: list, n: int) -> list:
    """Select n entries spread over the list, including both ends when n >= 2."""
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[0]]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("protocol", type=Path)
    parser.add_argument("results", type=Path, help="the run's --out directory")
    parser.add_argument("--jobs", type=int, default=2, help="how many jobs to rerun")
    parser.add_argument("--cell", action="append", help="only from this cell; repeatable")
    args = parser.parse_args(argv)

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if os.environ.get(run_experiment.PLANT_ENV, "").strip():
        parser.error(f"unset {run_experiment.PLANT_ENV} first: a rerun must not plant failures")

    protocol = protocol_module.load_protocol(args.protocol)
    finished = []
    for job in run_experiment.select_jobs(protocol, args.cell, "0/1"):
        try:
            path = run_experiment.job_result_path(protocol, job, args.results)
        except FileNotFoundError:
            continue  # no instance file: this job cannot have a result
        if storage.exists_complete(path):
            finished.append((job, path))
    if len(finished) < args.jobs:
        print(f"only {len(finished)} complete result(s) in {args.results}", file=sys.stderr)
        return 1

    fresh = Path(tempfile.mkdtemp(prefix=f"{args.results.name}-rerun-", dir=args.results.parent))
    manifest = storage.build_manifest(args.protocol, 1, milp.solver_versions())
    print(f"rerunning {args.jobs} of {len(finished)} finished jobs into {fresh}")

    failed = False
    for job, original in _spread(finished, args.jobs):
        source = run_experiment._instance_file(job, args.results)
        target = run_experiment._instance_file(job, fresh)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

        start = time.perf_counter()
        rerun = run_experiment.run_job(job, args.protocol, fresh, manifest)
        seconds = time.perf_counter() - start
        comparison = compare_results(storage.read_result(original), storage.read_result(rerun))

        key = protocol_module.job_key(job)
        verdict = "DIFFERS" if comparison.problems else "identical"
        print(f"{key}: {verdict} ({seconds:.0f} s)")
        for problem in comparison.problems:
            print(f"  FAIL {problem}")
        for note in comparison.notes:
            print(f"  note {note}")
        failed |= bool(comparison.problems)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
