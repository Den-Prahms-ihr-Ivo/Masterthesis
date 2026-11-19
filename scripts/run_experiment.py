"""Run the selected experiment jobs, with parallel workers and restart support.

For a prepared pilot protocol, with src on the import path:
    python scripts/run_experiment.py protocol/pilot.toml --out results/pilot --workers 8

--count prints the selected job count; --list prints the job keys.
--cell selects experiment cells and --shard i/N divides the selected list.
The main run additionally needs --main and a filled, dated protocol.

Each job loads its base instance, samples one shared event sequence, runs
the repair arms and writes a result. Complete result files are skipped,
so repeating the command continues an interrupted run.

Ordinary job exceptions are saved as failure files. SystematicError and
AssertionError also request ABORT. The same error type and message in five
failure files triggers ABORT too. Workers check the flag before starting
another job; jobs already running finish and save their output.

KeyboardInterrupt and SystemExit are allowed to stop the process. Atomic
writes protect finished results. Before a resume, compare the run manifest
so changed code, protocols or solver settings are not mixed silently.

For the Gate M3 rehearsal only, REPAIR_PLANT_FAILURE can contain
"<job key>:error,<job key>:assert". These raise an ordinary error or an
assertion before that job starts. Invalid entries, unselected keys and use
with --main are refused. The manifest records any planted failures.
"""

import argparse
import functools
import os
import re
import sys
import traceback
from collections.abc import Callable, Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from repair import milp, perturbations, runner, storage
from repair import protocol as protocol_module
from repair.model import SequenceKind, SequenceProfile, SystematicError
from repair.protocol import Job, Protocol

RunJob = Callable[[Job, Path, Path, dict[str, Any]], Path]

# Stop after this many failure files have the same error type and message.
BREAKER_THRESHOLD = 5

# Exception types available for the rehearsal's planted failures.
PLANT_ENV = "REPAIR_PLANT_FAILURE"
PLANTED: dict[str, type[Exception]] = {"error": RuntimeError, "assert": AssertionError}


def planted_failures(spec: str | None) -> dict[str, str]:
    """Parse a REPAIR_PLANT_FAILURE value: "key:kind[,key:kind...]" -> {key: kind}.

    None, "" or only whitespace -> {} (nothing planted). kind is "error" or "assert".
    ValueError for anything else (a typo must stop the rehearsal, not silently plant
    nothing): an entry without ":", an empty key, an unknown kind, a key named twice.
    """
    planted: dict[str, str] = {}
    if spec is None or not spec.strip():
        return planted
    for entry in spec.split(","):
        key, sep, kind = entry.strip().rpartition(":")
        if not sep or not key or kind not in PLANTED:
            raise ValueError(
                f"{PLANT_ENV}: {entry.strip()!r} is not <job key>:{'|'.join(PLANTED)}"
            )
        if key in planted:
            raise ValueError(f"{PLANT_ENV}: job key {key!r} named twice")
        planted[key] = kind
    return planted


def _raise_if_planted(job: Job) -> None:
    """Raise the planted exception if REPAIR_PLANT_FAILURE names this job. The message
    contains the job key, so planted errors never trip the breaker by themselves.
    """
    planted = planted_failures(os.environ.get(PLANT_ENV))
    if not planted:
        return
    key = protocol_module.job_key(job)
    if key in planted:
        kind = planted[key]
        raise PLANTED[kind](f"planted {kind} in {key} ({PLANT_ENV}, rehearsal only)")


@functools.lru_cache(maxsize=None)
def _protocol(protocol_path: Path) -> Protocol:
    """Load the protocol in a worker; the main process has already validated it."""
    return protocol_module.load_protocol(protocol_path)


def _instance_file(job: Job, out: Path) -> Path:
    return storage.instance_path(
        out, job.instance.size_class, job.instance.index, job.instance.under_supplied
    )


@functools.lru_cache(maxsize=None)
def _accepted_seed(instance_file: Path) -> int:
    """Cache the accepted seed because several sequence jobs share an instance."""
    _, acceptance, _ = storage.read_instance(instance_file)
    return acceptance.used_seed


def job_result_path(protocol: Protocol, job: Job, out: Path) -> Path:
    """Find the expected result path for a job.

    Use the accepted candidate seed from the saved instance, which can differ
    from the original base seed after rejection. The instance file must exist.
    """
    return storage.result_path(
        out,
        job.cell,
        job.instance.size_class,
        _accepted_seed(_instance_file(job, out)),
        protocol_module.sequence_seed(protocol, job),
    )


def run_job(job: Job, protocol_path: Path, out: Path, manifest: dict[str, Any]) -> Path:
    """Run one job, write its result, return the result path. Top level (and only plain
    arguments), so multiprocessing can pickle it.

    Sequence by job.kind: "rq1" -> rq1_sequence(targeting "s0"), "independent" ->
    rq1_sequence(targeting "independent"), "rq2" -> rq2_sequence; seed = protocol.
    sequence_seed, scale = job.scale, horizon = protocol.horizon. The instance file's
    S0 must be certified (ValueError otherwise). A planted failure (REPAIR_PLANT_FAILURE)
    is raised first, before anything is read.
    """
    _raise_if_planted(job)
    protocol = _protocol(protocol_path)
    instance, acceptance, s0 = storage.read_instance(_instance_file(job, out))
    if not s0.certified or s0.schedule is None:
        raise ValueError(f"S0 is not certified for {protocol_module.job_key(job)}")

    seed = protocol_module.sequence_seed(protocol, job)
    if job.kind == SequenceKind.RQ2:
        sequence = perturbations.rq2_sequence(
            instance, s0.schedule, seed, horizon=protocol.horizon, scale=job.scale
        )
    elif job.kind in (SequenceKind.RQ1, SequenceKind.INDEPENDENT):
        sequence = perturbations.rq1_sequence(
            instance,
            s0.schedule,
            SequenceProfile(job.profile),
            seed,
            horizon=protocol.horizon,
            targeting="s0" if job.kind == SequenceKind.RQ1 else "independent",
            scale=job.scale,
        )
    else:
        raise ValueError(f"unknown job kind {job.kind!r}")

    result = runner.run_sequence(
        job.cell,
        instance,
        acceptance,
        s0,
        sequence,
        job.arms,
        protocol.solver,
        milp.solve_stage,
        manifest,
    )
    path = job_result_path(protocol, job, out)
    storage.write_result(path, result)
    return path


def run_job_safely(
    job: Job,
    protocol_path: Path,
    out: Path,
    manifest: dict[str, Any],
    run: RunJob = run_job,
) -> Path | None:
    """run (default run_job) under the failure policy of the module docstring.

    Returns the result path, or None if the job was skipped (ABORT present: `run` is not
    called) or failed. The failure file is keyed by protocol.job_key(job); error_type is
    the exception's class name, message str(exception), traceback the full traceback
    text. After writing an ordinary failure, count the failure files with the same
    error_type and message (storage.read_failures); at BREAKER_THRESHOLD or more, request
    the abort too. The ABORT text names the reason (class, message, job key).
    """
    if storage.abort_requested(out):
        return None
    key = protocol_module.job_key(job)
    try:
        path = run(job, protocol_path, out, manifest)
    except Exception as error:  # Let KeyboardInterrupt and SystemExit stop the process.
        error_type, message = type(error).__name__, str(error)
        storage.write_failure(out, key, error_type, message, traceback.format_exc())
        reason = f"{error_type}: {message} (job {key})"
        if isinstance(error, (SystematicError, AssertionError)):
            storage.request_abort(out, reason)
        else:
            same = sum(
                1
                for failure in storage.read_failures(out)
                if failure["error_type"] == error_type and failure["message"] == message
            )
            if same >= BREAKER_THRESHOLD:
                storage.request_abort(out, f"{same} jobs failed with {reason}")
        return None
    storage.failure_path(out, key).unlink(missing_ok=True)  # A successful rerun removes its old failure file.
    return path


def select_jobs(
    protocol: Protocol, cells: Sequence[str] | None, shard_spec: str
) -> list[Job]:
    """protocol.jobs, filtered by cells (None = all), then protocol.shard by "i/N"
    (ValueError for a malformed spec).
    """
    match = re.fullmatch(r"([0-9]+)/([0-9]+)", shard_spec)
    if match is None:
        raise ValueError(f"shard must look like i/N, got {shard_spec!r}")
    index, count = int(match[1]), int(match[2])
    if count < 1 or index >= count:
        raise ValueError(f"shard needs 0 <= i < N, got {shard_spec!r}")
    jobs = protocol_module.jobs(protocol)
    if cells is not None:
        unknown = set(cells) - {job.cell for job in jobs}
        if unknown:
            raise ValueError(f"unknown cells: {sorted(unknown)}")
        jobs = [job for job in jobs if job.cell in cells]
    return list(protocol_module.shard(jobs, index, count))


def _run_one(protocol_path: Path, out: Path, manifest: dict[str, Any], job: Job) -> Path | None:
    """Arrange worker arguments for partial(); keep the function at module level."""
    return run_job_safely(job, protocol_path, out, manifest)


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments below and run. --main loads the protocol with
    require_frozen=True. --count prints len(selected jobs) and returns 0 without solving;
    --list prints their job keys, one per line, in run order. REPAIR_PLANT_FAILURE is
    checked as the module docstring says (refusal: argparse error, exit status 2).
    Otherwise: build_manifest once, storage.check_run_manifest (ManifestMismatch: print
    it, request the abort, return 1), skip complete results, run the rest with
    multiprocessing.Pool(workers) (workers = 1: in-process, easier to debug), print a
    summary (done / skipped / failed). Returns 1 if ABORT is present at the end.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cell", action="append", help="restrict to a cell; repeatable"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", default="0/1", help="i/N: run jobs[i::N]")
    parser.add_argument(
        "--main", action="store_true", help="require a frozen, dated protocol"
    )
    parser.add_argument(
        "--count", action="store_true", help="print the number of jobs and exit"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the job keys in run order and exit"
    )

    args = parser.parse_args(argv)

    plant_spec = os.environ.get(PLANT_ENV, "")
    if args.main and plant_spec.strip():
        parser.error(f"{PLANT_ENV} is set: planted failures are for the rehearsal only")

    protocol = protocol_module.load_protocol(args.protocol, require_frozen=args.main)
    jobs = select_jobs(protocol, args.cell, args.shard)
    if args.count:
        print(len(jobs))
        return 0
    if args.list:
        for job in jobs:
            print(protocol_module.job_key(job))
        return 0

    try:
        planted = planted_failures(plant_spec)
    except ValueError as error:
        parser.error(str(error))
    unknown = sorted(set(planted) - {protocol_module.job_key(job) for job in jobs})
    if unknown:
        parser.error(f"{PLANT_ENV} names jobs that are not selected: {', '.join(unknown)}")

    manifest = storage.build_manifest(
        args.protocol, args.workers, milp.solver_versions()
    )
    if planted:
        print(f"WARNING: planted failures active ({PLANT_ENV}={plant_spec})", file=sys.stderr)
        manifest["planted_failures"] = plant_spec
    try:
        storage.check_run_manifest(args.out, manifest)
    except storage.ManifestMismatch as error:
        print(f"refusing to resume: {error}", file=sys.stderr)
        storage.request_abort(args.out, f"ManifestMismatch: {error}")
        return 1

    pending = [
        job
        for job in jobs
        if not storage.exists_complete(job_result_path(protocol, job, args.out))
    ]
    skipped = len(jobs) - len(pending)
    work = functools.partial(_run_one, args.protocol, args.out, manifest)

    if args.workers <= 1:
        paths = [work(job) for job in pending]
    else:
        with Pool(args.workers, maxtasksperchild=1) as pool:
            paths = list(pool.imap_unordered(work, pending, chunksize=1))

    done = sum(path is not None for path in paths)
    failed_keys = {failure["job_key"] for failure in storage.read_failures(args.out)}
    failed = sum(protocol_module.job_key(job) in failed_keys for job in pending)
    not_started = len(pending) - done - failed
    print(f"done {done}, skipped {skipped}, failed {failed}, not started {not_started}")
    return 1 if storage.abort_requested(args.out) else 0


if __name__ == "__main__":
    sys.exit(main())
