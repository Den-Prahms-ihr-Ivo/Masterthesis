"""Generate the protocol's base instances and their starting schedules S0.

Example after preparing the pilot settings, with src on the import path:
    python scripts/make_instances.py protocol/pilot.toml --out results/pilot --workers 8

Accept each candidate for coverage, then solve S0. If S0 is not certified,
try the next unused reserve seed. Save the instance, acceptance log and S0
together. An instance file that can already be read is skipped on restart.

Finally, write acceptance.csv for all instance jobs of this protocol, not
just those generated in this invocation.
"""

import argparse
import csv
import dataclasses
import functools
import sys
from multiprocessing import Pool
from pathlib import Path

from repair import milp, runner, storage
from repair import protocol as protocol_module
from repair.protocol import GeneratorConfig, InstanceJob, Protocol

# What a file that cannot be read back raises (missing, truncated gzip, bad JSON, wrong
# format or kind). Anything else is a bug and must not be hidden by a recompute.
_UNREADABLE = (OSError, EOFError, ValueError, KeyError, TypeError)


def _generator_config(protocol: Protocol, job: InstanceJob) -> GeneratorConfig:
    if job.under_supplied:
        if protocol.workforce_factor is None:
            raise ValueError(
                "under-supplied instance but protocol.workforce_factor is missing"
            )
        return dataclasses.replace(
            protocol.generator, workforce_factor=protocol.workforce_factor
        )
    return protocol.generator


def make_instance(protocol: Protocol, instance_job: InstanceJob, out: Path) -> Path:
    """Produce one accepted instance with a certified S0, then save it.

    The under-supplied class takes its workforce factor from the protocol.
    If S0 is not certified, continue from the next unused reserve seed and
    record that replacement. Exhausting the reserve seeds raises RuntimeError.
    The function is at module level so multiprocessing can pass it to workers.
    """
    path = storage.instance_path(
        out, instance_job.size_class, instance_job.index, instance_job.under_supplied
    )
    try:
        storage.read_instance(path)
        return path  # restart: a file that reads back is done
    except _UNREADABLE:
        pass  # missing or unreadable: (re)compute it

    config = _generator_config(protocol, instance_job)
    settings = protocol.solver
    first_base, reserve = protocol_module.instance_seeds(protocol, instance_job)
    base, replaced, draws, rejected = first_base, False, 0, 0

    while True:
        instance, log = runner.accept_instance(
            config,
            instance_job.size_class,
            base,
            reserve,
            settings,
            milp.solve_stage,
            max_draws=protocol.max_draws,
            under_supplied=instance_job.under_supplied,
        )
        draws += log.draws
        rejected += log.rejected
        s0 = runner.initial_schedule(instance, settings, milp.solve_stage)
        if s0.certified:
            break
        # accept_instance tries max_draws candidates per seed family (base, reserve[0],
        # reserve[1], ...) before moving on, so its draw count says which family the
        # accepted candidate came from. The next attempt starts at the family after it,
        # so no seed is ever tried twice.
        family = (
            log.draws - 1
        ) // protocol.max_draws  # 0 = base, f >= 1 = reserve[f-1]
        if family >= len(reserve):
            raise RuntimeError(
                f"no reserve seed left for n={instance_job.size_class} "
                f"index={instance_job.index}"
                f"{' (under-supplied)' if instance_job.under_supplied else ''}: "
                "S0 was never certified"
            )
        base, reserve, replaced = reserve[family], reserve[family + 1 :], True

    log = dataclasses.replace(
        log,
        base_seed=first_base,
        draws=draws,
        rejected=rejected,
        replaced=log.replaced or replaced,
    )
    storage.write_instance(path, instance, log, s0)
    return path


ACCEPTANCE_COLUMNS = (
    "size_class",
    "under_supplied",
    "index",
    "base_seed",
    "used_seed",
    "draws",
    "rejected",
    "replaced",
    "lambda_star_0",
    "s0_certified",
)


def write_acceptance_csv(protocol: Protocol, out: Path) -> Path:
    """Write one acceptance row per protocol instance, in instance-job order."""
    rows = []
    for job in protocol_module.instance_jobs(protocol):
        path = storage.instance_path(out, job.size_class, job.index, job.under_supplied)
        _, log, s0 = storage.read_instance(
            path
        )  # a missing file raises: no partial table
        rows.append(
            {
                "size_class": job.size_class,
                "under_supplied": job.under_supplied,
                "index": job.index,
                "base_seed": log.base_seed,
                "used_seed": log.used_seed,
                "draws": log.draws,
                "rejected": log.rejected,
                "replaced": log.replaced,
                "lambda_star_0": log.lambda_star_0,
                "s0_certified": s0.certified,
            }
        )
    target = out / "acceptance.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ACCEPTANCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return target


def _make_one(protocol: Protocol, out: Path, job: InstanceJob) -> Path:
    """Arrange arguments for partial() while keeping the worker function picklable."""
    return make_instance(protocol, job, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    protocol = protocol_module.load_protocol(args.protocol)
    jobs = list(protocol_module.instance_jobs(protocol))
    work = functools.partial(_make_one, protocol, args.out)

    if args.workers <= 1:
        for done, path in enumerate(map(work, jobs), start=1):
            print(f"[{done}/{len(jobs)}] {path}")
    else:
        with Pool(args.workers) as pool:
            for done, path in enumerate(pool.imap_unordered(work, jobs), start=1):
                print(f"[{done}/{len(jobs)}] {path}")

    print(write_acceptance_csv(protocol, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
