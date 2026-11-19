"""Run a small integration check with real SCIP solves (Gate M2).

Accept one generated instance and obtain a certified S0. Then run one
balanced RQ1 sequence through all five policies and one RQ2 sequence through
both SF references. Write both results, load them again and compare them.

The runner checks the schedules with the oracle. This script also checks
that no arm ended early and each repair reaches the shared coverage value.
It prints one row per arm and returns status 1 when a check fails.

Run from the project root with src on the import path:
    python scripts/m2_gate.py
    python scripts/m2_gate.py --size 40 --seed 3

The default size is 20 and the default protocol is pilot.toml. Output goes
to results/m2-gate, separately from pilot and main results.
"""

import argparse
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import oracle
from repair import milp, storage
from repair.model import SequenceKind, SequenceProfile, SequenceResult, StageStatus
from repair.perturbations import instances_along, rq1_sequence, rq2_sequence
from repair.protocol import (
    InstanceJob,
    Job,
    gate_root,
    instance_seeds,
    load_protocol,
    main_arms,
    rq2_arms,
    sequence_seed,
)
from repair.runner import accept_instance, initial_schedule, run_sequence

ROOT = Path(__file__).resolve().parents[1]


def check(result: SequenceResult) -> list[str]:
    """List early endings, wrong trajectory lengths and coverage mismatches."""
    problems = []
    instances_t = instances_along(result.instance, result.sequence.events)
    horizon = len(result.sequence.events)
    for trajectory in result.trajectories:
        name = trajectory.arm.name
        if trajectory.ended_early or len(trajectory.steps) != horizon:
            problems.append(f"{name}: {len(trajectory.steps)}/{horizon} steps")
        for step, cover, instance_t in zip(trajectory.steps, result.coverage, instances_t):
            shortfall = oracle.shortfall(step.schedule, instance_t)
            if shortfall != cover.lambda_star:
                problems.append(
                    f"{name} t={step.t}: Lambda {shortfall} != Lambda* {cover.lambda_star}"
                )
    return problems


def print_rows(result: SequenceResult, seconds: float) -> None:
    """Print each arm's step counts, certification, solver calls, retries and time."""
    for trajectory in result.trajectories:
        stages = [r for s in trajectory.steps for r in s.stages]
        certified = sum(
            all(r.status == StageStatus.OPTIMAL for r in s.stages) for s in trajectory.steps
        )
        print(" | ".join(str(x) for x in (
            result.cell, trajectory.arm.name, len(trajectory.steps), certified,
            len(stages), sum(r.retried for r in stages),
            f"{sum(r.seconds for r in stages):.1f}",
        )))
    lambdas = [c.lambda_star for c in result.coverage]
    print(f"  coverage Lambda*_t = {lambdas}; sequence {seconds:.1f} s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol" / "pilot.toml")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="instance index")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "m2-gate")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    settings = protocol.solver
    manifest = {
        "script": "m2_gate.py",
        "protocol": protocol.name,
        "versions": milp.solver_versions(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    print(f"protocol {protocol.name}, n = {args.size}, instance index {args.seed}, "
          f"SCIP {manifest['versions']}, tau {settings.time_limit} s")

    # Use the normal instance and sequence scheme under a separate gate root.
    # test_protocol checks that the selected gate seeds do not overlap the run.
    gate = replace(protocol, root_seed=gate_root(protocol))
    instance_job = InstanceJob(args.size, args.seed)
    base, reserve = instance_seeds(gate, instance_job)
    instance, log = accept_instance(
        protocol.generator, args.size, base, reserve, settings, milp.solve_stage,
        max_draws=protocol.max_draws,
    )
    s0 = initial_schedule(instance, settings, milp.solve_stage)
    print(f"accepted seed {log.used_seed} after {log.draws} draw(s); "
          f"S0 certified: {s0.certified}")
    if not s0.certified or s0.schedule is None:
        print("FAIL: S0 not certified", file=sys.stderr)
        return 1

    k = instance.meta.scale
    main_job = Job("main", instance_job, SequenceKind.RQ1, SequenceProfile.BALANCED, 0,
                   main_arms(protocol.lam, protocol.alpha), k)
    rq2_job = Job("rq2", instance_job, SequenceKind.RQ2, SequenceProfile.RQ2, 0, rq2_arms(), k)
    jobs = [
        ("main", rq1_sequence(instance, s0.schedule, SequenceProfile.BALANCED,
                              sequence_seed(gate, main_job)), main_job.arms),
        ("rq2", rq2_sequence(instance, s0.schedule, sequence_seed(gate, rq2_job)), rq2_job.arms),
    ]

    print()
    print("cell | arm | steps | certified | stages | retries | solver s")
    print("--- | --- | --- | --- | --- | --- | ---")
    problems: list[str] = []
    for cell, sequence, arms in jobs:
        start = time.perf_counter()
        result = run_sequence(
            cell, instance, log, s0, sequence, arms, settings, milp.solve_stage, manifest
        )
        print_rows(result, time.perf_counter() - start)
        problems += check(result)

        path = storage.result_path(args.out, cell, args.size, instance.meta.seed, sequence.seed)
        storage.write_result(path, result)
        if not storage.exists_complete(path) or storage.read_result(path) != result:
            problems.append(f"{path}: does not round-trip")
        else:
            print(f"  round-trip OK: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")

    print()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print("Gate M2 passed: every oracle check, Lambda = Lambda*_t for every arm and step, "
          "both files round-trip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
