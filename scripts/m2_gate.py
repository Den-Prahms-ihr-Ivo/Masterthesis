"""Gate M2: one generated instance through every RQ1 and RQ2 arm with SCIP.

Steps: accept_instance -> initial_schedule (certified S0) -> one balanced RQ1 sequence
through the five main arms and one RQ2 sequence through both RQ2 arms -> write both
result files -> read them back and compare. run_arm checks every step with the oracle
(an OracleMismatch aborts the script); this script additionally checks that every arm
keeps Lambda = Lambda*_t at every step and that no arm ended early.

Prints one row per arm and exits with status 1 if any check fails.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/m2_gate.py                # n = 20, pilot.toml
    PYTHONPATH=src .venv/bin/python scripts/m2_gate.py --size 40 --seed 3

Files: results/m2-gate/<cell>-n<size>/<instance_seed>/<sequence_seed>.json.gz (the real
layout, in a directory of its own so it never mixes with pilot or main results).
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import oracle
from repair import milp, storage
from repair.model import SequenceProfile, SequenceResult, StageStatus
from repair.perturbations import instances_along, rq1_sequence, rq2_sequence
from repair.protocol import derive_seed, load_protocol, main_arms, rq2_arms
from repair.runner import accept_instance, initial_schedule, run_sequence

ROOT = Path(__file__).resolve().parents[1]


def check(result: SequenceResult) -> list[str]:
    """Problems in one result: arms that ended early, wrong length, Lambda != Lambda*_t."""
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
    """One row per arm: steps, certified steps, solver calls, retries, solver seconds."""
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
    root_seed = protocol.root_seed
    manifest = {
        "script": "m2_gate.py",
        "protocol": protocol.name,
        "versions": milp.solver_versions(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    print(f"protocol {protocol.name}, n = {args.size}, instance index {args.seed}, "
          f"SCIP {manifest['versions']}, tau {settings.time_limit} s")

    base = derive_seed(root_seed, "instance", args.size, args.seed)
    reserve = [
        derive_seed(root_seed, "reserve", args.size, args.seed, j)
        for j in range(protocol.n_reserve_seeds)
    ]
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

    sequence_seed = derive_seed(root_seed, "sequence", args.size, args.seed, 0)
    jobs = [
        ("main", rq1_sequence(instance, s0.schedule, SequenceProfile.BALANCED, sequence_seed),
         main_arms(protocol.lam, protocol.alpha)),
        ("rq2", rq2_sequence(instance, s0.schedule, sequence_seed), rq2_arms()),
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
