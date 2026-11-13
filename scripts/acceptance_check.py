"""Gate 2.5: accept base instances with SCIP and report the acceptance rates.

For every size class: N base seeds -> accept_instance -> initial_schedule. Prints one
row per size class (draws, rejections, reserve use, certified S0, time) and exits with
status 1 if any S0 is not certified.

Usage:
    .venv/bin/python scripts/acceptance_check.py                 # pilot.toml, 20 seeds
    .venv/bin/python scripts/acceptance_check.py --seeds 5 --sizes 20 40

Seeds: derive_seed(root_seed, "instance", size_class, i), reserve seeds
derive_seed(root_seed, "reserve", size_class, i, j): the same scheme make_instances.py
will use, so the pilot sees exactly these instances.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

from repair import milp
from repair.protocol import derive_seed, load_protocol
from repair.runner import accept_instance, initial_schedule

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol" / "pilot.toml")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    sizes = args.sizes or list(protocol.size_classes)
    settings = protocol.solver

    print(f"protocol {protocol.name}, root_seed {protocol.root_seed}, {args.seeds} seeds")
    print(f"SCIP {milp.solver_versions()}, tau {settings.time_limit} s, "
          f"S0 {settings.s0_time_limit} s, max_draws {protocol.max_draws}")
    print()
    header = (
        "size", "accepted", "draws", "mean draws", "max draws", "rejected",
        "reject rate", "replaced", "S0 certified", "accept s", "S0 s",
    )
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))

    all_certified = True
    for size in sizes:
        logs, certified, accept_s, s0_s = [], 0, 0.0, 0.0
        for i in range(args.seeds):
            base = derive_seed(protocol.root_seed, "instance", size, i)
            reserve = [
                derive_seed(protocol.root_seed, "reserve", size, i, j)
                for j in range(protocol.n_reserve_seeds)
            ]
            t0 = time.perf_counter()
            instance, log = accept_instance(
                protocol.generator, size, base, reserve, settings, milp.solve_stage,
                max_draws=protocol.max_draws,
            )
            t1 = time.perf_counter()
            s0 = initial_schedule(instance, settings, milp.solve_stage)
            t2 = time.perf_counter()

            logs.append(log)
            certified += s0.certified
            accept_s += t1 - t0
            s0_s += t2 - t1
            if not s0.certified:
                statuses = [r.status.value for r in s0.stages]
                print(f"  ! n={size} seed {i}: S0 not certified {statuses}", file=sys.stderr)

        draws = [log.draws for log in logs]
        rejected = sum(log.rejected for log in logs)
        replaced = sum(log.replaced for log in logs)
        all_certified &= certified == len(logs)
        print(" | ".join(str(x) for x in (
            size, len(logs), sum(draws), f"{statistics.mean(draws):.2f}", max(draws),
            rejected, f"{rejected / sum(draws):.1%}", replaced,
            f"{certified}/{len(logs)}", f"{accept_s:.1f}", f"{s0_s:.1f}",
        )))

    return 0 if all_certified else 1


if __name__ == "__main__":
    sys.exit(main())
