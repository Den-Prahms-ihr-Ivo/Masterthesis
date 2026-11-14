"""Calculate pilot calibration tables without changing the protocol file.

phi tests workforce factors and selects the largest tested one with proven
positive shortfall in at least 80% of raw under-supplied candidates.

intensity compares independent targeting with the S0-targeted balanced
design. It chooses the scale whose mean non-inert state count is closest
to the target (the proposed definition in DESIGN.md 13.5). It uses one
sequence per accepted instance and the same seeds across scales. A tie
between scales is resolved in favour of the smaller k.

background reports diagnostics for k=2. There is no selection target in
DESIGN.md 13.6, so this output is a default check, not a calibrated choice.

A required optimum must be certified. If it is not, stop instead of dropping
that observation or counting a timeout as an inert state.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import replace
from pathlib import Path

import oracle

from repair import milp
from repair.generator import generate_instance
from repair.model import SequenceProfile
from repair.objectives import MAX_QUALITY, MIN_SHORTFALL
from repair.perturbations import instances_along, rq1_sequence, rq2_sequence
from repair.protocol import derive_seed, load_protocol
from repair.runner import (
    accept_instance,
    check_oracles,
    initial_schedule,
    solve_lexicographic,
)


def _positive_count(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _solve(protocol, instance, specs):
    """Solve the requested stages and check their answer with the independent oracle."""

    result = solve_lexicographic(
        instance,
        specs,
        milp.solve_stage,
        time_limit=protocol.solver.time_limit,
        retry_factor=protocol.solver.retry_factor,
        start=frozenset(),
    )
    check_oracles(instance, specs, result)
    return result


def _require_certified(result, context: str) -> None:
    """Stop calibration when a required optimum has not been proven."""

    if not result.certified:
        statuses = [record.status.value for record in result.stages]
        raise RuntimeError(f"{context}: optimum not certified; statuses={statuses}")


def _accepted_cases(protocol, n_instances: int):
    """Yield accepted n=40 instances with a certified S0.

    Use the same base and reserve keys as the instance-generation scripts.
    Stop on an uncertified S0 so calibration does not silently change its sample.
    """
    _positive_count(n_instances, "n_instances")
    for i in range(n_instances):
        base = derive_seed(protocol.root_seed, "instance", 40, i)
        reserve = [
            derive_seed(protocol.root_seed, "reserve", 40, i, j)
            for j in range(protocol.n_reserve_seeds)
        ]
        instance, _ = accept_instance(
            protocol.generator,
            40,
            base,
            reserve,
            protocol.solver,
            milp.solve_stage,
            max_draws=protocol.max_draws,
        )
        result = initial_schedule(instance, protocol.solver, milp.solve_stage)
        _require_certified(result, f"S0, instance index={i}, seed={instance.meta.seed}")
        assert result.schedule is not None
        yield i, instance, result.schedule


def _non_inert(protocol, instance_t, s0, context: str) -> bool:
    """Check whether S0 is binding or worse than the best quality at minimum shortfall.

    This compares the current cumulative state with fixed S0. It does not
    isolate the effect of the latest event, and uses no repair-policy outcome.
    A hard-constraint violation already makes S0 binding; otherwise solve the
    coverage and quality stages to decide.
    """
    if oracle.violations(s0, instance_t):
        return True

    result = _solve(protocol, instance_t, (MIN_SHORTFALL, MAX_QUALITY))
    _require_certified(result, context)
    optimum = result.schedule
    assert optimum is not None
    return oracle.shortfall(s0, instance_t) > oracle.shortfall(
        optimum, instance_t
    ) or oracle.quality(s0, instance_t) < oracle.quality(optimum, instance_t)


def _count_non_inert(protocol, instance, s0, sequence, *, background_only=False):
    """Count non-inert states while applying the sequence in order.

    background_only skips counting steps 1 and 6, but still applies them. Their
    effects therefore remain present when later background states are checked.
    """

    count = 0
    for t, instance_t in enumerate(instances_along(instance, sequence.events), 1):
        if background_only and t in (1, 6):
            continue
        context = (
            f"instance seed={instance.meta.seed}, sequence seed={sequence.seed}, t={t}"
        )
        count += _non_inert(protocol, instance_t, s0, context)
    return count


def calibrate_phi(protocol, grid: list[float], n_candidates: int) -> list[dict]:
    """Test each workforce factor on the same candidate seeds.

    Evaluate raw candidates without rejecting those with missing coverage.
    The selected row is the largest tested factor meeting the 80% condition;
    no selected row means the grid did not contain a suitable factor.
    """

    _positive_count(n_candidates, "n_candidates")
    if not grid or any(not math.isfinite(phi) or not 0 < phi <= 1 for phi in grid):
        raise ValueError("grid must contain finite workforce factors in (0, 1]")

    rows = []
    for phi in sorted(set(grid), reverse=True):
        cfg = replace(protocol.generator, workforce_factor=phi)
        positive = 0
        for i in range(n_candidates):
            # Leave phi out of the seed so all factors use the same candidate draws.
            seed = derive_seed(protocol.root_seed, "calibration", "phi", i)
            instance = generate_instance(cfg, 40, seed, under_supplied=True)
            result = _solve(protocol, instance, (MIN_SHORTFALL,))

            # A feasible zero-shortfall incumbent proves the optimum is zero,
            # even when the solver reports a time limit. A positive incumbent
            # alone does not prove that every feasible schedule has positive shortfall.
            if result.schedule is not None:
                value = oracle.shortfall(result.schedule, instance)
                if value == 0:
                    continue
            _require_certified(result, f"phi={phi}, candidate={i}, seed={seed}")
            assert result.schedule is not None
            positive += oracle.shortfall(result.schedule, instance) > 0

        rows.append(
            {
                "root_seed": protocol.root_seed,
                "phi": phi,
                "n_volunteers": math.floor(40 * phi),
                "n_candidates": n_candidates,
                "positive_shortfall": positive,
                "rate": positive / n_candidates,
                "passes": 5 * positive >= 4 * n_candidates,
                "selected": False,
            }
        )

    # The grid is descending, so the first passing row is the largest tested phi.
    for row in rows:
        if row["passes"]:
            row["selected"] = True
            break
    return rows


def calibrate_intensity(protocol, scales: list[int], n_instances: int) -> list[dict]:
    """Match independent-targeting intensity to the S0-targeted design.

    Reuse each accepted instance, S0 and sequence seed for all candidate scales.
    Count non-inert cumulative states against S0 and select the closest total,
    preferring the smaller scale in a tie. Equal sequence lengths make matching
    totals equivalent to matching mean counts or rates.
    """

    _positive_count(n_instances, "n_instances")
    if not scales:
        raise ValueError("scales must not be empty")
    for k in scales:
        _positive_count(k, "scale")
    scales = sorted(set(scales))

    target_count = 0
    independent_counts = dict.fromkeys(scales, 0)
    for i, instance, s0 in _accepted_cases(protocol, n_instances):
        # Reuse the sequence seed across the targeted design and all scales.
        # Each comparison also starts from the same accepted instance and S0.
        seed = derive_seed(protocol.root_seed, "calibration", "intensity", i)
        target = rq1_sequence(
            instance,
            s0,
            SequenceProfile.BALANCED,
            seed,
            targeting="s0",
            scale=2,
        )
        target_count += _count_non_inert(protocol, instance, s0, target)
        for k in scales:
            independent = rq1_sequence(
                instance,
                s0,
                SequenceProfile.BALANCED,
                seed,
                targeting="independent",
                scale=k,
            )
            independent_counts[k] += _count_non_inert(
                protocol, instance, s0, independent
            )

    # All sequences have 15 steps and equal weight, so matching total counts,
    # mean counts per sequence, or pooled rates chooses the same scale.
    chosen = min(scales, key=lambda k: (abs(independent_counts[k] - target_count), k))
    return [
        {
            "root_seed": protocol.root_seed,
            "scale": k,
            "n_instances": n_instances,
            "n_sequences": n_instances,
            "n_events": 15 * n_instances,
            "target_non_inert": target_count,
            "independent_non_inert": independent_counts[k],
            "target_mean": target_count / n_instances,
            "independent_mean": independent_counts[k] / n_instances,
            "target_rate": target_count / (15 * n_instances),
            "independent_rate": independent_counts[k] / (15 * n_instances),
            "mean_difference": (independent_counts[k] - target_count) / n_instances,
            "selected": k == chosen,
        }
        for k in scales
    ]


def calibrate_background(protocol, n_instances: int) -> list[dict]:
    """Report RQ2 background diagnostics at the default k = 40 / 20 = 2.

    Steps 1 and 6 are left out of the count, but their effects remain in the
    evolving instance. For example, the forced absence can keep S0 infeasible
    during steps 2..5. This is not a separate estimate of background effects
    and is not used to choose a scale or compare the two repair arms.
    """
    _positive_count(n_instances, "n_instances")
    count = 0
    for i, instance, s0 in _accepted_cases(protocol, n_instances):
        seed = derive_seed(protocol.root_seed, "calibration", "background", i)
        sequence = rq2_sequence(instance, s0, seed, scale=2)
        count += _count_non_inert(
            protocol, instance, s0, sequence, background_only=True
        )
    return [
        {
            "root_seed": protocol.root_seed,
            "scale": 2,
            "n_instances": n_instances,
            "n_background_steps": 13 * n_instances,
            "non_inert_background_states": count,
            "mean_non_inert_background_states": count / n_instances,
            "calibrated": False,
            "selection_rule": "default_n_over_20_no_target_specified",
        }
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=["phi", "intensity", "background"])
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--n-instances", type=int, default=20)
    parser.add_argument(
        "--grid", type=float, nargs="+", default=[x / 100 for x in range(95, 59, -5)]
    )
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args(argv)

    try:
        protocol = load_protocol(args.protocol)
        if args.what == "phi":
            rows = calibrate_phi(protocol, args.grid, args.n_candidates)
        elif args.what == "intensity":
            rows = calibrate_intensity(protocol, args.scales, args.n_instances)
        else:
            rows = calibrate_background(protocol, args.n_instances)
    except (ValueError, RuntimeError) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1

    # Write the table only after all required solves have completed.
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"calibration_{args.what}.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(" | ".join(rows[0]))
    for row in rows:
        print(" | ".join(str(value) for value in row.values()))
    print(f"Written: {path}")

    if args.what == "background":
        print("k=2 is a documented default, not a calibrated value.")
    elif not any(row["selected"] for row in rows):
        print("No tested value meets the selection rule.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
