"""Protocol: every number the pilot sets, seed derivation, and the experiment matrix.

The TOML file holds NUMBERS ONLY (sizes, counts, time limits, calibrated values).
The structure of the experiment (which cells, which arms, which sequences) is fixed
by the exposé and therefore lives in code: jobs(protocol). The dated main.toml plus
the git commit in the manifest freeze the whole design.

This module and storage.py are the only src/ modules that read files.
"""

import hashlib
import json

import tomllib
from dataclasses import dataclass, field, fields
from fractions import Fraction
from pathlib import Path

from repair.model import (
    Arm,
    PolicyName,
    ReferenceMode,
    SequenceKind,
    SequenceProfile,
    UtilityWeights,
)


@dataclass(frozen=True)
class GeneratorConfig:
    """Table A1. Defaults are the exposé values; [generator] in the TOML may override."""

    n_days: int = 14
    periods_per_day: int = 2
    n_roles: int = 5
    role_activation: float = 0.6  # rho
    commitment_shares: tuple[float, float, float] = (0.24, 0.35, 0.41)
    availability_ranges: tuple[tuple[int, int], ...] = ((3, 7), (10, 16), (18, 26))
    seniority_shares: tuple[float, float, float] = (0.35, 0.40, 0.25)
    skill_shares: tuple[float, float, float, float] = (0.10, 0.30, 0.40, 0.20)
    caps: tuple[int, ...] = (10, 11, 12)
    min_skill_two_share: float = 0.25  # P(kappa_s = 2)
    expert_share: float = 0.20  # P(e_s = 1 | skill-3 volunteer available)
    workforce_factor: float = 1.0  # phi; set per job for the under-supplied class
    weights: UtilityWeights = field(default_factory=UtilityWeights)


@dataclass(frozen=True)
class SolverSettings:
    time_limit: float  # tau per stage, seconds (600 in the exposé)
    s0_time_limit: float
    seed: int = 0  # SCIP randomization seed shift
    threads: int = 1
    retry_factor: float = 2.0  # one retry with factor * tau


@dataclass(frozen=True)
class Protocol:
    name: str
    date: str  # ISO date of freezing; "" = not frozen
    root_seed: int
    horizon: int  # T = 15
    size_classes: tuple[int, ...]  # main classes still in (adequacy rule)
    cells: tuple[str, ...]  # subset of CELLS this protocol runs
    n_instances: dict[int, int]  # N_I per main size class
    n_instances_secondary: int  # robustness and under-supplied cells (exposé 4.7)
    sensitivity_instances: int | None  # None = all main n=40 instances (exposé 4.5)
    n_sequences: int  # N_Q
    lam: int  # main WQS value (2)
    alpha: Fraction  # main HWF value (1)
    lam_grid: tuple[int, ...]  # sensitivity values (1, 4)
    alpha_grid: tuple[Fraction, ...]  # sensitivity values (1/2, 2)
    robustness_scale: int | None  # calibrated k for independent targeting
    rq2_background_scale: int | None  # None -> default k = n/20
    workforce_factor: float | None  # phi, calibrated
    generator: GeneratorConfig
    solver: SolverSettings
    max_draws: int = 100
    n_reserve_seeds: int = 50


CELLS = ("main", "rq2", "robustness", "undersupplied")
# No separate sensitivity cell: exposé 4.5 runs the other lambda/alpha values "as additional arms
# on the main S0-targeted sequences of the n = 40 balanced profile", so they are extra arms in
# those main jobs and share their coverage solves and result files.


@dataclass(frozen=True)
class InstanceJob:
    size_class: int  # demand class
    index: int  # 0..N_I-1
    under_supplied: bool = False


@dataclass(frozen=True)
class Job:
    """One sequence to run: the unit of parallelism and of restart."""

    cell: str  # one of CELLS
    instance: InstanceJob
    kind: SequenceKind
    profile: str  # "balanced" | "availability" | "preference" | "rq2"
    sequence_index: int  # 0..N_Q-1
    arms: tuple[Arm, ...]
    scale: int  # perturbation scale k for this job


SEED_LIMIT = 2**63  # derive_seed returns 0 <= seed < SEED_LIMIT

# Protocol fields that may be left out of the TOML; missing -> None.
OPTIONAL_KEYS = (
    "sensitivity_instances",
    "robustness_scale",
    "rq2_background_scale",
    "workforce_factor",
)


def _tuples(value):
    """TOML arrays -> tuples, also nested ([[3, 7], [10, 16]] -> ((3, 7), (10, 16)))."""
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


def _validate_frozen(protocol: Protocol) -> None:
    if not protocol.date:
        raise ValueError("date is empty")

    if protocol.n_sequences <= 0:
        raise ValueError("n_sequences must be positive")

    for size_class in protocol.size_classes:
        if size_class not in protocol.n_instances:
            raise ValueError(f"n_instances[{size_class}] missing")

        if protocol.n_instances[size_class] <= 0:
            raise ValueError(f"n_instances[{size_class}] must be positive")

    has_robustness = "robustness" in protocol.cells
    has_undersupplied = "undersupplied" in protocol.cells

    if has_robustness or has_undersupplied:
        if protocol.n_instances_secondary <= 0:
            raise ValueError(
                "n_instances_secondary must be positive "
                "for robustness or undersupplied"
            )

    # "is None or <= 0": the template's placeholder 0 must not count as calibrated.
    if has_robustness and (
        protocol.robustness_scale is None or protocol.robustness_scale <= 0
    ):
        raise ValueError("robustness_scale missing or not positive")

    if has_undersupplied and (
        protocol.workforce_factor is None or protocol.workforce_factor <= 0
    ):
        raise ValueError("workforce_factor missing or not positive")

    has_sensitivity = "main" in protocol.cells and bool(
        protocol.lam_grid or protocol.alpha_grid
    )

    needs_40 = has_sensitivity or has_robustness or has_undersupplied

    if needs_40 and 40 not in protocol.size_classes:
        raise ValueError(
            "size class 40 is required for sensitivity, " "robustness, or undersupplied"
        )


def load_protocol(path: Path, require_frozen: bool = False) -> Protocol:
    """Parse a protocol TOML file (format: see protocol/pilot.toml).

    Conversions, because TOML has no fractions, int keys, tuples or null:
    - alpha and alpha_grid are strings ("1", "1/2") -> Fraction
    - [n_instances] keys are strings ("20") -> int
    - arrays -> tuples (also inside [generator], e.g. availability_ranges)
    - a missing optional key -> None (sensitivity_instances, robustness_scale,
      rq2_background_scale, workforce_factor); a missing [generator] -> GeneratorConfig()
    An unknown key raises ValueError (a typo must not silently fall back to a default).

    require_frozen=True (main run) raises ValueError if `date` is empty, any needed
    count is 0, or a calibrated value needed by a listed cell is missing:
    "robustness" needs robustness_scale, "undersupplied" needs workforce_factor.
    Also raises if 40 is not in size_classes while the sensitivity arms, "robustness" or
    "undersupplied" need the n = 40 main instances.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    allowed = {field.name for field in fields(Protocol) if field.init}

    unknown = raw.keys() - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown protocol keys: {names}")

    # Convert serialized representations into domain types.
    if "alpha" in raw:
        raw["alpha"] = Fraction(raw["alpha"])

    if "alpha_grid" in raw:
        raw["alpha_grid"] = tuple(Fraction(value) for value in raw["alpha_grid"])

    for name in ("size_classes", "cells", "lam_grid"):
        if name in raw:
            raw[name] = tuple(raw[name])

    if "n_instances" in raw:
        raw["n_instances"] = {
            int(size_class): count for size_class, count in raw["n_instances"].items()
        }

    # Optional values: left out of the TOML -> None.
    for name in OPTIONAL_KEYS:
        raw.setdefault(name, None)

    if "solver" not in raw:
        raise ValueError("solver table missing")
    try:
        raw["solver"] = SolverSettings(**raw["solver"])
    except TypeError as exc:
        raise ValueError(f"invalid solver settings: {exc}") from exc

    try:
        generator = {
            name: _tuples(value) for name, value in raw.get("generator", {}).items()
        }
        if "weights" in generator:
            generator["weights"] = UtilityWeights(**generator["weights"])
        raw["generator"] = GeneratorConfig(**generator)
    except TypeError as exc:
        raise ValueError(f"invalid generator settings: {exc}") from exc

    try:
        protocol = Protocol(**raw)
    except TypeError as exc:
        raise ValueError(f"invalid protocol: {exc}") from exc

    if require_frozen:
        _validate_frozen(protocol)

    return protocol


def derive_seed(root: int, *keys: str | int) -> int:
    """A seed for one purpose, derived from the protocol's root seed and a key path.

    Example: derive_seed(protocol.root_seed, "instance", 40, 3) seeds instance 3 of
    size class 40. Every random draw in the study gets its seed this way, so a single
    root_seed reproduces everything and no two purposes share a seed.

    Deterministic across runs and machines (no hash(): it is salted per process).
    Sensitive to the root, the order of the keys and their types ("1" != 1).
    Returns 0 <= seed < SEED_LIMIT.
    """
    data = json.dumps([root, *keys]).encode("utf-8")
    digest = hashlib.sha256(data).digest()

    return int.from_bytes(digest[:8], "big") % SEED_LIMIT


def main_arms(lam: int, alpha: Fraction) -> tuple[Arm, ...]:
    """QO plus SF, QL, WQS(lam), HWF(alpha), all chained (QO has no reference)."""
    return (
        Arm(policy=PolicyName.QO, reference=None),
        Arm(policy=PolicyName.SF, reference=ReferenceMode.CHAINED),
        Arm(policy=PolicyName.QL, reference=ReferenceMode.CHAINED),
        Arm(policy=PolicyName.WQS, reference=ReferenceMode.CHAINED, lam=lam),
        Arm(policy=PolicyName.HWF, reference=ReferenceMode.CHAINED, alpha=alpha),
    )


def rq2_arms() -> tuple[Arm, ...]:
    """SF-anchored and SF-chained: same policy, the reference is the only difference."""
    return tuple(
        Arm(policy=PolicyName.SF, reference=reference)
        for reference in (ReferenceMode.ANCHORED, ReferenceMode.CHAINED)
    )


def sensitivity_arms(
    lam_grid: tuple[int, ...], alpha_grid: tuple[Fraction, ...]
) -> tuple[Arm, ...]:
    """WQS at each lam and HWF at each alpha (the non-main grid values), chained."""
    wqs = tuple(
        Arm(policy=PolicyName.WQS, reference=ReferenceMode.CHAINED, lam=lam)
        for lam in lam_grid
    )
    hwf = tuple(
        Arm(policy=PolicyName.HWF, reference=ReferenceMode.CHAINED, alpha=alpha)
        for alpha in alpha_grid
    )
    return wqs + hwf


# The three main-cell profiles, in matrix order (rq2 has its own profile).
MAIN_PROFILES = (
    SequenceProfile.BALANCED,
    SequenceProfile.AVAILABILITY,
    SequenceProfile.PREFERENCE,
)


def _validate_job_inputs(protocol: Protocol) -> None:
    unknown = set(protocol.cells) - set(CELLS)
    if unknown:
        raise ValueError(f"unknown cells: {sorted(unknown)}")

    if "robustness" in protocol.cells:
        if protocol.robustness_scale is None:
            raise ValueError("robustness_scale missing")

        if protocol.n_instances_secondary > protocol.n_instances.get(40, 0):
            raise ValueError("n_instances_secondary exceeds n_instances[40]")

    if "undersupplied" in protocol.cells:
        if protocol.workforce_factor is None:
            raise ValueError("workforce_factor missing")


def instance_jobs(protocol: Protocol) -> list[InstanceJob]:
    """Every base instance the protocol needs, in a fixed order (make_instances.py).

    Order: main instances by size class, then index; the under-supplied instances
    InstanceJob(40, i, under_supplied=True), i < n_instances_secondary, last (only if
    "undersupplied" is a cell). Exactly the set {job.instance for job in jobs(protocol)},
    no duplicates. Robustness builds nothing extra: it reuses main n = 40 instances.
    """
    # dict.fromkeys drops duplicates but keeps the first-seen order (a set would not).
    return list(dict.fromkeys(job.instance for job in jobs(protocol)))


def jobs(protocol: Protocol) -> list[Job]:
    """The experiment matrix (DESIGN.md section 7), in a fixed order.

    main          size_classes x N_I x 3 profiles x N_Q, kind "rq1", main_arms;
                  for n=40, profile "balanced" and instance index < sensitivity_instances
                  (default: all N_I[40]), arms = main_arms + sensitivity_arms
    rq2           size_classes x N_I x 1, kind "rq2", rq2_arms
    robustness    n=40, N_secondary instances, balanced, kind "independent", robustness_scale
    undersupplied n=40 demand class, under_supplied=True, N_secondary, balanced, k=2

    Details the tests pin down:
    - cells in CELLS order, only those in protocol.cells; loops nested in the order above
      (size class -> instance -> profile -> sequence)
    - scale k: main n // 20; rq2 rq2_background_scale or n // 20; robustness
      robustness_scale; undersupplied 2
    - rq2: one sequence per instance, sequence_index 0, profile "rq2"
    - robustness: main instances InstanceJob(40, i), i < N_secondary, N_Q sequences each,
      main_arms; undersupplied: kind "rq1", N_Q sequences each, main_arms
    - ValueError if a listed cell lacks its calibrated value (robustness_scale,
      workforce_factor) or N_secondary > N_I[40] while "robustness" is listed
    """
    _validate_job_inputs(protocol)

    result: list[Job] = []

    main = main_arms(protocol.lam, protocol.alpha)
    sensitivity = sensitivity_arms(protocol.lam_grid, protocol.alpha_grid)
    rq2 = rq2_arms()

    def add(
        cell: str,
        instance: InstanceJob,
        kind: SequenceKind,
        profile: str,
        sequence_index: int,
        arms: tuple[Arm, ...],
        scale: int,
    ) -> None:
        result.append(
            Job(
                cell=cell,
                instance=instance,
                kind=kind,
                profile=profile,
                sequence_index=sequence_index,
                arms=arms,
                scale=scale,
            )
        )

    for cell in CELLS:
        if cell not in protocol.cells:
            continue

        if cell == "main":
            for size in protocol.size_classes:
                for i in range(protocol.n_instances[size]):
                    instance = InstanceJob(size, i)

                    for profile in MAIN_PROFILES:
                        arms = main

                        include_sensitivity = (
                            size == 40
                            and profile == SequenceProfile.BALANCED
                            and (
                                protocol.sensitivity_instances is None
                                or i < protocol.sensitivity_instances
                            )
                        )

                        if include_sensitivity:
                            arms = main + sensitivity

                        for q in range(protocol.n_sequences):
                            add(
                                cell=cell,
                                instance=instance,
                                kind=SequenceKind.RQ1,
                                profile=profile,
                                sequence_index=q,
                                arms=arms,
                                scale=size // 20,
                            )

        elif cell == "rq2":
            for size in protocol.size_classes:
                scale = (
                    size // 20
                    if protocol.rq2_background_scale is None
                    else protocol.rq2_background_scale
                )

                for i in range(protocol.n_instances[size]):
                    add(
                        cell=cell,
                        instance=InstanceJob(size, i),
                        kind=SequenceKind.RQ2,
                        profile=SequenceProfile.RQ2,
                        sequence_index=0,
                        arms=rq2,
                        scale=scale,
                    )

        elif cell == "robustness":
            robustness_scale = protocol.robustness_scale
            assert robustness_scale is not None  # checked by _validate_job_inputs

            for i in range(protocol.n_instances_secondary):
                instance = InstanceJob(40, i)

                for q in range(protocol.n_sequences):
                    add(
                        cell=cell,
                        instance=instance,
                        kind=SequenceKind.INDEPENDENT,
                        profile=SequenceProfile.BALANCED,
                        sequence_index=q,
                        arms=main,
                        scale=robustness_scale,
                    )

        elif cell == "undersupplied":
            for i in range(protocol.n_instances_secondary):
                instance = InstanceJob(
                    size_class=40,
                    index=i,
                    under_supplied=True,
                )

                for q in range(protocol.n_sequences):
                    add(
                        cell=cell,
                        instance=instance,
                        kind=SequenceKind.RQ1,
                        profile=SequenceProfile.BALANCED,
                        sequence_index=q,
                        arms=main,
                        scale=2,
                    )

    return result


def shard(all_jobs: list[Job], index: int, count: int) -> list[Job]:
    """all_jobs[index::count]; 0 <= index < count.

    One shard per cluster task (e.g. SLURM_ARRAY_TASK_ID of SLURM_ARRAY_TASK_COUNT).
    Strided, so every shard mixes cells and size classes. Raises ValueError for
    count <= 0 or index outside 0..count-1 (a negative index would slice silently).
    """
    if count <= 0:
        raise ValueError("shard count must be positive")

    if not 0 <= index < count:
        raise ValueError("shard index must satisfy 0 <= index < count")

    return all_jobs[index::count]
