"""Protocol: every number the pilot sets, seed derivation, and the experiment matrix.

The TOML file holds NUMBERS ONLY (sizes, counts, time limits, calibrated values).
The structure of the experiment (which cells, which arms, which sequences) is fixed
by the exposé and therefore lives in code: jobs(protocol). The dated main.toml plus
the git commit in the manifest freeze the whole design.

This module and storage.py are the only src/ modules that read files.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from repair.model import Arm, SequenceKind, UtilityWeights, PolicyName, ReferenceMode


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
    s0_time_limit: float  # TODO: IVO Design Decision
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


def load_protocol(path: Path, require_frozen: bool = False) -> Protocol:
    """Parse a protocol TOML file. Fractions are given as strings ("1/2").

    require_frozen=True (main run) raises ValueError if `date` is empty, any needed
    count is 0, or a calibrated value needed by a listed cell is missing.
    Also raises if 40 is not in size_classes while the sensitivity arms, "robustness" or
    "undersupplied" need the n = 40 main instances.
    """
    raise NotImplementedError


def derive_seed(root: int, *keys: object) -> int:
    raise NotImplementedError


def main_arms(lam: int, alpha: Fraction) -> tuple[Arm, ...]:
    """QO plus SF, QL, WQS(lam), HWF(alpha), all chained."""

    return tuple(
        Arm(policy=policy, reference=ReferenceMode.CHAINED, lam=lam, alpha=alpha)
        for policy in PolicyName
    )


def rq2_arms() -> tuple[Arm, ...]:
    """SF-anchored and SF-chained."""
    return tuple(
        Arm(policy=policy, reference=ReferenceMode.CHAINED)
        for policy in (PolicyName.WQS, PolicyName.HWF)
    )


def sensitivity_arms(
    lam_grid: tuple[int, ...], alpha_grid: tuple[Fraction, ...]
) -> tuple[Arm, ...]:
    """WQS and HWF at the non-main grid values, chained."""
    return tuple(
        Arm(
            policy=policy,
            reference=ReferenceMode.CHAINED,
            lam=lam,
            alpha=alpha,
        )
        for lam in lam_grid
        for alpha in alpha_grid
        for policy in (PolicyName.WQS, PolicyName.HWF)
    )


def instance_jobs(protocol: Protocol) -> list[InstanceJob]:
    """Every base instance the protocol needs, in a fixed order (make_instances.py)."""
    raise NotImplementedError


def jobs(protocol: Protocol) -> list[Job]:
    """The experiment matrix (DESIGN.md section 7), in a fixed order.

    main          size_classes x N_I x 3 profiles x N_Q, kind "rq1", main_arms;
                  for n=40, profile "balanced" and instance index < sensitivity_instances
                  (default: all N_I[40]), arms = main_arms + sensitivity_arms
    rq2           size_classes x N_I x 1, kind "rq2", rq2_arms
    robustness    n=40, N_secondary instances, balanced, kind "independent", robustness_scale
    undersupplied n=40 demand class, under_supplied=True, N_secondary, balanced, k=2
    """
    raise NotImplementedError


def shard(all_jobs: list[Job], index: int, count: int) -> list[Job]:
    """all_jobs[index::count]; 0 <= index < count."""
    raise NotImplementedError
