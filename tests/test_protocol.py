"""protocol.py: seeds, arms, loading the protocol TOML files, and the job matrix."""

import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest

from repair import protocol
from repair.model import (
    Arm,
    PolicyName,
    ReferenceMode,
    SequenceKind,
    SequenceProfile,
)
from repair.protocol import InstanceJob

PROTOCOL_DIR = Path(__file__).parents[1] / "protocol"
PILOT = PROTOCOL_DIR / "pilot.toml"
MAIN = PROTOCOL_DIR / "main.toml"


# --- derive_seed ---------------------------------------------------------------


def test_derive_seed_deterministic_and_in_range():
    """Same input -> same seed, also in a fresh process with a different hash salt."""
    seeds = [protocol.derive_seed(7, "instance", 40, i) for i in range(1000)]

    assert seeds == [protocol.derive_seed(7, "instance", 40, i) for i in range(1000)]
    assert all(0 <= seed < protocol.SEED_LIMIT for seed in seeds)

    # hash() of a str is salted per process (PYTHONHASHSEED): a seed built on it
    # would change between runs. Two subprocesses with different salts must agree.
    code = "from repair.protocol import derive_seed; print(derive_seed(7, 'instance', 40, 3))"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": salt, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            check=True,
            cwd=PROTOCOL_DIR.parent,
        ).stdout
        for salt in ("1", "2")
    }
    assert outputs == {f"{seeds[3]}\n"}


def test_derive_seed_sensitive_to_key_order_and_root():
    """Every part of the input matters, and no two key paths can be confused."""
    base = protocol.derive_seed(7, "a", "b")

    assert protocol.derive_seed(7, "b", "a") != base  # key order
    assert protocol.derive_seed(8, "a", "b") != base  # root
    assert protocol.derive_seed(7, "ab") != base  # no plain string joining
    assert protocol.derive_seed(7, "1") != protocol.derive_seed(7, 1)  # key type


def test_no_collisions_between_pilot_and_main_roots():
    """10,000 instance seeds per root: all distinct, and pilot and main share none."""
    pilot_root = protocol.load_protocol(PILOT).root_seed
    main_root = protocol.load_protocol(MAIN).root_seed
    assert pilot_root != main_root

    pilot = {protocol.derive_seed(pilot_root, "instance", i) for i in range(10_000)}
    main = {protocol.derive_seed(main_root, "instance", i) for i in range(10_000)}

    assert len(pilot) == len(main) == 10_000
    assert not pilot & main


# --- arms ----------------------------------------------------------------------


def test_main_arms_are_exactly_the_five_policies():
    """Each main arm carries only its own parameter; QO has no reference."""
    assert protocol.main_arms(2, Fraction(1)) == (
        Arm(PolicyName.QO, None),
        Arm(PolicyName.SF, ReferenceMode.CHAINED),
        Arm(PolicyName.QL, ReferenceMode.CHAINED),
        Arm(PolicyName.WQS, ReferenceMode.CHAINED, lam=2),
        Arm(PolicyName.HWF, ReferenceMode.CHAINED, alpha=Fraction(1)),
    )


def test_main_arm_names():
    """The names match the ones in expected.py and in stored results."""
    assert [arm.name for arm in protocol.main_arms(2, Fraction(1))] == [
        "QO",
        "SF-chained",
        "QL-chained",
        "WQS-chained-l2",
        "HWF-chained-a1",
    ]


def test_rq2_arms_differ_only_in_reference():
    assert protocol.rq2_arms() == (
        Arm(PolicyName.SF, ReferenceMode.ANCHORED),
        Arm(PolicyName.SF, ReferenceMode.CHAINED),
    )


def test_sensitivity_arms_vary_one_parameter_per_policy():
    """WQS varies lam only, HWF varies alpha only: one arm per grid value, no cross product."""
    arms = protocol.sensitivity_arms((1, 4), (Fraction(1, 2), Fraction(2)))

    assert arms == (
        Arm(PolicyName.WQS, ReferenceMode.CHAINED, lam=1),
        Arm(PolicyName.WQS, ReferenceMode.CHAINED, lam=4),
        Arm(PolicyName.HWF, ReferenceMode.CHAINED, alpha=Fraction(1, 2)),
        Arm(PolicyName.HWF, ReferenceMode.CHAINED, alpha=Fraction(2)),
    )
    assert [arm.name for arm in arms] == [
        "WQS-chained-l1",
        "WQS-chained-l4",
        "HWF-chained-a1/2",
        "HWF-chained-a2",
    ]


# --- load_protocol -------------------------------------------------------------


def test_load_pilot_toml():
    """TOML types become the Protocol's types: Fractions, int keys, tuples, None."""
    p = protocol.load_protocol(PILOT)

    assert p.name == "pilot"
    assert p.alpha == Fraction(1)
    assert p.alpha_grid == (Fraction(1, 2), Fraction(2))
    assert p.lam_grid == (1, 4)
    assert p.n_instances == {20: 2, 40: 2, 80: 2}
    assert all(isinstance(key, int) for key in p.n_instances)
    assert p.size_classes == (20, 40, 80)
    assert p.cells == protocol.CELLS
    assert p.robustness_scale is None  # left out -> None
    assert p.workforce_factor is None
    assert p.generator == protocol.GeneratorConfig()  # no [generator] -> defaults
    assert p.solver == protocol.SolverSettings(time_limit=60.0, s0_time_limit=60.0)


def test_load_rejects_unknown_key(tmp_path):
    """A typo must fail loudly instead of silently leaving a default in place."""
    path = tmp_path / "typo.toml"
    path.write_text("n_sequencs = 3\n" + PILOT.read_text())

    with pytest.raises(ValueError):
        protocol.load_protocol(path)


# Each entry fills one gap in main.toml; together they make a frozen protocol
FILLS = {
    # every pattern starts at a line start, so a comment mentioning it cannot match
    "date": ('\ndate = ""', '\ndate = "2026-12-01"'),
    "n_sequences": ("\nn_sequences = 0", "\nn_sequences = 5"),
    "n_instances_secondary": ("\nn_instances_secondary = 0", "\nn_instances_secondary = 10"),
    "n_instances": ("\n20 = 0\n40 = 0\n80 = 0", "\n20 = 10\n40 = 10\n80 = 10"),
    "robustness_scale": ("\n# robustness_scale = 0", "\nrobustness_scale = 2"),
    "workforce_factor": ("\n# workforce_factor = 0.0", "\nworkforce_factor = 0.8"),
}


def _filled_main(tmp_path, skip=None, extra=()):
    """A copy of main.toml in tmp_path with every gap filled except `skip`,
    then the (old, new) replacements in `extra`."""
    text = MAIN.read_text()
    for name, (old, new) in FILLS.items():
        assert old in text, f"main.toml no longer contains {old!r}"
        if name != skip:
            text = text.replace(old, new, 1)
    for old, new in extra:
        assert old in text
        text = text.replace(old, new, 1)
    path = tmp_path / "main.toml"
    path.write_text(text)
    return path


def test_require_frozen_rejects_unfilled_main():
    """The template itself is not frozen."""
    protocol.load_protocol(MAIN)  # loading without the check works

    with pytest.raises(ValueError):
        protocol.load_protocol(MAIN, require_frozen=True)


@pytest.mark.parametrize("skip", list(FILLS))
def test_require_frozen_rejects_each_missing_value(tmp_path, skip):
    """Every single gap is enough to refuse the file."""
    path = _filled_main(tmp_path, skip=skip)

    with pytest.raises(ValueError):
        protocol.load_protocol(path, require_frozen=True)


def test_require_frozen_rejects_missing_size_class_40(tmp_path):
    """Sensitivity, robustness and undersupplied all need the n = 40 main instances."""
    path = _filled_main(
        tmp_path,
        extra=[
            ("size_classes = [20, 40, 80]", "size_classes = [20, 80]"),
            ("20 = 10\n40 = 10\n80 = 10", "20 = 10\n80 = 10"),  # keep n_instances consistent
        ],
    )

    with pytest.raises(ValueError):
        protocol.load_protocol(path, require_frozen=True)


def test_require_frozen_accepts_filled_copy(tmp_path):
    """With every gap filled, the copy loads as a frozen protocol."""
    p = protocol.load_protocol(_filled_main(tmp_path), require_frozen=True)

    assert p.date == "2026-12-01"
    assert p.n_instances == {20: 10, 40: 10, 80: 10}
    assert p.robustness_scale == 2



# --- instance_jobs and jobs ----------------------------------------------------
#
# The matrix protocol uses different counts per size class (2, 3, 1), so a loop over
# the wrong count or the wrong size class changes the totals:
#   main          (2 + 3 + 1) instances x 3 profiles x N_Q = 2  -> 36 jobs
#   rq2           6 instances x 1 sequence                      ->  6 jobs
#   robustness    N_secondary = 2 main n = 40 instances x N_Q   ->  4 jobs
#   undersupplied 2 separate under-supplied instances x N_Q     ->  4 jobs

MAIN_PROFILES = (
    SequenceProfile.BALANCED,
    SequenceProfile.AVAILABILITY,
    SequenceProfile.PREFERENCE,
)


def _matrix_protocol(**changes):
    """The pilot with calibrated values set and unequal counts per size class."""
    base = replace(
        protocol.load_protocol(PILOT),
        n_instances={20: 2, 40: 3, 80: 1},
        n_sequences=2,
        n_instances_secondary=2,
        robustness_scale=3,
        workforce_factor=0.8,
    )
    return replace(base, **changes)


def _cell(p, cell):
    return [job for job in protocol.jobs(p) if job.cell == cell]


def test_instance_jobs_exact_order():
    """Main instances by size class, then index; the under-supplied instances last."""
    assert protocol.instance_jobs(_matrix_protocol()) == [
        InstanceJob(20, 0),
        InstanceJob(20, 1),
        InstanceJob(40, 0),
        InstanceJob(40, 1),
        InstanceJob(40, 2),
        InstanceJob(80, 0),
        InstanceJob(40, 0, under_supplied=True),
        InstanceJob(40, 1, under_supplied=True),
    ]


def test_instance_jobs_without_undersupplied_cell():
    """No under-supplied cell -> no under-supplied instances are built."""
    p = _matrix_protocol(cells=("main", "rq2", "robustness"))

    assert not any(i.under_supplied for i in protocol.instance_jobs(p))


@pytest.mark.parametrize(
    "cells", [protocol.CELLS, ("main",), ("rq2",), ("robustness",), ("undersupplied",)]
)
def test_instance_jobs_are_exactly_the_instances_jobs_use(cells):
    """make_instances.py builds instance_jobs; run_jobs.py loads job.instance.
    Every job must find its instance, and nothing is built that no job uses."""
    p = _matrix_protocol(cells=cells)
    built = protocol.instance_jobs(p)

    assert len(built) == len(set(built))  # no instance built twice
    assert set(built) == {job.instance for job in protocol.jobs(p)}


def test_job_counts_per_cell():
    """Each cell has the size given in the table above."""
    counts = Counter(job.cell for job in protocol.jobs(_matrix_protocol()))

    assert counts == {"main": 36, "rq2": 6, "robustness": 4, "undersupplied": 4}


def test_cells_appear_in_cells_order():
    """The matrix is grouped by cell, in the order of CELLS."""
    cells = [job.cell for job in protocol.jobs(_matrix_protocol())]
    groups = [c for i, c in enumerate(cells) if i == 0 or cells[i - 1] != c]

    assert groups == list(protocol.CELLS)


def test_cells_not_listed_produce_no_jobs():
    p = _matrix_protocol(cells=("rq2",))

    assert {job.cell for job in protocol.jobs(p)} == {"rq2"}


def test_main_jobs_nested_order_kind_and_scale():
    """size_classes x N_I x profiles x N_Q, nested in that order; kind rq1, k = n/20."""
    p = _matrix_protocol()
    main = _cell(p, "main")

    expected = [
        (InstanceJob(n, i), profile, q)
        for n in p.size_classes
        for i in range(p.n_instances[n])
        for profile, q in product(MAIN_PROFILES, range(p.n_sequences))
    ]
    assert [(j.instance, j.profile, j.sequence_index) for j in main] == expected
    assert all(j.kind == SequenceKind.RQ1 for j in main)
    assert all(j.scale == j.instance.size_class // 20 for j in main)


@pytest.mark.parametrize(
    "sensitivity_instances, n_extended",
    [
        pytest.param(None, 3, id="default_all_n40"),
        pytest.param(1, 1, id="first_instance_only"),
    ],
)
def test_sensitivity_arms_only_on_balanced_n40(sensitivity_instances, n_extended):
    """Exposé 4.5: the sensitivity arms run on the balanced n = 40 main sequences, for
    the first `sensitivity_instances` instances (None = all). Every other main job has
    exactly the main arms."""
    p = _matrix_protocol(sensitivity_instances=sensitivity_instances)
    main_arms = protocol.main_arms(p.lam, p.alpha)
    extended = main_arms + protocol.sensitivity_arms(p.lam_grid, p.alpha_grid)

    for job in _cell(p, "main"):
        gets_extra = (
            job.instance.size_class == 40
            and job.profile == SequenceProfile.BALANCED
            and job.instance.index < n_extended
        )
        assert job.arms == (extended if gets_extra else main_arms), job

    n_jobs = sum(len(job.arms) > len(main_arms) for job in _cell(p, "main"))
    assert n_jobs == n_extended * p.n_sequences


def test_rq2_jobs():
    """One rq2 sequence per main instance ("x 1"): sequence 0, rq2 arms, default k = n/20.
    CHECK against DESIGN.md section 7 if RQ2 should have N_Q sequences instead."""
    p = _matrix_protocol()
    rq2 = _cell(p, "rq2")

    assert [j.instance for j in rq2] == [
        InstanceJob(n, i) for n in p.size_classes for i in range(p.n_instances[n])
    ]
    for job in rq2:
        assert job.kind == SequenceKind.RQ2
        assert job.profile == SequenceProfile.RQ2
        assert job.sequence_index == 0
        assert job.arms == protocol.rq2_arms()
        assert job.scale == job.instance.size_class // 20


def test_rq2_background_scale_overrides_default():
    p = _matrix_protocol(rq2_background_scale=4)

    assert {job.scale for job in _cell(p, "rq2")} == {4}


def test_robustness_jobs():
    """Exposé 4.7: the first N_secondary main n = 40 instances, independent targeting
    at the calibrated scale, balanced profile, N_Q sequences each, main arms."""
    p = _matrix_protocol()
    jobs = _cell(p, "robustness")

    assert [(j.instance, j.sequence_index) for j in jobs] == [
        (InstanceJob(40, i), q) for i in range(2) for q in range(p.n_sequences)
    ]
    for job in jobs:
        assert job.kind == SequenceKind.INDEPENDENT
        assert job.profile == SequenceProfile.BALANCED
        assert job.scale == p.robustness_scale
        assert job.arms == protocol.main_arms(p.lam, p.alpha)


def test_undersupplied_jobs():
    """Exposé 4.7: separate under-supplied instances (n = 40 demand), S0-targeted,
    balanced profile, k = 2, N_Q sequences each, main arms."""
    p = _matrix_protocol()
    jobs = _cell(p, "undersupplied")

    assert [(j.instance, j.sequence_index) for j in jobs] == [
        (InstanceJob(40, i, under_supplied=True), q)
        for i in range(2)
        for q in range(p.n_sequences)
    ]
    for job in jobs:
        assert job.kind == SequenceKind.RQ1
        assert job.profile == SequenceProfile.BALANCED
        assert job.scale == 2
        assert job.arms == protocol.main_arms(p.lam, p.alpha)


def test_jobs_unique_and_deterministic():
    """No job twice, and the same protocol gives the same list (restart and shard rely
    on the order)."""
    first = protocol.jobs(_matrix_protocol())
    keys = [(j.cell, j.instance, j.kind, j.profile, j.sequence_index) for j in first]

    assert len(keys) == len(set(keys))
    assert protocol.jobs(_matrix_protocol()) == first


@pytest.mark.parametrize(
    "cells, missing",
    [
        pytest.param(("robustness",), {"robustness_scale": None}, id="robustness_scale"),
        pytest.param(("undersupplied",), {"workforce_factor": None}, id="workforce_factor"),
    ],
)
def test_jobs_rejects_missing_calibration(cells, missing):
    """A cell whose calibrated value is missing cannot be scheduled."""
    p = _matrix_protocol(cells=cells, **missing)

    with pytest.raises(ValueError):
        protocol.jobs(p)


def test_jobs_rejects_more_robustness_than_main_n40_instances():
    """Robustness reuses main n = 40 instances, so N_secondary <= N_I[40]."""
    p = _matrix_protocol(cells=("main", "robustness"), n_instances_secondary=4)

    with pytest.raises(ValueError):
        protocol.jobs(p)


# --- shard ---------------------------------------------------------------------


def _fake_jobs(n=50):
    """n distinct jobs, so the shard tests do not depend on jobs() working."""
    return [
        protocol.Job("main", InstanceJob(20, i), SequenceKind.RQ1, "balanced", 0, (), 1)
        for i in range(n)
    ]


@pytest.mark.parametrize("count", [1, 3, 7, 50, 60])
def test_shards_partition_the_jobs(count):
    """Every job is in exactly one shard, and shard sizes differ by at most one."""
    all_jobs = _fake_jobs()
    shards = [protocol.shard(all_jobs, i, count) for i in range(count)]

    assert sorted(
        (all_jobs.index(job) for s in shards for job in s)
    ) == list(range(len(all_jobs)))
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1


def test_shard_is_strided():
    """Strided (index, index + count, ...), so each shard mixes cells and sizes."""
    all_jobs = _fake_jobs()

    assert protocol.shard(all_jobs, 1, 3) == all_jobs[1::3]
    assert protocol.shard(all_jobs, 0, 1) == all_jobs


@pytest.mark.parametrize(
    "index, count", [(-1, 3), (3, 3), (0, 0), (0, -2)], ids=str
)
def test_shard_rejects_bad_index(index, count):
    """all_jobs[-1::3] would silently return the wrong jobs, so bad values raise."""
    with pytest.raises(ValueError):
        protocol.shard(_fake_jobs(), index, count)
