"""protocol.py: seeds, arms, and loading the protocol TOML files."""

import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from repair import protocol
from repair.model import (
    Arm,
    PolicyName,
    ReferenceMode,
)

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

