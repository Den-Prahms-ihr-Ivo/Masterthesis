import pytest

from dataclasses import replace
from fractions import Fraction
from repair import protocol
from repair.model import (
    Arm,
    PolicyName,
    ReferenceMode,
)


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
