"""Check the Table 1 stage order, repair references and HWF history weights."""

from fractions import Fraction

import pytest

from repair.expected import REPAIR_SLICE_S0, REPAIR_SLICE_SF
from repair.model import Arm, PolicyName, ReferenceMode
from repair.objectives import (
    MAX_QUALITY,
    MAX_TIE_BREAK,
    MIN_SHORTFALL,
    max_quality_minus_distance,
    min_distance,
)
from repair.policies import history_weights, policy_stages, reference_for, validate_arm
from repair.protocol import main_arms, rq2_arms, sensitivity_arms

QO, SF, QL, WQS, HWF = (
    PolicyName.QO,
    PolicyName.SF,
    PolicyName.QL,
    PolicyName.WQS,
    PolicyName.HWF,
)
CHAINED, ANCHORED = ReferenceMode.CHAINED, ReferenceMode.ANCHORED

ALL_ARMS = (
    *main_arms(2, Fraction(1)),
    *rq2_arms(),
    *sensitivity_arms((1, 4), (Fraction(1, 2), Fraction(2))),
    Arm(HWF, CHAINED, alpha=Fraction(0)),  # the identity arm of the runner tests
)


# Validate_arm


@pytest.mark.parametrize(
    "arm",
    [
        pytest.param(Arm(QO, CHAINED), id="QO_with_reference"),
        pytest.param(Arm(QO, None, lam=2), id="QO_with_lam"),
        pytest.param(Arm(SF, None), id="SF_without_reference"),
        pytest.param(Arm(SF, CHAINED, lam=2), id="SF_with_lam"),
        pytest.param(Arm(QL, CHAINED, alpha=Fraction(1)), id="QL_with_alpha"),
        pytest.param(Arm(WQS, CHAINED), id="WQS_without_lam"),
        pytest.param(Arm(WQS, CHAINED, lam=-1), id="WQS_negative_lam"),
        pytest.param(Arm(WQS, CHAINED, lam=True), id="WQS_bool_lam"),
        pytest.param(Arm(WQS, CHAINED, lam=2, alpha=Fraction(1)), id="WQS_with_alpha"),
        pytest.param(Arm(HWF, CHAINED), id="HWF_without_alpha"),
        pytest.param(Arm(HWF, CHAINED, alpha=Fraction(-1, 2)), id="HWF_negative_alpha"),
        pytest.param(Arm(HWF, CHAINED, alpha=0.5), id="HWF_float_alpha"),  # type: ignore[arg-type]
        pytest.param(Arm(HWF, CHAINED, lam=2, alpha=Fraction(1)), id="HWF_with_lam"),
        pytest.param(Arm(HWF, "chained-ish", alpha=Fraction(1)), id="unknown_reference"),  # type: ignore[arg-type]
    ],
)
def test_validate_arm_rejects_inconsistent_arms(arm):
    with pytest.raises(ValueError):
        validate_arm(arm)


@pytest.mark.parametrize("arm", ALL_ARMS, ids=lambda a: a.name)
def test_validate_arm_accepts_every_protocol_arm(arm):
    validate_arm(arm)


# Reference_for


def test_reference_for_anchored_chained_qo():
    """anchored -> S0, chained -> the arm's own previous schedule, QO -> None."""
    s0, previous = REPAIR_SLICE_S0, REPAIR_SLICE_SF

    assert reference_for(Arm(SF, ANCHORED), s0, previous) == s0
    assert reference_for(Arm(SF, CHAINED), s0, previous) == previous
    assert reference_for(Arm(QO, None), s0, previous) is None
    # t = 1: previous is S0, so both conventions coincide
    assert reference_for(Arm(SF, ANCHORED), s0, s0) == reference_for(
        Arm(SF, CHAINED), s0, s0
    )


# History_weights

VOLUNTEERS = (0, 1, 2)


def test_history_weights_alpha_1():
    """1 + b_w."""
    assert history_weights(Fraction(1), {0: 0, 1: 3, 2: 1}, VOLUNTEERS) == {0: 1, 1: 4, 2: 2}


def test_history_weights_alpha_half():
    """Scale 2: 2 * (1 + b / 2) = 2 + b -> {0: 2, 1: 5} for b = (0, 3)."""
    assert history_weights(Fraction(1, 2), {0: 0, 1: 3}, (0, 1)) == {0: 2, 1: 5}


def test_history_weights_alpha_2():
    """1 + 2 b_w."""
    assert history_weights(Fraction(2), {0: 0, 1: 3}, (0, 1)) == {0: 1, 1: 7}


def test_history_weights_missing_volunteer_is_zero():
    """A volunteer without a burden entry was never affected: b = 0."""
    assert history_weights(Fraction(1), {1: 2}, VOLUNTEERS) == {0: 1, 1: 3, 2: 1}


@pytest.mark.parametrize("burden", [{}, {0: 5, 1: 1}, {0: 9, 1: 9, 2: 9}])
def test_history_weights_alpha_0_all_equal(burden):
    """alpha = 0: every weight is 1 whatever the burden, so HWF(0) is SF."""
    assert history_weights(Fraction(0), burden, VOLUNTEERS) == {0: 1, 1: 1, 2: 1}


@pytest.mark.parametrize(
    "alpha, burden",
    [
        pytest.param(Fraction(-1), {}, id="negative_alpha"),
        pytest.param(Fraction(1), {0: -1}, id="negative_burden"),
        pytest.param(Fraction(1), {7: 1}, id="unknown_volunteer"),
    ],
)
def test_history_weights_rejects_bad_input(alpha, burden):
    with pytest.raises(ValueError):
        history_weights(alpha, burden, VOLUNTEERS)


# Policy_stages

REF = REPAIR_SLICE_SF
BURDEN = {0: 2, 2: 1}


# Built inside the test (not at import), so a missing objective fails one test
# instead of breaking the collection of the whole file
EXPECTED_STAGES = {
    "QO": lambda: (MAX_QUALITY,),
    "SF": lambda: (min_distance(REF), MAX_QUALITY),
    "QL": lambda: (MAX_QUALITY, min_distance(REF)),
    "WQS": lambda: (max_quality_minus_distance(REF, 2),),
    "HWF": lambda: (min_distance(REF, weights={0: 4, 1: 2, 2: 3}), MAX_QUALITY),
}
ARMS_BY_POLICY = {
    "QO": Arm(QO, None),
    "SF": Arm(SF, CHAINED),
    "QL": Arm(QL, CHAINED),
    "WQS": Arm(WQS, CHAINED, lam=2),
    "HWF": Arm(HWF, CHAINED, alpha=Fraction(1, 2)),
}


@pytest.mark.parametrize("policy", list(EXPECTED_STAGES))
def test_policy_stages_per_policy(repair_slice, policy):
    """Table 1. HWF: alpha = 1/2 and b = (2, 0, 1) -> weights 2 + b = (4, 2, 3)."""
    arm = ARMS_BY_POLICY[policy]
    reference = None if arm.policy == QO else REF

    assert policy_stages(arm, reference, BURDEN, repair_slice) == EXPECTED_STAGES[policy]()


@pytest.mark.parametrize("arm", ALL_ARMS, ids=lambda a: a.name)
def test_stages_contain_no_coverage_or_tiebreak(repair_slice, arm):
    """The runner adds the coverage cap and the tie-break; a policy never does."""
    reference = None if arm.policy == QO else REF
    names = [o.name for o in policy_stages(arm, reference, {}, repair_slice)]

    assert MIN_SHORTFALL.name not in names
    assert MAX_TIE_BREAK.name not in names
    assert names  # at least one stage


@pytest.mark.parametrize(
    "arm, reference",
    [
        pytest.param(Arm(SF, CHAINED), None, id="SF_without_reference"),
        pytest.param(Arm(QO, None), REF, id="QO_with_reference"),
        pytest.param(Arm(SF, CHAINED, lam=1), REF, id="invalid_arm"),
    ],
)
def test_policy_stages_rejects_bad_reference_or_arm(repair_slice, arm, reference):
    with pytest.raises(ValueError):
        policy_stages(arm, reference, {}, repair_slice)
