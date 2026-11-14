"""Test the analysis tables, confidence intervals and H1-H3 decisions.

The artificial tables have known effects, so the expected decision does
not depend on a scheduling solver. Zero-effect checks with noise use
several seeds because a 95% interval can exclude zero by chance.
"""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from repair import analysis
from repair.model import StageStatus, Step, Trajectory
from tests.fake_results import ARMS, PROFILES, planted_instance_table, planted_rq2_table
from tests.test_metrics import (
    LIMIT,
    QO_ARM,
    SF_CHAINED,
    _fillers,
    _line,
    _records,
    _result,
    _steps,
)

N, SD = 60, 1.0  # instances per planted table; standard error of a mean ~ 0.13
NOISE_SEEDS = range(20)  # zero-effect checks run on several seeds (see _mostly)


def _mostly(verdicts, expected="inconclusive", at_least=16):
    """Check the expected verdict across several seeds.

    Even with no population effect, a 95% interval can exclude zero. Requiring
    at least 16 of 20 seeds to agree avoids relying on one favourable draw.
    """
    verdicts = list(verdicts)
    assert sum(v == expected for v in verdicts) >= at_least, verdicts

# Verdict and bootstrap


@pytest.mark.parametrize(
    "ci, sign, expected",
    [
        pytest.param((1.0, 0.5, 1.5), +1, "supported", id="positive_predicted_positive"),
        pytest.param((-1.0, -1.5, -0.5), +1, "contradicted", id="negative_predicted_positive"),
        pytest.param((0.1, -0.5, 0.7), +1, "inconclusive", id="contains_zero"),
        pytest.param((-1.0, -1.5, -0.5), -1, "supported", id="negative_predicted_negative"),
        pytest.param((1.0, 0.5, 1.5), -1, "contradicted", id="positive_predicted_negative"),
        pytest.param((0.5, 0.0, 1.0), +1, "inconclusive", id="lower_exactly_zero"),
        pytest.param((-0.5, -1.0, 0.0), +1, "inconclusive", id="upper_exactly_zero"),
        pytest.param((-0.5, -1.0, 0.0), -1, "inconclusive", id="upper_zero_predicted_negative"),
        pytest.param((0.0, 0.0, 0.0), +1, "inconclusive", id="all_zero"),
        pytest.param((0.0, 0.0, 0.0), -1, "inconclusive", id="all_zero_predicted_negative"),
    ],
)
def test_verdict_three_cases(ci, sign, expected):
    assert analysis.verdict(ci, sign) == expected


@pytest.mark.parametrize(
    "ci, sign",
    [((0.0, 1.0, -1.0), +1), ((0.0, -1.0, 1.0), 0), ((0.0, -1.0, 1.0), 2)],
    ids=["lower_above_upper", "sign_0", "sign_2"],
)
def test_verdict_rejects_bad_input(ci, sign):
    with pytest.raises(ValueError):
        analysis.verdict(ci, sign)


def test_bootstrap_ci_seeded_and_constant_data():
    """Same seed -> same interval; the mean is the sample mean; the interval contains it;
    constant data -> a zero-width interval at the constant.
    """
    values = list(np.random.default_rng(1).normal(2.0, 1.0, 50))

    first = analysis.bootstrap_ci(values, seed=3)
    assert analysis.bootstrap_ci(values, seed=3) == first
    assert analysis.bootstrap_ci(values, seed=4) != first
    mean, lower, upper = first
    assert mean == pytest.approx(np.mean(values))
    assert lower < mean < upper
    assert upper - lower == pytest.approx(2 * 1.96 * np.std(values) / math.sqrt(50), rel=0.25)

    assert analysis.bootstrap_ci([4.0] * 10) == (4.0, 4.0, 4.0)


def test_bootstrap_ci_level_widens_interval():
    values = list(np.random.default_rng(2).normal(0.0, 1.0, 40))
    _, lo90, hi90 = analysis.bootstrap_ci(values, level=0.90)
    _, lo99, hi99 = analysis.bootstrap_ci(values, level=0.99)

    assert lo99 < lo90 and hi90 < hi99


@pytest.mark.parametrize(
    "values, level", [([], 0.95), ([1.0, 2.0], 1.0), ([1.0, 2.0], 0.0)],
    ids=["empty", "level_1", "level_0"],
)
def test_bootstrap_ci_rejects_bad_input(values, level):
    with pytest.raises(ValueError):
        analysis.bootstrap_ci(values, level=level)


# Certified prefix


def _fifteen(qo_records=None, sf_steps=15):
    """A 15-step result with a QO and an SF arm, all stages optimal unless changed."""
    instance = _line()
    s0 = frozenset({(0, 0)})
    qo = Trajectory(
        QO_ARM,
        _steps([s0] * 15, records=qo_records or (lambda t: _records(("MaxQuality", "optimal")))),
    )
    sf = Trajectory(SF_CHAINED, _steps([s0] * sf_steps), ended_early=sf_steps < 15)
    return _result(instance, s0, [qo, sf], _fillers(15, shift=3), lambda_star=(9,) * 15)


def test_certified_prefix_cases():
    """all optimal -> 15; a time limit at t = 7 in any arm -> 6; a coverage timeout at
    t = 3 -> 2; an arm that ended early after step 9 -> 9.
    """
    assert analysis.certified_prefix(_fifteen()) == 15

    limit_at_7 = _fifteen(
        qo_records=lambda t: _records(("MaxQuality", LIMIT if t == 7 else StageStatus.OPTIMAL))
    )
    assert analysis.certified_prefix(limit_at_7) == 6

    full = _fifteen()
    coverage = list(full.coverage)
    coverage[2] = replace(coverage[2], record=replace(coverage[2].record, status=LIMIT))
    assert analysis.certified_prefix(replace(full, coverage=tuple(coverage))) == 2

    assert analysis.certified_prefix(_fifteen(sf_steps=9)) == 9


def test_certified_prefix_counts_the_tiebreak_stage():
    """Every stage counts, also the last one (an unproven tie-break at t = 1 -> 0)."""
    full = _fifteen()
    sf = full.trajectories[1]
    first = Step(1, sf.steps[0].schedule, _records(("MinDistance", StageStatus.OPTIMAL),
                                                    ("MaxQuality", StageStatus.OPTIMAL),
                                                    ("MaxTieBreak", LIMIT)))
    broken = replace(sf, steps=(first, *sf.steps[1:]))

    assert analysis.certified_prefix(replace(full, trajectories=(full.trajectories[0], broken))) == 0


# Sequence and instance tables


def test_sequence_table_rows_from_results():
    """One row per (result, arm); eligible_primary if and only if certified through T; metrics are
    cut at the prefix.
    """
    certified = _fifteen()
    cut = _fifteen(sf_steps=9)

    seq = analysis.sequence_table([certified, cut])

    assert len(seq) == 4
    assert set(analysis.METRICS) <= set(seq.columns)
    assert list(seq["prefix"]) == [15, 15, 9, 9]
    assert list(seq["eligible_primary"]) == [True, True, False, False]
    assert set(seq["arm"]) == {"QO", "SF-chained"}
    assert set(seq["profile"]) == {"balanced"}
    assert (seq["sum_C"] == 0).all()  # nothing ever changes in these trajectories
    assert seq["spearman_burden"].isna().all()  # constant (all-zero) burden vectors


def _seq_rows(rows):
    """A sequence table by hand: every METRICS column 0 unless given."""
    base = {m: 0.0 for m in analysis.METRICS}
    return pd.DataFrame([{**base, "under_supplied": False, "horizon": 15, **r} for r in rows])


def test_instance_table_averages_sequences():
    """Two sequences of instance 1 are averaged; with primary=True the uncertified
    sequence of instance 2 is dropped (and with it the instance).
    """
    key = {"cell": "main", "size_class": 20, "profile": "balanced", "arm": "SF-chained"}
    seq = _seq_rows([
        {**key, "instance": 1, "sequence": 0, "prefix": 15, "eligible_primary": True,
         "sum_g": 2.0, "spearman_burden": 0.5},
        {**key, "instance": 1, "sequence": 1, "prefix": 15, "eligible_primary": True,
         "sum_g": 4.0, "spearman_burden": float("nan")},  # constant burden vectors
        {**key, "instance": 2, "sequence": 0, "prefix": 7, "eligible_primary": False,
         "sum_g": 9.0},
    ])

    primary = analysis.instance_table(seq)
    everything = analysis.instance_table(seq, primary=False)

    assert list(primary["instance"]) == [1]
    assert primary["sum_g"].iloc[0] == 3.0 and primary["n_sequences"].iloc[0] == 2
    # the undefined Spearman is skipped, and how many values the mean rests on is kept
    assert primary["spearman_burden"].iloc[0] == 0.5 and primary["n_spearman"].iloc[0] == 1
    assert list(everything["instance"]) == [1, 2]
    assert list(everything["sum_g"]) == [3.0, 9.0]


def test_paired_differences_matches_instances():
    """a - b per instance; an instance with only one of the arms is dropped."""
    inst = planted_instance_table(5, {}, SD, seed=0)
    inst = inst[~((inst["instance"] == 4) & (inst["arm"] == "QL-chained"))]

    diffs = analysis.paired_differences(inst, "SF-chained", "QL-chained", "sum_g")

    assert set(diffs.columns) >= {"size_class", "profile", "instance", "diff"}
    assert sorted(set(diffs["instance"])) == [0, 1, 2, 3]
    row = diffs[(diffs["instance"] == 0) & (diffs["profile"] == "balanced")]
    sf = inst[(inst["instance"] == 0) & (inst["profile"] == "balanced") & (inst["arm"] == "SF-chained")]
    ql = inst[(inst["instance"] == 0) & (inst["profile"] == "balanced") & (inst["arm"] == "QL-chained")]
    assert row["diff"].iloc[0] == pytest.approx(sf["sum_g"].iloc[0] - ql["sum_g"].iloc[0])


# The fake tables themselves


def test_fake_table_has_instance_table_format_and_planted_effect():
    """The helper's output looks like instance_table's, and the planted contrast is
    there: its mean is within 4 standard errors of the effect.
    """
    inst = planted_instance_table(N, {"h1_gap": 2.0}, SD, seed=0, size_classes=(20, 40))

    assert set(analysis.INSTANCE_KEYS) | {"arm", "n_sequences"} | set(analysis.METRICS) <= set(inst.columns)
    assert len(inst) == 2 * len(PROFILES) * N * len(ARMS)
    gap = analysis.paired_differences(inst, "SF-chained", "QL-chained", "sum_g")
    wide = gap.pivot_table(index=["size_class", "instance"], columns="profile", values="diff")
    contrast = wide["preference"] - wide["availability"]
    assert abs(contrast.mean() - 2.0) < 4 * SD / math.sqrt(len(contrast))


# H1


def _h1(effects, seed=0):
    result = analysis.evaluate_h1(planted_instance_table(N, effects, SD, seed=seed), seed=seed)
    return {row.component: row for row in result.itertuples()}


def _h1_effects(gap, churn, size=2.0, **extra):
    """Planted H1 effects relative to the prediction: +1 = as H1_EXPECTED predicts,
    -1 = the opposite, 0 = none. So these tests follow H1_EXPECTED if it changes.
    """
    expected = analysis.H1_EXPECTED
    return {"h1_gap": gap * size * expected["gap"],
            "h1_churn": churn * size * expected["churn"], **extra}


def test_h1_planted_supported():
    """Both components planted positive -> both and the joint verdict supported, and
    the interval covers the planted value.
    """
    rows = _h1(_h1_effects(+1, +1))
    planted = 2.0 * analysis.H1_EXPECTED["gap"]

    assert rows["gap"].verdict == rows["churn"].verdict == rows["joint"].verdict == "supported"
    assert rows["gap"].lower < planted < rows["gap"].upper
    assert rows["churn"].n == N


def test_h1_compares_preference_with_availability():
    """A large offset at "balanced" must not matter: the contrast is preference minus
    availability (preference minus balanced would be contradicted).
    """
    offset = -10.0 * analysis.H1_EXPECTED["gap"]  # pref - balanced would point the wrong way
    rows = _h1(_h1_effects(+1, +1, h1_gap_balanced=offset, h1_churn_balanced=offset))
    planted = 2.0 * analysis.H1_EXPECTED["gap"]

    assert rows["joint"].verdict == "supported"
    assert rows["gap"].lower < planted < rows["gap"].upper


def test_h1_follows_h1_expected(monkeypatch):
    """If the exposé's H1 sentence predicts the opposite direction, flipping H1_EXPECTED
    (and with it PREDICTED_SIGN) must flip what counts as support: no hard-coded +1.
    """
    flipped = {"gap": -1, "churn": -1}
    monkeypatch.setattr(analysis, "H1_EXPECTED", flipped)
    monkeypatch.setitem(analysis.PREDICTED_SIGN, "h1_gap", -1)
    monkeypatch.setitem(analysis.PREDICTED_SIGN, "h1_churn", -1)

    rows = _h1(_h1_effects(+1, +1))  # now planted negative, as the flipped H1 predicts

    assert rows["joint"].verdict == "supported"


def test_h1_flipped_contradicted():
    """Both components planted negative -> contradicted."""
    rows = _h1(_h1_effects(-1, -1))

    assert rows["gap"].verdict == rows["churn"].verdict == rows["joint"].verdict == "contradicted"


def test_h1_noise_inconclusive():
    """No effect -> inconclusive (planted noise only), on most seeds."""
    runs = [_h1({}, seed=seed) for seed in NOISE_SEEDS]

    _mostly(r["gap"].verdict for r in runs)
    _mostly(r["churn"].verdict for r in runs)
    _mostly(r["joint"].verdict for r in runs)


@pytest.mark.parametrize(
    "gap, churn, joint",
    [
        pytest.param(+1, 0, "inconclusive", id="gap_only"),
        pytest.param(0, +1, "inconclusive", id="churn_only"),
    ],
)
def test_h1_needs_both_components(gap, churn, joint):
    """Both components are needed for H1 support; use several seeds for the noisy component."""
    _mostly((_h1(_h1_effects(gap, churn), seed=seed)["joint"].verdict for seed in NOISE_SEEDS),
            joint)


def test_h1_supported_plus_contradicted_is_contradicted():
    """One component supported, the other contradicted: the joint prediction is
    contradicted, never supported.
    """
    rows = _h1(_h1_effects(+1, -1))

    assert (rows["gap"].verdict, rows["churn"].verdict) == ("supported", "contradicted")
    assert rows["joint"].verdict == "contradicted"


def test_h1_per_size_class():
    inst = planted_instance_table(N, _h1_effects(+1, +1), SD, seed=0, size_classes=(20, 40))
    result = analysis.evaluate_h1(inst)

    assert sorted(set(result["size_class"])) == [20, 40]
    assert len(result) == 2 * 3


# H2, H3


def test_h2_planted_per_profile():
    """HWF lowers the Gini in balanced, does nothing in availability and raises it in
    preference: supported / inconclusive / contradicted, one row per profile.
    """
    effects = {"h2": {"balanced": -0.5, "availability": 0.0, "preference": 0.5}}
    runs = []
    for seed in NOISE_SEEDS:
        result = analysis.evaluate_h2(planted_instance_table(N, effects, SD / 4, seed=seed))
        verdicts = dict(zip(result["profile"], result["verdict"]))
        assert verdicts["balanced"] == "supported" and verdicts["preference"] == "contradicted"
        runs.append(verdicts["availability"])

    _mostly(runs)
    assert len(result) == 3 and (result["n"] == N).all()
    assert {"total_burden_diff", "sum_g_diff"} <= set(result.columns)


def test_h3_planted():
    """D planted positive -> supported; negative -> contradicted; zero -> inconclusive."""
    for effect, expected in [(3.0, "supported"), (-3.0, "contradicted")]:
        result = analysis.evaluate_h3(planted_rq2_table(N, effect, SD, seed=1))
        assert list(result["verdict"]) == [expected], effect
        assert result["n"].iloc[0] == N

    _mostly(
        analysis.evaluate_h3(planted_rq2_table(N, 0.0, SD, seed=seed))["verdict"].iloc[0]
        for seed in NOISE_SEEDS
    )


def test_h3_uses_only_certified_sequences():
    rq2 = planted_rq2_table(N, 3.0, SD, seed=1)
    rq2.loc[rq2["instance"] < 10, "eligible_primary"] = False

    assert analysis.evaluate_h3(rq2)["n"].iloc[0] == N - 10


# Burden sensitivity and precision


def test_burden_sensitivity_identical_vectors():
    """With identical burden variants (gini_cells = gini, rank agreement 1), the repeated
    H2 contrast equals the main one exactly.
    """
    inst = planted_instance_table(N, {"h2": -0.5}, SD / 4, seed=0)
    inst["spearman_burden"] = 1.0

    out = analysis.burden_sensitivity(inst)
    main = analysis.evaluate_h2(inst)

    assert set(out) == {"agreement", "h2_cells"}
    assert (out["agreement"]["spearman"] == 1.0).all()
    assert set(out["agreement"]["arm"]) == set(ARMS)
    pd.testing.assert_frame_equal(
        out["h2_cells"].reset_index(drop=True), main.reset_index(drop=True)
    )


def test_burden_agreement_reports_undefined_instances():
    """Instances without a defined Spearman (constant burden vectors) are counted, not
    silently averaged away; the mean uses the defined ones only.
    """
    inst = planted_instance_table(10, {}, SD, seed=0)
    sf = inst["arm"] == "SF-chained"
    inst.loc[sf, "spearman_burden"] = 0.8
    inst.loc[sf & (inst["instance"] < 3), "spearman_burden"] = float("nan")

    agreement = analysis.burden_sensitivity(inst)["agreement"]
    row = agreement[(agreement["arm"] == "SF-chained") & (agreement["profile"] == "balanced")]

    assert {"spearman", "n", "n_undefined"} <= set(agreement.columns)
    assert row["spearman"].iloc[0] == pytest.approx(0.8)
    assert (row["n"].iloc[0], row["n_undefined"].iloc[0]) == (7, 3)


def test_precision_plan_385():
    """sd_dif and only if = pooled SD -> ceil(1.96^2 / 0.1^2) = ceil(384.16) = 385."""
    assert analysis.required_instances(1.0, 1.0) == 385
    assert analysis.required_instances(2.0, 1.0) == 1537  # ceil(4 * 384.16)
    assert analysis.required_instances(1.0, 1.0, fraction=0.2) == 97  # ceil(96.04)


@pytest.mark.parametrize("args", [(0.0, 1.0), (1.0, 0.0), (1.0, 1.0, 0.0)])
def test_required_instances_rejects_non_positive(args):
    with pytest.raises(ValueError):
        analysis.required_instances(*args)


def test_precision_plan_uses_difference_of_differences_sd():
    """The sample-size calculation must use the spread of the paired contrast.

    A large instance term appears in both profiles. It increases the pooled
    SD but cancels when taking their difference. Using that pooled SD as the
    contrast SD would therefore ask for far too many instances here.
    """
    inst = planted_instance_table(40, {}, SD, seed=0)
    common = np.random.default_rng(9).normal(0.0, 10.0, 40)
    rows = (inst["arm"] == "SF-chained") & inst["profile"].isin(["preference", "availability"])
    inst.loc[rows, "sum_g"] += common[inst.loc[rows, "instance"].to_numpy()]

    plan = analysis.precision_plan(inst)
    gap = plan[plan["contrast"] == "h1_gap"].iloc[0]

    diff = analysis.paired_differences(inst, "SF-chained", "QL-chained", "sum_g")
    wide = diff.pivot_table(index="instance", columns="profile", values="diff")
    did = wide["preference"] - wide["availability"]
    per_profile = math.sqrt((wide["preference"].std() ** 2 + wide["availability"].std() ** 2) / 2)

    assert gap["sd_diff"] == pytest.approx(did.std(ddof=1))
    assert gap["pooled_sd"] == pytest.approx(per_profile)
    assert gap["sd_diff"] < gap["pooled_sd"] / 5  # the covariance between profiles counts
    assert gap["n_required"] == analysis.required_instances(gap["sd_diff"], gap["pooled_sd"])
    assert gap["n_required"] < 50


def test_precision_plan_takes_the_most_demanding_contrast():
    """One row per contrast and size class, n_required from the table's own SDs, and the
    "max" row is the largest of them.
    """
    inst = planted_instance_table(30, {"h1_gap": 1.0}, SD, seed=0)

    plan = analysis.precision_plan(inst)

    contrasts = set(plan["contrast"])
    assert {"h1_gap", "h1_churn", "max"} | {f"h2_{p}" for p in PROFILES} == contrasts
    real = plan[plan["contrast"] != "max"]
    for row in real.itertuples():
        assert row.n_required == analysis.required_instances(row.sd_diff, row.pooled_sd)
    assert plan[plan["contrast"] == "max"]["n_required"].iloc[0] == real["n_required"].max()
