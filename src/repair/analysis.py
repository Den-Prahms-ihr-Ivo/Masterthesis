"""Build analysis tables from saved results (exposé sections 5 and 4.6).

First calculate one row per sequence and arm. Then average the sequences of
each base instance before comparing arms. The base instance is the analysis
unit; treating its sequences as independent observations would overstate the
sample size. Paired differences then enter the bootstrap and verdict functions.

The primary analysis uses sequences certified through T. The secondary tables
also include shorter certified prefixes and mark them. Confidence intervals
come from a seeded percentile bootstrap over base instances (DESIGN.md 13.3).

Contrasts (per size class; H2 also per profile). PREDICTED_SIGN fixes their direction:
    H1 gap    [sum_g(SF) - sum_g(QL)]  at preference  minus  at availability   H1_EXPECTED
    H1 churn  [sum_C(QL) - sum_C(SF)]  at preference  minus  at availability   H1_EXPECTED
    H1 joint  intersection-union rule: supported if and only if both components are supported;
              contradicted if and only if at least one is contradicted; inconclusive otherwise.
              Support needs no multiplicity correction (an IU test keeps its level), and
              one supported component is not enough when the other is unsupported.
    H2        G_T(HWF) - G_T(SF) per profile                                    < 0
    H3        D_i (anchored minus chained, RQ2)                                  > 0

The expected H1 mechanism is that P3 changes quality without changing feasibility.
QL can then choose a better schedule, while chained SF keeps its previous one
if it is still allowed. P2 can force both policies to repair assignments, which
may reduce their difference. This motivates comparing preference-dominated
(3/3/9) and availability-dominated (3/9/3) profiles; it is not guaranteed at
every step, and the experiment tests whether the expected difference appears.

Verdicts compare the interval with 0. If the frozen exposé's falsification rule has a
materiality threshold, the comparison must use it instead (TODO IVO: check).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
from typing import Literal

import numpy as np
import pandas as pd

from repair import metrics
from repair.model import SequenceResult, StageStatus

Verdict = Literal["supported", "contradicted", "inconclusive"]

# TODO IVO: set from the exposé's H1 sentence, word for word, then drop this TODO
H1_EXPECTED = {"gap": +1, "churn": +1}
PREDICTED_SIGN = {
    "h1_gap": H1_EXPECTED["gap"],
    "h1_churn": H1_EXPECTED["churn"],
    "h2": -1,
    "h3": +1,
}

# Default arm names (Arm.name of the main arms with lam = 2, alpha = 1)
SF, QL, HWF = "SF-chained", "QL-chained", "HWF-chained-a1"

# Identifier columns of an instance-level row; a pair of arms is matched on these
INSTANCE_KEYS = ("cell", "size_class", "profile", "instance")

# Numeric terminal metrics carried from metrics.terminal_rows into the tables
METRICS = (
    "sum_g", "sum_C", "B", "total_burden", "gini", "max_share", "gini_cells",
    "binding_rate", "spearman_burden",
)


# Verdicts and intervals


def verdict(ci: tuple[float, float, float], predicted_sign: int) -> Verdict:
    """(mean, lower, upper) against the predicted sign (+1 or -1):

    supported     the whole interval lies strictly on the predicted side of 0
    contradicted  the whole interval lies strictly on the other side
    inconclusive  the interval contains 0 (also when it only touches it: the outcomes
                  are integer sums, so a percentile bound can land exactly on 0)

    ValueError for a sign other than +1 / -1 or lower > upper.
    """
    if isinstance(predicted_sign, bool) or predicted_sign not in (-1, 1):
        raise ValueError("predicted_sign must be +1 or -1")
    mean, lower, upper = ci
    if not all(math.isfinite(value) for value in (mean, lower, upper)):
        raise ValueError("interval values must be finite")
    if lower > upper:
        raise ValueError("lower must not exceed upper")
    if lower > 0:
        return "supported" if predicted_sign == 1 else "contradicted"
    if upper < 0:
        return "supported" if predicted_sign == -1 else "contradicted"
    return "inconclusive"


def bootstrap_ci(
    values: Sequence[float], level: float = 0.95, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Estimate a confidence interval for the mean by resampling instances.

    Draw n_boot samples with replacement using default_rng(seed). Return the
    original sample mean and the lower/upper percentile bounds at `level`.
    Reusing the seed gives the same interval. Constant data gives (c, c, c).
    Empty data or a level outside (0, 1) is rejected.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("values must be a nonempty one-dimensional sequence")
    if not np.isfinite(data).all():
        raise ValueError("values must all be finite")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if np.all(data == data[0]):
        constant = float(data[0])
        return constant, constant, constant

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    # Bound temporary memory while retaining one RNG stream for all draws.
    batch_size = max(1, min(n_boot, 1_000_000 // len(data)))
    for start in range(0, n_boot, batch_size):
        stop = min(start + batch_size, n_boot)
        samples = rng.choice(data, size=(stop - start, len(data)), replace=True)
        means[start:stop] = samples.mean(axis=1)
    lower, upper = np.quantile(
        means, [(1 - level) / 2, (1 + level) / 2], method="linear"
    )
    return float(data.mean()), float(lower), float(upper)


# Tables


def certified_prefix(result: SequenceResult) -> int:
    """Count the consecutive steps that are certified for every arm.

    Stop before the first missing or unproven stage, including shared coverage
    and tie-breaking. All arms are cut at this same step for a paired comparison.
    For example, a timeout at step 7 gives a prefix of 6.
    """
    if not result.trajectories:
        return 0
    prefix = 0
    for index in range(len(result.sequence.events)):
        t = index + 1
        if index >= len(result.coverage):
            break
        coverage = result.coverage[index]
        if coverage.t != t or coverage.record.status != StageStatus.OPTIMAL:
            break
        for trajectory in result.trajectories:
            if index >= len(trajectory.steps):
                return prefix
            step = trajectory.steps[index]
            if (
                step.t != t
                or not step.stages
                or any(stage.status != StageStatus.OPTIMAL for stage in step.stages)
            ):
                return prefix
        prefix = t
    return prefix


def sequence_table(results: Iterable[SequenceResult]) -> pd.DataFrame:
    """One row per (result, arm) with columns:

    cell, size_class (instance.meta.size_class), instance (instance.meta.seed),
    under_supplied, profile (sequence.profile value), sequence (sequence.seed),
    arm (Arm.name), horizon (len(events)), prefix (certified_prefix),
    eligible_primary (prefix == horizon), every column in METRICS.

    The metrics are metrics.terminal_rows(result, until=prefix) (NaN when prefix = 0);
    spearman_burden is the Spearman rank correlation of the two burden vectors
    (burden_repairs vs. burden_cells; NaN if either is constant). RQ2 results are not
    included here (see rq2_table).
    """
    columns = [
        "cell", "size_class", "instance", "under_supplied", "profile", "sequence",
        "arm", "horizon", "prefix", "eligible_primary", *METRICS,
    ]
    rows = []
    for result in results:
        if result.cell == "rq2":
            continue
        prefix = certified_prefix(result)
        horizon = len(result.sequence.events)
        terminal = metrics.terminal_rows(result, until=prefix) if prefix else None
        for index, trajectory in enumerate(result.trajectories):
            row = {
                "cell": result.cell,
                "size_class": result.instance.meta.size_class,
                "instance": result.instance.meta.seed,
                "under_supplied": result.instance.meta.under_supplied,
                "profile": _enum_value(result.sequence.profile),
                "sequence": result.sequence.seed,
                "arm": trajectory.arm.name,
                "horizon": horizon,
                "prefix": prefix,
                "eligible_primary": prefix == horizon,
                **dict.fromkeys(METRICS, np.nan),
            }
            if terminal is not None:
                values = terminal[index]
                for metric in METRICS:
                    if metric != "spearman_burden":
                        row[metric] = _as_float(values[metric])
                row["spearman_burden"] = _spearman(
                    values["burden_repairs"], values["burden_cells"]
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def rq2_table(results: Iterable[SequenceResult]) -> pd.DataFrame:
    """One row per RQ2 result: cell, size_class, instance, sequence, prefix,
    eligible_primary, and metrics.rq2_row's D, D_pre, P, lock_in, delta_u_6 (P as float,
    NaN for None). Non-RQ2 results are skipped.
    """
    columns = [
        "cell", "size_class", "instance", "sequence", "prefix", "eligible_primary",
        "D", "D_pre", "P", "lock_in", "delta_u_6",
    ]
    rows = []
    for result in results:
        if result.cell != "rq2":
            continue
        horizon = len(result.sequence.events)
        if horizon != 15:
            raise ValueError("RQ2 requires a 15-event sequence")
        prefix = certified_prefix(result)
        row = {
            "cell": result.cell,
            "size_class": result.instance.meta.size_class,
            "instance": result.instance.meta.seed,
            "sequence": result.sequence.seed,
            "prefix": prefix,
            "eligible_primary": prefix == horizon,
            **dict.fromkeys(("D", "D_pre", "P", "lock_in", "delta_u_6"), np.nan),
        }
        # An early stop has no complete D or P. Keep its metadata for exclusions.
        if result.trajectories and all(len(t.steps) >= 15 for t in result.trajectories):
            values = metrics.rq2_row(result)
            row.update({key: values[key] for key in ("D", "D_pre", "lock_in", "delta_u_6")})
            row["P"] = _as_float(values["P"])
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def instance_table(seq: pd.DataFrame, primary: bool = True) -> pd.DataFrame:
    """Average the METRICS over sequences within (INSTANCE_KEYS, arm).

    primary=True keeps only eligible_primary rows first. Adds n_sequences (rows
    averaged) and n_spearman (rows with a defined spearman_burden; the mean skips NaN).
    One row per (cell, size_class, profile, instance, arm), sorted by them.
    """
    keys = [*INSTANCE_KEYS, "arm"]
    columns = [*keys, *METRICS, "n_sequences", "n_spearman"]
    if seq.empty:
        return pd.DataFrame(columns=columns)
    selected = seq.loc[seq["eligible_primary"].eq(True)].copy() if primary else seq.copy()
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected[list(METRICS)] = selected[list(METRICS)].astype(float)
    grouped = selected.groupby(keys, sort=True, dropna=False, observed=True)
    means = grouped[list(METRICS)].mean()
    means["n_sequences"] = grouped.size()
    means["n_spearman"] = grouped["spearman_burden"].count()
    return means.reset_index().loc[:, columns]


def paired_differences(
    inst: pd.DataFrame,
    arm_a: str,
    arm_b: str,
    metric: str,
    by: tuple[str, ...] = ("size_class", "profile"),
) -> pd.DataFrame:
    """Subtract arm_b from arm_a for the same base instance.

    Match on all INSTANCE_KEYS, so different cells and profiles cannot be mixed.
    Drop an instance if one arm is missing. The result columns are the grouping
    keys, instance and diff.
    """
    if len(set(by)) != len(by) or any(key not in INSTANCE_KEYS[:-1] for key in by):
        raise ValueError("by must contain distinct INSTANCE_KEYS other than instance")
    pairs = _arm_pairs(inst, arm_a, arm_b, metric)
    columns = [*by, "instance", "diff"]
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    pairs["diff"] = pairs["a"] - pairs["b"]
    return pairs.loc[:, columns].reset_index(drop=True)


# Hypotheses


def evaluate_h1(
    inst: pd.DataFrame, sf: str = SF, ql: str = QL, seed: int = 0
) -> pd.DataFrame:
    """Per size class of cell "main": three rows, component "gap", "churn", "joint".

    gap:   per instance, [sum_g(sf) - sum_g(ql)] at "preference" minus at "availability"
    churn: per instance, [sum_C(ql) - sum_C(sf)] at "preference" minus at "availability"
    Columns: size_class, component, mean, lower, upper, n, verdict (the joint row has
    NaN for mean / lower / upper / n and the joint rule of the module docstring).
    """
    columns = ["size_class", "component", "mean", "lower", "upper", "n", "verdict"]
    main = _main(inst)
    if main.empty:
        return pd.DataFrame(columns=columns)
    contrasts = {
        "gap": _h1_pairs(main, sf, ql, "sum_g"),
        "churn": _h1_pairs(main, ql, sf, "sum_C"),
    }
    rows = []
    for size in sorted(main["size_class"].unique()):
        outcomes = []
        for component, pairs in contrasts.items():
            selected = pairs.loc[pairs["size_class"] == size]
            summary = _summarize(
                selected["preference"] - selected["availability"],
                PREDICTED_SIGN[f"h1_{component}"], seed,
            )
            outcomes.append(summary["verdict"])
            rows.append({"size_class": size, "component": component, **summary})
        joint = (
            "supported" if all(outcome == "supported" for outcome in outcomes)
            else "contradicted" if "contradicted" in outcomes
            else "inconclusive"
        )
        rows.append({
            "size_class": size, "component": "joint", "mean": np.nan,
            "lower": np.nan, "upper": np.nan, "n": np.nan, "verdict": joint,
        })
    return pd.DataFrame(rows, columns=columns)


def evaluate_h2(
    inst: pd.DataFrame, hwf: str = HWF, sf: str = SF, seed: int = 0
) -> pd.DataFrame:
    """Per size class and profile of cell "main": gini(hwf) - gini(sf), predicted < 0.

    Columns: size_class, profile, mean, lower, upper, n, verdict, and alongside (no
    verdict) total_burden_dif and only if and sum_g_dif and only if (mean paired differences hwf - sf).
    """
    return _evaluate_h2(inst, hwf, sf, "gini", seed)


def evaluate_h3(rq2: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Per size class: D averaged within instance (eligible_primary rows only), then the
    bootstrap over instances, predicted > 0. Columns: size_class, mean, lower, upper, n,
    verdict.
    """
    columns = ["size_class", "mean", "lower", "upper", "n", "verdict"]
    if rq2.empty:
        return pd.DataFrame(columns=columns)
    selected = rq2.loc[rq2["eligible_primary"].eq(True)].copy()
    selected["D"] = selected["D"].astype(float)
    per_instance = selected.groupby(
        ["size_class", "instance"], sort=True, observed=True
    )["D"].mean().reset_index()
    rows = []
    for size in sorted(rq2["size_class"].unique()):
        values = per_instance.loc[per_instance["size_class"] == size, "D"]
        rows.append({"size_class": size, **_summarize(values, PREDICTED_SIGN["h3"], seed)})
    return pd.DataFrame(rows, columns=columns)


def burden_sensitivity(
    inst: pd.DataFrame, hwf: str = HWF, sf: str = SF, seed: int = 0
) -> dict[str, pd.DataFrame]:
    """Exposé 4.6, a sensitivity check that never changes the H2 verdict:

    "agreement": per size class, profile and arm, the mean spearman_burden over the
                 instances where it is defined (do both burden counts rank the
                 volunteers alike?), with the counts it rests on; columns size_class,
                 profile, arm, spearman, n (instances with a value), n_undefined
                 (instances whose burden vectors were constant in every sequence).
                 Constant vectors are common (SF at n = 20 leaves most volunteers at
                 0), so the count is reported, never silently averaged away. Same
                 horizon as the Gini comparison: terminal values of certified sequences.
                 Descriptive only: no Fisher z or similar.
    "h2_cells":  evaluate_h2 repeated with gini_cells in place of gini (same columns)
    """
    main = _main(inst)
    columns = ["size_class", "profile", "arm", "spearman", "n", "n_undefined"]
    rows = []
    if not main.empty:
        for (size, profile, arm), group in main.groupby(
            ["size_class", "profile", "arm"], sort=True, observed=True
        ):
            values = group["spearman_burden"].astype(float)
            rows.append({
                "size_class": size, "profile": profile, "arm": arm,
                "spearman": float(values.mean()), "n": int(values.count()),
                "n_undefined": int(values.isna().sum()),
            })
    return {
        "agreement": pd.DataFrame(rows, columns=columns),
        "h2_cells": _evaluate_h2(inst, hwf, sf, "gini_cells", seed),
    }


# Precision plan


def required_instances(
    sd_diff: float, pooled_sd: float, fraction: float = 0.1, z: float = 1.96
) -> int:
    """Calculate ceil((z * sd_dif and only if / half_width)^2).

    half_width = fraction * pooled_sd sets the precision target. sd_dif and only if is the
    spread of the paired contrast itself and determines the sample size. These
    two standard deviations have different roles. With the defaults and equal
    SDs, the result is 385. Both SDs and fraction must be positive.
    """
    if not all(math.isfinite(value) and value > 0 for value in (sd_diff, pooled_sd, fraction, z)):
        raise ValueError("SDs, fraction and z must be finite and positive")
    return math.ceil((z * sd_diff / (fraction * pooled_sd)) ** 2)


def precision_plan(pilot_inst: pd.DataFrame, fraction: float = 0.1) -> pd.DataFrame:
    """Required N_I per contrast and size class from a pilot instance table.

    Contrasts: H1 gap and churn (their per-instance difference of differences), H2 per
    profile. sd_dif and only if = sample SD (ddof = 1) of the per-instance contrast. For H1 that is
    sd of (SF - QL)_pref - (SF - QL)_avail per instance: the same instances run both
    profiles, so it includes their covariance and is NOT sqrt(2) times a per-profile
    SD. pooled_sd =
    sqrt((sd_a^2 + sd_b^2) / 2) of the two per-instance quantities whose difference the
    contrast is (DESIGN.md 13.8): for H2 gini(hwf) and gini(sf); for H1 the arm
    difference at "preference" and at "availability".
    Contrast names: "h1_gap", "h1_churn", "h2_<profile>".
    Columns: size_class, contrast, sd_diff, pooled_sd, n_required.
    One extra row per size class with contrast "max": the N_I to plan with.
    """
    if not math.isfinite(fraction) or fraction <= 0:
        raise ValueError("fraction must be finite and positive")
    columns = ["size_class", "contrast", "sd_diff", "pooled_sd", "n_required"]
    main = _main(pilot_inst)
    if main.empty:
        return pd.DataFrame(columns=columns)
    h1_gap = _h1_pairs(main, SF, QL, "sum_g")
    h1_churn = _h1_pairs(main, QL, SF, "sum_C")
    rows = []
    for size in sorted(main["size_class"].unique()):
        size_rows = []
        for name, pairs in (("h1_gap", h1_gap), ("h1_churn", h1_churn)):
            selected = pairs.loc[pairs["size_class"] == size]
            size_rows.append(_precision_row(
                size, name, selected["preference"], selected["availability"], fraction
            ))
        size_data = main.loc[main["size_class"] == size]
        for profile in sorted(size_data["profile"].unique()):
            pairs = _arm_pairs(size_data.loc[size_data["profile"] == profile], HWF, SF, "gini")
            size_rows.append(_precision_row(
                size, f"h2_{profile}", pairs["a"], pairs["b"], fraction
            ))
        rows.extend(size_rows)
        requirements = [row["n_required"] for row in size_rows]
        # An undefined contrast prevents a defensible overall maximum.
        maximum = max(requirements) if all(pd.notna(n) for n in requirements) else np.nan
        rows.append({
            "size_class": size, "contrast": "max", "sd_diff": np.nan,
            "pooled_sd": np.nan, "n_required": maximum,
        })
    return pd.DataFrame(rows, columns=columns)


# Shared calculations


def _enum_value(value):
    return value.value if hasattr(value, "value") else value


def _as_float(value) -> float:
    return np.nan if value is None else float(value)


def _spearman(repairs: Sequence[int], cells: Sequence[int]) -> float:
    if len(repairs) != len(cells):
        raise ValueError("burden vectors must have the same length")
    a, b = pd.Series(repairs, dtype=float), pd.Series(cells, dtype=float)
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    # Spearman is Pearson correlation of ranks; ties get their average rank.
    return float(a.rank(method="average").corr(b.rank(method="average")))


def _main(inst: pd.DataFrame) -> pd.DataFrame:
    return inst if inst.empty else inst.loc[inst["cell"] == "main"].copy()


def _arm_pairs(inst: pd.DataFrame, arm_a: str, arm_b: str, metric: str) -> pd.DataFrame:
    """Match complete arm pairs on every instance key, including the experiment cell."""
    keys = list(INSTANCE_KEYS)
    columns = [*keys, "a", "b"]
    if inst.empty:
        return pd.DataFrame(columns=columns)
    selected = inst.loc[inst["arm"].isin([arm_a, arm_b])]
    if selected.duplicated([*keys, "arm"]).any():
        raise ValueError("duplicate instance/arm rows; call instance_table first")
    a = selected.loc[selected["arm"] == arm_a, [*keys, metric]].rename(columns={metric: "a"})
    b = selected.loc[selected["arm"] == arm_b, [*keys, metric]].rename(columns={metric: "b"})
    pairs = a.merge(b, on=keys, how="inner", validate="one_to_one")
    pairs[["a", "b"]] = pairs[["a", "b"]].astype(float)
    pairs = pairs.dropna(subset=["a", "b"])
    if not np.isfinite(pairs[["a", "b"]].to_numpy()).all():
        raise ValueError("paired outcomes must be finite")
    return pairs.sort_values(keys).reset_index(drop=True).loc[:, columns]


def _h1_pairs(inst: pd.DataFrame, arm_a: str, arm_b: str, metric: str) -> pd.DataFrame:
    columns = ["size_class", "instance", "preference", "availability"]
    if inst.empty:
        return pd.DataFrame(columns=columns)
    selected = inst.loc[inst["profile"].isin(["preference", "availability"])]
    differences = paired_differences(selected, arm_a, arm_b, metric)
    if differences.empty:
        return pd.DataFrame(columns=columns)
    wide = differences.pivot(index=["size_class", "instance"], columns="profile", values="diff")
    wide = wide.reindex(columns=["preference", "availability"]).dropna()
    wide.columns.name = None
    return wide.reset_index().loc[:, columns]


def _summarize(values, sign: int, seed: int) -> dict:
    data = pd.Series(values, dtype=float).dropna().to_numpy()
    if data.size == 0:
        return {"mean": np.nan, "lower": np.nan, "upper": np.nan, "n": 0, "verdict": "inconclusive"}
    ci = bootstrap_ci(data, seed=seed)
    return {"mean": ci[0], "lower": ci[1], "upper": ci[2], "n": len(data), "verdict": verdict(ci, sign)}


def _evaluate_h2(inst: pd.DataFrame, hwf: str, sf: str, metric: str, seed: int) -> pd.DataFrame:
    columns = [
        "size_class", "profile", "mean", "lower", "upper", "n", "verdict",
        "total_burden_diff", "sum_g_diff",
    ]
    main = _main(inst)
    if main.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (size, profile), group in main.groupby(["size_class", "profile"], sort=True, observed=True):
        pairs = _arm_pairs(group, hwf, sf, metric)
        summary = _summarize(pairs["a"] - pairs["b"], PREDICTED_SIGN["h2"], seed)
        row = {"size_class": size, "profile": profile, **summary}
        # Report side outcomes for the same instance sample as the Gini contrast.
        for side in ("total_burden", "sum_g"):
            side_pairs = _arm_pairs(group, hwf, sf, side).merge(
                pairs[list(INSTANCE_KEYS)], on=list(INSTANCE_KEYS), how="inner", validate="one_to_one"
            )
            row[f"{side}_diff"] = float((side_pairs["a"] - side_pairs["b"]).mean())
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _precision_row(size, name: str, a: pd.Series, b: pd.Series, fraction: float) -> dict:
    a, b = a.astype(float), b.astype(float)
    sd_diff = float((a - b).std(ddof=1)) if len(a) >= 2 else np.nan
    pooled = math.sqrt((float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2) if len(a) >= 2 else np.nan
    n = (
        required_instances(sd_diff, pooled, fraction)
        if math.isfinite(sd_diff) and math.isfinite(pooled) and sd_diff > 0 and pooled > 0
        else np.nan
    )
    return {"size_class": size, "contrast": name, "sd_diff": sd_diff, "pooled_sd": pooled, "n_required": n}
