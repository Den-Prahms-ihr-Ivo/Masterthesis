"""Build artificial result tables with effects that are known in advance.

These tables test the analysis, not the scheduling model. Normal draws may
give values outside the range of a real metric. Clipping them would change
the effect being tested. Use sd=0 when a test needs exactly zero effect and
no random variation; with noise, some intervals can exclude zero by chance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import pandas as pd

from repair.analysis import INSTANCE_KEYS, METRICS

ARMS = ("QO", "SF-chained", "QL-chained", "WQS-chained-l2", "HWF-chained-a1")
PROFILES = ("balanced", "availability", "preference")


def _validate(n_instances: int, sd: float) -> None:
    if not isinstance(n_instances, int) or isinstance(n_instances, bool) or n_instances < 0:
        raise ValueError("n_instances must be a non-negative integer")
    if not math.isfinite(sd) or sd < 0:
        raise ValueError("sd must be finite and non-negative")


def _effect(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("effects must be finite")
    return number


def planted_instance_table(
    n_instances: int,
    effects: Mapping[str, float | Mapping[str, float]],
    sd: float,
    seed: int,
    size_classes: Sequence[int] = (20,),
) -> pd.DataFrame:
    """Create tables with chosen H1 and H2 contrasts.

    Keep the random draw order unchanged when effect parameters change.
    This lets tests compare modified effects with the same background noise.
    Rows are ordered by size, profile, instance and arm.
    """
    _validate(n_instances, sd)
    allowed = {"h1_gap", "h1_churn", "h2", "h1_gap_balanced", "h1_churn_balanced"}
    unknown = set(effects) - allowed
    if unknown:
        raise ValueError(f"unknown effects: {sorted(unknown)}")

    def scalar(name: str) -> float:
        value = effects.get(name, 0.0)
        if isinstance(value, Mapping):
            raise ValueError(f"{name} must be a scalar effect")
        return _effect(value)

    gap_effect = scalar("h1_gap")
    churn_effect = scalar("h1_churn")
    balanced_gap = scalar("h1_gap_balanced")
    balanced_churn = scalar("h1_churn_balanced")
    h2 = effects.get("h2", 0.0)
    if isinstance(h2, Mapping):
        if set(h2) - set(PROFILES):
            raise ValueError("h2 contains an unknown profile")
        h2_effects = {profile: _effect(h2.get(profile, 0.0)) for profile in PROFILES}
    else:
        h2_effects = dict.fromkeys(PROFILES, _effect(h2))

    rng = np.random.default_rng(seed)
    columns = [*INSTANCE_KEYS, "arm", *METRICS, "n_sequences", "n_spearman"]
    rows = []
    for size in size_classes:
        # Each baseline offset belongs to an instance, not to a profile.
        gap_offsets = rng.normal(0.0, sd, size=n_instances)
        churn_offsets = rng.normal(0.0, sd, size=n_instances)
        gap_draws = rng.normal(gap_effect, sd, size=n_instances)
        churn_draws = rng.normal(churn_effect, sd, size=n_instances)

        for profile in PROFILES:
            for instance in range(n_instances):
                values = {
                    arm: {metric: float(rng.normal(0.0, sd)) for metric in METRICS}
                    for arm in ARMS
                }
                sf = values["SF-chained"]
                ql = values["QL-chained"]
                hwf = values["HWF-chained-a1"]

                gap_difference = gap_offsets[instance]
                churn_difference = churn_offsets[instance]
                if profile == "preference":
                    gap_difference += gap_draws[instance]
                    churn_difference += churn_draws[instance]
                elif profile == "balanced":
                    gap_difference += balanced_gap
                    churn_difference += balanced_churn

                sf["sum_g"] = ql["sum_g"] + float(gap_difference)
                ql["sum_C"] = sf["sum_C"] + float(churn_difference)
                hwf["gini"] = sf["gini"] + float(rng.normal(h2_effects[profile], sd))

                for arm in ARMS:
                    values[arm]["gini_cells"] = values[arm]["gini"]
                    rows.append({
                        "cell": "main", "size_class": size, "profile": profile,
                        "instance": instance, "arm": arm, **values[arm],
                        "n_sequences": 1,
                        "n_spearman": 1,  # the synthetic spearman field is defined
                    })
    return pd.DataFrame(rows, columns=columns)


def planted_rq2_table(
    n_instances: int,
    effect: float,
    sd: float,
    seed: int,
    size_classes: Sequence[int] = (20,),
) -> pd.DataFrame:
    """Create one complete RQ2 row per instance, with D ~ Normal(effect, sd).

    Only D represents a planted effect. Other outcomes are placeholders;
    there is no reconstructed schedule trajectory. Persistence is NaN.
    """
    _validate(n_instances, sd)
    effect = _effect(effect)
    rng = np.random.default_rng(seed)
    columns = [
        "cell", "size_class", "instance", "sequence", "prefix", "eligible_primary",
        "D", "D_pre", "P", "lock_in", "delta_u_6",
    ]
    rows = []
    for size in size_classes:
        draws = rng.normal(effect, sd, size=n_instances)
        for instance, draw in enumerate(draws):
            rows.append({
                "cell": "rq2", "size_class": size, "instance": instance,
                "sequence": 0, "prefix": 15, "eligible_primary": True,
                "D": float(draw), "D_pre": 0.0, "P": np.nan,
                "lock_in": False, "delta_u_6": 0.0,
            })
    return pd.DataFrame(rows, columns=columns)
