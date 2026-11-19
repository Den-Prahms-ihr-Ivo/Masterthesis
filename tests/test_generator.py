"""Check that generated instances follow Table A1 and reuse seeds correctly.

Small checks cover structure and reproducibility. The slow tests use many
draws to check distributions with wider tolerances, so normal sampling
variation is unlikely to cause a failure.
"""

from collections import Counter
from dataclasses import replace
from math import floor

import numpy as np
import pytest

from repair import generator
from repair.model import InstanceMeta
from repair.protocol import GeneratorConfig

CFG = GeneratorConfig()
SEEDS = range(5)
N_PERIODS = CFG.n_days * CFG.periods_per_day  # 28


def _instances(size_class=40, seeds=SEEDS, cfg=CFG, **kwargs):
    return [generator.generate_instance(cfg, size_class, s, **kwargs) for s in seeds]


def _has_skill3_available(instance, shift):
    return any(
        v.skill[shift.role] == 3 and shift.period in v.available
        for v in instance.volunteers
    )


# Determinism


def test_same_seed_same_instance():
    """Same (cfg, size class, seed) -> equal Instances; another seed -> another one."""
    a = generator.generate_instance(CFG, 40, 7)
    b = generator.generate_instance(CFG, 40, 7)

    assert a == b
    assert a.meta.seed == 7
    assert generator.generate_instance(CFG, 40, 8) != a


def test_streams_are_reproducible_and_distinct():
    """make_streams: same seed -> same draws; different names -> different draws."""
    first = {name: rng.random(3).tolist() for name, rng in generator.make_streams(3).items()}
    again = {name: rng.random(3).tolist() for name, rng in generator.make_streams(3).items()}

    assert list(first) == list(generator.SUBSTREAMS)
    assert first == again
    assert len({tuple(draws) for draws in first.values()}) == len(generator.SUBSTREAMS)


# Structure


@pytest.mark.parametrize("size_class", [20, 40, 80])
def test_structure_28_periods_5_roles(size_class):
    """28 periods, 5 roles, |W| = n; shifts period-major; demand >= 1, kappa in {1, 2}."""
    for instance in _instances(size_class):
        assert instance.n_periods == 28
        assert instance.n_roles == 5
        assert len(instance.volunteers) == size_class
        assert instance.weights == CFG.weights
        assert instance.meta == InstanceMeta(
            size_class, generator.scale_for(size_class), instance.meta.seed
        )

        slots = [(s.period, s.role) for s in instance.shifts]
        assert slots == sorted(slots)  # period-major (and no duplicates: Instance checks)
        assert 0 < len(instance.shifts) < 28 * 5  # rho = 0.6: neither empty nor full
        assert all(s.demand >= 1 and s.min_skill in {1, 2} for s in instance.shifts)


@pytest.mark.parametrize(
    "size_class, support", [(20, {1, 2}), (40, {2, 3}), (80, {4, 5, 6})]
)
def test_demand_support_per_size_class(size_class, support):
    """Over a few instances the demands cover exactly the support of the class."""
    demands = {s.demand for i in _instances(size_class) for s in i.shifts}

    assert demands == support


@pytest.mark.parametrize("size_class", [10, 30, 160])
def test_demand_draw_rejects_unknown_size_class(size_class):
    with pytest.raises(ValueError):
        generator.demand_draw(size_class, np.random.default_rng(0))


def test_volunteer_field_ranges():
    """3..26 available periods, cap in {10, 11, 12}, skills 0..3, seniority 0..2,
    prefs 0..2, and 5 roles / 28 periods long.
    """
    for instance in _instances(80):
        for v in instance.volunteers:
            assert 3 <= len(v.available) <= 26
            assert v.available <= set(range(N_PERIODS))
            assert v.cap in {10, 11, 12}
            assert v.seniority in {0, 1, 2}
            assert len(v.skill) == len(v.pref_role) == 5
            assert len(v.pref_time) == N_PERIODS
            assert set(v.skill) <= {0, 1, 2, 3}
            assert set(v.pref_role) | set(v.pref_time) <= {0, 1, 2}


def test_availability_count_inside_a_commitment_range():
    """The three ranges are disjoint, so every count lies in exactly one of them."""
    ranges = CFG.availability_ranges

    for instance in _instances(80):
        for v in instance.volunteers:
            assert sum(lo <= len(v.available) <= hi for lo, hi in ranges) == 1


def test_expert_only_where_skill3_available():
    """expert=True only if some volunteer with skill 3 in that role is available in
    that period. Not vacuous: experts do occur.
    """
    instances = _instances(40, seeds=range(10))

    for instance in instances:
        for shift in instance.shifts:
            if shift.expert:
                assert _has_skill3_available(instance, shift), shift

    assert any(s.expert for i in instances for s in i.shifts)


def test_no_expert_without_skill3():
    """skill_shares without skill 3 -> nobody can be an expert -> no expert shifts."""
    cfg = replace(CFG, skill_shares=(0.25, 0.25, 0.5, 0.0))

    for instance in _instances(40, cfg=cfg):
        assert not any(s.expert for s in instance.shifts)


# Size and scale


@pytest.mark.parametrize("size_class, k", [(20, 1), (40, 2), (80, 4)])
def test_scale_for(size_class, k):
    """k = n / 20."""
    assert generator.scale_for(size_class) == k


@pytest.mark.parametrize("size_class", [10, 30, 160])
def test_scale_for_rejects_unknown_size_class(size_class):
    with pytest.raises(ValueError):
        generator.scale_for(size_class)


@pytest.mark.parametrize("phi", [0.8, 0.75, 0.9])
def test_under_supplied_size_and_scale(phi):
    """phi -> floor(40 * phi) volunteers, but demand class 40 and scale 2."""
    cfg = replace(CFG, workforce_factor=phi)
    instance = generator.generate_instance(cfg, 40, 3, under_supplied=True)

    assert generator.n_volunteers_for(40, cfg, under_supplied=True) == floor(40 * phi)
    assert len(instance.volunteers) == floor(40 * phi)
    assert instance.meta == InstanceMeta(40, 2, 3, under_supplied=True)
    assert {s.demand for s in instance.shifts} <= {2, 3}  # the n = 40 demand


def test_workforce_factor_ignored_for_main_classes():
    cfg = replace(CFG, workforce_factor=0.8)

    assert [generator.n_volunteers_for(n, cfg) for n in (20, 40, 80)] == [20, 40, 80]


@pytest.mark.parametrize("size_class", [20, 80])
def test_under_supplied_only_for_class_40(size_class):
    with pytest.raises(ValueError):
        generator.n_volunteers_for(size_class, CFG, under_supplied=True)


@pytest.mark.parametrize(
    "phi",
    [
        pytest.param(1.0, id="default_1_not_under_supplied"),
        pytest.param(1.2, id="above_1"),
        pytest.param(0.0, id="zero"),
        pytest.param(-0.5, id="negative"),
        pytest.param(0.02, id="no_volunteer_left"),  # floor(0.8) = 0
    ],
)
def test_under_supplied_rejects_phi_outside_0_1(phi):
    """Table A1: the under-supplied class has fewer volunteers, and at least one."""
    cfg = replace(CFG, workforce_factor=phi)

    with pytest.raises(ValueError):
        generator.n_volunteers_for(40, cfg, under_supplied=True)
    with pytest.raises(ValueError):
        generator.generate_instance(cfg, 40, 0, under_supplied=True)


# Sub-stream independence


def _components(instance):
    """Every generated component, by name (per volunteer / per shift)."""
    vs, ss = instance.volunteers, instance.shifts
    return {
        "shifts": [(s.period, s.role) for s in ss],
        "demand": [s.demand for s in ss],
        "min_skill": [s.min_skill for s in ss],
        "expert": [s.expert for s in ss],
        "availability": [v.available for v in vs],
        "skill": [v.skill for v in vs],
        "seniority": [v.seniority for v in vs],
        "cap": [v.cap for v in vs],
        "pref_role": [v.pref_role for v in vs],
        "pref_time": [v.pref_time for v in vs],
    }


@pytest.mark.parametrize(
    "change, may_differ",
    [
        # experts depend on who has skill 3, so they may change with the skills
        pytest.param({"skill_shares": (0.0, 0.0, 0.5, 0.5)}, {"skill", "expert"}, id="skill"),
        pytest.param({"seniority_shares": (0.0, 0.0, 1.0)}, {"seniority"}, id="seniority"),
        pytest.param({"caps": (12,)}, {"cap"}, id="cap"),
        pytest.param({"expert_share": 1.0}, {"expert"}, id="expert"),
        # These change the NUMBER of draws, which also catches two components sharing
        # one stream (a changed distribution with the same draw count would not)
        pytest.param(
            {"availability_ranges": ((3, 3), (10, 10), (26, 26))},
            {"availability", "expert"},  # availability decides expert eligibility
            id="availability_count",
        ),
        pytest.param(
            {"role_activation": 0.9},
            {"shifts", "demand", "min_skill", "expert"},
            id="shift_count",
        ),
    ],
)
def test_substreams_independent(change, may_differ):
    """Changing one component's distribution leaves every other component identical
    (in particular availability), and does change the component itself.
    """
    for seed in SEEDS:
        before = _components(generator.generate_instance(CFG, 40, seed))
        after = _components(generator.generate_instance(replace(CFG, **change), 40, seed))

        for name in before:
            if name not in may_differ:
                assert after[name] == before[name], (seed, name)

        assert any(after[name] != before[name] for name in may_differ), seed


def test_expert_draws_aligned_with_shift_ids():
    """One expert draw per shift, eligible or not: a shift that is eligible both before
    and after a skill change keeps its expert flag (a draw only for eligible shifts would
    shift every later draw).
    """
    cfg = replace(CFG, skill_shares=(0.05, 0.25, 0.40, 0.30))
    compared = 0

    for seed in range(10):
        before = generator.generate_instance(CFG, 40, seed)
        after = generator.generate_instance(cfg, 40, seed)

        for s_before, s_after in zip(before.shifts, after.shifts):
            if _has_skill3_available(before, s_before) and _has_skill3_available(
                after, s_after
            ):
                assert s_after.expert == s_before.expert, (seed, s_before.id)
                compared += 1

    assert compared > 0


# Capacity_ratio


def test_capacity_ratio_hand_instance(min_crew):
    """min_crew: caps 10 + 10, one shift with d = 3 -> 20 / 3."""
    assert generator.capacity_ratio(min_crew) == pytest.approx(20 / 3)


def test_capacity_ratio_generated():
    instance = generator.generate_instance(CFG, 40, 0)
    caps = sum(v.cap for v in instance.volunteers)
    demand = sum(s.demand for s in instance.shifts)

    assert generator.capacity_ratio(instance) == pytest.approx(caps / demand)


def test_capacity_ratio_without_shifts(min_crew):
    with pytest.raises(ValueError):
        generator.capacity_ratio(replace(min_crew, shifts=()))


# Distributions (slow)

N_DRAWS = 2000


@pytest.fixture(scope="module")
def many_instances():
    """N_DRAWS instances of class 40 (shared by the slow tests below)."""
    return _instances(40, seeds=range(N_DRAWS))


def _shares(values, categories):
    counts = Counter(values)
    total = sum(counts.values())
    return [counts[c] / total for c in categories]


@pytest.mark.slow
def test_shift_density(many_instances):
    """Mean shift count ~ rho * 140 = 84 (standard error ~ 0.13)."""
    mean = np.mean([len(i.shifts) for i in many_instances])

    assert mean == pytest.approx(0.6 * 140, abs=1.0)


@pytest.mark.slow
@pytest.mark.parametrize("size_class, mean", [(20, 1.25), (40, 2.5), (80, 5.0)])
def test_demand_mean_per_size_class(size_class, mean):
    """1 + Bern(0.25), 2 + Bern(0.5), 4 + Bin(2, 0.5): the support alone would not catch
    a wrong probability.
    """
    demands = [s.demand for i in _instances(size_class, seeds=range(300)) for s in i.shifts]

    assert np.mean(demands) == pytest.approx(mean, abs=0.02)


@pytest.mark.slow
def test_commitment_shares(many_instances):
    """Commitment classes ~ (0.24, 0.35, 0.41), recovered from the availability count
    (the ranges are disjoint); within a class the count is uniform -> mean = midpoint.
    """
    ranges = CFG.availability_ranges
    counts = [len(v.available) for i in many_instances for v in i.volunteers]
    classes = [next(c for c, (lo, hi) in enumerate(ranges) if lo <= n <= hi) for n in counts]

    assert _shares(classes, range(3)) == pytest.approx(CFG.commitment_shares, abs=0.01)
    for c, (lo, hi) in enumerate(ranges):
        in_class = [n for n, k in zip(counts, classes) if k == c]
        assert np.mean(in_class) == pytest.approx((lo + hi) / 2, abs=0.1)


@pytest.mark.slow
def test_seniority_skill_kappa_shares(many_instances):
    """Seniority ~ (0.35, 0.40, 0.25), skill ~ (0.10, 0.30, 0.40, 0.20) per
    (volunteer, role), P(kappa = 2) ~ 0.25.
    """
    volunteers = [v for i in many_instances for v in i.volunteers]
    shifts = [s for i in many_instances for s in i.shifts]

    seniority = _shares([v.seniority for v in volunteers], range(3))
    skill = _shares([k for v in volunteers for k in v.skill], range(4))
    kappa_two = np.mean([s.min_skill == 2 for s in shifts])

    assert seniority == pytest.approx(CFG.seniority_shares, abs=0.01)
    assert skill == pytest.approx(CFG.skill_shares, abs=0.01)
    assert kappa_two == pytest.approx(CFG.min_skill_two_share, abs=0.01)


@pytest.mark.slow
def test_preference_shares(many_instances):
    """Table A1: p^role and p^time uniform over {0, 1, 2}."""
    volunteers = [v for i in many_instances for v in i.volunteers]
    pref_role = _shares([p for v in volunteers for p in v.pref_role], range(3))
    pref_time = _shares([p for v in volunteers for p in v.pref_time], range(3))

    assert pref_role == pytest.approx([1 / 3] * 3, abs=0.01)
    assert pref_time == pytest.approx([1 / 3] * 3, abs=0.01)


@pytest.mark.slow
def test_expert_share_among_eligible(many_instances):
    """P(e_s = 1 | some skill-3 volunteer available) ~ 0.20."""
    eligible = [
        s for i in many_instances for s in i.shifts if _has_skill3_available(i, s)
    ]

    assert np.mean([s.expert for s in eligible]) == pytest.approx(CFG.expert_share, abs=0.01)
