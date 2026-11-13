"""perturbations.py: events produce a new instance; samplers follow the exposé 4.4 rules.

The samplers only read which cells S0 contains, so most tests pass a small hand-made
S0 (or _pseudo_s0 on a generated instance) instead of solving for the real one.
"""

import copy
import inspect
import typing
from collections import Counter
from functools import reduce

import numpy as np
import pytest

from repair import perturbations as pert
from repair.expected import REPAIR_SLICE_EVENT
from repair.generator import generate_instance
from repair.model import (
    AvailabilityRemoval,
    AvailabilityRestoration,
    DemandChange,
    PreferenceChange,
    Schedule,
    SequenceKind,
    SequenceProfile,
)
from repair.perturbations import apply_event, instances_along
from repair.protocol import GeneratorConfig
from tests.builders import Grid, greedy_schedule, make_shift

SEEDS = range(200)  # sampler draws are cheap; enough to see every eligible target


def _rng(seed):
    return np.random.default_rng(seed)


def _generated(seed=0, size_class=20):
    return generate_instance(GeneratorConfig(), size_class, seed)


_pseudo_s0 = greedy_schedule  # the samplers only read which cells S0 contains


def _s0_pairs(instance, s0):
    """(w, p): w works a shift in period p in S0 (the P2 targets)."""
    return {(w, instance.shifts[s].period) for w, s in s0}


def _s0_roles(instance, s0):
    """(w, r): w works a shift of role r in S0 (the P3 targets)."""
    return {(w, instance.shifts[s].role) for w, s in s0}


def _staffed(instance, s0):
    """Shift -> number of S0 assignees."""
    counts = Counter(s for _, s in s0)
    return {s.id: counts[s.id] for s in instance.shifts}


def _kind(event):
    return {
        DemandChange: "P1",
        AvailabilityRemoval: "P2",
        PreferenceChange: "P3",
        AvailabilityRestoration: "R",
    }[type(event)]


# --- apply_event ---------------------------------------------------------------


def test_apply_removal_returns_new_instance(repair_slice):
    """A loses period 0; nothing else changes, and repair_slice itself is unchanged."""
    after = apply_event(repair_slice, REPAIR_SLICE_EVENT)

    assert after is not repair_slice
    assert after.volunteers[0].available == frozenset({1})
    assert repair_slice.volunteers[0].available == frozenset({0, 1})  # not mutated

    # Everything else is identical
    assert after.volunteers[1:] == repair_slice.volunteers[1:]
    assert after.shifts == repair_slice.shifts
    assert after.volunteers[0].skill == repair_slice.volunteers[0].skill


@pytest.mark.parametrize(
    "event",
    [
        DemandChange(shift=0, delta=1),
        AvailabilityRemoval(pairs=((0, 0),)),
        PreferenceChange(changes=((1, 1, 0),)),
        AvailabilityRestoration(volunteer=0, period=0),
    ],
    ids=_kind,
)
def test_apply_event_never_mutates(repair_slice, event):
    """Every event type returns a new, different instance and leaves its input alone."""
    base = repair_slice
    if isinstance(event, AvailabilityRestoration):  # needs a removed period first
        base = apply_event(repair_slice, AvailabilityRemoval(pairs=((0, 0),)))
    snapshot = copy.deepcopy(base)

    after = apply_event(base, event)

    assert base == snapshot
    assert after != base


def test_demand_change_only_target_and_min_one():
    """d_s += delta on the target only; a result below 1 is refused."""
    grid = Grid(n_periods=1, n_roles=3)
    instance = grid.instance(
        grid.volunteers(2), [make_shift(i, role=i, demand=2) for i in range(3)]
    )

    up = apply_event(instance, DemandChange(shift=1, delta=2))
    assert [s.demand for s in up.shifts] == [2, 4, 2]
    assert up.volunteers == instance.volunteers

    down = apply_event(instance, DemandChange(shift=1, delta=-1))
    assert [s.demand for s in down.shifts] == [2, 1, 2]  # exactly 1 is allowed

    with pytest.raises(ValueError):
        apply_event(instance, DemandChange(shift=1, delta=-2))  # 2 - 2 = 0


def test_removal_and_restoration_are_inverse():
    """Removal then restoration of the same pair gives back the original instance.
    Removing an unavailable period or restoring an available one is refused."""
    instance = _generated()
    volunteer = instance.volunteers[0]
    period = min(volunteer.available)
    missing = min(set(range(instance.n_periods)) - volunteer.available)

    removed = apply_event(instance, AvailabilityRemoval(pairs=((0, period),)))
    restored = apply_event(removed, AvailabilityRestoration(volunteer=0, period=period))

    assert period not in removed.volunteers[0].available
    assert restored == instance

    with pytest.raises(ValueError):
        apply_event(instance, AvailabilityRestoration(volunteer=0, period=period))
    with pytest.raises(ValueError):
        apply_event(instance, AvailabilityRemoval(pairs=((0, missing),)))


@pytest.mark.parametrize("new", [0, 1])
def test_preference_change_new_value_valid(repair_slice, new):
    """B's role-1 preference is 2: the new value is set; nothing else changes."""
    after = apply_event(repair_slice, PreferenceChange(changes=((1, 1, new),)))

    assert after.volunteers[1].pref_role == (0, new)
    assert after.volunteers[0] == repair_slice.volunteers[0]


@pytest.mark.parametrize(
    "new",
    [
        pytest.param(2, id="same_as_current"),  # inert: a sampler bug
        pytest.param(3, id="above_2"),
        pytest.param(-1, id="negative"),
    ],
)
def test_preference_change_rejects_invalid_value(repair_slice, new):
    with pytest.raises(ValueError):
        apply_event(repair_slice, PreferenceChange(changes=((1, 1, new),)))


def test_instances_along_is_prefix_fold(repair_slice):
    """instance_t = events 1..t applied in order; the base instance is not included."""
    events = (
        AvailabilityRemoval(pairs=((0, 0),)),
        DemandChange(shift=1, delta=1),
        PreferenceChange(changes=((1, 1, 0),)),
        AvailabilityRestoration(volunteer=0, period=0),
    )

    along = instances_along(repair_slice, events)

    assert len(along) == len(events)
    for t in range(len(events)):
        assert along[t] == reduce(apply_event, events[: t + 1], repair_slice)
    assert instances_along(repair_slice, ()) == []


# --- P2 and P3 -----------------------------------------------------------------


def test_p2_targets_assigned_available_distinct():
    """k distinct pairs, each S0-assigned and currently available; an already removed
    pair and the excluded pair are never chosen."""
    instance = _generated()
    s0 = _pseudo_s0(instance)
    gone, excluded = sorted(_s0_pairs(instance, s0))[:2]
    current = apply_event(instance, AvailabilityRemoval(pairs=(gone,)))

    for seed in SEEDS:
        event = pert.sample_p2(current, s0, 2, _rng(seed), exclude=excluded)

        assert len(event.pairs) == len(set(event.pairs)) == 2
        assert not event.fallback
        for w, p in event.pairs:
            assert (w, p) in _s0_pairs(instance, s0)
            assert p in current.volunteers[w].available
            assert (w, p) not in {gone, excluded}


def test_p2_independent_any_available_pair():
    """s0=None: any currently available pair, also ones S0 does not use."""
    instance = _generated()
    s0_pairs = _s0_pairs(instance, _pseudo_s0(instance))
    chosen = {
        pair for seed in SEEDS for pair in pert.sample_p2(instance, None, 2, _rng(seed)).pairs
    }

    assert all(p in instance.volunteers[w].available for w, p in chosen)
    assert chosen - s0_pairs  # not restricted to S0


def test_p3_pairs_assigned_new_value_differs():
    """k distinct (w, r) pairs worked together in S0; the new value differs from the
    current one, and both other values occur."""
    instance = _generated()
    s0 = _pseudo_s0(instance)
    excluded = min(w for w, _ in s0)
    seen = set()

    for seed in SEEDS:
        event = pert.sample_p3(instance, s0, 2, _rng(seed), exclude_volunteer=excluded)

        assert len({(w, r) for w, r, _ in event.changes}) == 2
        assert not event.fallback
        for w, r, new in event.changes:
            current = instance.volunteers[w].pref_role[r]
            assert (w, r) in _s0_roles(instance, s0)
            assert w != excluded
            assert new in {0, 1, 2} and new != current
            seen.add((current, new))

    for current in {c for c, _ in seen}:  # redrawn from BOTH other values
        assert {new for c, new in seen if c == current} == {0, 1, 2} - {current}


# --- P1 ------------------------------------------------------------------------


def _p1_instance():
    """One period, four roles. d = (1, 2, 1, 1); S0 staffs them (2, 2, 0, 1).

    shift  d  staffed  increase (staffed <= d)  decrease (d - 1 >= 1)
    s0     1  2        no                       no
    s1     2  2        yes                      yes
    s2     1  0        yes (cancelled in S0)    no
    s3     1  1        yes                      no
    """
    grid = Grid(n_periods=1, n_roles=4)
    demands = (1, 2, 1, 1)
    instance = grid.instance(
        grid.volunteers(5), [make_shift(i, role=i, demand=d) for i, d in enumerate(demands)]
    )
    s0 = frozenset({(0, 0), (1, 0), (2, 1), (3, 1), (4, 3)})
    return instance, s0


def _targets(instance, s0, sign, seeds=SEEDS):
    """Targets of the non-fallback P1 events in direction `sign`."""
    events = [pert.sample_p1(instance, s0, 1, _rng(seed)) for seed in seeds]
    return {e.shift for e in events if (e.delta > 0) == (sign > 0) and not e.fallback}


def test_p1_increase_target_rule():
    """+k only where S0 staffing <= current d_s; -k only where d_s - k >= 1."""
    instance, s0 = _p1_instance()
    events = [pert.sample_p1(instance, s0, 1, _rng(seed)) for seed in SEEDS]

    assert {abs(e.delta) for e in events} == {1}
    assert not any(e.fallback for e in events)
    assert _targets(instance, s0, +1) == {1, 2, 3}
    assert _targets(instance, s0, -1) == {1}
    assert 0.35 < np.mean([e.delta > 0 for e in events]) < 0.65  # probability 1/2


def test_p1_independent_any_shift():
    """s0=None: every shift is an increase target."""
    instance, _ = _p1_instance()

    assert _targets(instance, None, +1) == {0, 1, 2, 3}


def test_p1_builds_on_current_demand():
    """After d_0: 1 -> 2, shift 0 qualifies for both directions (staffed 2 <= 2, 2 - 1 >= 1)."""
    instance, s0 = _p1_instance()
    current = apply_event(instance, DemandChange(shift=0, delta=1))

    assert _targets(current, s0, +1) == {0, 1, 2, 3}
    assert _targets(current, s0, -1) == {0, 1}


def test_p1_increase_fallback_flagged():
    """Every shift is overstaffed in S0 (3 > d = 2): increases fall back to any shift open
    in S0 and are flagged; decreases are possible and unflagged."""
    grid = Grid(n_periods=1, n_roles=2)
    instance = grid.instance(
        grid.volunteers(6), [make_shift(i, role=i, demand=2) for i in range(2)]
    )
    s0 = frozenset({(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)})
    events = [pert.sample_p1(instance, s0, 1, _rng(seed)) for seed in SEEDS]

    increases = [e for e in events if e.delta > 0]
    decreases = [e for e in events if e.delta < 0]
    assert increases and decreases
    assert all(e.fallback and e.shift in {0, 1} for e in increases)
    assert not any(e.fallback for e in decreases)


def test_p1_decrease_fallback_becomes_increase():
    """All d = 1 and k = 1: no decrease is possible, so every event is +1, and the
    ones that started as a decrease (about half) are flagged."""
    grid = Grid(n_periods=1, n_roles=2)
    instance = grid.instance(grid.volunteers(2), grid.shifts(2))
    s0 = frozenset({(0, 0), (1, 1)})
    events = [pert.sample_p1(instance, s0, 1, _rng(seed)) for seed in SEEDS]

    assert all(e.delta == 1 for e in events)
    assert 0.35 < np.mean([e.fallback for e in events]) < 0.65


# --- 13.9: fewer than k eligible targets ---------------------------------------


def _short_instance():
    """3 periods, 3 roles, 2 volunteers, one shift per period (role = period).
    S0 has the single cell (v0, s0): one P2 target (0, 0), one P3 target (0, 0)."""
    grid = Grid(n_periods=3, n_roles=3)
    shifts = [make_shift(i, period=i, role=i) for i in range(3)]
    return grid.instance(grid.volunteers(2), shifts), frozenset({(0, 0)})


def test_fewer_than_k_targets_p2():
    """13.9: all eligible targets, topped up from any available pair, flagged."""
    instance, s0 = _short_instance()

    for seed in range(20):
        enough = pert.sample_p2(instance, s0, 1, _rng(seed))
        assert enough.pairs == ((0, 0),) and not enough.fallback

        short = pert.sample_p2(instance, s0, 3, _rng(seed))
        assert short.fallback
        assert short.pairs[0] == (0, 0)  # eligible ones first
        assert len(set(short.pairs)) == 3
        assert all(p in instance.volunteers[w].available for w, p in short.pairs)

        excluded = pert.sample_p2(instance, s0, 2, _rng(seed), exclude=(0, 0))
        assert excluded.fallback and (0, 0) not in excluded.pairs


def test_fewer_than_k_targets_p3():
    """13.9: all eligible (w, r) pairs, topped up from any (w, r) pair, flagged."""
    instance, s0 = _short_instance()

    for seed in range(20):
        enough = pert.sample_p3(instance, s0, 1, _rng(seed))
        assert [(w, r) for w, r, _ in enough.changes] == [(0, 0)]
        assert not enough.fallback

        short = pert.sample_p3(instance, s0, 3, _rng(seed))
        assert short.fallback
        assert short.changes[0][:2] == (0, 0)  # eligible ones first
        assert len({(w, r) for w, r, _ in short.changes}) == 3
        assert all(new != instance.volunteers[w].pref_role[r] for w, r, new in short.changes)


@pytest.mark.parametrize(
    "sample, k, kwargs",
    [
        pytest.param(pert.sample_p2, 7, {}, id="p2_more_than_all_6_pairs"),
        pytest.param(pert.sample_p3, 7, {}, id="p3_more_than_all_6_pairs"),
        pytest.param(pert.sample_p3, 4, {"exclude_volunteer": 0}, id="p3_only_v1_left"),
    ],
)
def test_too_few_targets_even_after_top_up_raises(sample, k, kwargs):
    instance, s0 = _short_instance()

    with pytest.raises(ValueError):
        sample(instance, s0, k, _rng(0), **kwargs)


# --- RQ1 and independent targeting --------------------------------------------


@pytest.fixture(scope="module")
def base():
    """A generated n = 20 instance (k = 1) with a pseudo S0."""
    instance = _generated()
    return instance, _pseudo_s0(instance)


@pytest.mark.parametrize(
    "profile, counts",
    [
        (SequenceProfile.BALANCED, {"P1": 5, "P2": 5, "P3": 5}),
        (SequenceProfile.AVAILABILITY, {"P1": 3, "P2": 9, "P3": 3}),
        (SequenceProfile.PREFERENCE, {"P1": 3, "P2": 3, "P3": 9}),
    ],
)
def test_rq1_profile_counts(base, profile, counts):
    """★ The profile counts, in an order that varies with the seed; every event has
    size k; the sequence applies cleanly step by step."""
    instance, s0 = base
    orders = set()

    for seed in range(10):
        seq = pert.rq1_sequence(instance, s0, profile, seed)

        assert (seq.kind, seq.profile, seq.seed, seq.selected) == (
            SequenceKind.RQ1, profile, seed, None
        )
        assert len(seq.events) == 15
        assert Counter(_kind(e) for e in seq.events) == counts
        instances_along(instance, seq.events)  # raises if an event was sampled stale
        orders.add(tuple(_kind(e) for e in seq.events))

    assert len(orders) > 1


def test_rq1_scale_sets_event_size(base):
    """scale=2 -> |delta| = 2, two pairs per P2, two changes per P3."""
    instance, s0 = base
    seq = pert.rq1_sequence(instance, s0, SequenceProfile.BALANCED, 0, scale=2)

    assert {abs(e.delta) for e in seq.events if isinstance(e, DemandChange)} == {2}
    assert {len(e.pairs) for e in seq.events if isinstance(e, AvailabilityRemoval)} == {2}
    assert {len(e.changes) for e in seq.events if isinstance(e, PreferenceChange)} == {2}


def test_rq1_same_seed_same_digest(base):
    instance, s0 = base
    a = pert.rq1_sequence(instance, s0, SequenceProfile.BALANCED, 4)
    b = pert.rq1_sequence(instance, s0, SequenceProfile.BALANCED, 4)
    c = pert.rq1_sequence(instance, s0, SequenceProfile.BALANCED, 5)

    assert a == b and a.digest() == b.digest()
    assert c.digest() != a.digest()


def test_independent_targeting_ignores_s0(base):
    """With targeting "independent" the sequence is the same for any S0 (kind
    independent); with "s0" a different S0 gives a different sequence."""
    instance, s0 = base
    other = _pseudo_s0(instance, seed=1)
    profile = SequenceProfile.BALANCED

    for seed in range(5):
        a = pert.rq1_sequence(instance, s0, profile, seed, targeting="independent")
        b = pert.rq1_sequence(instance, other, profile, seed, targeting="independent")
        empty = pert.rq1_sequence(instance, frozenset(), profile, seed, targeting="independent")

        assert a == b == empty
        assert a.kind == SequenceKind.INDEPENDENT

    assert any(
        pert.rq1_sequence(instance, s0, profile, seed)
        != pert.rq1_sequence(instance, other, profile, seed)
        for seed in range(5)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"targeting": "random"}, id="unknown_targeting"),
        pytest.param({"horizon": 14}, id="horizon_not_15"),
    ],
)
def test_rq1_rejects_bad_arguments(base, kwargs):
    instance, s0 = base

    with pytest.raises(ValueError):
        pert.rq1_sequence(instance, s0, SequenceProfile.BALANCED, 0, **kwargs)


# --- RQ2 -----------------------------------------------------------------------


def _background(seq):
    """The 13 events other than step 1 and step 6."""
    return [e for t, e in enumerate(seq.events, start=1) if t not in (1, 6)]


def test_rq2_step1_removal_step6_restoration(base):
    """★ Step 1 removes the selected S0 pair, step 6 restores exactly that pair; in
    between it stays unavailable."""
    instance, s0 = base

    for seed in range(10):
        seq = pert.rq2_sequence(instance, s0, seed)
        assert seq.selected is not None
        w, p = seq.selected

        assert (seq.kind, seq.profile) == (SequenceKind.RQ2, SequenceProfile.RQ2)
        assert len(seq.events) == 15
        assert (w, p) in _s0_pairs(instance, s0)
        assert seq.events[0] == AvailabilityRemoval(pairs=((w, p),))
        assert seq.events[5] == AvailabilityRestoration(volunteer=w, period=p)

        along = instances_along(instance, seq.events)
        assert all(p not in along[t].volunteers[w].available for t in range(5))
        assert p in along[5].volunteers[w].available


def test_rq2_background_counts_and_selected_pair_untouched(base):
    """4 P1, 5 P2, 4 P3 on the other 13 steps; no P2 removes the selected pair and no
    P3 changes the selected volunteer."""
    instance, s0 = base

    for seed in range(10):
        seq = pert.rq2_sequence(instance, s0, seed)
        assert seq.selected is not None
        w, _ = seq.selected
        background = _background(seq)

        assert Counter(_kind(e) for e in background) == {"P1": 4, "P2": 5, "P3": 4}
        for e in background:
            if isinstance(e, AvailabilityRemoval):
                assert seq.selected not in e.pairs
            if isinstance(e, PreferenceChange):
                assert all(v != w for v, _, _ in e.changes)


def test_rq2_selected_pair_untouched_on_small_instance():
    """Same rule where it bites: 2 volunteers x 4 periods leave only a few pairs, so a
    background P2 without `exclude` would soon remove the restored pair again."""
    grid = Grid(n_periods=4, n_roles=2)
    instance = grid.instance(
        grid.volunteers(2), [make_shift(i, period=i, role=0, demand=2) for i in range(4)]
    )
    s0 = frozenset({(0, 0)})  # forces selected = (0, 0)

    for seed in range(30):
        seq = pert.rq2_sequence(instance, s0, seed)

        assert seq.selected == (0, 0)
        for e in _background(seq):
            if isinstance(e, AvailabilityRemoval):
                assert (0, 0) not in e.pairs, seed
            if isinstance(e, PreferenceChange):
                assert all(v != 0 for v, _, _ in e.changes), seed


def test_rq2_background_independent_of_s0(base):
    """★ Two different S0s that select the same pair give the same sequence: the
    background is drawn without S0 (and from its own stream)."""
    instance, _ = base
    by_period = Counter(s.period for s in instance.shifts)
    s1, s2 = next(
        (a, b)
        for a in instance.shifts
        for b in instance.shifts
        if a.id < b.id and a.period == b.period
    )
    assert by_period[s1.period] >= 2
    w = next(v.id for v in instance.volunteers if s1.period in v.available)
    s0_a, s0_b = frozenset({(w, s1.id)}), frozenset({(w, s2.id)})

    for seed in range(10):
        a = pert.rq2_sequence(instance, s0_a, seed)
        b = pert.rq2_sequence(instance, s0_b, seed)

        assert a.selected == b.selected == (w, s1.period)
        assert a == b
        # S0 has no other pair, so S0-targeted P2 would have to top up (fallback)
        assert not any(getattr(e, "fallback", False) for e in _background(a))


# --- the rule behind all samplers ---------------------------------------------

SAMPLERS = {
    pert.sample_p1: {"instance", "s0", "k", "rng"},
    pert.sample_p2: {"instance", "s0", "k", "rng", "exclude"},
    pert.sample_p3: {"instance", "s0", "k", "rng", "exclude_volunteer"},
    pert.rq1_sequence: {"instance", "s0", "profile", "seed", "horizon", "targeting", "scale"},
    pert.rq2_sequence: {"instance", "s0", "seed", "horizon", "scale"},
}


@pytest.mark.parametrize("sampler", list(SAMPLERS), ids=lambda f: f.__name__)
def test_no_sampler_accepts_an_arm_schedule(sampler):
    """★ DESIGN.md rule 5: every arm faces the same events, so no sampler may see an
    arm's schedule. The only Schedule parameter is s0, and no other parameter exists."""
    assert set(inspect.signature(sampler).parameters) == SAMPLERS[sampler]

    hints = typing.get_type_hints(sampler)
    schedule_params = {
        name
        for name, hint in hints.items()
        if name != "return" and (hint == Schedule or Schedule in typing.get_args(hint))
    }
    assert schedule_params == {"s0"}
