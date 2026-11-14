"""Check the metrics on small trajectories written out by hand.

No solver output is needed here. Default volunteers have skill 1 and
preference 0, so one assignment per shift gives U(S) = |S|. Filler events
change demand on an unused shift and do not change schedule quality.
"""

from fractions import Fraction

import pytest

from repair import metrics
from repair.expected import (
    REPAIR_SLICE_DISTANCE,
    REPAIR_SLICE_EVENT,
    REPAIR_SLICE_S0,
    REPAIR_SLICE_SF,
)
from repair.metrics import affected, burden, distance
from repair.model import (
    AcceptanceLog,
    Arm,
    AvailabilityRemoval,
    AvailabilityRestoration,
    Classification,
    CoverageStep,
    DemandChange,
    PolicyName,
    ReferenceMode,
    Sequence,
    SequenceKind,
    SequenceProfile,
    SequenceResult,
    StageRecord,
    StageStatus,
    Step,
    Trajectory,
)
from repair.objectives import MAX_QUALITY, min_distance
from repair.perturbations import apply_event
from tests.builders import Grid, make_shift


@pytest.mark.parametrize("schedule, expected", list(REPAIR_SLICE_DISTANCE.items()))
def test_distance_symmetric_difference(schedule, expected):
    """H(S, S0) = |S xor S0| from the value sheet; symmetric in its arguments."""
    assert distance(schedule, REPAIR_SLICE_S0) == expected
    assert distance(REPAIR_SLICE_S0, schedule) == expected


def test_distance_counts_added_and_removed_cells():
    """One removed cell and one added cell: H = 2, not 1 (unlike a plain set difference)."""
    a = frozenset({(0, 0), (1, 1)})
    b = frozenset({(1, 1), (2, 0)})

    assert distance(a, b) == 2
    assert distance(a, a) == 0


# Affected and burden (HWF needs them at run time)

_ARM = Arm(PolicyName.SF, ReferenceMode.CHAINED)


def _trajectory(*schedules):
    """A trajectory whose step t has schedule schedules[t - 1] (stages do not matter)."""
    return Trajectory(
        _ARM, tuple(Step(t, s, ()) for t, s in enumerate(schedules, start=1))
    )


def test_affected_volunteers_with_a_changed_cell():
    """v0 moved (removed), v2 added; v1 kept its cell."""
    assert affected(frozenset({(0, 0), (1, 1)}), frozenset({(1, 1), (2, 0)})) == {0, 2}
    assert affected(frozenset({(1, 1)}), frozenset({(1, 1)})) == frozenset()


def test_affected_counts_a_swap_once():
    """v0 moves from s0 to s2: two changed cells, one affected volunteer."""
    assert affected(frozenset({(0, 0)}), frozenset({(0, 2)})) == {0}


def test_burden_counts_affected_repairs():
    """S0 -> S1 affects v0, v2; S1 -> S2 changes nothing; S2 -> S3 affects v0."""
    s0 = frozenset({(0, 0), (1, 1)})
    s1 = frozenset({(1, 1), (2, 0)})
    s3 = frozenset({(0, 2), (1, 1), (2, 0)})
    trajectory = _trajectory(s1, s1, s3)

    assert burden(trajectory, s0, (0, 1, 2), until=0) == {0: 0, 1: 0, 2: 0}
    assert burden(trajectory, s0, (0, 1, 2), until=1) == {0: 1, 1: 0, 2: 1}
    assert burden(trajectory, s0, (0, 1, 2), until=2) == {0: 1, 1: 0, 2: 1}
    assert burden(trajectory, s0, (0, 1, 2), until=3) == {0: 2, 1: 0, 2: 1}


@pytest.mark.parametrize("until", [-1, 4])
def test_burden_rejects_until_outside_the_trajectory(until):
    with pytest.raises(ValueError):
        burden(_trajectory(frozenset(), frozenset(), frozenset()), frozenset(), (0,), until)



# Helpers for hand-built results

OPT, LIMIT = StageStatus.OPTIMAL, StageStatus.TIME_LIMIT
QO_ARM = Arm(PolicyName.QO, None)
SF_CHAINED = Arm(PolicyName.SF, ReferenceMode.CHAINED)
SF_ANCHORED = Arm(PolicyName.SF, ReferenceMode.ANCHORED)


def _records(*specs):
    """StageRecords from (name, status[, value]) tuples; bound and seconds irrelevant."""
    return tuple(
        StageRecord(s[0], s[1], s[2] if len(s) > 2 else 0, None, 0.0) for s in specs
    )


def _sf_records(status=OPT):
    return _records(("MinDistance", status), ("MaxQuality", status), ("MaxTieBreak", OPT))


def _steps(schedules, records=None):
    return tuple(
        Step(t, s, records(t) if records else _sf_records())
        for t, s in enumerate(schedules, start=1)
    )


def _fillers(n, shift):
    """n valid DemandChange events on `shift`: +1, -1, +1, ... (demand stays >= 1)."""
    return tuple(DemandChange(shift=shift, delta=1 if i % 2 == 0 else -1) for i in range(n))


def _result(instance, s0, trajectories, events, lambda_star=None, kind=SequenceKind.RQ1,
            profile=SequenceProfile.BALANCED, selected=None):
    horizon = len(events)
    lambdas = lambda_star or (0,) * horizon
    return SequenceResult(
        cell="test",
        instance=instance,
        acceptance=AcceptanceLog(0, 0, 1, 0, False, 0),
        sequence=Sequence(kind, profile, 0, tuple(events), selected),
        s0=s0,
        s0_stages=(),
        coverage=tuple(
            CoverageStep(t, lam, StageRecord("MinShortfall", OPT, lam, None, 0.0))
            for t, lam in enumerate(lambdas, start=1)
        ),
        trajectories=tuple(trajectories),
        manifest={},
    )


def _line(n_volunteers=2, n_shifts=3, filler=True, fields=None):
    """One shift per period (role 0), d = 1, plus an extra filler shift if asked.
    fields: volunteer id -> builder overrides, e.g. {1: {"available": [2]}}.
    """
    n = n_shifts + (1 if filler else 0)
    grid = Grid(n_periods=n, n_roles=1)
    fields = fields or {}
    volunteers = [grid.volunteer(i, **fields.get(i, {})) for i in range(n_volunteers)]
    return grid.instance(volunteers, [make_shift(i, period=i) for i in range(n)])


# Affected, changed cells, burden variants


def test_affected_and_changed_cells():
    """v0 swaps s0 for s2 (2 cells, 1 repair); v1 is added (1 cell); v2 keeps its cell."""
    previous = frozenset({(0, 0), (2, 1)})
    current = frozenset({(0, 2), (1, 1), (2, 1)})

    assert affected(previous, current) == {0, 1}
    assert metrics.changed_cells_by_volunteer(previous, current) == {0: 2, 1: 1}
    assert metrics.changed_cells_by_volunteer(previous, previous) == {}


def test_burden_repairs_vs_cells():
    """Changing three cells for one volunteer counts as one affected repair or three cells."""
    s0 = frozenset({(0, 0)})
    trajectory = Trajectory(SF_CHAINED, _steps([frozenset({(0, 1), (0, 2)})]))

    assert burden(trajectory, s0, (0, 1), until=1) == {0: 1, 1: 0}
    assert burden(trajectory, s0, (0, 1), until=1, variant="cells") == {0: 3, 1: 0}
    with pytest.raises(ValueError):
        burden(trajectory, s0, (0, 1), until=1, variant="shifts")  # type: ignore[arg-type]


# Inequality


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 0, 0], Fraction(0)),
        ([1, 1, 1], Fraction(0)),
        ([0, 0, 3], Fraction(2, 3)),
        ([1, 3], Fraction(1, 4)),  # |1 - 3| * 2 / (2 * 4 * 2)
        ([5], Fraction(0)),
    ],
)
def test_gini_values(values, expected):
    got = metrics.gini(values)
    assert got == expected and isinstance(got, Fraction)


@pytest.mark.parametrize(
    "values, expected",
    [([0, 0, 0], None), ([1, 1, 2], Fraction(1, 2)), ([4], Fraction(1))],
)
def test_max_share_values(values, expected):
    assert metrics.max_share(values) == expected


@pytest.mark.parametrize("function", [metrics.gini, metrics.max_share])
@pytest.mark.parametrize("values", [[], [1, -1]], ids=["empty", "negative"])
def test_inequality_rejects_bad_vectors(function, values):
    with pytest.raises(ValueError):
        function(values)


# Classify (repair_slice; U values from the value sheet)

# three schedules with U = 16 under the initial instance (REPAIR_SLICE_ROWS_0)
U16_A = frozenset({(0, 2), (1, 1), (2, 0)})
U16_B = frozenset({(0, 0), (1, 1), (1, 2)})


def test_classify_binding_infeasible(repair_slice):
    """After the event A cannot keep s0, so S0 is infeasible: binding."""
    after = apply_event(repair_slice, REPAIR_SLICE_EVENT)

    assert metrics.is_binding(REPAIR_SLICE_S0, after, lambda_star=0)
    got = metrics.classify((MAX_QUALITY,), REPAIR_SLICE_S0, REPAIR_SLICE_SF, after, 0,
                           _records(("MaxQuality", OPT)))
    assert got == Classification.BINDING


def test_classify_binding_over_cap(repair_slice):
    """Feasible, but s0 cancelled (Lambda = 1) while Lambda* = 0: binding."""
    previous = frozenset({(1, 1), (2, 2)})

    assert metrics.is_binding(previous, repair_slice, lambda_star=0)
    assert not metrics.is_binding(previous, repair_slice, lambda_star=1)
    got = metrics.classify((MAX_QUALITY,), previous, REPAIR_SLICE_S0, repair_slice, 0,
                           _records(("MaxQuality", OPT)))
    assert got == Classification.BINDING


def test_classify_relevant(repair_slice):
    """SF* (U = 13) is still feasible and covered, but S0 (U = 18) is better."""
    got = metrics.classify((MAX_QUALITY,), REPAIR_SLICE_SF, REPAIR_SLICE_S0, repair_slice, 0,
                           _records(("MaxQuality", OPT)))
    assert got == Classification.RELEVANT


def test_classify_inert_despite_tiebreak_change(repair_slice):
    """The schedule changed, but only between two U = 16 schedules: inert."""
    got = metrics.classify((MAX_QUALITY,), U16_A, U16_B, repair_slice, 0,
                           _records(("MaxQuality", OPT)))
    assert U16_A != U16_B
    assert got == Classification.INERT


def test_classify_undetermined(repair_slice):
    """Without an improvement or proven optimality, the step must remain undetermined."""
    got = metrics.classify((MAX_QUALITY,), U16_A, U16_B, repair_slice, 0,
                           _records(("MaxQuality", LIMIT)))
    assert got == Classification.UNDETERMINED


def test_classify_first_differing_stage_decides(repair_slice):
    """SF stages: previous U16_A has H = 0 to itself, current S0 has H > 0: current is
    worse on stage 1, so its higher U (18 > 16) does not count: inert, not relevant.
    """
    specs = (min_distance(U16_A), MAX_QUALITY)
    got = metrics.classify(specs, U16_A, REPAIR_SLICE_S0, repair_slice, 0,
                           _records(("MinDistance", OPT), ("MaxQuality", OPT)))
    assert got == Classification.INERT


def test_classify_rejects_mismatched_stages(repair_slice):
    with pytest.raises(ValueError):
        metrics.classify((MAX_QUALITY,), U16_A, U16_B, repair_slice, 0, ())


# Classifications (rebuilt from stored data)


def test_sf_chained_non_binding_is_inert():
    """Chained SF keeps the previous schedule whenever it is not binding (H = 0 is
    optimal), so every non-binding step is inert.
    """
    instance = _line()
    s0 = frozenset({(0, 0), (1, 1)})
    trajectory = Trajectory(SF_CHAINED, _steps([s0, s0, s0]))
    # Lambda(S0) = 2 here (s2 and the filler shift are empty): a cap of 2 keeps S0 non-binding
    result = _result(instance, s0, [trajectory], _fillers(3, shift=3), lambda_star=(2, 2, 2))
    instances_t = [instance] * 3  # the fillers never touch a used shift

    assert metrics.classifications(result, trajectory, instances_t) == [Classification.INERT] * 3


def test_classifications_rebuild_hwf_burden(monkeypatch):
    """HWF-anchored, alpha = 1, S0 = {(0, 0)}: step 1 drops v0 (burden b_0 = 1), step 2
    adds v1 on s0 next to v0.

    t = 2 with the rebuilt weights {0: 2, 1: 1}: previous H_w = 2, current H_w = 1 ->
    current better -> RELEVANT. With weights all 1 (burden ignored) H ties at 1 and U
    decides: U({}) = 0 > U(current) = 1 + 1 + 5 - 10 = -3 -> it would be INERT.
    """
    instance = _line(n_shifts=2, filler=True)
    s0 = frozenset({(0, 0)})
    s1: frozenset[tuple[int, int]] = frozenset()
    s2 = frozenset({(0, 0), (1, 0)})
    arm = Arm(PolicyName.HWF, ReferenceMode.ANCHORED, alpha=Fraction(1))
    trajectory = Trajectory(arm, _steps([s1, s2]))
    # caps equal to the previous schedules' shortfall (2, then 3): nothing is binding
    result = _result(instance, s0, [trajectory], _fillers(2, shift=2), lambda_star=(2, 3))

    import repair.policies

    seen = []
    original = repair.policies.policy_stages

    def spy(arm, reference, burden, instance):
        seen.append(dict(burden))
        return original(arm, reference, burden, instance)

    monkeypatch.setattr(repair.policies, "policy_stages", spy)

    assert metrics.classifications(result, trajectory, [instance, instance]) == [
        Classification.INERT,  # t = 1: H(S0) = 0 beats H({}) = 1
        Classification.RELEVANT,
    ]
    # the burden BEFORE each step: b_{t-1}, never including step t itself
    assert seen == [{0: 0, 1: 0}, {0: 1, 1: 0}]


# Step rows


def test_churn_and_drift_divide_by_actual_cells():
    """J = |W| * |S| = 2 * 4 = 8 although v1 is available in one period only (5 available
    cells): C_t = H(S_t, S_{t-1}) / J, B_t = H(S_t, S_0) / J; A_t counts the affected
    volunteers (Table A2: sum_w c_{w,t}, not a share).
    """
    instance = _line(n_shifts=3, filler=True, fields={1: {"available": [2]}})
    s0 = frozenset({(0, 0)})
    s1 = frozenset({(0, 1)})  # v0 moves: H = 2
    s2 = frozenset({(0, 1), (1, 2)})  # v1 added: H = 1, drift H(S2, S0) = 3
    result = _result(instance, s0, [Trajectory(SF_CHAINED, _steps([s1, s2]))],
                     _fillers(2, shift=3), lambda_star=(2, 1))

    rows = metrics.step_rows(result)

    assert instance.n_cells == 8
    assert [(r["t"], r["C"], r["B"], r["A"]) for r in rows] == [
        (1, Fraction(2, 8), Fraction(2, 8), 1),  # v0 moved
        (2, Fraction(1, 8), Fraction(3, 8), 1),  # v1 added
    ]
    assert [r["U"] for r in rows] == [1, 2]
    assert set(rows[0]) == set(metrics.STEP_KEYS)
    assert rows[0]["arm"] == "SF-chained" and rows[0]["certified"]


def test_gap_uses_qo_value_none_when_unproven():
    """U*_t is QO's recorded MaxQuality value: proven at t = 1 (10), unproven at t = 2,
    so g_2 is None for every arm. An unproven SF stage makes that step uncertified.
    """
    instance = _line(n_shifts=3, filler=True)
    s0 = frozenset({(0, 0)})
    qo_steps = (
        Step(1, frozenset({(0, 0), (1, 1)}), _records(("MaxQuality", OPT, 10), ("MaxTieBreak", OPT))),
        Step(2, frozenset({(0, 0)}), _records(("MaxQuality", LIMIT, 9), ("MaxTieBreak", OPT))),
    )
    sf_steps = (
        Step(1, frozenset({(0, 0)}), _sf_records()),
        Step(2, frozenset({(0, 0)}), _sf_records(LIMIT)),
    )
    result = _result(instance, s0, [Trajectory(QO_ARM, qo_steps), Trajectory(SF_CHAINED, sf_steps)],
                     _fillers(2, shift=3), lambda_star=(2, 3))

    rows = {(r["arm"], r["t"]): r for r in metrics.step_rows(result)}

    assert rows[("QO", 1)]["g"] == 10 - 2
    assert rows[("SF-chained", 1)]["g"] == 10 - 1
    assert rows[("QO", 2)]["g"] is None and rows[("SF-chained", 2)]["g"] is None
    assert rows[("SF-chained", 1)]["certified"] and not rows[("SF-chained", 2)]["certified"]
    assert not rows[("QO", 2)]["certified"]


def test_certified_needs_every_stage_including_tiebreak():
    """A step whose only unproven stage is the tie-break is not certified."""
    instance = _line()
    s0 = frozenset({(0, 0)})
    records = _records(("MinDistance", OPT), ("MaxQuality", OPT), ("MaxTieBreak", LIMIT))
    result = _result(instance, s0, [Trajectory(SF_CHAINED, (Step(1, s0, records),))],
                     _fillers(1, shift=3), lambda_star=(9,))

    assert [r["certified"] for r in metrics.step_rows(result)] == [False]


def test_gap_none_without_qo_arm():
    instance = _line()
    s0 = frozenset({(0, 0)})
    result = _result(instance, s0, [Trajectory(SF_CHAINED, _steps([s0]))], _fillers(1, shift=3))

    assert [r["g"] for r in metrics.step_rows(result)] == [None]


# Terminal rows


def test_terminal_rows_until_and_burden_vectors():
    """Two SF arms over 3 steps; until = 2 ignores step 3.

    chained:  S0 {(0,0)} -> {(0,1)} (v0, 2 cells) -> {(0,1),(1,2)} (v1, 1 cell)
    anchored: S0 -> S0 -> S0 (nothing changes)
    """
    instance = _line(n_shifts=3, filler=True)
    s0 = frozenset({(0, 0)})
    chained = Trajectory(SF_CHAINED, _steps([frozenset({(0, 1)}), frozenset({(0, 1), (1, 2)}),
                                             frozenset()]))
    anchored = Trajectory(SF_ANCHORED, _steps([s0, s0, s0]))
    # a cap no schedule exceeds: nothing is binding, binding_rate = 0
    result = _result(instance, s0, [chained, anchored], _fillers(3, shift=3),
                     lambda_star=(9, 9, 9))

    rows = {r["arm"]: r for r in metrics.terminal_rows(result, until=2)}
    c, a = rows["SF-chained"], rows["SF-anchored"]

    assert set(c) == set(metrics.TERMINAL_KEYS) and c["until"] == 2
    assert c["sum_C"] == Fraction(2, 8) + Fraction(1, 8)
    assert c["B"] == Fraction(3, 8)
    assert c["burden_repairs"] == [1, 1] and c["burden_cells"] == [2, 1]
    assert c["total_burden"] == 2
    assert c["gini"] == 0 and c["max_share"] == Fraction(1, 2)
    assert c["gini_cells"] == Fraction(1, 6)  # [2, 1]: 2 * 1 / (2 * 4 * 1.5)
    assert c["sum_g"] is None  # no QO arm
    assert a["burden_repairs"] == [0, 0] and a["total_burden"] == 0
    assert a["gini"] == 0 and a["max_share"] is None
    assert a["sum_C"] == 0 and a["binding_rate"] == 0


@pytest.mark.parametrize("until", [0, 4])
def test_terminal_rows_rejects_until_outside_the_steps(until):
    instance = _line()
    s0 = frozenset({(0, 0)})
    result = _result(instance, s0, [Trajectory(SF_CHAINED, _steps([s0, s0, s0]))],
                     _fillers(3, shift=3))

    with pytest.raises(ValueError):
        metrics.terminal_rows(result, until)


# Persistence and RQ2

P_S0 = frozenset({(0, 0), (1, 1)})
P_C1 = frozenset({(1, 1), (2, 2)})  # forcing changed (0, 0) and (2, 2)


def _six(*schedules):
    """A 6-step trajectory; schedules[i] is S_{i+1}, the last one repeated."""
    padded = list(schedules) + [schedules[-1]] * (6 - len(schedules))
    return _steps(padded)


@pytest.mark.parametrize(
    "chained, anchored_6, expected",
    [
        # both changed cells kept by chained; anchored at t = 6 differs on (0, 0) only
        pytest.param([P_C1], P_S0 | {(2, 2)}, Fraction(1, 2), id="half"),
        # chained takes (0, 0) back at t = 3 only (and drops it again), so at t = 6 it
        # still differs from anchored, but it was not kept through 1..6; (2, 2): anchored
        # agrees with chained -> nothing kept
        pytest.param([P_C1, P_C1, P_C1 | {(0, 0)}, P_C1], P_S0 | {(2, 2)}, Fraction(0),
                     id="zero"),
        # forcing changed nothing: undefined
        pytest.param([P_S0], P_S0, None, id="none"),
    ],
)
def test_persistence_cases(chained, anchored_6, expected):
    anchored = Trajectory(SF_ANCHORED, _six(*([P_S0] * 5), anchored_6))

    got = metrics.persistence(P_S0, anchored, Trajectory(SF_CHAINED, _six(*chained)))
    assert got == expected


def test_persistence_needs_six_steps():
    short = Trajectory(SF_CHAINED, _steps([P_C1] * 5))

    with pytest.raises(ValueError):
        metrics.persistence(P_S0, short, short)


def _rq2_instance():
    """v0 likes role 0 (u = 5 there, 1 in role 1), v1 likes role 1 (5 there, 1 in
    role 0). Shifts: s0 (p0, r0), s1 (p1, r1), s2 (p0, r1) is the filler shift.
    """
    grid = Grid(n_periods=2, n_roles=2)
    volunteers = [grid.volunteer(0, pref_role=(2, 0)), grid.volunteer(1, pref_role=(0, 2))]
    shifts = [make_shift(0, period=0, role=0), make_shift(1, period=1, role=1),
              make_shift(2, period=0, role=1)]
    return grid.instance(volunteers, shifts)


RQ2_EVENTS = (
    AvailabilityRemoval(pairs=((0, 0),)),  # t = 1: v0 loses period 0
    *_fillers(4, shift=2),  # t = 2..5
    AvailabilityRestoration(volunteer=0, period=0),  # t = 6
    *_fillers(9, shift=2),  # t = 7..15
)
RQ2_S0 = frozenset({(0, 0), (1, 1)})  # U = 10
ONLY_V1 = frozenset({(1, 1)})  # U = 5
V1_BOTH = frozenset({(1, 0), (1, 1)})  # U = 6


def _rq2_result(anchored_from_6=RQ2_S0, unproven_last=False):
    """chained keeps ONLY_V1 throughout; anchored: V1_BOTH, ONLY_V1, V1_BOTH, ONLY_V1,
    ONLY_V1, then anchored_from_6 for t = 6..15. (Delta U_1 = 1 must NOT enter D_pre.)
    """
    anchored = [V1_BOTH, ONLY_V1, V1_BOTH, ONLY_V1, ONLY_V1] + [anchored_from_6] * 10
    chained_steps = list(_steps([ONLY_V1] * 15))
    if unproven_last:
        chained_steps[-1] = Step(15, ONLY_V1, _sf_records(LIMIT))
    return _result(
        _rq2_instance(), RQ2_S0,
        [Trajectory(SF_ANCHORED, _steps(anchored)), Trajectory(SF_CHAINED, tuple(chained_steps))],
        RQ2_EVENTS, kind=SequenceKind.RQ2, profile=SequenceProfile.RQ2, selected=(0, 0),
    )


def test_rq2_row_sums_and_lock_in():
    """Delta U = U(anchored) - U(chained): t = 3: 6 - 5 = 1, other pre steps 0 -> D_pre = 1;
    t = 6..15: 10 - 5 = 5 each -> D = 50. Forcing changed (0, 0); chained never takes it
    back, anchored has it at t = 6 -> P = 1; P > 0 and Delta U_6 = 5 > 0 -> lock-in.
    """
    row = metrics.rq2_row(_rq2_result())

    assert set(row) == set(metrics.RQ2_KEYS)
    assert (row["D"], row["D_pre"], row["delta_u_6"]) == (50, 1, 5)
    assert row["P"] == Fraction(1)
    assert row["lock_in"] is True
    assert row["certified"] is True


def test_rq2_row_no_lock_in_and_uncertified():
    """Anchored also stays at ONLY_V1 from t = 6: P = 0, Delta U_6 = 0 -> no lock-in;
    an unproven stage at t = 15 -> not certified.
    """
    row = metrics.rq2_row(_rq2_result(anchored_from_6=ONLY_V1, unproven_last=True))

    assert (row["D"], row["delta_u_6"], row["P"], row["lock_in"]) == (0, 0, Fraction(0), False)
    assert row["certified"] is False


def test_rq2_lock_in_needs_a_better_alternative():
    """Persistence alone is not enough to call the outcome lock-in.

    Anchored restores the changed cell, so P=1, but both schedules have U=5.
    Without a quality advantage at step 6, the lock-in flag should stay False.
    """
    row = metrics.rq2_row(_rq2_result(anchored_from_6=frozenset({(0, 0)})))

    assert row["P"] == Fraction(1) and row["delta_u_6"] == 0
    assert row["lock_in"] is False


def test_rq2_row_needs_one_anchored_and_one_chained_arm():
    result = _rq2_result()
    two_chained = _result(result.instance, result.s0,
                          [result.trajectories[1], result.trajectories[1]], RQ2_EVENTS,
                          kind=SequenceKind.RQ2, profile=SequenceProfile.RQ2, selected=(0, 0))

    with pytest.raises(ValueError):
        metrics.rq2_row(two_chained)
