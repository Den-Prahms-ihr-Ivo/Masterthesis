"""Check the data model, validation, immutable updates and sequence digests.

The examples deliberately include empty schedules, unavailable cells and
invalid values. These checks should fail before bad data reaches a solver.
"""

import json
import pytest
import dataclasses

from dataclasses import replace
from fractions import Fraction
import inspect
from repair import protocol
from repair.model import (
    Arm,
    LexResult,
    Sequence,
    StageRecord,
    Step,
    Shift,
    Volunteer,
    Instance,
    InstanceMeta,
    AvailabilityRemoval,
    AvailabilityRestoration,
    DemandChange,
    PreferenceChange,
    PolicyName,
    ReferenceMode,
    SequenceKind,
    SequenceProfile,
    StageStatus,
    _stable_json,
)
from tests.builders import Grid
import repair.model as model_module

# Helper

META = InstanceMeta(size_class=20, scale=1, seed=0)

# A small valid instance: 2 volunteers, 3 shifts, 2 periods x 2 roles


def _volunteers() -> list[Volunteer]:
    return [
        Volunteer(
            id=0,
            available=frozenset({0, 1}),
            skill=(1, 2),
            seniority=0,
            cap=10,
            pref_role=(0, 1),
            pref_time=(0, 2),
        ),
        Volunteer(
            id=1,
            available=frozenset({0}),
            skill=(3, 0),
            seniority=2,
            cap=11,
            pref_role=(2, 0),
            pref_time=(1, 1),
        ),
    ]


def _shifts() -> list[Shift]:
    return [
        Shift(id=0, period=0, role=0, demand=1, min_skill=1, expert=False),
        Shift(id=1, period=1, role=1, demand=2, min_skill=2, expert=True),
        Shift(id=2, period=0, role=1, demand=3, min_skill=1, expert=False),
    ]


def _kwargs(volunteers=None, shifts=None, **overrides) -> dict:
    kwargs = dict(
        volunteers=tuple(_volunteers() if volunteers is None else volunteers),
        shifts=tuple(_shifts() if shifts is None else tuple(shifts)),
        n_periods=2,
        n_roles=2,
        meta=META,
    )
    kwargs.update(overrides)
    return kwargs


def _vol(i: int, **changes) -> dict:
    """Valid instance kwargs with volunteer i changed."""
    vs = _volunteers()
    vs[i] = replace(vs[i], **changes)
    return _kwargs(volunteers=vs)


def _shift(j: int, **changes) -> dict:
    """Valid instance kwargs with shift j changed."""
    ss = _shifts()
    ss[j] = replace(ss[j], **changes)
    return _kwargs(shifts=ss)


def _replace_event(seq, index, new_event):
    events = list(seq.events)
    events[index] = new_event
    return dataclasses.replace(seq, events=tuple(events))


def _example_sequence() -> Sequence:
    """Builds a fresh object on every call, covering every event type."""
    return Sequence(
        kind=SequenceKind.RQ2,
        profile=SequenceProfile.RQ2,
        seed=12345,
        events=(
            AvailabilityRemoval(pairs=((3, 7),)),
            DemandChange(shift=5, delta=2, fallback=True),
            PreferenceChange(changes=((1, 2, 0),)),
            AvailabilityRestoration(volunteer=3, period=7),
        ),
        selected=(3, 7),
    )


def _model_dataclasses() -> list[type]:
    return [
        cls
        for _, cls in inspect.getmembers(model_module, inspect.isclass)
        if dataclasses.is_dataclass(cls) and cls.__module__ == model_module.__name__
    ]


def _stage(status, retried=False, name="stage") -> StageRecord:
    return StageRecord(
        name=name, status=status, value=5, bound=5.0, seconds=1.0, retried=retried
    )


# 1.1 model.py


@pytest.mark.parametrize(
    "n_volunteers, n_shifts, n_periods, n_roles, expected",
    [
        # 2x2 grid, size_class 40, partial availability: rules out all four wrong formulas
        (2, 3, 2, 2, 6),
        (1, 1, 2, 2, 1),
        (3, 0, 0, 0, 0),
    ],
)
def test_n_cells_is_volunteers_times_shifts(
    n_volunteers, n_shifts, n_periods, n_roles, expected
):
    """J counts every volunteer-shift combination, including unavailable cells."""
    grid = Grid(n_periods=n_periods, n_roles=n_roles)
    instance = grid.instance(
        grid.volunteers(n_volunteers), grid.shifts(n_shifts), size_class=40
    )

    assert instance.n_cells == expected


def test_dataclasses_are_frozen():
    """Changing a stored attribute should raise FrozenInstanceError."""

    grid = Grid(n_periods=28, n_roles=2)
    instance = grid.instance(grid.volunteers(1), grid.shifts(1), size_class=40)

    arm = Arm(policy=PolicyName.WQS, reference=ReferenceMode.CHAINED, lam=2)

    sequence = _example_sequence()

    step = Step(t=0, schedule=frozenset(), stages=())

    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.shifts = ()  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        arm.lam = -1  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        sequence.selected = (0, 0)  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        step.t = -1  # type: ignore[misc]


@pytest.mark.parametrize(
    "arm, expected",
    [
        (Arm(PolicyName.QO, None), "QO"),
        (Arm(PolicyName.SF, ReferenceMode.CHAINED), "SF-chained"),
        (Arm(PolicyName.SF, ReferenceMode.ANCHORED), "SF-anchored"),
        (Arm(PolicyName.WQS, ReferenceMode.CHAINED, lam=2), "WQS-chained-l2"),
        (
            Arm(PolicyName.HWF, ReferenceMode.CHAINED, alpha=Fraction(1)),
            "HWF-chained-a1",
        ),
        (
            Arm(PolicyName.HWF, ReferenceMode.CHAINED, alpha=Fraction(1, 2)),
            "HWF-chained-a1/2",
        ),
    ],
)
def test_arm_names(arm, expected):
    """Arm.name follows the naming scheme used in stored results."""
    assert arm.name == expected


def test_arm_names_unique_across_protocol_arms():
    """Distinct arms get distinct names (the naming scheme is injective).

    The protocol lists legitimately contain the same arm twice (SF-chained is in
    main_arms and rq2_arms), so uniqueness is checked over the *set* of arms, not
    over the concatenated list.
    """
    protocol_arms = (
        protocol.main_arms(2, Fraction(1))
        + protocol.rq2_arms()
        + protocol.sensitivity_arms((1, 4), (Fraction(1, 2), Fraction(2)))
    )
    # Arms the protocol doesn't use today but the scheme must still keep apart.
    extra_arms = (
        Arm(PolicyName.WQS, ReferenceMode.ANCHORED, lam=2),
        Arm(PolicyName.HWF, ReferenceMode.ANCHORED, alpha=Fraction(1)),
        Arm(PolicyName.QL, ReferenceMode.ANCHORED),
    )

    arms = set(protocol_arms) | set(extra_arms)
    names = {arm.name: arm for arm in arms}

    collisions = [arm for arm in arms if names[arm.name] != arm]
    assert not collisions, f"name collisions: {[(a, a.name) for a in collisions]}"
    assert len(names) == len(arms)


def test_lexresult_certified_false_without_schedule():
    """LexResult(schedule=None, ...) is not certified, even if every recorded stage is 'optimal'."""
    stages = (
        StageRecord(
            name="MinShortfall",
            status=StageStatus.OPTIMAL,
            value=0,
            bound=0.0,
            seconds=0.1,
        ),
        StageRecord(
            name="MaxQuality",
            status=StageStatus.OPTIMAL,
            value=6,
            bound=6.0,
            seconds=0.2,
        ),
    )
    result = LexResult(schedule=None, stages=stages)

    assert result.certified is False


@pytest.mark.parametrize("timed_out_stage", [0, 1, 2])
def test_lexresult_certified_false_with_time_limit_stage(timed_out_stage):
    """A schedule exists, but one stage hit the time limit -> not certified, wherever that stage is."""
    statuses = [StageStatus.OPTIMAL, StageStatus.OPTIMAL, StageStatus.OPTIMAL]
    statuses[timed_out_stage] = StageStatus.TIME_LIMIT
    stages = tuple(
        StageRecord(name=f"stage{i}", status=status, value=5, bound=7.0, seconds=600.0)
        for i, status in enumerate(statuses)
    )
    result = LexResult(schedule=frozenset({(0, 0)}), stages=stages)

    assert result.certified is False


def test_lexresult_certified_true_when_all_optimal():
    """A schedule and only 'optimal' stages -> certified."""
    stages = (
        StageRecord(
            name="MinShortfall",
            status=StageStatus.OPTIMAL,
            value=0,
            bound=0.0,
            seconds=0.1,
        ),
        StageRecord(
            name="MaxQuality",
            status=StageStatus.OPTIMAL,
            value=6,
            bound=6.0,
            seconds=0.2,
        ),
        StageRecord(
            name="MaxTieBreak",
            status=StageStatus.OPTIMAL,
            value=3,
            bound=3.0,
            seconds=0.1,
        ),
    )
    result = LexResult(schedule=frozenset({(0, 0)}), stages=stages)

    assert result.certified is True


def test_digest_equal_for_equal_sequences():
    """Two Sequences built from equal fields have the same digest."""
    a, b = _example_sequence(), _example_sequence()
    assert a is not b  # two separate objects, not one reused

    assert a.digest() == b.digest()
    assert isinstance(a.digest(), str) and len(a.digest()) == 64  # sha256 hex


@pytest.mark.parametrize(
    "make_variant",
    [
        pytest.param(
            lambda s: _replace_event(
                s, 1, DemandChange(shift=5, delta=-2, fallback=True)
            ),
            id="delta_flipped",
        ),
        pytest.param(
            lambda s: _replace_event(s, 0, AvailabilityRemoval(pairs=((3, 8),))),
            id="removal_pair_changed",
        ),
        pytest.param(
            lambda s: _replace_event(
                s, 1, DemandChange(shift=5, delta=2, fallback=False)
            ),
            id="fallback_flag_changed",
        ),
        pytest.param(
            lambda s: dataclasses.replace(
                s, events=(s.events[1], s.events[0]) + s.events[2:]
            ),
            id="event_order_swapped",
        ),
        pytest.param(
            lambda s: dataclasses.replace(s, seed=s.seed + 1), id="seed_changed"
        ),
        pytest.param(
            lambda s: dataclasses.replace(s, selected=(3, 8)),
            id="selected_pair_changed",
        ),
    ],
)
def test_digest_changes_when_one_event_changes(make_variant):
    """Any single change to the sequence changes its digest."""
    original = _example_sequence()
    variant = make_variant(original)

    assert variant != original  # guard: the variant really differs
    assert variant.digest() != original.digest()


def test_digest_pinned():
    """Keep one known sequence digest as a regression check.

    Changing event fields, serialization or class names can change old and new
    sequence hashes. This check makes that visible so the change can be
    reviewed and recorded instead of passing unnoticed.
    """
    # 2026-11-13: changed on purpose, `fallback` added to AvailabilityRemoval and
    # PreferenceChange (open choice 13.9). No stored results existed yet.
    EXAMPLE_DIGEST = "297e5471244d1ea5165f382b431f56cd6a869219df2781b54468825fa31e9eda"

    assert _example_sequence().digest() == EXAMPLE_DIGEST


def test_digest_input_is_canonical_json():
    """The digest hashes real JSON with plain values and tagged events, not a repr()."""
    payload = json.loads(_stable_json(_example_sequence()))

    assert payload["kind"] == "rq2"
    assert payload["selected"] == [3, 7]
    assert payload["events"][0] == {
        "__type__": "AvailabilityRemoval",
        "pairs": [[3, 7]],
        "fallback": False,
    }


def test_digest_equal_for_enum_and_its_string_value():
    """A sequence read back with plain strings still matches the original."""
    original = _example_sequence()
    reloaded = replace(original, kind="rq2", profile="rq2")  # type: ignore[arg-type]

    assert reloaded == original  # guard: StrEnum members equal their values
    assert reloaded.digest() == original.digest()


def test_stable_json_rejects_unknown_types():
    """An unsupported value raises instead of being hashed through str()."""
    with pytest.raises(TypeError):
        _stable_json(frozenset({(0, 0)}))


# 1.1 validation


def test_valid_instance_accepted():
    """A correct instance constructs, is hashable, and zero shifts are allowed."""
    instance = Instance(**_kwargs())
    hash(instance)  # all fields are immutable
    Instance(**_kwargs(shifts=[]))  # matches the 2 x 0 row of the n_cells test
    assert Instance(**_kwargs(shifts=[])).n_cells == 0


@pytest.mark.parametrize(
    "make_kwargs",
    [
        # ids must equal positions
        pytest.param(lambda: _vol(1, id=5), id="volunteer_id_not_position"),
        pytest.param(lambda: _shift(2, id=0), id="shift_id_not_position"),
        # shift fields
        pytest.param(lambda: _shift(0, period=2), id="shift_period_out_of_range"),
        pytest.param(lambda: _shift(0, role=-1), id="shift_role_negative"),
        pytest.param(lambda: _shift(0, demand=0), id="demand_zero"),
        pytest.param(lambda: _shift(0, min_skill=3), id="min_skill_three"),
        pytest.param(lambda: _shift(2, period=0, role=0), id="duplicate_period_role"),
        # volunteer fields
        pytest.param(
            lambda: _vol(0, available=frozenset({0, 2})), id="available_out_of_range"
        ),
        pytest.param(lambda: _vol(0, cap=0), id="cap_zero"),
        pytest.param(lambda: _vol(0, seniority=3), id="seniority_three"),
        pytest.param(lambda: _vol(0, skill=(1,)), id="skill_wrong_length"),
        pytest.param(lambda: _vol(0, pref_role=(0, 1, 2)), id="pref_role_wrong_length"),
        pytest.param(lambda: _vol(0, pref_time=(0,)), id="pref_time_wrong_length"),
        pytest.param(lambda: _vol(0, skill=(4, 1)), id="skill_value_four"),
        pytest.param(lambda: _vol(0, pref_role=(3, 0)), id="pref_role_value_three"),
        pytest.param(lambda: _vol(0, pref_time=(0, 3)), id="pref_time_value_three"),
        # container types (JSON gives lists; lists and sets break equality and hashing)
        pytest.param(
            lambda: {**_kwargs(), "volunteers": _volunteers()}, id="volunteers_is_list"
        ),
        pytest.param(lambda: {**_kwargs(), "shifts": _shifts()}, id="shifts_is_list"),
        pytest.param(lambda: _vol(0, available={0, 1}), id="available_is_set"),
        pytest.param(lambda: _vol(0, skill=[1, 2]), id="skill_is_list"),
        pytest.param(lambda: _vol(0, pref_time=[0, 2]), id="pref_time_is_list"),
        # lower bounds (the existing rows only test values that are too high)
        pytest.param(lambda: _shift(0, period=-1), id="shift_period_negative"),
        pytest.param(lambda: _shift(0, role=2), id="shift_role_out_of_range"),
        pytest.param(lambda: _shift(0, min_skill=0), id="min_skill_zero"),
        pytest.param(
            lambda: _vol(0, available=frozenset({-1})), id="available_negative"
        ),
        pytest.param(lambda: _vol(0, seniority=-1), id="seniority_negative"),
        pytest.param(lambda: _vol(0, skill=(-1, 1)), id="skill_negative"),
        pytest.param(lambda: _vol(0, pref_role=(-1, 0)), id="pref_role_negative"),
        pytest.param(lambda: _vol(0, pref_time=(0, -1)), id="pref_time_negative"),
    ],
)
def test_invalid_instance_rejected(make_kwargs):
    """Each row breaks exactly one invariant; construction must raise ValueError."""
    with pytest.raises(ValueError):
        Instance(**make_kwargs())


@pytest.mark.parametrize(
    "meta",
    [
        pytest.param(
            InstanceMeta(size_class=30, scale=1, seed=0), id="unknown_size_class"
        ),
        pytest.param(InstanceMeta(size_class=20, scale=0, seed=0), id="scale_zero"),
        pytest.param(
            InstanceMeta(size_class=20, scale=1, seed=0, under_supplied=True),
            id="under_supplied_not_40",
        ),
    ],
)
def test_invalid_meta_rejected(meta):
    """Invalid size classes and perturbation scales must fail instance validation."""
    with pytest.raises(ValueError):
        Instance(**_kwargs(meta=meta))


# 1.1 replace helpers


def test_with_volunteer_replaces_only_that_volunteer():
    """Only position volunteer.id changes; everything else and the original stay as they were."""
    instance = Instance(**_kwargs())
    new_volunteer = replace(instance.volunteers[1], cap=12)

    result = instance.with_volunteer(new_volunteer)

    assert result is not instance
    assert result.volunteers[1] == new_volunteer
    assert result.volunteers[0] == instance.volunteers[0]
    assert (
        result.shifts,
        result.n_periods,
        result.n_roles,
        result.meta,
        result.weights,
    ) == (
        instance.shifts,
        instance.n_periods,
        instance.n_roles,
        instance.meta,
        instance.weights,
    )
    assert instance.volunteers[1].cap == 11  # the original is untouched


def test_with_shift_replaces_only_that_shift():
    """Only position shift.id changes; everything else and the original stay as they were."""
    instance = Instance(**_kwargs())
    new_shift = replace(instance.shifts[2], demand=5)

    result = instance.with_shift(new_shift)

    assert result is not instance
    assert result.shifts[2] == new_shift
    assert result.shifts[:2] == instance.shifts[:2]
    assert result.volunteers == instance.volunteers
    assert instance.shifts[2].demand == 3  # the original is untouched


@pytest.mark.parametrize("bad_id", [2, -1])
def test_with_volunteer_unknown_id_raises(bad_id):
    """An id outside 0..|W|-1 raises; -1 must not silently replace the last volunteer."""
    instance = Instance(**_kwargs())
    with pytest.raises(ValueError):
        instance.with_volunteer(replace(instance.volunteers[0], id=bad_id))


@pytest.mark.parametrize("bad_id", [3, -1])
def test_with_shift_unknown_id_raises(bad_id):
    """An id outside 0..|S|-1 raises; -1 must not silently replace the last shift."""
    instance = Instance(**_kwargs())
    with pytest.raises(ValueError):
        instance.with_shift(replace(instance.shifts[0], id=bad_id))


def test_with_shift_revalidates():
    """Replacing with an invalid shift raises, because the new instance is validated too."""
    instance = Instance(**_kwargs())
    with pytest.raises(ValueError):
        instance.with_shift(replace(instance.shifts[0], demand=0))


def test_with_volunteer_revalidaates():
    """Replacing with an invalid volunteer raises, because the new instance is validated too."""
    instance = Instance(**_kwargs())
    with pytest.raises(ValueError):
        instance.with_volunteer(
            replace(instance.volunteers[0], available=frozenset({5}))
        )


@pytest.mark.parametrize("cls", _model_dataclasses(), ids=lambda cls: cls.__name__)
def test_every_model_dataclass_is_frozen(cls):
    """Instance hashing and the digest assume nothing in the model can change."""
    assert cls.__dataclass_params__.frozen


def test_frozen_check_sees_the_model_classes():
    """Check that the parametrised immutability test actually includes these classes."""
    names = {cls.__name__ for cls in _model_dataclasses()}
    assert {
        "Volunteer",
        "Shift",
        "Instance",
        "InstanceMeta",
        "Arm",
        "Sequence",
        "StageRecord",
        "LexResult",
        "Step",
        "DemandChange",
        "AvailabilityRemoval",
        "PreferenceChange",
        "AvailabilityRestoration",
    } <= names


def test_lexresult_certified_false_without_stages():
    """A schedule with no recorded stage proves nothing (all(()) would say True)."""
    assert LexResult(schedule=frozenset({(0, 0)}), stages=()).certified is False


def test_lexresult_certified_false_with_no_solution_stage():
    """NO_SOLUTION is not optimal, even if a schedule is present."""
    stages = (_stage(StageStatus.OPTIMAL), _stage(StageStatus.NO_SOLUTION))
    assert LexResult(schedule=frozenset({(0, 0)}), stages=stages).certified is False


def test_lexresult_certified_true_after_successful_retry():
    """A stage proven optimal on the 2τ retry counts as certified; the flag only records it."""
    stages = (_stage(StageStatus.OPTIMAL), _stage(StageStatus.OPTIMAL, retried=True))
    assert LexResult(schedule=frozenset({(0, 0)}), stages=stages).certified is True
