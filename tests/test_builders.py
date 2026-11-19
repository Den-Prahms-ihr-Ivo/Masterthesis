"""Check the fixture builders before relying on them in the other tests."""

import pytest

from repair.model import Instance, InstanceMeta, Shift, UtilityWeights, Volunteer
from tests.builders import Grid, make_instance, make_shift, make_volunteer

# Make_volunteer


def test_make_volunteer_defaults():
    """Available in every period, skill 1 in every role, preferences 0, seniority 0, cap 10."""
    v = make_volunteer(3, n_periods=2, n_roles=3)

    assert isinstance(v, Volunteer)
    assert v.id == 3
    assert v.available == frozenset({0, 1})
    assert v.skill == (1, 1, 1)
    assert v.pref_role == (0, 0, 0)
    assert v.pref_time == (0, 0)
    assert (v.seniority, v.cap) == (0, 10)


def test_make_volunteer_returns_tuples_and_frozenset():
    """Containers have the types the Instance validator requires."""
    v = make_volunteer(0, n_periods=2, n_roles=2)

    assert type(v.available) is frozenset
    assert (
        type(v.skill) is tuple
        and type(v.pref_role) is tuple
        and type(v.pref_time) is tuple
    )


def test_make_volunteer_requires_dimensions():
    """n_periods and n_roles have no defaults: a silent default would give wrong-length tuples."""
    with pytest.raises(TypeError):
        make_volunteer(0)  # type: ignore[call-arg]  # pylint: disable=missing-kwoa


@pytest.mark.parametrize(
    "field, given, expected",
    [
        pytest.param("skill", {1: 3}, (1, 3), id="skill_default_1_elsewhere"),
        pytest.param("pref_role", {0: 2}, (2, 0), id="pref_role_default_0_elsewhere"),
    ],
)
def test_dict_per_role_is_converted(field, given, expected):
    """A {role: value} dict becomes a dense per-role tuple; missing roles get the default."""
    v = make_volunteer(0, n_periods=1, n_roles=2, **{field: given})
    assert getattr(v, field) == expected


def test_dict_pref_time_is_converted():
    """A {period: value} dict becomes a dense per-period tuple; missing periods get 0."""
    v = make_volunteer(0, n_periods=3, n_roles=1, pref_time={2: 1})
    assert v.pref_time == (0, 0, 1)


@pytest.mark.parametrize(
    "field, given",
    [
        pytest.param("skill", {2: 1}, id="skill_role_out_of_range"),
        pytest.param("pref_role", {-1: 1}, id="pref_role_negative"),
        pytest.param("pref_time", {5: 1}, id="pref_time_period_out_of_range"),
    ],
)
def test_dict_index_out_of_range_raises(field, given):
    """A dict key outside the role/period range is a fixture typo, not something to drop silently."""
    with pytest.raises(ValueError):
        make_volunteer(0, n_periods=2, n_roles=2, **{field: given})


def test_explicit_values_pass_through():
    """Tuples, seniority and cap given explicitly are used unchanged."""
    v = make_volunteer(
        1,
        n_periods=2,
        n_roles=2,
        skill=(3, 0),
        pref_role=(2, 1),
        pref_time=(0, 2),
        seniority=2,
        cap=1,
    )
    assert (v.skill, v.pref_role, v.pref_time, v.seniority, v.cap) == (
        (3, 0),
        (2, 1),
        (0, 2),
        2,
        1,
    )


@pytest.mark.parametrize("given", [[0], {0}, (0,), frozenset({0})])
def test_available_is_converted_to_frozenset(given):
    """Any iterable of periods becomes a frozenset; the empty set means never available."""
    v = make_volunteer(0, n_periods=2, n_roles=1, available=given)
    assert v.available == frozenset({0}) and type(v.available) is frozenset
    assert (
        make_volunteer(0, n_periods=2, n_roles=1, available=[]).available == frozenset()
    )


# Make_shift


def test_make_shift_defaults():
    """Period 0, role 0, demand 1, min_skill 1, no expert."""
    s = make_shift(4)
    assert isinstance(s, Shift)
    assert (s.id, s.period, s.role, s.demand, s.min_skill, s.expert) == (
        4,
        0,
        0,
        1,
        1,
        False,
    )


def test_make_shift_explicit_values():
    s = make_shift(1, period=1, role=1, demand=3, min_skill=2, expert=True)
    assert (s.period, s.role, s.demand, s.min_skill, s.expert) == (1, 1, 3, 2, True)


# Make_instance


def _two_by_two():
    volunteers = [make_volunteer(w, n_periods=2, n_roles=2) for w in range(2)]
    shifts = [make_shift(0, period=0, role=0), make_shift(1, period=1, role=1)]
    return volunteers, shifts


def test_make_instance_converts_lists_to_tuples():
    """Lists of volunteers and shifts become tuples; the result is a valid, hashable Instance."""
    volunteers, shifts = _two_by_two()
    instance = make_instance(volunteers, shifts, n_periods=2, n_roles=2)

    assert isinstance(instance, Instance)
    assert type(instance.volunteers) is tuple and type(instance.shifts) is tuple
    assert instance.volunteers == tuple(volunteers) and instance.shifts == tuple(shifts)
    hash(instance)


def test_make_instance_meta_defaults():
    """Default meta: size class 20, scale 1, seed 0, not under-supplied."""
    instance = make_instance(*_two_by_two(), n_periods=2, n_roles=2)
    assert instance.meta == InstanceMeta(
        size_class=20, scale=1, seed=0, under_supplied=False
    )


def test_make_instance_passes_meta_fields():
    instance = make_instance(
        *_two_by_two(),
        n_periods=2,
        n_roles=2,
        size_class=40,
        scale=2,
        seed=7,
        under_supplied=True,
    )
    assert instance.meta == InstanceMeta(
        size_class=40, scale=2, seed=7, under_supplied=True
    )


def test_make_instance_default_weights():
    """weights=None means the Table A1 defaults."""
    instance = make_instance(*_two_by_two(), n_periods=2, n_roles=2)
    assert instance.weights == UtilityWeights()


def test_make_instance_is_validated():
    """The builder does not bypass the Instance validator: a duplicate (period, role) raises."""
    volunteers, _ = _two_by_two()
    duplicate = [make_shift(0, period=0, role=0), make_shift(1, period=0, role=0)]
    with pytest.raises(ValueError):
        make_instance(volunteers, duplicate, n_periods=2, n_roles=2)


@pytest.mark.parametrize(
    "make_kwargs",
    [
        pytest.param(
            lambda vs: {"volunteers": vs, "shifts": [make_shift(0, period=2)]},
            id="period_out_of_range",
        ),
        pytest.param(
            lambda vs: {"volunteers": vs, "shifts": [make_shift(0, role=2)]},
            id="role_out_of_range",
        ),
        pytest.param(
            lambda vs: {"volunteers": vs, "shifts": [make_shift(1)]},
            id="shift_id_not_position",
        ),
        pytest.param(
            lambda vs: {"volunteers": vs[::-1], "shifts": []},
            id="volunteer_id_not_position",
        ),
    ],
)
def test_make_instance_rejects_bad_ids_and_slots(make_kwargs):
    """With the builder's own checks gone, these still raise through the validator."""
    volunteers, _ = _two_by_two()
    with pytest.raises(ValueError):
        make_instance(**make_kwargs(volunteers), n_periods=2, n_roles=2)


# Grid


def test_grid_builds_consistent_instance():
    """Volunteers and shifts take their shape from the grid, and the validator accepts them."""
    grid = Grid(n_periods=3, n_roles=2)
    instance = grid.instance(grid.volunteers(2), grid.shifts(4), size_class=40)

    assert (instance.n_periods, instance.n_roles) == (3, 2)
    assert all(len(v.pref_time) == 3 for v in instance.volunteers)
    assert all(len(v.skill) == 2 for v in instance.volunteers)
    assert instance.meta.size_class == 40


def test_grid_volunteer_passes_fields_on():
    v = Grid(n_periods=1, n_roles=2).volunteer(0, skill=(3, 2), pref_role={1: 2})
    assert (v.skill, v.pref_role) == ((3, 2), (0, 2))


def test_grid_shifts_use_distinct_slots():
    shifts = Grid(n_periods=2, n_roles=2).shifts(4)

    assert [s.id for s in shifts] == [0, 1, 2, 3]
    assert len({(s.period, s.role) for s in shifts}) == 4


def test_grid_shifts_can_fill_exactly_and_no_more():
    """Asking for too many raises, instead of quietly returning fewer."""
    grid = Grid(n_periods=2, n_roles=2)
    assert len(grid.shifts(4)) == 4
    with pytest.raises(ValueError):
        grid.shifts(5)
