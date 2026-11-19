"""scripts/calibrate.py: the pilot calibrations (phi, robustness scale k, RQ2 background).

Most tests replace the solver and the samplers with scripted fakes, so they check the
SELECTION LOGIC and the BOOKKEEPING without SCIP:
  - phi: the 80 % rule (inclusive), the largest passing factor, common candidate seeds,
    and the certification rule (a zero incumbent proves zero; a positive one proves nothing);
  - intensity: closest scale, ties to the smaller k, paired seeds, the row arithmetic;
  - background: never reported as calibrated, steps 1 and 6 skipped;
  - _accepted_cases: the same seed scheme as make_instances (protocol.instance_seeds);
  - _non_inert: the DESIGN 13.5 definition (binding, or a better schedule exists);
  - main(): no CSV after a failure, exit codes.
Two tests use the brute force on the repair_slice fixture (real oracle, no SCIP), and two
`solver`/`slow` tests run the real thing once at n = 40.
"""

import csv
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from repair import perturbations
from repair.generator import generate_instance
from repair.model import (
    AvailabilityRemoval,
    LexResult,
    SequenceProfile,
    StageRecord,
    StageStatus,
)
from repair.objectives import MAX_QUALITY, MIN_SHORTFALL
from repair.protocol import InstanceJob, derive_seed, instance_seeds, load_protocol
from repair.runner import initial_schedule
from scripts import calibrate
from tests import bruteforce

PILOT = Path(__file__).parents[1] / "protocol" / "pilot.toml"
PROTOCOL = load_protocol(PILOT)
ROOT = PROTOCOL.root_seed
OPT, TL, NONE = StageStatus.OPTIMAL, StageStatus.TIME_LIMIT, StageStatus.NO_SOLUTION


def _lex(schedule, *statuses):
    return LexResult(
        schedule, tuple(StageRecord("MinShortfall", s, 0, 0.0, 0.1) for s in statuses)
    )


def _schedule_with_shortfall(value):
    """A fake schedule that carries its own shortfall (read by the fake oracle)."""
    return frozenset({(value, -1)})


# --- phi ---------------------------------------------------------------------------------


def _script_phi(monkeypatch, outcomes):
    """outcomes[phi] = [(shortfall or None, status), ...], one entry per candidate."""
    index_of = {derive_seed(ROOT, "calibration", "phi", i): i for i in range(50)}
    calls = []

    def fake_generate(cfg, size_class, seed, under_supplied=False):
        calls.append((cfg.workforce_factor, size_class, seed, under_supplied))
        return SimpleNamespace(phi=cfg.workforce_factor, index=index_of[seed])

    def fake_solve(protocol, instance, specs):
        assert specs == (MIN_SHORTFALL,)
        value, status = outcomes[instance.phi][instance.index]
        schedule = None if value is None else _schedule_with_shortfall(value)
        return _lex(schedule, status)

    monkeypatch.setattr(calibrate, "generate_instance", fake_generate)
    monkeypatch.setattr(calibrate, "_solve", fake_solve)
    monkeypatch.setattr(
        calibrate, "oracle", SimpleNamespace(shortfall=lambda s, inst: next(iter(s))[0])
    )
    return calls


def _proven(*shortfalls):
    return [(v, OPT) for v in shortfalls]


def test_phi_selects_the_largest_passing_factor(monkeypatch):
    """0.9: 3/5 positive (60 %, fails); 0.8: 4/5 (80 %, passes); 0.7: 5/5 (passes)."""
    _script_phi(monkeypatch, {
        0.9: _proven(1, 1, 1, 0, 0),
        0.8: _proven(2, 1, 1, 1, 0),
        0.7: _proven(3, 2, 2, 1, 1),
    })

    rows = calibrate.calibrate_phi(PROTOCOL, [0.9, 0.8, 0.7], 5)

    assert [r["phi"] for r in rows] == [0.9, 0.8, 0.7]
    assert [r["positive_shortfall"] for r in rows] == [3, 4, 5]
    assert [r["rate"] for r in rows] == [0.6, 0.8, 1.0]
    assert [r["passes"] for r in rows] == [False, True, True]
    assert [r["selected"] for r in rows] == [False, True, False]
    assert [r["n_volunteers"] for r in rows] == [36, 32, 28]
    assert all(r["root_seed"] == ROOT and r["n_candidates"] == 5 for r in rows)


@pytest.mark.parametrize("positives, passes", [(0, False), (3, False), (4, True), (5, True)])
def test_phi_80_percent_rule_is_inclusive(monkeypatch, positives, passes):
    _script_phi(monkeypatch, {0.8: _proven(*([1] * positives + [0] * (5 - positives)))})

    [row] = calibrate.calibrate_phi(PROTOCOL, [0.8], 5)

    assert row["passes"] is passes and row["selected"] is passes


def test_phi_selects_nothing_when_no_factor_passes(monkeypatch):
    _script_phi(monkeypatch, {0.9: _proven(0, 0), 0.8: _proven(1, 0)})

    rows = calibrate.calibrate_phi(PROTOCOL, [0.9, 0.8], 2)

    assert not any(r["selected"] for r in rows)


def test_phi_grid_is_sorted_descending_and_deduplicated(monkeypatch):
    _script_phi(monkeypatch, {0.9: _proven(1), 0.7: _proven(1)})

    rows = calibrate.calibrate_phi(PROTOCOL, [0.7, 0.9, 0.7], 1)

    assert [r["phi"] for r in rows] == [0.9, 0.7]
    assert [r["selected"] for r in rows] == [True, False]  # the LARGEST passing factor


def test_phi_uses_the_same_candidate_seeds_for_every_factor(monkeypatch):
    """phi is deliberately not in the seed: the factors differ only in workforce size."""
    calls = _script_phi(monkeypatch, {0.9: _proven(1, 1, 1), 0.6: _proven(1, 1, 1)})

    calibrate.calibrate_phi(PROTOCOL, [0.9, 0.6], 3)

    expected = [derive_seed(ROOT, "calibration", "phi", i) for i in range(3)]
    assert [seed for phi, _, seed, _ in calls if phi == 0.9] == expected
    assert [seed for phi, _, seed, _ in calls if phi == 0.6] == expected
    assert {(size, under) for _, size, _, under in calls} == {(40, True)}


def test_phi_zero_incumbent_at_time_limit_counts_as_zero(monkeypatch):
    """Shortfall is >= 0, so a feasible incumbent with 0 proves the optimum is 0."""
    _script_phi(monkeypatch, {0.8: [(0, TL), (1, OPT)]})

    [row] = calibrate.calibrate_phi(PROTOCOL, [0.8], 2)

    assert row["positive_shortfall"] == 1


@pytest.mark.parametrize(
    "outcome", [(2, TL), (None, NONE)], ids=["positive-incumbent-at-time-limit", "no-schedule"]
)
def test_phi_unproven_positive_shortfall_stops_the_calibration(monkeypatch, outcome):
    """A positive incumbent does not prove a positive optimum: never count it, never drop it."""
    _script_phi(monkeypatch, {0.8: [(1, OPT), outcome]})

    with pytest.raises(RuntimeError, match="not certified"):
        calibrate.calibrate_phi(PROTOCOL, [0.8], 2)


@pytest.mark.parametrize("grid", [[], [0.0], [-0.1], [1.5], [math.nan], [math.inf]])
def test_phi_rejects_an_invalid_grid(grid):
    with pytest.raises(ValueError, match="grid"):
        calibrate.calibrate_phi(PROTOCOL, grid, 1)


def test_phi_of_one_is_rejected_by_the_grid_check():
    """phi = 1 is not under-supplied (the generator refuses it, Table A1). The grid check
    should say so itself instead of passing it on to the generator.

    calibrate_phi accepted (0, 1] until 2026-11-18, so the error came from
    n_volunteers_for and did not mention the grid; it now checks 0 < phi < 1."""
    with pytest.raises(ValueError, match="grid"):
        calibrate.calibrate_phi(PROTOCOL, [1.0], 1)


@pytest.mark.parametrize("n", [0, -1, True, 2.0])
def test_phi_rejects_a_bad_candidate_count(n):
    with pytest.raises(ValueError, match="n_candidates"):
        calibrate.calibrate_phi(PROTOCOL, [0.8], n)


@pytest.mark.parametrize("phi", [x / 100 for x in range(95, 59, -5)])
def test_n_volunteers_column_matches_the_generator(phi):
    """The CSV column is computed as floor(40 * phi); the generator must agree."""
    cfg = replace(PROTOCOL.generator, workforce_factor=phi)
    instance = generate_instance(cfg, 40, 1, under_supplied=True)
    assert len(instance.volunteers) == math.floor(40 * phi)


# --- intensity ---------------------------------------------------------------------------


def _script_intensity(monkeypatch, target, independent):
    """target[i] and independent[k][i]: non-inert counts of instance i's sequences."""
    n = len(target)
    cases = [
        (i, SimpleNamespace(meta=SimpleNamespace(seed=100 + i)), frozenset({(i, 0)}))
        for i in range(n)
    ]
    calls = []

    def fake_cases(protocol, n_instances):
        assert n_instances == n
        return iter(cases)

    def fake_rq1(instance, s0, profile, seed, **kwargs):
        i = instance.meta.seed - 100
        calls.append(dict(i=i, s0=s0, profile=profile, seed=seed, **kwargs))
        return SimpleNamespace(i=i, **kwargs)

    def fake_count(protocol, instance, s0, sequence, *, background_only=False):
        assert not background_only
        if sequence.targeting == "s0":
            return target[sequence.i]
        return independent[sequence.scale][sequence.i]

    monkeypatch.setattr(calibrate, "_accepted_cases", fake_cases)
    monkeypatch.setattr(calibrate, "rq1_sequence", fake_rq1)
    monkeypatch.setattr(calibrate, "_count_non_inert", fake_count)
    return cases, calls


def test_intensity_selects_the_scale_closest_to_the_target(monkeypatch):
    """Target total 8. Independent totals 2, 7, 11, 16: scale 2 is 1 away."""
    _script_intensity(
        monkeypatch, [3, 5], {1: [1, 1], 2: [3, 4], 3: [5, 6], 4: [8, 8]}
    )

    rows = calibrate.calibrate_intensity(PROTOCOL, [1, 2, 3, 4], 2)

    assert [r["scale"] for r in rows] == [1, 2, 3, 4]
    assert [r["selected"] for r in rows] == [False, True, False, False]


def test_intensity_tie_goes_to_the_smaller_scale(monkeypatch):
    """Target 8; scale 1 gives 6 and scale 2 gives 10: both 2 away."""
    _script_intensity(monkeypatch, [4, 4], {1: [3, 3], 2: [5, 5]})

    rows = calibrate.calibrate_intensity(PROTOCOL, [2, 1], 2)

    assert [(r["scale"], r["selected"]) for r in rows] == [(1, True), (2, False)]


def test_intensity_row_arithmetic(monkeypatch):
    _script_intensity(monkeypatch, [3, 5], {3: [6, 6]})

    [row] = calibrate.calibrate_intensity(PROTOCOL, [3], 2)

    assert row == {
        "root_seed": ROOT,
        "scale": 3,
        "n_instances": 2,
        "n_sequences": 2,
        "n_events": 30,
        "target_non_inert": 8,
        "independent_non_inert": 12,
        "target_mean": 4.0,
        "independent_mean": 6.0,
        "target_rate": 8 / 30,
        "independent_rate": 12 / 30,
        "mean_difference": 2.0,
        "selected": True,
    }


def test_intensity_scales_are_sorted_and_deduplicated(monkeypatch):
    _script_intensity(monkeypatch, [1], {1: [1], 3: [1]})

    rows = calibrate.calibrate_intensity(PROTOCOL, [3, 1, 3], 1)

    assert [r["scale"] for r in rows] == [1, 3]


def test_intensity_sequences_are_paired_within_an_instance(monkeypatch):
    """One seed per instance, shared by the targeted sequence and every scale; the
    targeted design runs at k = n/20 = 2 against the instance's own S0."""
    cases, calls = _script_intensity(monkeypatch, [0, 0], {1: [0, 0], 4: [0, 0]})

    calibrate.calibrate_intensity(PROTOCOL, [1, 4], 2)

    for i, _, s0 in cases:
        mine = [c for c in calls if c["i"] == i]
        assert {c["seed"] for c in mine} == {derive_seed(ROOT, "calibration", "intensity", i)}
        assert all(c["s0"] == s0 and c["profile"] == SequenceProfile.BALANCED for c in mine)
        assert sorted((c["targeting"], c["scale"]) for c in mine) == [
            ("independent", 1), ("independent", 4), ("s0", 2)
        ]


@pytest.mark.parametrize("scales", [[], [0], [1, -2], [2.0]])
def test_intensity_rejects_bad_scales(scales):
    with pytest.raises(ValueError):
        calibrate.calibrate_intensity(PROTOCOL, scales, 1)


@pytest.mark.parametrize("n", [0, -3, True])
def test_intensity_rejects_a_bad_instance_count(n):
    with pytest.raises(ValueError, match="n_instances"):
        calibrate.calibrate_intensity(PROTOCOL, [1], n)


# --- background --------------------------------------------------------------------------


def test_background_is_a_diagnostic_at_k2_never_a_calibration(monkeypatch):
    cases: list[tuple[int, SimpleNamespace, frozenset]] = [
        (i, SimpleNamespace(meta=SimpleNamespace(seed=i)), frozenset()) for i in range(3)
    ]
    rq2_calls: list = []
    count_calls: list = []
    monkeypatch.setattr(calibrate, "_accepted_cases", lambda p, n: iter(cases))

    def fake_rq2(instance, s0, seed, *, scale):
        rq2_calls.append((instance.meta.seed, seed, scale))
        return "sequence"

    def fake_count(protocol, instance, s0, sequence, *, background_only=False):
        count_calls.append(background_only)
        return 2

    monkeypatch.setattr(calibrate, "rq2_sequence", fake_rq2)
    monkeypatch.setattr(calibrate, "_count_non_inert", fake_count)

    [row] = calibrate.calibrate_background(PROTOCOL, 3)

    assert rq2_calls == [(i, derive_seed(ROOT, "calibration", "background", i), 2) for i in range(3)]
    assert count_calls == [True, True, True]
    assert row["calibrated"] is False and row["scale"] == 2
    assert row["n_background_steps"] == 39
    assert row["non_inert_background_states"] == 6
    assert row["mean_non_inert_background_states"] == 2.0
    assert row["selection_rule"] == "default_n_over_20_no_target_specified"


# --- _count_non_inert --------------------------------------------------------------------


@pytest.mark.parametrize("background_only, counted", [(False, list(range(1, 16))),
                                                      (True, [2, 3, 4, 5] + list(range(7, 16)))])
def test_count_non_inert_visits_every_step_or_only_background_steps(
    monkeypatch, background_only, counted
):
    """Steps 1 and 6 are skipped in the count but still applied to later instances."""
    steps = [f"instance_{t}" for t in range(1, 16)]
    seen: list[str] = []

    def non_inert(p, inst_t, s0, ctx):
        seen.append(inst_t)
        return True

    monkeypatch.setattr(calibrate, "instances_along", lambda instance, events: steps)
    monkeypatch.setattr(calibrate, "_non_inert", non_inert)
    instance = SimpleNamespace(meta=SimpleNamespace(seed=1))
    sequence = SimpleNamespace(events=(), seed=2)

    got = calibrate._count_non_inert(
        PROTOCOL, instance, frozenset(), sequence, background_only=background_only
    )

    assert got == len(counted)
    assert seen == [f"instance_{t}" for t in counted]


# --- _non_inert (DESIGN 13.5) ------------------------------------------------------------


def _script_non_inert(monkeypatch, violations, s0_values, opt_values, status=OPT):
    """s0_values / opt_values = (shortfall, quality)."""
    s0, optimum = frozenset({("s0", 0)}), frozenset({("opt", 0)})
    values = {s0: s0_values, optimum: opt_values}
    solves = []

    def fake_solve(protocol, instance, specs):
        solves.append(specs)
        return _lex(optimum, status, status)

    monkeypatch.setattr(calibrate, "_solve", fake_solve)
    monkeypatch.setattr(calibrate, "oracle", SimpleNamespace(
        violations=lambda s, inst: violations if s == s0 else [],
        shortfall=lambda s, inst: values[s][0],
        quality=lambda s, inst: values[s][1],
    ))
    return s0, solves


def test_non_inert_when_s0_breaks_a_rule_without_solving(monkeypatch):
    s0, solves = _script_non_inert(monkeypatch, ["unavailable"], (0, 5), (0, 5))

    assert calibrate._non_inert(PROTOCOL, "instance", s0, "ctx") is True
    assert solves == []


@pytest.mark.parametrize(
    "s0_values, opt_values, expected",
    [((1, 9), (0, 5), True), ((0, 4), (0, 5), True), ((0, 5), (0, 5), False)],
    ids=["better-shortfall-exists", "better-quality-exists", "s0-still-optimal"],
)
def test_non_inert_compares_s0_with_the_lexicographic_optimum(
    monkeypatch, s0_values, opt_values, expected
):
    s0, solves = _script_non_inert(monkeypatch, [], s0_values, opt_values)

    assert calibrate._non_inert(PROTOCOL, "instance", s0, "ctx") is expected
    assert solves == [(MIN_SHORTFALL, MAX_QUALITY)]


def test_non_inert_refuses_an_unproven_optimum(monkeypatch):
    """A timeout must never be read as an inert event."""
    s0, _ = _script_non_inert(monkeypatch, [], (0, 5), (0, 5), status=TL)

    with pytest.raises(RuntimeError, match="ctx.*not certified"):
        calibrate._non_inert(PROTOCOL, "instance", s0, "ctx")


# --- _non_inert with the real oracle and the brute force (no SCIP) -----------------------


@pytest.fixture
def brute_force_calibrate(monkeypatch):
    monkeypatch.setattr(calibrate, "milp", SimpleNamespace(solve_stage=bruteforce.solve_stage))


def test_s0_is_inert_on_its_own_instance(repair_slice, brute_force_calibrate):
    s0 = initial_schedule(repair_slice, PROTOCOL.solver, bruteforce.solve_stage).schedule

    assert calibrate._non_inert(PROTOCOL, repair_slice, s0, "base") is False


def test_removing_an_assigned_pair_makes_s0_non_inert(repair_slice, brute_force_calibrate):
    s0 = initial_schedule(repair_slice, PROTOCOL.solver, bruteforce.solve_stage).schedule
    assert s0 is not None
    w, s = min(s0)
    period = next(shift.period for shift in repair_slice.shifts if shift.id == s)
    after = perturbations.apply_event(repair_slice, AvailabilityRemoval(pairs=((w, period),)))

    assert calibrate._non_inert(PROTOCOL, after, s0, "removal") is True


# --- _accepted_cases ---------------------------------------------------------------------


def _script_acceptance(monkeypatch, s0_status=OPT):
    calls = []

    def fake_accept(cfg, size_class, base, reserve, settings, solve, *, max_draws):
        calls.append((size_class, base, list(reserve), max_draws))
        return SimpleNamespace(meta=SimpleNamespace(seed=base)), None

    monkeypatch.setattr(calibrate, "accept_instance", fake_accept)
    monkeypatch.setattr(
        calibrate, "initial_schedule",
        lambda instance, settings, solve: _lex(frozenset({(0, 0)}), s0_status),
    )
    return calls


def test_accepted_cases_use_the_run_seed_scheme(monkeypatch):
    """The docstring promises make_instances' scheme: protocol.instance_seeds at n = 40.
    So the calibration instances ARE the pilot's main n = 40 instances (pilot root only;
    the main run's root is never touched)."""
    calls = _script_acceptance(monkeypatch)

    cases = list(calibrate._accepted_cases(PROTOCOL, 3))

    assert [i for i, _, _ in cases] == [0, 1, 2]
    for i, (size, base, reserve, max_draws) in enumerate(calls):
        assert size == 40 and max_draws == PROTOCOL.max_draws
        assert (base, reserve) == instance_seeds(PROTOCOL, InstanceJob(40, i))


def test_accepted_cases_stop_on_an_uncertified_s0(monkeypatch):
    _script_acceptance(monkeypatch, s0_status=TL)

    with pytest.raises(RuntimeError, match="S0.*not certified"):
        list(calibrate._accepted_cases(PROTOCOL, 1))


# --- main() ------------------------------------------------------------------------------


ROWS = [{"phi": 0.9, "selected": False}, {"phi": 0.8, "selected": True}]


def _main(tmp_path, what, *extra):
    out = tmp_path / "calibration"
    return out, calibrate.main([what, str(PILOT), "--out", str(out), *extra])


def test_main_writes_the_table_and_exits_0_when_a_value_is_selected(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(calibrate, "calibrate_phi", lambda p, grid, n: ROWS)

    out, code = _main(tmp_path, "phi")

    assert code == 0
    with (out / "calibration_phi.csv").open() as file:
        assert list(csv.DictReader(file)) == [
            {"phi": "0.9", "selected": "False"}, {"phi": "0.8", "selected": "True"}
        ]
    assert "Written:" in capsys.readouterr().out


def test_main_passes_the_cli_arguments_through(tmp_path, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        calibrate, "calibrate_intensity",
        lambda p, scales, n: seen.update(scales=scales, n=n, root=p.root_seed) or ROWS,
    )

    _main(tmp_path, "intensity", "--scales", "2", "3", "--n-instances", "7")

    assert seen == {"scales": [2, 3], "n": 7, "root": ROOT}


def test_main_exits_1_when_nothing_is_selected_but_keeps_the_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        calibrate, "calibrate_phi", lambda p, grid, n: [{"phi": 0.9, "selected": False}]
    )

    out, code = _main(tmp_path, "phi")

    assert code == 1
    assert (out / "calibration_phi.csv").exists()
    assert "No tested value meets the selection rule" in capsys.readouterr().err


def test_main_failure_writes_no_table(tmp_path, monkeypatch, capsys):
    def fail(p, grid, n):
        raise RuntimeError("phi=0.8, candidate=3: optimum not certified")

    monkeypatch.setattr(calibrate, "calibrate_phi", fail)

    out, code = _main(tmp_path, "phi")

    assert code == 1
    assert not out.exists()
    assert "Calibration failed: phi=0.8" in capsys.readouterr().err


def test_main_background_exits_0_and_says_it_is_a_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        calibrate, "calibrate_background", lambda p, n: [{"scale": 2, "calibrated": False}]
    )

    _, code = _main(tmp_path, "background")

    assert code == 0
    assert "k=2 is a documented default, not a calibrated value." in capsys.readouterr().out


# --- the real thing, once ----------------------------------------------------------------


@pytest.mark.solver
@pytest.mark.slow
def test_phi_with_scip_is_deterministic():
    first = calibrate.calibrate_phi(PROTOCOL, [0.7], 2)
    second = calibrate.calibrate_phi(PROTOCOL, [0.7], 2)

    assert first == second
    assert first[0]["n_volunteers"] == 28 and 0 <= first[0]["positive_shortfall"] <= 2


@pytest.mark.solver
@pytest.mark.slow
def test_intensity_with_scip_on_one_instance():
    rows = calibrate.calibrate_intensity(PROTOCOL, [1, 2], 1)

    assert [r["scale"] for r in rows] == [1, 2]
    assert sum(r["selected"] for r in rows) == 1
    assert all(0 <= r["independent_non_inert"] <= 15 for r in rows)
    assert len({r["target_non_inert"] for r in rows}) == 1  # one targeted count for all
