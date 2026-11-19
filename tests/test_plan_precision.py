"""scripts/plan_precision.py: N_I, certification share and core-hour projection from a pilot.

Unit tests with hand values for the three pure helpers (projected_core_hours,
pilot_timing, planned_matrix), then main() on fake pilot results. Fake results carry the
real record types (CoverageStep, Step, StageRecord) and the real analysis.precision_plan
runs on a planted instance table (tests/fake_results.py), so the shapes are the real ones.

Fixed on 2026-11-18 (these tests found it):
  - _stages read `step.stages` on coverage steps; a CoverageStep has one `record`.
  - main() iterated precision_plan's DataFrame as if it were a list of rows.
  - projected_core_hours looked up stages with zero planned solves (KeyError).
Two further tests state behaviour to decide (marked DECISION).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from repair import analysis
from repair.model import CoverageStep, StageRecord, StageStatus, Step
from scripts import plan_precision
from tests.fake_results import planted_instance_table, planted_rq2_table

OPT = StageStatus.OPTIMAL


def _rec(name, seconds, status=OPT):
    return StageRecord(name, status, 0, 0.0, seconds)


def _result(cell, size, n_arms, *, coverage_s=1.0, arm_stages=(("MaxQuality", 2.0),), prefix=15):
    """A SequenceResult look-alike: one step, one coverage record, n_arms trajectories."""
    stages = tuple(_rec(name, s) for name, s in arm_stages)
    return SimpleNamespace(
        cell=cell,
        instance=SimpleNamespace(meta=SimpleNamespace(size_class=size)),
        coverage=(CoverageStep(1, 0, _rec("MinShortfall", coverage_s)),),
        trajectories=tuple(
            SimpleNamespace(steps=(Step(1, frozenset(), stages),)) for _ in range(n_arms)
        ),
        prefix=prefix,  # read by the fake certified_prefix in the main() tests
    )


# --- projected_core_hours ----------------------------------------------------------------


def test_projected_core_hours_hand_value():
    """2 s x 1,800 solves + 10 s x 360 solves = 7,200 s = 2 core-hours."""
    seconds = {(20, "A"): 2.0, (20, "B"): 10.0}
    matrix = {(20, "A"): 1800.0, (20, "B"): 360.0}

    assert plan_precision.projected_core_hours(seconds, matrix) == 2.0


def test_projected_core_hours_names_a_stage_without_pilot_timing():
    with pytest.raises(ValueError, match=r"\(80, 'MaxQuality'\)"):
        plan_precision.projected_core_hours({(20, "A"): 1.0}, {(20, "A"): 1, (80, "MaxQuality"): 5})


def test_projected_core_hours_ignores_untimed_stages_with_zero_solves():
    assert plan_precision.projected_core_hours({(20, "A"): 1.0}, {(20, "A"): 3600, (80, "B"): 0}) == 1.0


# --- _stages and pilot_timing ------------------------------------------------------------


def test_stages_include_the_coverage_record():
    """Every solve is costed: the shared coverage stage, then each arm's stages."""
    result = _result("main", 20, 2, arm_stages=(("MinDistance", 1.0), ("MaxQuality", 2.0)))

    names = [s.name for s in plan_precision._stages(result)]

    assert names == ["MinShortfall", "MinDistance", "MaxQuality", "MinDistance", "MaxQuality"]


def test_pilot_timing_means_per_size_and_stage():
    results = [
        _result("main", 20, 5, coverage_s=1.0, arm_stages=(("MaxQuality", 2.0),)),
        _result("main", 20, 5, coverage_s=3.0, arm_stages=(("MaxQuality", 4.0),)),
        _result("main", 40, 5, coverage_s=5.0, arm_stages=(("MaxQuality", 6.0),)),
    ]

    seconds, solves = plan_precision.pilot_timing(results)

    assert seconds == {
        (20, "MinShortfall"): 2.0, (20, "MaxQuality"): 3.0,
        (40, "MinShortfall"): 5.0, (40, "MaxQuality"): 6.0,
    }
    assert solves[("main", 20, 5)] == {"MinShortfall": 1.0, "MaxQuality": 5.0}


def test_pilot_timing_keeps_job_types_with_different_arm_counts_apart():
    """Main n = 40 balanced jobs carry nine arms, the others five (exposé §4.5)."""
    results = [_result("main", 40, 5), _result("main", 40, 9), _result("rq2", 40, 2)]

    _, solves = plan_precision.pilot_timing(results)

    assert solves[("main", 40, 5)]["MaxQuality"] == 5.0
    assert solves[("main", 40, 9)]["MaxQuality"] == 9.0
    assert solves[("rq2", 40, 2)]["MaxQuality"] == 2.0


# --- planned_matrix ----------------------------------------------------------------------


def _job(cell, size, n_arms):
    return SimpleNamespace(cell=cell, instance=SimpleNamespace(size_class=size), arms=(None,) * n_arms)


def test_planned_matrix_multiplies_jobs_by_solves_per_job():
    solves = {("main", 20, 5): {"MinShortfall": 15.0, "MaxQuality": 75.0}}
    jobs = [_job("main", 20, 5)] * 4

    assert plan_precision.planned_matrix(jobs, solves) == {
        (20, "MinShortfall"): 60.0, (20, "MaxQuality"): 300.0
    }


def test_planned_matrix_costs_a_cell_missing_from_the_pilot_like_main():
    """A robustness job has the five main arms, so it is costed like a main n = 40 job."""
    solves = {("main", 40, 5): {"MaxQuality": 10.0}}

    assert plan_precision.planned_matrix([_job("robustness", 40, 5)], solves) == {(40, "MaxQuality"): 10.0}


def test_planned_matrix_prefers_the_cells_own_pilot_counts():
    solves = {("main", 40, 5): {"MaxQuality": 10.0}, ("robustness", 40, 5): {"MaxQuality": 99.0}}

    assert plan_precision.planned_matrix([_job("robustness", 40, 5)], solves) == {(40, "MaxQuality"): 99.0}


def test_planned_matrix_refuses_a_job_type_the_pilot_never_ran():
    with pytest.raises(ValueError, match="no job like"):
        plan_precision.planned_matrix([_job("main", 80, 5)], {("main", 40, 5): {"A": 1.0}})


# --- main() ------------------------------------------------------------------------------


PILOT_TOML = """
name = "pilot-test"
date = ""
root_seed = 20261109
horizon = 15
size_classes = [20, 40]
cells = ["main", "rq2", "robustness", "undersupplied"]
n_instances_secondary = 2
n_sequences = 1
lam = 2
alpha = "1"
lam_grid = [1, 4]
alpha_grid = ["1/2", "2"]
robustness_scale = 2
workforce_factor = 0.8

[n_instances]
20 = 10
40 = 10

[solver]
time_limit = 600.0
s0_time_limit = 600.0
"""

PILOT_RESULTS = [
    _result("main", 20, 5, prefix=15),
    _result("main", 20, 5, prefix=7),
    _result("rq2", 20, 2),
    _result("main", 40, 5),
    _result("main", 40, 9),
    _result("rq2", 40, 2),
]


def _run_main(tmp_path, monkeypatch, plan, *, budget=1000.0, results=PILOT_RESULTS):
    protocol_file = tmp_path / "pilot.toml"
    protocol_file.write_text(PILOT_TOML)
    planned = {}
    real_jobs = plan_precision.protocol_module.jobs

    def spy_jobs(protocol):
        planned["n_instances"] = protocol.n_instances
        planned["cells"] = protocol.cells
        return real_jobs(protocol)

    monkeypatch.setattr(plan_precision, "storage", SimpleNamespace(read_results=lambda path: iter(results)))
    def precision_plan(inst, rq2=None):
        planned["precision_inputs"] = (inst, rq2)
        return plan

    monkeypatch.setattr(plan_precision, "analysis", SimpleNamespace(
        sequence_table=lambda rs: "sequence table",
        instance_table=lambda seq: "instance table",
        rq2_table=lambda rs: "rq2 table",
        precision_plan=precision_plan,
        certified_prefix=lambda r: r.prefix,
    ))
    monkeypatch.setattr(plan_precision.protocol_module, "jobs", spy_jobs)
    monkeypatch.setattr(plan_precision, "BUDGET_CORE_HOURS", budget)
    code = plan_precision.main([str(tmp_path / "results"), "--protocol", str(protocol_file)])
    return code, planned


def _plan(n_instances=10):
    """The real precision_plan on planted tables: the DataFrame main() really receives,
    H3 included."""
    return analysis.precision_plan(
        planted_instance_table(n_instances, {}, sd=1.0, seed=0, size_classes=(20, 40)),
        rq2=planted_rq2_table(n_instances, 0.0, sd=1.0, seed=1, size_classes=(20, 40)),
    )


def test_main_plans_with_each_size_classes_max_row(tmp_path, monkeypatch, capsys):
    plan = _plan()
    expected = {
        size: int(plan.loc[(plan.size_class == size) & (plan.contrast == "max"), "n_required"].item())
        for size in (20, 40)
    }

    code, planned = _run_main(tmp_path, monkeypatch, plan, budget=1e12)

    out = capsys.readouterr().out
    assert code == 0
    assert planned["n_instances"] == expected
    assert all(type(n) is int for n in planned["n_instances"].values())
    assert f"n=20  N_I = {expected[20]}" in out and f"n=40  N_I = {expected[40]}" in out
    assert "h1_gap" in out and "h2_balanced" in out and "h3" in out


def test_main_passes_the_rq2_table_for_h3(tmp_path, monkeypatch):
    """N_I must also cover H3 (DESIGN.md 13.8), so the pilot's RQ2 rows reach the plan."""
    _, planned = _run_main(tmp_path, monkeypatch, _plan(), budget=1e12)

    assert planned["precision_inputs"] == ("instance table", "rq2 table")


def test_main_projects_the_whole_matrix_with_all_four_cells(tmp_path, monkeypatch):
    _, planned = _run_main(tmp_path, monkeypatch, _plan(), budget=1e12)

    assert tuple(planned["cells"]) == tuple(plan_precision.protocol_module.CELLS)


def test_main_prints_the_certified_share_per_size_class(tmp_path, monkeypatch, capsys):
    """n = 20: prefixes 15, 7 and 15 (rq2) -> 2 of 3 certified through T = 15."""
    _run_main(tmp_path, monkeypatch, _plan(), budget=1e12)

    out = capsys.readouterr().out
    assert "n=20  certified through T=15: 67% of 3" in out
    assert "n=40  certified through T=15: 100% of 3" in out


def test_main_prints_seconds_per_solve_and_the_projection(tmp_path, monkeypatch, capsys):
    _run_main(tmp_path, monkeypatch, _plan(), budget=1e12)

    out = capsys.readouterr().out
    assert "n=20  MaxQuality" in out and "2.0 s per solve" in out
    assert "projected" in out and "core-hours of" in out


@pytest.mark.parametrize("budget, code", [(1e12, 0), (1e-6, 1)], ids=["within-budget", "over-budget"])
def test_main_exit_status_is_the_budget_check(tmp_path, monkeypatch, budget, code):
    assert _run_main(tmp_path, monkeypatch, _plan(), budget=budget)[0] == code


def test_main_refuses_an_undefined_n_i_with_a_clear_message(tmp_path, monkeypatch):
    """DECISION: precision_plan sets the "max" row to NaN when any contrast is undefined
    (e.g. a pilot with one instance per class). main() must not plan with NaN; it should
    stop and name the size class."""
    plan = _plan()
    plan.loc[(plan.size_class == 40) & (plan.contrast == "max"), "n_required"] = float("nan")

    with pytest.raises(ValueError, match="n=40"):
        _run_main(tmp_path, monkeypatch, plan)


def test_main_refuses_a_size_class_without_pilot_results(tmp_path, monkeypatch):
    """DECISION: a size class in the protocol with no pilot results (e.g. every n = 40 job
    failed) should stop with a message naming it, not with ZeroDivisionError or an empty
    max()."""
    only_20 = [r for r in PILOT_RESULTS if r.instance.meta.size_class == 20]
    plan = _plan()
    plan = plan[plan.size_class == 20]

    with pytest.raises(ValueError, match="n=40"):
        _run_main(tmp_path, monkeypatch, plan, results=only_20)
