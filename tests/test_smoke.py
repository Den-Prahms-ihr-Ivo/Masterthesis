"""Run a small experiment through the whole pipeline with SCIP.

Use one n=20 instance over three days and four sequence jobs: the three
main profiles plus RQ2. Generate inputs, run jobs, resume, write analysis
tables and repeat two finished jobs. Everything is written under tmp_path.
This checks the wiring of the scripts, not statistical conclusions.
"""

import pytest

from repair import storage
from scripts import analyse, make_instances, rerun_check, run_experiment

pytestmark = [pytest.mark.solver, pytest.mark.slow]

TINY = """
name = "smoke"
date = ""
root_seed = 4242
horizon = 15
size_classes = [20]
cells = ["main", "rq2"]
n_instances_secondary = 0
n_sequences = 1
lam = 2
alpha = "1"
lam_grid = [1, 4]
alpha_grid = ["1/2", "2"]

[n_instances]
20 = 1

[solver]
time_limit = 30.0
s0_time_limit = 30.0

[generator]
n_days = 3
availability_ranges = [[1, 2], [2, 4], [4, 6]]
"""


def test_pipeline_end_to_end(tmp_path, capsys):
    protocol_file = tmp_path / "smoke.toml"
    protocol_file.write_text(TINY)
    out = tmp_path / "results"

    # 1. instances, restartable
    assert make_instances.main([str(protocol_file), "--out", str(out)]) == 0
    instance_file = storage.instance_path(out, 20, 0)
    _, acceptance, s0 = storage.read_instance(instance_file)
    assert s0.certified
    assert (out / "acceptance.csv").read_text().splitlines()[0].split(",") == list(
        make_instances.ACCEPTANCE_COLUMNS
    )
    first_mtime = instance_file.stat().st_mtime_ns
    assert make_instances.main([str(protocol_file), "--out", str(out)]) == 0
    assert instance_file.stat().st_mtime_ns == first_mtime  # skipped, not recomputed

    # 2. the job count for monitor.py --total
    capsys.readouterr()
    assert run_experiment.main([str(protocol_file), "--out", str(out), "--count"]) == 0
    assert capsys.readouterr().out.strip() == "4"

    # 3. the run, then a restart that finds everything done
    assert run_experiment.main([str(protocol_file), "--out", str(out)]) == 0
    assert storage.count_results(out)[0] == 4
    assert storage.read_failures(out) == [] and not storage.abort_requested(out)
    mtimes = {p: p.stat().st_mtime_ns for p in out.glob("*-n*/*/*.json.gz")}
    assert run_experiment.main([str(protocol_file), "--out", str(out)]) == 0
    assert {p: p.stat().st_mtime_ns for p in out.glob("*-n*/*/*.json.gz")} == mtimes

    results = list(storage.read_results(out))
    assert {r.cell for r in results} == {"main", "rq2"}
    assert all(r.manifest["workers"] == 1 for r in results)
    assert all(r.manifest["protocol"] == "smoke.toml" for r in results)

    # 4. the tables
    assert analyse.main([str(out), "--out", str(out / "analysis")]) == 0
    for name in analyse.TABLES:
        path = out / "analysis" / name
        assert path.exists() and path.read_text().strip(), name
    for name in ("step_table.csv", "sequence_table.csv", "rq2_table.csv", "h1.csv", "h3.csv"):
        lines = (out / "analysis" / name).read_text().strip().splitlines()
        assert len(lines) >= 2, name  # a header and at least one row
    assert len((out / "analysis" / "step_table.csv").read_text().splitlines()) == 1 + 3 * 5 * 15

    # 5. Gate M3's determinism check: two finished jobs rerun into a fresh directory
    capsys.readouterr()
    assert rerun_check.main([str(protocol_file), str(out), "--jobs", "2"]) == 0
    report = capsys.readouterr().out
    assert report.count(": identical") == 2, report
    [fresh] = tmp_path.glob("results-rerun-*")
    assert storage.count_results(fresh)[0] == 2
