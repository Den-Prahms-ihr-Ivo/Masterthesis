"""Check command-line helpers without running a mail server or experiment.

The tests cover failure handling, job selection, planted errors, the dry-run
protocol and rerun comparisons. Mail and health-check calls use fakes.
test_smoke.py separately runs the full pipeline with SCIP.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from repair import storage
from repair.model import SequenceKind
from repair.monitoring import initial_state
from repair.protocol import InstanceJob, Job, job_key, load_protocol
from repair.runner import OracleMismatch
from repair.storage import ManifestMismatch
from scripts import monitor, rerun_check, run_experiment
from scripts.rerun_check import compare_results
from scripts.run_experiment import PLANT_ENV, planted_failures
from tests.test_storage import _full_result

PROTOCOLS = Path(__file__).parents[1] / "protocol"
PILOT = PROTOCOLS / "pilot.toml"
DRY_RUN = PROTOCOLS / "dry-run.toml"
JOB = Job("main", InstanceJob(20, 0), SequenceKind.RQ1, "balanced", 0, (), 1)
MANIFEST = {"test": True}


# Run_job_safely: the failure policy


def _raising(error):
    def run(job, protocol_path, out, manifest):
        raise error

    return run


def test_run_job_safely_returns_the_result_path(tmp_path):
    calls = []

    def run(job, protocol_path, out, manifest):
        calls.append((job, protocol_path, out, manifest))
        return out / "result.json.gz"

    got = run_experiment.run_job_safely(JOB, PILOT, tmp_path, MANIFEST, run=run)

    assert got == tmp_path / "result.json.gz"
    assert calls == [(JOB, PILOT, tmp_path, MANIFEST)]
    assert storage.read_failures(tmp_path) == []


def test_ordinary_error_is_logged_and_the_run_continues(tmp_path):
    """Any exception: a failure file with type and traceback, no ABORT."""
    got = run_experiment.run_job_safely(
        JOB, PILOT, tmp_path, MANIFEST, run=_raising(ValueError("disk full"))
    )

    assert got is None
    [failure] = storage.read_failures(tmp_path)
    assert failure["job_key"] == job_key(JOB)
    assert failure["error_type"] == "ValueError"
    assert "disk full" in failure["traceback"] and "Traceback" in failure["traceback"]
    assert not storage.abort_requested(tmp_path)


@pytest.mark.parametrize(
    "error",
    [OracleMismatch("value: U recorded 7"), ManifestMismatch("git_commit"), AssertionError("oracle")],
    ids=["OracleMismatch", "ManifestMismatch", "AssertionError"],
)
def test_systematic_error_aborts_the_run(tmp_path, error):
    """An oracle mismatch is a bug in the code, not in this job: failure AND ABORT."""
    assert run_experiment.run_job_safely(JOB, PILOT, tmp_path, MANIFEST, run=_raising(error)) is None

    assert [f["error_type"] for f in storage.read_failures(tmp_path)] == [type(error).__name__]
    assert storage.abort_requested(tmp_path)


def test_no_job_starts_after_an_abort(tmp_path):
    storage.request_abort(tmp_path, "earlier bug")

    def must_not_run(job, protocol_path, out, manifest):
        raise AssertionError("ran despite ABORT")

    assert run_experiment.run_job_safely(JOB, PILOT, tmp_path, MANIFEST, run=must_not_run) is None
    assert storage.read_failures(tmp_path) == []


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(2)], ids=["KeyboardInterrupt", "SystemExit"])
def test_interrupts_are_not_swallowed(tmp_path, error):
    with pytest.raises(type(error)):
        run_experiment.run_job_safely(JOB, PILOT, tmp_path, MANIFEST, run=_raising(error))
    assert storage.read_failures(tmp_path) == []


def _job(i):
    return Job("main", InstanceJob(20, i), SequenceKind.RQ1, "balanced", 0, (), 1)


def test_breaker_aborts_on_the_same_unclassified_error(tmp_path):
    """The same KeyError in BREAKER_THRESHOLD jobs is a code bug: abort, although
    KeyError itself is not a SystematicError. One job fewer does not trip it.
    """
    k = run_experiment.BREAKER_THRESHOLD
    for i in range(k - 1):
        run_experiment.run_job_safely(_job(i), PILOT, tmp_path, MANIFEST, run=_raising(KeyError("arm")))
    assert not storage.abort_requested(tmp_path)

    run_experiment.run_job_safely(_job(k - 1), PILOT, tmp_path, MANIFEST, run=_raising(KeyError("arm")))
    assert storage.abort_requested(tmp_path)
    assert "KeyError" in storage.abort_path(tmp_path).read_text()


def test_breaker_ignores_different_errors(tmp_path):
    """Many failures with different messages (e.g. memory on different jobs) are logged,
    not treated as one bug.
    """
    for i in range(2 * run_experiment.BREAKER_THRESHOLD):
        run_experiment.run_job_safely(
            _job(i), PILOT, tmp_path, MANIFEST, run=_raising(MemoryError(f"job {i}"))
        )

    assert len(storage.read_failures(tmp_path)) == 2 * run_experiment.BREAKER_THRESHOLD
    assert not storage.abort_requested(tmp_path)
    assert {f["message"] for f in storage.read_failures(tmp_path)} >= {"job 0", "job 1"}


# Job selection


def test_select_jobs_filters_cells_then_shards():
    from dataclasses import replace

    from repair import protocol as protocol_module

    p = replace(load_protocol(PILOT), robustness_scale=3, workforce_factor=0.8)
    every = protocol_module.jobs(p)

    assert run_experiment.select_jobs(p, None, "0/1") == every
    rq2 = run_experiment.select_jobs(p, ["rq2"], "0/1")
    assert rq2 and {j.cell for j in rq2} == {"rq2"}
    assert run_experiment.select_jobs(p, ["rq2"], "1/2") == rq2[1::2]
    for bad in ("2/2", "x/2", "1", "0/0"):
        with pytest.raises(ValueError):
            run_experiment.select_jobs(p, None, bad)


# Monitor.py helpers

ENV = {
    "SMTP_HOST": "smtp.example.org",
    "SMTP_PORT": "587",
    "SMTP_USER": "me@example.org",
    "SMTP_PASSWORD": "app-password",
    "MAIL_TO": "me@example.org",
}


def test_mail_settings_name_every_missing_key():
    assert monitor.mail_settings({**ENV, "OTHER": "x"}) == ENV

    with pytest.raises(ValueError) as error:
        monitor.mail_settings({"SMTP_HOST": "h"})
    for key in ("SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "MAIL_TO"):
        assert key in str(error.value)


class FakeSMTP:
    """Records what monitor.send_mail does with smtplib.SMTP."""

    log: list = []

    def __init__(self, host, port, *args, **kwargs):
        FakeSMTP.log.append(("connect", host, port))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        FakeSMTP.log.append(("close",))

    def starttls(self, *args, **kwargs):
        FakeSMTP.log.append(("starttls",))

    def login(self, user, password):
        FakeSMTP.log.append(("login", user, password))

    def send_message(self, message):
        FakeSMTP.log.append(("send", message["Subject"], message["To"], message.get_content()))

    def quit(self):
        FakeSMTP.log.append(("close",))


def test_send_mail_uses_starttls_and_login(monkeypatch):
    import smtplib

    FakeSMTP.log = []
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    monitor.send_mail(ENV, "[repair-study] pilot: 10 %", "10 of 100 done")

    steps = [entry[0] for entry in FakeSMTP.log]
    assert steps[:4] == ["connect", "starttls", "login", "send"]
    assert FakeSMTP.log[0] == ("connect", "smtp.example.org", 587)
    assert FakeSMTP.log[2] == ("login", "me@example.org", "app-password")
    _, subject, to, body = FakeSMTP.log[3]
    assert (subject, to) == ("[repair-study] pilot: 10 %", "me@example.org")
    assert "10 of 100 done" in body


def test_ping_healthcheck_never_raises(monkeypatch):
    import urllib.request

    urls = []

    def fake_urlopen(url, *args, **kwargs):
        urls.append(url if isinstance(url, str) else url.full_url)
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    monitor.ping_healthcheck("https://hc.example/abc")
    monitor.ping_healthcheck("https://hc.example/abc", fail=True)
    monitor.ping_healthcheck(None)
    monitor.ping_healthcheck("")

    assert urls == ["https://hc.example/abc", "https://hc.example/abc/fail"]


def test_snapshot_from_the_results_directory(tmp_path):
    """Counts complete results only, lists failure keys, sees ABORT."""
    storage.write_result(storage.result_path(tmp_path, "main", 20, 1, 0), _full_result())
    storage.result_path(tmp_path, "main", 20, 1, 1).write_bytes(b"truncated")
    storage.write_failure(tmp_path, "main-n20-i1-balanced-q1", "ValueError", "boom", "")
    storage.request_abort(tmp_path, "bug")

    snap = monitor.snapshot(tmp_path, total=10, now=123.0)

    assert (snap.time, snap.done, snap.total) == (123.0, 1, 10)
    assert snap.failures == (("main-n20-i1-balanced-q1", "ValueError"),)
    assert snap.aborted and snap.last_result_time is not None


def test_state_save_and_load(tmp_path):
    assert monitor.load_state(tmp_path) is None

    state = initial_state(monitor.snapshot(tmp_path, total=5, now=1.0), "pilot")
    monitor.save_state(tmp_path, state)

    assert monitor.state_path(tmp_path) == tmp_path / ".monitor" / "state.json"
    assert monitor.load_state(tmp_path) == state
    assert storage.count_results(tmp_path) == (0, None)  # the state file is no result


# The Gate M3 rehearsal: dry-run protocol, --list, planted failures


def test_dry_run_protocol_is_what_gate_m3_asks_for():
    """n = 20, 3 instances, tau = 60 s, never frozen, a root seed of its own."""
    from repair import protocol as protocol_module

    p = load_protocol(DRY_RUN)

    assert tuple(p.size_classes) == (20,) and dict(p.n_instances) == {20: 3}
    assert p.solver.time_limit == p.solver.s0_time_limit == 60.0
    assert p.date == ""
    names = ("dry-run", "pilot", "main")
    roots = {load_protocol(PROTOCOLS / f"{name}.toml").root_seed for name in names}
    assert len(roots) == 3
    assert len(protocol_module.jobs(p)) == 183  # the size the rehearsal plan assumes


def test_list_prints_the_job_keys_in_run_order(tmp_path, capsys):
    assert run_experiment.main([str(DRY_RUN), "--out", str(tmp_path), "--list"]) == 0

    keys = capsys.readouterr().out.split()
    selected = run_experiment.select_jobs(load_protocol(DRY_RUN), None, "0/1")
    assert keys == [job_key(j) for j in selected]
    assert len(set(keys)) == len(keys) == 183
    assert not any(tmp_path.iterdir())  # listing writes nothing


@pytest.mark.parametrize(
    "spec, expected",
    [
        (None, {}),
        ("", {}),
        ("  ", {}),
        ("main-n20-i0-balanced-q0:error", {"main-n20-i0-balanced-q0": "error"}),
        ("a:error, b:assert", {"a": "error", "b": "assert"}),
    ],
)
def test_planted_failures_parse(spec, expected):
    assert planted_failures(spec) == expected


@pytest.mark.parametrize("spec", ["a", "a:", ":error", "a:boom", "a:error,a:assert", "a:error,"])
def test_planted_failures_reject_typos(spec):
    """A typo must stop the rehearsal, not silently plant nothing."""
    with pytest.raises(ValueError):
        planted_failures(spec)


@pytest.mark.parametrize(
    "kind, error_type, aborts",
    [("error", "RuntimeError", False), ("assert", "AssertionError", True)],
)
def test_planted_failure_goes_through_the_failure_policy(
    tmp_path, monkeypatch, kind, error_type, aborts
):
    """The REAL run_job raises the planted exception before reading anything (tmp_path
    has no instance file): "error" -> failure, the run continues; "assert" -> failure AND
    abort. The message names the job, so planted errors never trip the breaker.
    """
    monkeypatch.setenv(PLANT_ENV, f"{job_key(JOB)}:{kind}")

    assert run_experiment.run_job_safely(JOB, PILOT, tmp_path, MANIFEST) is None

    [failure] = storage.read_failures(tmp_path)
    assert failure["error_type"] == error_type
    assert "planted" in failure["message"] and job_key(JOB) in failure["message"]
    assert storage.abort_requested(tmp_path) == aborts


def test_unplanted_jobs_run_normally(tmp_path, monkeypatch):
    monkeypatch.setenv(PLANT_ENV, f"{job_key(_job(1))}:error")

    with pytest.raises(FileNotFoundError):  # the real work starts: no instance file here
        run_experiment.run_job(JOB, PILOT, tmp_path, MANIFEST)


def test_main_refuses_planted_failures_in_the_main_run(tmp_path, monkeypatch):
    monkeypatch.setenv(PLANT_ENV, "main-n20-i0-balanced-q0:error")

    with pytest.raises(SystemExit) as stop:
        run_experiment.main([str(PILOT), "--out", str(tmp_path / "out"), "--main"])

    assert stop.value.code == 2
    assert not (tmp_path / "out").exists()  # refused before anything was written


@pytest.mark.parametrize(
    "spec", ["oops", "main-n20-i9-balanced-q0:error"], ids=["malformed", "unknown key"]
)
def test_main_refuses_a_plant_that_would_not_fire(tmp_path, monkeypatch, spec):
    monkeypatch.setenv(PLANT_ENV, spec)

    with pytest.raises(SystemExit) as stop:
        run_experiment.main([str(DRY_RUN), "--out", str(tmp_path / "out")])

    assert stop.value.code == 2
    assert not (tmp_path / "out").exists()


# Rerun_check.compare_results
# _full_result: QO step t=1 and coverage t=1 are proven; HWF's step t=1 and coverage t=2
# hit the time limit; WQS has no steps.


def _with_trajectory(result, index, trajectory):
    trajectories = list(result.trajectories)
    trajectories[index] = trajectory
    return replace(result, trajectories=tuple(trajectories))


def _retimed(result):
    """Same result, but every timing-dependent field changed."""

    def retime(record):
        return replace(record, seconds=record.seconds + 9.0, bound=None, retried=not record.retried)

    def restage(step):
        return replace(step, stages=tuple(map(retime, step.stages)))

    return replace(
        result,
        manifest={"host": "another machine"},
        s0_stages=tuple(map(retime, result.s0_stages)),
        coverage=tuple(replace(c, record=retime(c.record)) for c in result.coverage),
        trajectories=tuple(
            replace(t, steps=tuple(map(restage, t.steps))) for t in result.trajectories
        ),
    )


def test_compare_ignores_timing_and_machine():
    a = _full_result()

    assert compare_results(a, a).problems == []
    comparison = compare_results(a, _retimed(a))
    assert comparison.problems == []
    assert "coverage t=2: not compared (time limit)" in comparison.notes


def test_compare_reports_a_different_schedule_at_a_proven_step():
    a = _full_result()
    qo = a.trajectories[1]
    b = _with_trajectory(a, 1, replace(qo, steps=(replace(qo.steps[0], schedule=frozenset()),)))

    assert compare_results(a, b).problems == ["QO t=1: schedules differ"]


def test_compare_reports_a_different_stage_value():
    a = _full_result()
    qo = a.trajectories[1]
    [proven] = qo.steps[0].stages
    changed = replace(qo.steps[0], stages=(replace(proven, value=proven.value + 1),))
    b = _with_trajectory(a, 1, replace(qo, steps=(changed,)))

    [problem] = compare_results(a, b).problems
    assert problem.startswith("QO t=1: stage values differ")


def test_compare_reports_a_different_lambda_star_and_length():
    a = _full_result()
    b = replace(a, coverage=(replace(a.coverage[0], lambda_star=1), a.coverage[1]))
    assert compare_results(a, b).problems == ["coverage t=1: Lambda* 0 vs 1"]

    qo = a.trajectories[1]
    longer = _with_trajectory(a, 1, replace(qo, steps=qo.steps * 2, ended_early=True))
    [problem] = compare_results(a, longer).problems
    assert problem.startswith("QO: 1 vs 2 steps")


def test_compare_skips_what_a_time_limit_made_run_dependent():
    """HWF's step t=1 hit its limit: a different schedule there is a note, not a problem."""
    a = _full_result()
    hwf = a.trajectories[0]
    first = replace(hwf.steps[0], schedule=frozenset({(1, 1)}))
    b = _with_trajectory(a, 0, replace(hwf, steps=(first, *hwf.steps[1:])))

    comparison = compare_results(a, b)
    assert comparison.problems == []
    assert any(note.startswith(hwf.arm.name) and "t=1" in note for note in comparison.notes)


def test_compare_skips_steps_whose_coverage_hit_the_time_limit():
    """Lambda*_t caps every arm at t: if the coverage solve at t was unproven, a proven
    arm step at t may still differ between runs.
    """
    a = _full_result()
    unproven = replace(a.coverage[0], record=a.coverage[1].record)  # t=1, now TIME_LIMIT
    qo = a.trajectories[1]
    b = _with_trajectory(a, 1, replace(qo, steps=(replace(qo.steps[0], schedule=frozenset()),)))

    comparison = compare_results(replace(a, coverage=(unproven, a.coverage[1])), b)

    assert comparison.problems == []
    assert "QO: steps from t=1 on not compared (time limit)" in comparison.notes


def test_compare_different_inputs():
    problems = compare_results(_full_result(seed=7), _full_result(seed=8)).problems

    assert {"input differs: instance", "input differs: sequence"} <= set(problems)

    a = _full_result()  # S0 is copied with the instance file, so it must match exactly
    assert compare_results(a, replace(a, s0=frozenset())).problems == ["S0 differs"]


def test_spread_picks_first_and_last():
    assert rerun_check._spread(list(range(10)), 2) == [0, 9]
    assert rerun_check._spread(list(range(10)), 3) == [0, 4, 9]
    assert rerun_check._spread(list(range(10)), 1) == [0]
    assert rerun_check._spread([1, 2], 5) == [1, 2]

