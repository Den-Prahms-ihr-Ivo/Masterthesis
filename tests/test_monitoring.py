"""Check alert decisions using hand-made snapshots, without sending any mail."""

from dataclasses import replace

import pytest

from repair.monitoring import (
    SUBJECT_PREFIX,
    MonitorState,
    Snapshot,
    decide,
    deciles_reached,
    eta_seconds,
    initial_state,
    is_stalled,
    startup_alert,
    state_from_dict,
    state_to_dict,
)

HOUR = 3600.0
STALL = 3 * HOUR


def _snap(time=0.0, done=0, total=100, failures=(), aborted=False, last=None):
    """failures: job keys (error type "ValueError") or (key, error type) pairs."""
    pairs = tuple((f, "ValueError") if isinstance(f, str) else tuple(f) for f in failures)
    return Snapshot(time, done, total, pairs, aborted, last)


def _kinds(alerts):
    return [a.kind for a in alerts]


def _start(**kwargs):
    return initial_state(_snap(**kwargs), "pilot")


# Deciles and the initial state


@pytest.mark.parametrize(
    "done, total, expected",
    [
        (0, 100, ()),
        (9, 100, ()),
        (10, 100, (10,)),
        (25, 100, (10, 20)),
        (100, 100, tuple(range(10, 101, 10))),
        (150, 100, tuple(range(10, 101, 10))),
        (1, 3, (10, 20, 30)),  # 33.3 %: exact, no float rounding
        (2, 3, (10, 20, 30, 40, 50, 60)),
        (5, 0, ()),
    ],
)
def test_deciles_reached(done, total, expected):
    assert deciles_reached(done, total) == expected


def test_initial_state_counts_passed_deciles_as_sent():
    """A monitor (re)started at 35 % with two failures and ABORT present sends none of
    that again.
    """
    state = _start(time=50.0, done=35, failures=("a", "b"), aborted=True)

    assert state == MonitorState(
        name="pilot", start_time=50.0, start_done=35, last_done=35,
        deciles_sent=(10, 20, 30), failures_sent=("a", "b"), abort_sent=True,
    )
    alerts, _ = decide(state, _snap(time=60.0, done=35, failures=("a", "b"), aborted=True), STALL)
    assert alerts == []


def test_initial_state_of_a_finished_run():
    state = _start(done=100)

    assert state.finished_sent
    assert decide(state, _snap(time=10.0, done=100), STALL)[0] == []


# Decide


def test_one_mail_for_several_new_deciles():
    """From 5 % to 34 % between two polls: ONE progress mail (for 10, 20, 30), then
    nothing more until 40 %.
    """
    state = _start(done=5)

    alerts, state = decide(state, _snap(time=600.0, done=34, last=590.0), STALL)
    assert _kinds(alerts) == ["progress"]
    assert "30" in alerts[0].subject
    assert state.deciles_sent == (10, 20, 30)

    alerts, state = decide(state, _snap(time=1200.0, done=39, last=1190.0), STALL)
    assert alerts == []

    alerts, state = decide(state, _snap(time=1800.0, done=40, last=1790.0), STALL)
    assert _kinds(alerts) == ["progress"] and state.deciles_sent[-1] == 40


def test_failure_alert_once_per_key():
    """Each failure key is reported exactly once; new failures of one poll share ONE
    mail that lists every new key with its error type (and no old key).
    """
    state = _start()

    alerts, state = decide(state, _snap(time=1.0, failures=("job-a",)), STALL)
    assert _kinds(alerts) == ["failure"] and "job-a" in alerts[0].body

    alerts, state = decide(state, _snap(time=2.0, failures=("job-a",)), STALL)
    assert alerts == []

    new = (("job-b", "KeyError"), ("job-c", "TypeError"))
    alerts, state = decide(state, _snap(time=3.0, failures=("job-a", *new)), STALL)
    assert _kinds(alerts) == ["failure"]
    body = alerts[0].body
    assert all(key in body and kind in body for key, kind in new)
    assert "job-a" not in body
    assert set(state.failures_sent) == {"job-a", "job-b", "job-c"}


def test_hundreds_of_failures_one_mail():
    """A bug failing 300 jobs between two polls: one mail, not 300 (SMTP rate limits)."""
    state = _start()
    keys = [f"main-n20-i{i}-balanced-q0" for i in range(300)]

    alerts, _ = decide(state, _snap(time=1.0, failures=keys), STALL)

    assert _kinds(alerts) == ["failure"]
    assert all(key in alerts[0].body for key in keys)


def test_abort_alert_once():
    state = _start()

    alerts, state = decide(state, _snap(time=1.0, aborted=True), STALL)
    assert _kinds(alerts) == ["abort"]

    alerts, state = decide(state, _snap(time=2.0, aborted=True), STALL)
    assert alerts == []


def test_abort_and_failures_rearm_after_a_fix():
    """The abort procedure: ABORT removed and the failed job rerun successfully (its
    failure file is deleted), monitor restarted from its saved state. A second abort, or
    the same job failing again, must be mailed again, not silently swallowed.
    """
    state = _start()
    alerts, state = decide(state, _snap(time=1.0, failures=("job-a",), aborted=True), STALL)
    assert _kinds(alerts) == ["failure", "abort"]

    alerts, state = decide(state, _snap(time=2.0, failures=("job-a",), aborted=True), STALL)
    assert alerts == []  # still present: never twice

    _, state = decide(state, _snap(time=3.0), STALL)  # rm ABORT, job-a rerun fine
    state = state_from_dict(state_to_dict(state))  # the monitor restarts from its file

    alerts, _ = decide(state, _snap(time=4.0, failures=("job-a",), aborted=True), STALL)
    assert _kinds(alerts) == ["failure", "abort"]


def test_stall_alert_and_reset():
    """No new result for more than 3 h: one stall mail; a new result re-arms it; a second
    stall mails again.
    """
    state = _start(time=0.0, done=10, last=0.0)

    alerts, state = decide(state, _snap(time=2 * HOUR, done=10, last=0.0), STALL)
    assert alerts == []

    alerts, state = decide(state, _snap(time=3 * HOUR + 1, done=10, last=0.0), STALL)
    assert _kinds(alerts) == ["stall"] and state.stall_sent

    alerts, state = decide(state, _snap(time=4 * HOUR, done=10, last=0.0), STALL)
    assert alerts == []  # still stalled, already mailed

    alerts, state = decide(state, _snap(time=5 * HOUR, done=11, last=5 * HOUR), STALL)
    assert "stall" not in _kinds(alerts) and not state.stall_sent  # re-armed

    alerts, state = decide(state, _snap(time=8 * HOUR + 1, done=11, last=5 * HOUR), STALL)
    assert _kinds(alerts) == ["stall"]


def test_no_stall_when_finished_or_aborted():
    old = _snap(time=10 * HOUR, done=10, last=0.0)
    state = _start(time=0.0, done=10, last=0.0)

    assert is_stalled(state, old, STALL)
    assert not is_stalled(state, replace(old, aborted=True), STALL)
    assert not is_stalled(state, replace(old, done=100), STALL)


def test_old_results_before_a_restart_are_no_stall():
    """Results from yesterday's run: the clock starts when monitoring starts."""
    state = _start(time=100 * HOUR, done=10, last=0.0)

    assert not is_stalled(state, _snap(time=101 * HOUR, done=10, last=0.0), STALL)


def test_finished_alert_once():
    state = _start(done=95)

    alerts, state = decide(state, _snap(time=60.0, done=100, last=59.0), STALL)
    assert _kinds(alerts) == ["progress", "finished"]

    alerts, state = decide(state, _snap(time=120.0, done=100, last=59.0), STALL)
    assert alerts == []


def test_alert_order_and_subjects():
    """All kinds in one poll, in the documented order; every subject is prefixed."""
    state = _start(time=0.0, done=0, last=0.0)

    alerts, _ = decide(
        state, _snap(time=1.0, done=100, failures=("job-a",), aborted=True, last=1.0), STALL
    )

    assert _kinds(alerts) == ["progress", "failure", "abort", "finished"]
    assert all(a.subject.startswith(f"{SUBJECT_PREFIX} pilot") for a in alerts)


def test_startup_alert_summarises_what_happened_while_down():
    """One "started" mail per start: counts, failures by type, ABORT; it is the only way
    a failure from while the monitor was down gets reported after a restart.
    """
    first = _snap(time=5.0, done=412, total=2430,
                  failures=(("a", "KeyError"), ("b", "KeyError"), ("c", "OSError")), aborted=True)

    alert = startup_alert(first, "main")

    assert alert.kind == "started"
    assert alert.subject.startswith(f"{SUBJECT_PREFIX} main")
    for text in ("412", "2430", "3", "KeyError", "OSError", "ABORT"):
        assert text in alert.body, text
    quiet = startup_alert(_snap(done=0, total=10), "main")
    assert "ABORT" not in quiet.body or "no ABORT" in quiet.body


def test_decide_is_pure():
    state = _start(done=5)
    snap = _snap(time=600.0, done=34, last=590.0)

    assert decide(state, snap, STALL) == decide(state, snap, STALL)
    assert state == _start(done=5)


# ETA and the saved state


def test_eta():
    """20 new results in 1 h -> 20 / h; 70 left -> 3.5 h. Unknown before the first new
    result; 0 when finished.
    """
    state = _start(time=0.0, done=10)

    assert eta_seconds(state, _snap(time=HOUR, done=10)) is None
    assert eta_seconds(state, _snap(time=HOUR, done=30)) == pytest.approx(3.5 * HOUR)
    assert eta_seconds(state, _snap(time=HOUR, done=100)) == 0.0


def test_state_dict_round_trip():
    state = replace(_start(done=35, failures=("a",)), stall_sent=True)
    data = state_to_dict(state)

    assert state_from_dict(data) == state
    assert data["deciles_sent"] == [10, 20, 30]
    with pytest.raises(ValueError):
        state_from_dict({**data, "surprise": 1})
    with pytest.raises(ValueError):
        state_from_dict({k: v for k, v in data.items() if k != "name"})
