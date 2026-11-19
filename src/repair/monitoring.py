"""Decide which progress and failure messages the monitor should send.

This module only calculates decisions. scripts/monitor.py reads files,
sends the messages and saves the state. The tests can therefore check the
alert rules with small snapshots and no mail server.

Send one progress message for newly reached 10% marks, and one combined
failure message for new failure files in a poll. Combining failures avoids
a large burst of mail when many jobs fail for the same reason.

An ABORT flag triggers one alert. A stall triggers one alert after the
threshold without a new result, unless the run is finished or aborted.
New results reset the stall alert. Removing an abort or failure file lets
a later occurrence be reported again. Completion also gets one message.

On restart, existing events count as reported. A startup summary lists the
current progress, failures and ABORT state instead of repeating each alert.
The stall timer starts again at monitor startup, giving a full grace period.
Subjects start with "[repair-study] <run name>".
"""

from collections import Counter
from dataclasses import asdict, dataclass, fields, replace

from dataclasses import dataclass
from typing import Any, Literal

AlertKind = Literal["started", "progress", "failure", "abort", "stall", "finished"]
SUBJECT_PREFIX = "[repair-study]"


@dataclass(frozen=True)
class Snapshot:
    """The result count, failures and abort state seen during one poll."""

    time: float  # unix seconds
    done: int  # complete result files (storage.count_results)
    total: int  # jobs of this run (run_experiment.py --count)
    failures: tuple[tuple[str, str], ...]  # (job key, error type) per failure file
    aborted: bool  # ABORT present
    last_result_time: float | None  # mtime of the newest complete result


@dataclass(frozen=True)
class MonitorState:
    """Remember progress and sent alerts between polls."""

    name: str  # run name for the subjects, e.g. "main"
    start_time: float
    start_done: int  # results already present when monitoring started (for the ETA)
    last_done: int  # done at the previous poll (a rise re-arms the stall alert)
    deciles_sent: tuple[int, ...]  # e.g. (10, 20, 30)
    failures_sent: tuple[str, ...]
    abort_sent: bool = False
    stall_sent: bool = False
    finished_sent: bool = False


@dataclass(frozen=True)
class Alert:
    kind: AlertKind
    subject: str  # starts with SUBJECT_PREFIX + " " + state.name
    body: str  # done / total, percent and, if known, the ETA


def _finished(snap: Snapshot) -> bool:
    return snap.total > 0 and snap.done >= snap.total


def _duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days} d {hours} h"
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"


def _progress_line(snap: Snapshot, eta: float | None = None) -> str:
    percent = snap.done * 100 // snap.total if snap.total > 0 else 0
    line = f"{snap.done}/{snap.total} results ({percent} %)"
    if eta is not None and eta > 0:
        line += f", about {_duration(eta)} left"
    return line


def _alert(kind: AlertKind, name: str, what: str, body: str) -> Alert:
    return Alert(kind=kind, subject=f"{SUBJECT_PREFIX} {name}: {what}", body=body)


def deciles_reached(done: int, total: int) -> tuple[int, ...]:
    """Return all 10% marks reached by this result count.

    For done=25 and total=100, return (10, 20). Integer comparisons avoid
    rounding near a threshold. With no jobs, return an empty tuple.
    """
    if total <= 0:
        return ()
    return tuple(d for d in range(10, 101, 10) if done * 100 >= d * total)


def initial_state(first: Snapshot, name: str) -> MonitorState:
    """Start monitoring from the first snapshot.

    Treat existing progress marks, failures and ABORT as already reported.
    startup_alert() summarises them. Save the starting count and time for ETA.
    """
    return MonitorState(
        name=name,
        start_time=first.time,
        start_done=first.done,
        last_done=first.done,
        deciles_sent=deciles_reached(first.done, first.total),
        failures_sent=tuple(key for key, _ in first.failures),
        abort_sent=first.aborted,
        stall_sent=False,
        finished_sent=_finished(first),
    )


def startup_alert(first: Snapshot, name: str) -> Alert:
    """Build the startup summary with result counts, failures and the ABORT state."""
    lines = [_progress_line(first), f"failures: {len(first.failures)}"]
    by_type = Counter(error_type for _, error_type in first.failures)
    lines += [
        f"  {count} x {error_type}" for error_type, count in sorted(by_type.items())
    ]
    # "no ABORT", never "ABORT: no": a quiet mail must not match a search for "ABORT"
    lines.append("ABORT is set: no new jobs start" if first.aborted else "no ABORT")
    if _finished(first):
        lines.append("The run is already finished.")
    what = (
        f"monitor started: {first.done}/{first.total} results, "
        f"{len(first.failures)} failures, {'ABORT' if first.aborted else 'no ABORT'}"
    )
    return _alert("started", name, what, "\n".join(lines))


def is_stalled(state: MonitorState, snap: Snapshot, threshold_seconds: float) -> bool:
    """Check whether the run has waited too long for another result.

    Use the later of monitor startup and the last result time. Old files alone
    therefore do not cause an immediate stall alert after restarting.
    """
    if _finished(snap) or snap.aborted:
        return False
    since = state.start_time
    if snap.last_result_time is not None:
        since = max(since, snap.last_result_time)
    return snap.time - since > threshold_seconds


def eta_seconds(state: MonitorState, snap: Snapshot) -> float | None:
    """Estimate remaining time from results completed since monitor startup.

    Return None before a rate can be measured, and 0.0 when all jobs are done.
    This is an estimate from recent throughput, not from solver time limits.
    """
    if snap.done >= snap.total:
        return 0.0
    new_results = snap.done - state.start_done
    elapsed = snap.time - state.start_time
    if new_results <= 0 or elapsed <= 0:
        return None
    return (snap.total - snap.done) * elapsed / new_results


def decide(
    state: MonitorState, snap: Snapshot, stall_threshold: float
) -> tuple[list[Alert], MonitorState]:
    """Return this poll's alerts and the updated state without sending anything.

    The order is progress, failure, abort, stall, finished, with at most one
    message of each kind. Equal inputs give equal decisions, which also lets
    the caller retry alerts when sending fails.
    """
    alerts: list[Alert] = []
    status = _progress_line(snap, eta_seconds(state, snap))

    reached = deciles_reached(snap.done, snap.total)
    new_deciles = tuple(d for d in reached if d not in state.deciles_sent)
    if new_deciles:
        alerts.append(
            _alert("progress", state.name, f"{new_deciles[-1]} % done", status)
        )

    new_failures = [(k, t) for k, t in snap.failures if k not in state.failures_sent]
    if new_failures:
        listing = "\n".join(
            f"  {key}: {error_type}" for key, error_type in new_failures
        )
        alerts.append(
            _alert(
                "failure",
                state.name,
                f"{len(new_failures)} new failed job(s)",
                f"{listing}\n\n{len(snap.failures)} failures in total. {status}",
            )
        )

    abort_now = snap.aborted and not state.abort_sent
    if abort_now:
        alerts.append(
            _alert(
                "abort",
                state.name,
                "ABORT, no new jobs start",
                f"The ABORT flag is set. {status}",
            )
        )

    stall_sent = (
        state.stall_sent and snap.done <= state.last_done
    )  # a new result re-arms
    if not stall_sent and is_stalled(state, snap, stall_threshold):
        alerts.append(
            _alert(
                "stall",
                state.name,
                f"no new result for {_duration(stall_threshold)}",
                status,
            )
        )
        stall_sent = True

    finished_now = _finished(snap) and not state.finished_sent
    if finished_now:
        alerts.append(
            _alert(
                "finished",
                state.name,
                "finished",
                f"{status}\n{len(snap.failures)} failures.",
            )
        )

    # failures_sent and abort_sent mirror what is on disk now: everything present has been
    # mailed (now or earlier), and whatever disappeared is re-armed. So after a fix (rm
    # ABORT, failed jobs rerun) a second abort or a repeated failure mails again.
    new_state = replace(
        state,
        last_done=snap.done,
        deciles_sent=state.deciles_sent + new_deciles,
        failures_sent=tuple(key for key, _ in snap.failures),
        abort_sent=snap.aborted,
        stall_sent=stall_sent,
        finished_sent=state.finished_sent or finished_now,
    )
    return alerts, new_state


def state_to_dict(state: MonitorState) -> dict[str, Any]:
    """Convert monitor state to JSON-compatible fields for saving."""
    data = asdict(state)
    data["deciles_sent"] = list(state.deciles_sent)
    data["failures_sent"] = list(state.failures_sent)
    return data


def state_from_dict(data: dict[str, Any]) -> MonitorState:
    """Load monitor state, rejecting missing or unexpected fields."""
    names = {field.name for field in fields(MonitorState)}
    if type(data) is not dict or set(data) != names:
        raise ValueError(f"monitor state needs exactly the keys {sorted(names)}")
    return MonitorState(
        **{
            **data,
            "deciles_sent": tuple(data["deciles_sent"]),
            "failures_sent": tuple(data["failures_sent"]),
        }
    )
