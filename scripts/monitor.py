"""Watch result files and send progress, failure, abort and stall messages.

Run this beside the experiment, for example in another tmux window:
    python scripts/monitor.py results/main --total 1234 --interval 600 --stall-hours 3
    python scripts/monitor.py results/test --total 1 --test-mail

Each poll reads a snapshot, asks monitoring.decide for alerts, sends them,
saves the state and pings an optional external health check. The health
check can report missing pings when this process or the server stops.
A restart sends a summary of the current state.

Read these settings from the environment, not from the repository:
    SMTP_HOST, SMTP_PORT: mail server and STARTTLS port
    SMTP_USER, SMTP_PASSWORD: login details
    MAIL_TO: destination address
    HEALTHCHECK_URL: optional ping URL

SERVER.md explains the run setup. This script uses standard-library mail
and HTTP clients; the alert decisions are tested separately without them.
"""

import argparse
import json
import os
import smtplib
import ssl
import sys
import time
import urllib.request
from collections.abc import Iterable, Mapping
from email.message import EmailMessage
from pathlib import Path

from repair import monitoring, storage
from repair.monitoring import Alert, MonitorState, Snapshot

MAIL_KEYS = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "MAIL_TO")


def state_path(results: Path) -> Path:
    return results / ".monitor" / "state.json"


def snapshot(results: Path, total: int, now: float) -> Snapshot:
    done, last_result_time = storage.count_results(results)
    failures = tuple(
        (f["job_key"], f["error_type"]) for f in storage.read_failures(results)
    )
    return Snapshot(
        time=now,
        done=done,
        total=total,
        failures=failures,
        aborted=storage.abort_requested(results),
        last_result_time=last_result_time,
    )


def load_state(results: Path) -> MonitorState | None:
    path = state_path(results)
    if not path.exists():
        return None
    try:
        return monitoring.state_from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(
            f"warning: unreadable monitor state ({exc}); starting from the results on disk"
        )
        return None


def save_state(results: Path, state: MonitorState) -> None:
    path = state_path(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(monitoring.state_to_dict(state), indent=2))
    os.replace(tmp, path)


def mail_settings(env: Mapping[str, str]) -> dict[str, str]:
    missing = [key for key in MAIL_KEYS if not env.get(key)]
    if missing:
        raise ValueError(f"missing environment variables: {', '.join(missing)}")
    if not env["SMTP_PORT"].isdigit():
        raise ValueError(f"SMTP_PORT is not a number: {env['SMTP_PORT']!r}")
    return {key: env[key] for key in MAIL_KEYS}


def send_mail(settings: Mapping[str, str], subject: str, body: str) -> None:
    message = EmailMessage()
    message["From"] = settings["SMTP_USER"]
    message["To"] = settings["MAIL_TO"]
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP(
        settings["SMTP_HOST"], int(settings["SMTP_PORT"]), timeout=30
    ) as smtp:
        smtp.starttls(context=ssl.create_default_context())
        smtp.login(settings["SMTP_USER"], settings["SMTP_PASSWORD"])
        smtp.send_message(message)


def ping_healthcheck(url: str | None, fail: bool = False) -> None:
    if not url:
        return
    target = url.rstrip("/") + "/fail" if fail else url
    try:
        with urllib.request.urlopen(target, timeout=10):
            pass
    except (
        Exception
    ) as exc:  # A failed health-check request should not stop progress monitoring.
        print(f"warning: health-check ping failed: {exc}")


def _send_all(settings: Mapping[str, str], alerts: Iterable[Alert]) -> bool:
    """Send the alerts in order; print the first sending error and return False."""
    try:
        for alert in alerts:
            send_mail(settings, alert.subject, alert.body)
    except (OSError, smtplib.SMTPException) as exc:
        print(f"warning: sending mail failed: {exc}", file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    """Send one test message with --test-mail, otherwise start the polling loop.

    Send the startup summary first. Poll at the requested interval until the
    experiment is finished or aborted.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--total", type=int, required=True, help="number of jobs in this run"
    )
    parser.add_argument(
        "--interval", type=int, default=600, help="seconds between polls"
    )
    parser.add_argument("--stall-hours", type=float, default=3.0)
    parser.add_argument(
        "--name", default=None, help="run name in subjects (default: results dir)"
    )
    parser.add_argument(
        "--test-mail", action="store_true", help="send one test mail and exit"
    )
    args = parser.parse_args(argv)

    settings = mail_settings(os.environ)
    health_url = os.environ.get("HEALTHCHECK_URL")
    name = args.name or args.results.name
    stall_seconds = args.stall_hours * 3600

    if args.test_mail:
        send_mail(
            settings,
            f"{monitoring.SUBJECT_PREFIX} {name}: test mail",
            "The monitor can send mail.",
        )
        return 0

    snap = snapshot(args.results, args.total, time.time())
    state = load_state(args.results) or monitoring.initial_state(snap, name)
    _send_all(
        settings, [monitoring.startup_alert(snap, name)]
    )  # A failed startup summary is not retried here.
    save_state(args.results, state)

    while True:
        snap = snapshot(args.results, args.total, time.time())
        alerts, new_state = monitoring.decide(state, snap, stall_seconds)
        if _send_all(settings, alerts):
            state = new_state
            save_state(args.results, state)
            ping_healthcheck(health_url)
            if snap.aborted or snap.done >= snap.total:
                return 0
        else:
            # Keep the old state so these alerts can be tried again on the next poll.
            ping_healthcheck(health_url, fail=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
