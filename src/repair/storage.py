"""Save and load experiment data as typed JSON, usually compressed with gzip.

Each file has a format version, a kind and its data. Loading checks these
fields before rebuilding the dataclasses, enums and exact fractions.

Writes first finish the conversion to JSON, then use a temporary file in
the same directory and replace the destination. This avoids leaving a
half-written result at its final path after an interrupted write. No SCIP
model or other live solver state is stored.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import tempfile
import types
import typing
from collections.abc import Iterator
from dataclasses import MISSING, fields, is_dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from repair import model
from repair.model import (
    AcceptanceLog,
    Instance,
    LexResult,
    SequenceResult,
    SystematicError,
)

import hashlib
import platform
import socket
import subprocess
from datetime import datetime, timezone

_UNREADABLE = "unreadable failure file"

FORMAT_VERSION = 1

TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        model.UtilityWeights,
        model.Volunteer,
        model.Shift,
        model.InstanceMeta,
        model.Instance,
        model.DemandChange,
        model.AvailabilityRemoval,
        model.PreferenceChange,
        model.AvailabilityRestoration,
        model.Sequence,
        model.Arm,
        model.StageRecord,
        model.Step,
        model.LexResult,
        model.Trajectory,
        model.CoverageStep,
        model.AcceptanceLog,
        model.SequenceResult,
    )
}


def encode(value: Any) -> Any:
    """Convert supported model objects to JSON values; reject unknown types."""
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        if TYPES.get(cls.__name__) is not cls:
            raise TypeError(f"unregistered dataclass {cls.__name__!r}")
        return {
            "__type__": cls.__name__,
            **{
                field.name: encode(getattr(value, field.name))
                for field in fields(value)
                if field.init
            },
        }
    if type(value) is frozenset:
        return [encode(item) for item in sorted(value)]
    # A NamedTuple needs its own type information, so do not treat it as a plain tuple.
    if type(value) is tuple:
        return [encode(item) for item in value]
    # Lists only occur in plain JSON data (the manifest); the model itself uses tuples.
    if type(value) is list:
        return [encode(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("stored dictionaries require string keys")
        return {key: encode(item) for key, item in value.items()}
    if value is None or type(value) in (int, float, str, bool):
        return value
    raise TypeError(f"cannot encode {type(value).__name__}")


def _plain_json(data: Any) -> Any:
    """Check that untyped data, such as a manifest, contains only JSON values."""
    if data is None or type(data) in (str, int, bool):
        return data
    if type(data) is float and math.isfinite(data):
        return data
    if type(data) is list:
        return [_plain_json(item) for item in data]
    if type(data) is dict and all(type(key) is str for key in data):
        return {key: _plain_json(value) for key, value in data.items()}
    raise ValueError(f"not a JSON value: {data!r}")


def decode(data: Any, hint: Any) -> Any:
    """Rebuild the expected Python type from JSON and reject incompatible data."""
    # Python 3.12+ also allows aliases declared with `type Schedule = ...`.
    alias_type = getattr(typing, "TypeAliasType", ())
    if isinstance(hint, alias_type):
        return decode(data, hint.__value__)

    origin = get_origin(hint)
    args = get_args(hint)

    if hint is Any:
        return _plain_json(data)
    if origin is typing.Annotated:
        return decode(data, args[0])
    if origin in (typing.Union, types.UnionType):
        for candidate in args:
            try:
                return decode(data, candidate)
            except ValueError:
                pass
        raise ValueError(f"value {data!r} does not match {hint!r}")
    if origin is typing.Literal:
        if any(type(data) is type(value) and data == value for value in args):
            return data
        raise ValueError(f"expected one of {args!r}, got {data!r}")

    if hint is None or hint is type(None):
        if data is not None:
            raise ValueError(f"expected None, got {data!r}")
        return None
    if hint in (str, int, bool):
        if type(data) is not hint:
            raise ValueError(f"expected {hint.__name__}, got {data!r}")
        return data
    if hint is float:
        if type(data) not in (int, float):
            raise ValueError(f"expected a number, got {data!r}")
        try:
            value = float(data)
        except OverflowError as error:
            raise ValueError("number is too large for float") from error
        if not math.isfinite(value):
            raise ValueError("expected a finite number")
        return value
    if hint is Fraction:
        if type(data) is not str:
            raise ValueError(f"expected a Fraction string, got {data!r}")
        try:
            return Fraction(data)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"invalid Fraction {data!r}") from error
    if isinstance(hint, type) and issubclass(hint, StrEnum):
        if type(data) is not str:
            raise ValueError(f"expected a string for {hint.__name__}")
        try:
            return hint(data)
        except ValueError as error:
            raise ValueError(f"invalid {hint.__name__}: {data!r}") from error

    if isinstance(hint, type) and is_dataclass(hint):
        return _decode_dataclass(data, hint)

    if hint is tuple or origin is tuple:
        if type(data) is not list:
            raise ValueError("expected a JSON array for tuple")
        if hint is tuple:
            return tuple(decode(item, Any) for item in data)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(item, args[0]) for item in data)
        if len(data) != len(args):
            raise ValueError(f"expected tuple length {len(args)}, got {len(data)}")
        return tuple(decode(item, item_hint) for item, item_hint in zip(data, args))

    if hint is frozenset or origin is frozenset:
        if type(data) is not list:
            raise ValueError("expected a JSON array for frozenset")
        item_hint = args[0] if args else Any
        try:
            return frozenset(decode(item, item_hint) for item in data)
        except TypeError as error:
            raise ValueError("frozenset elements must be hashable") from error

    if hint is dict or origin is dict:
        if type(data) is not dict or any(type(key) is not str for key in data):
            raise ValueError("expected a JSON object with string keys")
        key_hint, value_hint = args if args else (str, Any)
        return {
            decode(key, key_hint): decode(value, value_hint)
            for key, value in data.items()
        }

    raise ValueError(f"unsupported type hint {hint!r}")


def _decode_dataclass(data: Any, hint: type) -> Any:
    if type(data) is not dict:
        raise ValueError(f"expected an object for {hint.__name__}")
    tag = data.get("__type__")
    if type(tag) is not str or tag not in TYPES:
        raise ValueError(f"unknown dataclass tag {tag!r}")
    if TYPES[tag] is not hint:
        raise ValueError(f"tag {tag!r} does not match {hint.__name__}")

    init_fields = {field.name: field for field in fields(hint) if field.init}
    extra = data.keys() - init_fields.keys() - {"__type__"}
    if extra:
        raise ValueError(f"unexpected fields for {hint.__name__}: {extra!r}")
    hints = get_type_hints(hint)
    values = {}
    for name, field in init_fields.items():
        if name in data:
            values[name] = decode(data[name], hints[name])
        elif field.default is MISSING and field.default_factory is MISSING:
            raise ValueError(f"missing field {hint.__name__}.{name}")
    try:
        return hint(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {hint.__name__}: {error}") from error


def _payload(data: Any, kind: str) -> Any:
    """Check the file format and kind before reading the model data."""
    if type(data) is not dict or set(data) != {"format", "kind", "data"}:
        raise ValueError("expected an envelope with format, kind and data")
    if type(data["format"]) is not int or data["format"] != FORMAT_VERSION:
        raise ValueError(f"unsupported format version {data['format']!r}")
    if data["kind"] != kind:
        raise ValueError(f"expected kind {kind!r}, got {data['kind']!r}")
    return data["data"]


def to_dict(result: SequenceResult) -> dict[str, Any]:
    """Put an encoded sequence result inside the versioned file structure."""
    return {"format": FORMAT_VERSION, "kind": "sequence_result", "data": encode(result)}


def from_dict(data: dict[str, Any]) -> SequenceResult:
    """Check the format and rebuild a SequenceResult from its stored fields."""
    return decode(_payload(data, "sequence_result"), SequenceResult)


def _write_json(path: Path, document: dict[str, Any]) -> None:
    # Finish encoding first; an unsupported value must not damage an existing file.
    raw = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=stream, mtime=0
            ) as zipped:
                zipped.write(raw)
        # Close gzip and the file before replacing the destination.
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def result_path(
    root: Path, cell: str, size_class: int, instance_seed: int, sequence_seed: int
) -> Path:
    """root/<cell>-n<size>/<instance_seed>/<sequence_seed>.json.gz"""
    return (
        root / f"{cell}-n{size_class}" / str(instance_seed) / f"{sequence_seed}.json.gz"
    )


def write_result(path: Path, result: SequenceResult) -> None:
    """Write a complete compressed result, replacing the destination atomically."""
    _write_json(path, to_dict(result))


def read_result(path: Path) -> SequenceResult:
    """Decompress one result file and reconstruct its typed contents."""
    return from_dict(_read_json(path))


def exists_complete(path: Path) -> bool:
    """Return whether the whole result file can be decoded successfully.

    A path existing is not sufficient for restart. Truncated or incompatible
    files return False, so their jobs can be run again.
    """
    try:
        read_result(path)
    except Exception:
        # A broken file means rerun the job; it should not break the restart check.
        return False
    return True


def read_results(root: Path, cell: str | None = None) -> Iterator[SequenceResult]:
    """Read stored results in a fixed path order, optionally selecting a cell.

    Ignore instance files, temporary files and unrelated paths. A malformed
    file in the result layout raises an error; silently skipping it could
    change the analysis sample without being noticed.
    """
    for path in sorted(root.glob("*-n*/*/*.json.gz")):
        if (
            re.fullmatch(r".+-n[0-9]+", path.parent.parent.name) is None
            or re.fullmatch(r"[0-9]+", path.parent.name) is None
            or re.fullmatch(r"[0-9]+\.json\.gz", path.name) is None
            or not path.is_file()
        ):
            continue
        result = read_result(path)
        if cell is None or result.cell == cell:
            yield result


def instance_path(
    root: Path, size_class: int, index: int, under_supplied: bool = False
) -> Path:
    """root/instances/<size or u40>/<index>.json.gz"""
    if under_supplied and size_class != 40:
        raise ValueError("under-supplied instances require size class 40")
    directory = "u40" if under_supplied else str(size_class)
    return root / "instances" / directory / f"{index}.json.gz"


def write_instance(
    path: Path, instance: Instance, acceptance: AcceptanceLog, s0: LexResult
) -> None:
    """Save the accepted instance, acceptance log and S0 in one atomic write."""
    document = {
        "format": FORMAT_VERSION,
        "kind": "instance",
        "data": {
            "instance": encode(instance),
            "acceptance": encode(acceptance),
            "s0": encode(s0),
        },
    }
    _write_json(path, document)


def read_instance(path: Path) -> tuple[Instance, AcceptanceLog, LexResult]:
    """Read a base-instance file and check its format version and file kind."""
    data = _payload(_read_json(path), "instance")
    if type(data) is not dict or set(data) != {"instance", "acceptance", "s0"}:
        raise ValueError("instance data must contain instance, acceptance and s0")
    return (
        decode(data["instance"], Instance),
        decode(data["acceptance"], AcceptanceLog),
        decode(data["s0"], LexResult),
    )


# Run control (step 4.4)
#
#   root/failures/<job_key>.json    one file per failed job (plain JSON, not gzip)
#   root/ABORT                      present = no new job may start (text: the reason)
#   root/manifest.json              the run's manifest, written by the first start; a
#                                   resumed run must match it (check_run_manifest)
#
# Run-control files use atomic writes too. The experiment writes them,
# and the monitor reads them to report progress and failures.


def _write_text_atomic(path: Path, text: str) -> None:
    """Use a temporary file and replacement for plain text, as for JSON writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def failure_path(root: Path, job_key: str) -> Path:
    """root/failures/<job_key>.json"""
    if not job_key or job_key.startswith(".") or "/" in job_key or "\\" in job_key:
        raise ValueError(f"job key cannot be used as a file name: {job_key!r}")
    return root / "failures" / f"{job_key}.json"


def write_failure(
    root: Path, job_key: str, error_type: str, message: str, traceback_text: str
) -> None:
    """Atomic. The JSON object has exactly the keys job_key, error_type, message (str(e),
    used by run_experiment's breaker to spot the same bug in many jobs), traceback, host
    (socket.gethostname()) and time (UTC, ISO 8601). A second failure of the same job
    replaces the first.
    """
    document = {
        "job_key": job_key,
        "error_type": error_type,
        "message": message,
        "traceback": traceback_text,
        "host": socket.gethostname(),
        "time": _utc_now(),
    }
    _write_text_atomic(failure_path(root, job_key), json.dumps(document, indent=2))


def read_failures(root: Path) -> list[dict[str, Any]]:
    """Every failure file, sorted by job_key; [] if there is no failures directory.

    Never raises (the monitor calls it every poll): a file that is not a JSON object
    with a string "job_key" (unparsable, truncated, "[]", ...) still appears, as
    {"job_key": <file stem>, "error_type": "unreadable failure file", "message": "",
    "traceback": "", "host": "", "time": ""}. Temp files are ignored.
    """
    try:
        paths = [
            p
            for p in (root / "failures").glob("*.json")
            if p.is_file() and not p.name.startswith(".")
        ]
    except OSError:
        return []
    failures = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if type(data) is not dict or type(data.get("job_key")) is not str:
                raise ValueError("not a failure object")
        except (
            OSError,
            ValueError,
        ):  # JSONDecodeError and UnicodeDecodeError are ValueErrors
            data = {
                "job_key": path.stem,
                "error_type": _UNREADABLE,
                "message": "",
                "traceback": "",
                "host": "",
                "time": "",
            }
        failures.append(data)
    return sorted(failures, key=lambda failure: failure["job_key"])


def abort_path(root: Path) -> Path:
    """root/ABORT"""
    return root / "ABORT"


def _result_files(root: Path) -> list[Path]:
    """Find files in the result layout and return them in sorted order."""
    return [
        path
        for path in sorted(root.glob("*-n*/*/*.json.gz"))
        if re.fullmatch(r".+-n[0-9]+", path.parent.parent.name) is not None
        and re.fullmatch(r"[0-9]+", path.parent.name) is not None
        and re.fullmatch(r"[0-9]+\.json\.gz", path.name) is not None
        and path.is_file()
    ]


def request_abort(root: Path, reason: str) -> None:
    """Create root/ABORT (atomically) with `reason` as its text; creates root if needed.
    Idempotent: a second request keeps the flag (and replaces the reason).
    """
    _write_text_atomic(
        abort_path(root), reason
    )  # creates root; replaces an older reason


def abort_requested(root: Path) -> bool:
    """Return whether ABORT is present; removing the file clears the request."""
    return abort_path(root).exists()


def count_results(root: Path) -> tuple[int, float | None]:
    """(number of COMPLETE result files below root, mtime of the newest complete one).

    Same layout as read_results; truncated or unparsable files are not counted (and do
    not set the mtime); no results -> (0, None). Never raises.
    """
    count, newest = 0, None
    try:
        paths = _result_files(root)
    except OSError:
        return 0, None
    for path in paths:
        try:
            _payload(
                _read_json(path), "sequence_result"
            )  # decompresses and parses fully
            mtime = path.stat().st_mtime
        except Exception:  # Keep the monitor running even if one result file is broken.
            continue
        count += 1
        newest = mtime if newest is None else max(newest, mtime)
    return count, newest


def cpu_model() -> str:
    """The CPU model name: "model name" from /proc/cpuinfo on Linux,
    `sysctl -n machdep.cpu.brand_string` on macOS, else platform.processor(); "unknown"
    if all of them fail. Never raises.
    """
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        name = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        if name:
            return name
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return platform.processor() or "unknown"
    except Exception:
        return "unknown"


def _git(directory: Path, *arguments: str) -> str | None:
    """Return git output, or None when git is unavailable or the command fails."""
    try:
        done = subprocess.run(
            ["git", *arguments],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def build_manifest(
    protocol_path: Path, workers: int, solver_versions: dict[str, str]
) -> dict[str, Any]:
    """What a run's results were produced with (stored in every SequenceResult):

    protocol       protocol_path.name
    protocol_sha256  hex SHA-256 of the protocol file's bytes
    git_commit     `git rev-parse HEAD`, or "unknown" (no git / not a repository)
    git_clean      True / False from `git status --porcelain` being empty, None if unknown
    python         platform.python_version()
    solver         solver_versions as given (milp.solver_versions(); passed in, so this
                   module stays solver-free)
    host           socket.gethostname()
    cpu_model      cpu_model();  logical_cpus: os.cpu_count()
    workers        the --workers value: pilot and main run must show the same CPU model
                   and worker count (DESIGN.md 13.13)
    started        UTC, ISO 8601

    Only JSON values (encode accepts it). git is run in the protocol file's directory.
    """
    directory = protocol_path.resolve().parent
    commit = _git(directory, "rev-parse", "HEAD")
    status = _git(directory, "status", "--porcelain")
    return {
        "protocol": protocol_path.name,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "git_commit": commit.strip() if commit else "unknown",
        "git_clean": None if status is None else status.strip() == "",
        "python": platform.python_version(),
        "solver": dict(solver_versions),
        "host": socket.gethostname(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "workers": workers,
        "started": _utc_now(),
    }


class ManifestMismatch(SystematicError):
    """The settings of a resumed run differ from its stored manifest."""


# What a resumed run must share with the run it continues: results written under a
# different commit, protocol, solver or Python must not be mixed. cpu_model and workers
# are included because timing decides certification (DESIGN.md 13.13).
RESUME_KEYS = (
    "protocol_sha256",
    "git_commit",
    "git_clean",
    "solver",
    "python",
    "cpu_model",
    "workers",
)


def run_manifest_path(root: Path) -> Path:
    """root/manifest.json"""
    return root / "manifest.json"


def check_run_manifest(root: Path, manifest: dict[str, Any]) -> None:
    """First start (no root/manifest.json): write `manifest` there (atomically).
    Later starts: compare the RESUME_KEYS with the stored manifest and raise
    ManifestMismatch naming every differing key with both values. On a match nothing
    is written (the stored manifest keeps the first start time).

    To continue on purpose after such a change (e.g. a documented bug fix that cannot
    change schedules), move manifest.json away by hand and record why in NOTES.md.
    """
    path = run_manifest_path(root)
    if not path.exists():
        _write_text_atomic(path, json.dumps(encode(manifest), indent=2, sort_keys=True))
        return
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if type(stored) is not dict:
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as error:
        raise ManifestMismatch(f"cannot read {path}: {error}") from error
    differing = [
        f"{key}: stored {stored.get(key)!r}, now {manifest.get(key)!r}"
        for key in RESUME_KEYS
        if stored.get(key) != manifest.get(key)
    ]
    if differing:
        raise ManifestMismatch(
            "resumed run differs from " + str(path) + "; " + "; ".join(differing)
        )
