"""Typed JSON/gzip storage for sequence results and accepted base instances.

Files have the envelope {"format": FORMAT_VERSION, "kind": ..., "data": ...}.
Writes serialize first, then write a temporary file in the destination directory
and atomically replace the destination. No solver state is stored.
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
from repair.model import AcceptanceLog, Instance, LexResult, SequenceResult

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
    """Convert supported model values to plain JSON values; reject other objects."""
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
    # Exact type deliberately excludes NamedTuple, which would lose its type.
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
    """Validate untyped JSON data, such as values inside a manifest."""
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
    """Restore a value using its field annotation; reject incompatible JSON data."""
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
    """Validate a file envelope before decoding its contents."""
    if type(data) is not dict or set(data) != {"format", "kind", "data"}:
        raise ValueError("expected an envelope with format, kind and data")
    if type(data["format"]) is not int or data["format"] != FORMAT_VERSION:
        raise ValueError(f"unsupported format version {data['format']!r}")
    if data["kind"] != kind:
        raise ValueError(f"expected kind {kind!r}, got {data['kind']!r}")
    return data["data"]


def to_dict(result: SequenceResult) -> dict[str, Any]:
    """Wrap an encoded SequenceResult in the current file format."""
    return {"format": FORMAT_VERSION, "kind": "sequence_result", "data": encode(result)}


def from_dict(data: dict[str, Any]) -> SequenceResult:
    """Decode a sequence result after checking its format version and kind."""
    return decode(_payload(data, "sequence_result"), SequenceResult)


def _write_json(path: Path, document: dict[str, Any]) -> None:
    # Finish serialization before creating directories or touching the destination.
    raw = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
                zipped.write(raw)
        # Both gzip and the underlying file are closed before replacement.
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
    return root / f"{cell}-n{size_class}" / str(instance_seed) / f"{sequence_seed}.json.gz"


def write_result(path: Path, result: SequenceResult) -> None:
    """Serialize a result and atomically replace its destination file."""
    _write_json(path, to_dict(result))


def read_result(path: Path) -> SequenceResult:
    """Read and decode one gzip-compressed result."""
    return from_dict(_read_json(path))


def exists_complete(path: Path) -> bool:
    """Whether a result fully decompresses, parses and decodes in this format."""
    try:
        read_result(path)
    except Exception:
        # This is deliberately a non-raising predicate for restart decisions.
        return False
    return True


def read_results(root: Path, cell: str | None = None) -> Iterator[SequenceResult]:
    """Read result-layout files in sorted path order, filtering by stored cell.

    Malformed files in the result layout raise; they are not silently omitted.
    Instance files, temporary files and unrelated layouts are ignored.
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
    """Atomically store an accepted instance, its acceptance log and S0 result."""
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
    """Read an instance file, requiring the current format and instance kind."""
    data = _payload(_read_json(path), "instance")
    if type(data) is not dict or set(data) != {"instance", "acceptance", "s0"}:
        raise ValueError("instance data must contain instance, acceptance and s0")
    return (
        decode(data["instance"], Instance),
        decode(data["acceptance"], AcceptanceLog),
        decode(data["s0"], LexResult),
    )
