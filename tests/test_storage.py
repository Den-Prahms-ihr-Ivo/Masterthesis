"""Check saved data, atomic writes and the result-file layout.

The hand-built example includes every event type, fractions, enums, None,
empty and interrupted trajectories, and a nested manifest. This helps catch
values that JSON cannot restore by itself. The solver-marked M2 test also
saves and reloads real sequence results.
"""

import gzip
import json
import os
from fractions import Fraction

import pytest

import oracle
from repair import milp, storage
from repair.generator import generate_instance
from repair.model import (
    AcceptanceLog,
    Arm,
    AvailabilityRemoval,
    AvailabilityRestoration,
    CoverageStep,
    DemandChange,
    LexResult,
    PolicyName,
    PreferenceChange,
    ReferenceMode,
    Sequence,
    SequenceKind,
    SequenceProfile,
    SequenceResult,
    StageRecord,
    StageResult,
    StageStatus,
    Step,
    Trajectory,
)
from repair.perturbations import instances_along, rq1_sequence, rq2_sequence
from repair.protocol import GeneratorConfig, SolverSettings, derive_seed, main_arms, rq2_arms
from repair.runner import accept_instance, initial_schedule, run_sequence
from tests.builders import Grid, make_shift


def _full_result(cell="main", seed=7):
    """A SequenceResult with every awkward value (see the module docstring)."""
    grid = Grid(n_periods=2, n_roles=2)
    volunteers = [
        grid.volunteer(
            0, available=[1], skill=(3, 1), seniority=2, cap=11, pref_role=(2, 0), pref_time=(1, 2)
        ),
        grid.volunteer(1),
    ]
    shifts = [
        make_shift(0, period=0, role=0, demand=2, min_skill=2, expert=True),
        make_shift(1, period=1, role=1),
    ]
    instance = grid.instance(
        volunteers, shifts, size_class=40, scale=2, seed=seed, under_supplied=True
    )
    sequence = Sequence(
        SequenceKind.RQ2,
        SequenceProfile.RQ2,
        seed,
        (
            DemandChange(shift=1, delta=2, fallback=True),
            AvailabilityRemoval(pairs=((1, 0), (1, 1)), fallback=True),
            PreferenceChange(changes=((0, 1, 2),)),
            AvailabilityRestoration(volunteer=1, period=0),
        ),
        selected=(1, 0),
    )
    s0 = frozenset({(0, 1), (1, 0)})
    unproven = StageRecord("MinDistance", StageStatus.TIME_LIMIT, 3, 2.5, 0.125, retried=True)
    nothing = StageRecord("MaxQuality", StageStatus.NO_SOLUTION, None, None, 1.0)
    proven = StageRecord("MinShortfall", StageStatus.OPTIMAL, 3, 3.0, 0.01)
    trajectories = (
        Trajectory(
            Arm(PolicyName.HWF, ReferenceMode.CHAINED, alpha=Fraction(1, 2)),
            (Step(1, s0, (unproven,)), Step(2, frozenset(), (unproven, nothing))),
            ended_early=True,
        ),
        Trajectory(Arm(PolicyName.QO, None), (Step(1, frozenset({(1, 1)}), (proven,)),)),
        Trajectory(Arm(PolicyName.WQS, ReferenceMode.ANCHORED, lam=4), ()),
    )
    manifest = {
        "git": "abc123",
        "clean": False,
        "versions": {"scip": "10.0.2", "pyscipopt": "6.3.0"},
        "cpus": 8,
        "load": 0.5,
        "missing": None,
        "list": [1, "a"],
    }
    return SequenceResult(
        cell=cell,
        instance=instance,
        acceptance=AcceptanceLog(seed, seed + 1, 2, 1, True, 0),
        sequence=sequence,
        s0=s0,
        s0_stages=(proven, unproven),
        coverage=(CoverageStep(1, 0, proven), CoverageStep(2, 3, unproven)),
        trajectories=trajectories,
        manifest=manifest,
    )


def _through_json(result):
    """to_dict -> real JSON text -> from_dict (what a file does, without the file)."""
    text = json.dumps(storage.to_dict(result), allow_nan=False)
    return storage.from_dict(json.loads(text))


# Encode / decode


def test_round_trip_full_result():
    """Equal after the round trip, and with the same types, not just equal values:
    a str is == its StrEnum and a list is not a tuple, so check those explicitly.
    """
    original = _full_result()
    back = _through_json(original)

    assert back == original
    assert back.sequence.digest() == original.sequence.digest()

    assert type(back.s0) is frozenset and type(next(iter(back.s0))) is tuple
    assert type(back.instance.volunteers[0].available) is frozenset
    assert type(back.instance.volunteers[0].skill) is tuple
    assert type(back.sequence.events[1]) is AvailabilityRemoval
    assert type(back.sequence.events[1].pairs[0]) is tuple
    assert type(back.sequence.selected) is tuple
    assert back.sequence.kind is SequenceKind.RQ2
    assert back.trajectories[0].arm.policy is PolicyName.HWF
    assert back.trajectories[0].arm.reference is ReferenceMode.CHAINED
    assert back.trajectories[1].arm.reference is None
    assert type(back.trajectories[0].arm.alpha) is Fraction
    assert back.trajectories[0].arm.alpha == Fraction(1, 2)
    assert back.trajectories[0].ended_early and back.trajectories[2].steps == ()
    assert back.s0_stages[1].status is StageStatus.TIME_LIMIT
    assert back.trajectories[0].steps[1].stages[1].value is None
    assert type(back.coverage[1].record.bound) is float
    assert back.manifest == original.manifest


def test_encode_format_details():
    """The documented representations (they are what a reader of the files sees)."""
    assert storage.encode(Fraction(1)) == "1/1"
    assert storage.encode(Fraction(1, 2)) == "1/2"
    assert storage.encode(frozenset({(1, 0), (0, 1)})) == [[0, 1], [1, 0]]
    assert storage.encode(ReferenceMode.CHAINED) == "chained"
    assert storage.encode(DemandChange(shift=3, delta=-1)) == {
        "__type__": "DemandChange",
        "shift": 3,
        "delta": -1,
        "fallback": False,
    }


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({1, 2}, id="set"),
        pytest.param(object(), id="object"),
        pytest.param(StageResult(None, None), id="namedtuple"),
        pytest.param({1: "int key"}, id="int_dict_key"),
    ],
)
def test_encode_rejects_unsupported_values(value):
    with pytest.raises(TypeError):
        storage.encode(value)


def test_decode_rejects_unknown_or_mismatched_tags():
    with pytest.raises(ValueError):
        storage.decode({"__type__": "Nonsense"}, DemandChange)
    with pytest.raises(ValueError):  # a valid tag, but not what the field declares
        storage.decode(storage.encode(AvailabilityRestoration(0, 0)), DemandChange)


def test_from_dict_rejects_other_format_or_kind():
    data = storage.to_dict(_full_result())

    with pytest.raises(ValueError):
        storage.from_dict({**data, "format": storage.FORMAT_VERSION + 1})
    with pytest.raises(ValueError):
        storage.from_dict({**data, "kind": "instance"})


# Result files


def test_result_path_layout(tmp_path):
    assert storage.result_path(tmp_path, "main", 40, 123, 45) == (
        tmp_path / "main-n40" / "123" / "45.json.gz"
    )


def test_write_then_read_results(tmp_path):
    """Results come back in sorted path order; cell filters by result.cell; instance
    files, temp files and other files are ignored.
    """
    a = _full_result("main", seed=1)
    b = _full_result("main", seed=2)
    c = _full_result("rq2", seed=1)
    storage.write_result(storage.result_path(tmp_path, "main", 20, 1, 0), a)
    storage.write_result(storage.result_path(tmp_path, "main", 20, 1, 1), b)
    storage.write_result(storage.result_path(tmp_path, "rq2", 20, 1, 0), c)
    storage.write_instance(
        storage.instance_path(tmp_path, 20, 0), a.instance, a.acceptance, LexResult(a.s0, ())
    )
    (tmp_path / "main-n20" / "1" / ".0.json.gz.123.tmp").write_bytes(b"half")
    (tmp_path / "notes.txt").write_text("not a result")

    assert list(storage.read_results(tmp_path)) == [a, b, c]
    assert list(storage.read_results(tmp_path, cell="rq2")) == [c]
    assert storage.read_result(storage.result_path(tmp_path, "main", 20, 1, 1)) == b


def test_exists_complete_false_for_truncated_and_missing(tmp_path):
    """Only a complete, current-format result file counts; nothing here raises."""
    good = storage.result_path(tmp_path, "main", 20, 1, 0)
    storage.write_result(good, _full_result())
    raw = good.read_bytes()

    truncated = tmp_path / "truncated.json.gz"
    truncated.write_bytes(raw[: len(raw) // 2])
    garbage = tmp_path / "garbage.json.gz"
    garbage.write_bytes(b"not gzip at all")
    bad_json = tmp_path / "bad_json.json.gz"
    bad_json.write_bytes(gzip.compress(b'{"format": 1, "kind": '))
    old_format = tmp_path / "old.json.gz"
    old_format.write_bytes(
        gzip.compress(json.dumps({**json.loads(gzip.decompress(raw)), "format": 0}).encode())
    )
    instance_file = storage.instance_path(tmp_path, 20, 0)
    storage.write_instance(instance_file, generate_instance(GeneratorConfig(), 20, 0),
                           AcceptanceLog(0, 0, 1, 0, False, 0), LexResult(None, ()))

    assert storage.exists_complete(good)
    for path in (tmp_path / "missing.json.gz", truncated, garbage, bad_json, old_format,
                 instance_file, tmp_path):
        assert not storage.exists_complete(path), path


def test_atomic_write_leaves_no_partial_file(tmp_path, monkeypatch):
    """A crash just before the rename keeps the old file intact and leaves no temp file;
    an encoding error leaves no file at all.
    """
    path = storage.result_path(tmp_path, "main", 20, 1, 0)
    old, new = _full_result(seed=1), _full_result(seed=2)
    storage.write_result(path, old)

    def crash(src, dst):
        raise OSError("simulated crash before the rename")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError):
        storage.write_result(path, new)
    monkeypatch.undo()

    assert storage.read_result(path) == old
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]

    other = storage.result_path(tmp_path, "main", 20, 2, 0)
    unencodable = SequenceResult(**{**vars(new), "manifest": {"x": object()}})
    with pytest.raises(TypeError):
        storage.write_result(other, unencodable)
    assert not other.exists()
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == [path]


# Instance files


def test_instance_path_layout(tmp_path):
    assert storage.instance_path(tmp_path, 20, 3) == tmp_path / "instances" / "20" / "3.json.gz"
    assert storage.instance_path(tmp_path, 40, 3, under_supplied=True) == (
        tmp_path / "instances" / "u40" / "3.json.gz"
    )


def test_instance_file_round_trip(tmp_path):
    """A generated instance, its log and S0 come back equal; a result reader refuses
    the instance file and vice versa.
    """
    instance = generate_instance(GeneratorConfig(), 20, 0)
    acceptance = AcceptanceLog(0, 0, 3, 2, False, 0)
    s0 = LexResult(
        frozenset({(0, 0), (3, 1)}),
        (StageRecord("MinShortfall", StageStatus.OPTIMAL, 0, 0.0, 0.1),),
    )
    path = storage.instance_path(tmp_path, 20, 0)

    storage.write_instance(path, instance, acceptance, s0)

    assert storage.read_instance(path) == (instance, acceptance, s0)
    with pytest.raises(ValueError):
        storage.read_result(path)
    result_file = storage.result_path(tmp_path, "main", 20, 0, 0)
    storage.write_result(result_file, _full_result())
    with pytest.raises(ValueError):
        storage.read_instance(result_file)


# Gate M2


@pytest.mark.slow
@pytest.mark.solver
def test_m2_gate_n20_sequences_with_scip(tmp_path):
    """Gate M2: a generated n = 20 instance with SCIP: accepted, certified S0, one RQ1
    sequence through the five main arms and one RQ2 sequence through both RQ2 arms.
    run_arm's oracle checks run on every step; every arm keeps Lambda = Lambda*_t; both
    result files round-trip.
    """
    settings = SolverSettings(time_limit=60.0, s0_time_limit=60.0)
    base = derive_seed(2, "instance", 20, 0)
    reserve = [derive_seed(2, "reserve", 20, 0, j) for j in range(3)]

    instance, log = accept_instance(GeneratorConfig(), 20, base, reserve, settings, milp.solve_stage)
    s0 = initial_schedule(instance, settings, milp.solve_stage)
    assert s0.certified and s0.schedule is not None

    jobs = [
        ("main", rq1_sequence(instance, s0.schedule, SequenceProfile.BALANCED, 1),
         main_arms(2, Fraction(1))),
        ("rq2", rq2_sequence(instance, s0.schedule, 1), rq2_arms()),
    ]
    for cell, sequence, arms in jobs:
        result = run_sequence(
            cell, instance, log, s0, sequence, arms, settings, milp.solve_stage, {"gate": "M2"}
        )

        for trajectory in result.trajectories:
            assert not trajectory.ended_early, trajectory.arm.name
            assert len(trajectory.steps) == 15
        instances_t = instances_along(result.instance, result.sequence.events)
        for trajectory in result.trajectories:
            for step, cover, instance_t in zip(trajectory.steps, result.coverage, instances_t):
                assert step.t == cover.t
                assert oracle.shortfall(step.schedule, instance_t) == cover.lambda_star

        path = storage.result_path(tmp_path, cell, 20, instance.meta.seed, sequence.seed)
        storage.write_result(path, result)
        assert storage.exists_complete(path)
        assert storage.read_result(path) == result



# Run control (step 4.4)


def test_failure_round_trip(tmp_path):
    """write_failure -> read_failures gives the same content back, sorted by job key; a
    second failure of the same job replaces the first.
    """
    storage.write_failure(tmp_path, "main-n20-i1-balanced-q0", "ValueError", "boom", "Traceback ...\nboom")
    storage.write_failure(tmp_path, "main-n20-i0-balanced-q0", "OracleMismatch", "value: U", "value: ...")
    storage.write_failure(tmp_path, "main-n20-i1-balanced-q0", "KeyError", "'x'", "second try")

    failures = storage.read_failures(tmp_path)

    assert storage.failure_path(tmp_path, "x") == tmp_path / "failures" / "x.json"
    assert [f["job_key"] for f in failures] == ["main-n20-i0-balanced-q0", "main-n20-i1-balanced-q0"]
    assert set(failures[0]) == {"job_key", "error_type", "message", "traceback", "host", "time"}
    assert (failures[0]["error_type"], failures[0]["message"]) == ("OracleMismatch", "value: U")
    assert (failures[1]["error_type"], failures[1]["traceback"]) == ("KeyError", "second try")
    assert failures[0]["host"] and failures[0]["time"].startswith("20")


def test_read_failures_never_raises(tmp_path):
    """No directory -> []; an unreadable file is still reported; temp files are ignored."""
    assert storage.read_failures(tmp_path) == []

    (tmp_path / "failures").mkdir()
    (tmp_path / "failures" / "broken-job.json").write_text("{not json")
    (tmp_path / "failures" / "list-job.json").write_text("[]")  # JSON, but no record
    (tmp_path / "failures" / "keyless-job.json").write_text('{"error_type": "X"}')
    (tmp_path / "failures" / ".x.json.123.tmp").write_text("half")

    failures = storage.read_failures(tmp_path)
    assert [(f["job_key"], f["error_type"]) for f in failures] == [
        ("broken-job", "unreadable failure file"),
        ("keyless-job", "unreadable failure file"),
        ("list-job", "unreadable failure file"),
    ]


def test_abort_flag(tmp_path):
    """The flag appears on request, keeps the reason, survives a second request, and is
    cleared by deleting the file (by hand, after fixing the cause).
    """
    root = tmp_path / "results"
    assert not storage.abort_requested(root)

    storage.request_abort(root, "OracleMismatch in main-n20-i0-balanced-q0")
    assert storage.abort_requested(root)
    assert storage.abort_path(root) == root / "ABORT"
    assert "OracleMismatch" in storage.abort_path(root).read_text()

    storage.request_abort(root, "second")
    assert storage.abort_requested(root)

    storage.abort_path(root).unlink()
    assert not storage.abort_requested(root)


def test_count_results_ignores_truncated(tmp_path):
    """Only complete result files count, and only they set the newest mtime."""
    assert storage.count_results(tmp_path) == (0, None)

    a = storage.result_path(tmp_path, "main", 20, 1, 0)
    b = storage.result_path(tmp_path, "main", 20, 1, 1)
    storage.write_result(a, _full_result(seed=1))
    storage.write_result(b, _full_result(seed=2))
    os.utime(a, (1_000, 1_000))
    os.utime(b, (2_000, 2_000))
    truncated = storage.result_path(tmp_path, "main", 20, 1, 2)
    truncated.write_bytes(b.read_bytes()[:100])
    os.utime(truncated, (3_000, 3_000))
    storage.write_failure(tmp_path, "some-job", "ValueError", "", "")  # not a result

    assert storage.count_results(tmp_path) == (2, 2_000.0)


def test_build_manifest_records_cpu_and_workers(tmp_path):
    """Every documented key; the protocol hash is the file's SHA-256; CPU model and worker
    count are recorded (pilot and main must match, DESIGN.md 13.13); plain JSON.
    """
    import hashlib

    protocol_file = tmp_path / "pilot.toml"
    protocol_file.write_text('name = "pilot"\n')

    manifest = storage.build_manifest(protocol_file, workers=7, solver_versions={"scip": "10"})

    assert set(manifest) == {
        "protocol", "protocol_sha256", "git_commit", "git_clean", "python", "solver", "host",
        "cpu_model", "logical_cpus", "workers", "started",
    }
    assert manifest["protocol"] == "pilot.toml"
    assert manifest["protocol_sha256"] == hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    assert manifest["workers"] == 7 and manifest["solver"] == {"scip": "10"}
    assert isinstance(manifest["cpu_model"], str) and manifest["cpu_model"]
    assert manifest["git_commit"] == "unknown" and manifest["git_clean"] is None  # tmp: no git
    assert json.loads(json.dumps(storage.encode(manifest))) == manifest


def test_build_manifest_inside_the_repository():
    """In this repository the commit is a 40-character hex hash."""
    from pathlib import Path

    manifest = storage.build_manifest(
        Path(__file__).parents[1] / "protocol" / "pilot.toml", workers=1, solver_versions={}
    )

    assert len(manifest["git_commit"]) == 40 and int(manifest["git_commit"], 16) >= 0
    assert isinstance(manifest["git_clean"], bool)


# Resumed runs must match the run they continue


def _manifest(**changes):
    base = {
        "protocol": "pilot.toml", "protocol_sha256": "aa", "git_commit": "c" * 40,
        "git_clean": True, "python": "3.14.0", "solver": {"scip": "10.0.2"},
        "host": "server-1", "cpu_model": "EPYC", "logical_cpus": 48, "workers": 24,
        "started": "2026-12-01T00:00:00+00:00",
    }
    return {**base, **changes}


def test_run_manifest_written_on_first_start_and_kept(tmp_path):
    """The first start writes it; an identical resume passes and keeps the first start
    time; keys outside RESUME_KEYS (host, start time) may differ.
    """
    storage.check_run_manifest(tmp_path, _manifest())
    assert storage.run_manifest_path(tmp_path) == tmp_path / "manifest.json"

    storage.check_run_manifest(tmp_path, _manifest(host="server-2", started="2026-12-02"))

    stored = json.loads(storage.run_manifest_path(tmp_path).read_text())
    assert stored["started"] == "2026-12-01T00:00:00+00:00"


@pytest.mark.parametrize(
    "key, value",
    [
        ("protocol_sha256", "bb"),
        ("git_commit", "d" * 40),
        ("git_clean", False),
        ("solver", {"scip": "10.0.3"}),
        ("python", "3.14.1"),
        ("cpu_model", "Xeon"),
        ("workers", 48),
    ],
)
def test_resume_with_a_changed_setup_is_refused(tmp_path, key, value):
    """Every RESUME_KEY counts; the error names the key and is a SystematicError (so the
    worker policy treats it like an oracle mismatch).
    """
    from repair.model import SystematicError

    storage.check_run_manifest(tmp_path, _manifest())

    with pytest.raises(storage.ManifestMismatch, match=key) as error:
        storage.check_run_manifest(tmp_path, _manifest(**{key: value}))
    assert isinstance(error.value, SystematicError)
    assert set(storage.RESUME_KEYS) == {
        "protocol_sha256", "git_commit", "git_clean", "solver", "python", "cpu_model", "workers"
    }
