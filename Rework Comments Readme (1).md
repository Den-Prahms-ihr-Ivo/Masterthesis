# Sequential Schedule Repair

My thesis code for repairing volunteer schedules. I compare schedule quality,
how much changes, and whether the same people keep getting affected.

The data is generated, so conclusions are about this model. No UI, just Python
scripts and result files.

## Basic idea

Default: 14 days, two periods/day, five possible roles, 20/40/80 volunteers.
Not every role is active in every period.

Generate an instance, solve the starting schedule `S0`, then apply 15 changes:

- P1: change staffing demand.
- P2: remove availability.
- P3: change a role preference.

Same inputs and events for all policies. Solve minimum shortfall once per step;
every policy gets that coverage cap.

| Policy | What it does after coverage                                                          |
| ------ | ------------------------------------------------------------------------------------ |
| QO     | Best quality.                                                                        |
| SF     | Fewest changed cells, then best quality.                                             |
| QL     | Best quality, then fewest changed cells.                                             |
| WQS    | Quality minus `lambda * changed cells`.                                              |
| HWF    | Weighted changes first, then quality. Volunteers affected before get higher weights. |

A cell is a volunteer-shift assignment. Moving someone changes two cells.

Main comparison: changes against the previous schedule (chained).
RQ2: SF-chained versus SF using `S0` throughout (anchored). Remove one assigned
availability pair at step 1, restore it at step 6, with background events around
it. Used to study persistence and lock-in.

A seeded tie-break helps with repeatability. Equal rank sums are possible,
so the final schedule is not necessarily unique.

## Setup

Python 3.12+, NumPy, pandas, PySCIPOpt/SCIP and pytest.
Dependencies in `pyproject.toml`, pinned versions in `uv.lock`.

From the project root on macOS/Linux:

```bash
uv sync --locked
source .venv/bin/activate
export PYTHONPATH=src:.
```

Keep this environment active. `PYTHONPATH` includes the separate `src/oracle.py`.

Without uv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export PYTHONPATH=src:.
```

Pip does not use the locked versions.

## Tests

Erstmal check that things run:

```bash
python -m pytest -m "not slow and not todo"
python -m pytest tests/test_smoke.py -m ""
```

Smoke test: inputs, repairs, tables and two reruns, using SCIP and a temporary
directory.

```bash
# Full suite, including slow tests
python -m pytest -m "not todo"

# Skip solver and slow tests
python -m pytest -m "not solver and not slow and not todo"
```

Hand values: [tests/fixture_value_sheet.md](tests/fixture_value_sheet.md) and
`src/repair/expected.py`. Check brute force against these, then MILP against
brute force. The oracle checks feasibility and scores separately.

One storage test needs a Git commit hash and fails in a ZIP without Git history.

## Try a run

Small M2 check: one instance, RQ1 + RQ2, separate gate seeds and output:

```bash
python scripts/m2_gate.py --size 20
```

Larger rehearsal: `dry-run.toml`, 183 jobs. Three n=20 instances, each with
20 sequences per main profile and one RQ2 sequence.

```bash
python scripts/make_instances.py protocol/dry-run.toml --out results/dry-run --workers 2
python scripts/run_experiment.py protocol/dry-run.toml --out results/dry-run --count
python scripts/run_experiment.py protocol/dry-run.toml --out results/dry-run --workers 2
python scripts/analyse.py results/dry-run --out results/dry-run/analysis
python scripts/rerun_check.py protocol/dry-run.toml results/dry-run --jobs 2
```

Two workers is the local example; keep the count when resuming. Rehearsal results
stay out of the main analysis. Server setup and monitoring: [SERVER.md](SERVER.md).

## Pilot and main run

Status from [NOTES.md](NOTES.md), 19 November 2026:

| Protocol       | Purpose                                                          |
| -------------- | ---------------------------------------------------------------- |
| `dry-run.toml` | Rehearsal, 60 s per stage.                                       |
| `pilot.toml`   | Main + RQ2 cells, 600 s per stage. Counts still need deciding.   |
| `main.toml`    | Main experiment. Fill counts and calibrated values, then freeze. |

Calibration writes CSV files without editing the protocol:

```bash
python scripts/calibrate.py phi protocol/pilot.toml --out results/pilot
python scripts/calibrate.py intensity protocol/pilot.toml --out results/pilot
python scripts/calibrate.py background protocol/pilot.toml --out results/pilot
```

`phi`: workforce reduction from coverage solves. `intensity`: match independent
targeting to the S0-targeted design using non-inert states against S0.
`background`: diagnostics at `k=2`, no scale calibration.

Before main: finish [MAIN_RUN_CHECKLIST.md](MAIN_RUN_CHECKLIST.md), fill/date
`main.toml`, record the code version. `--main` checks the frozen protocol.
No trial settings there.

## Where things are

Most code is in `src/repair/`:

| File                               | Job                                            |
| ---------------------------------- | ---------------------------------------------- |
| `model.py`                         | Data objects and validation.                   |
| `generator.py`, `perturbations.py` | Generate instances and events.                 |
| `protocol.py`                      | Settings, seeds and job list.                  |
| `objectives.py`, `policies.py`     | What each policy optimises.                    |
| `milp.py`                          | SCIP model and solves.                         |
| `runner.py`                        | Connect the stages and repairs.                |
| `metrics.py`, `analysis.py`        | Calculate metrics and compare results.         |
| `storage.py`, `monitoring.py`      | Save results, check resumes and decide alerts. |
| `src/oracle.py`                    | Separate feasibility and score checks.         |
| `scripts/`                         | Commands for running the project.              |
| `tests/`, `conftest.py`            | Tests, fixtures and brute-force solver.        |

Start with `tests/test_slice.py` to follow one repair. Then `runner.py`,
`policies.py` and `milp.py`.

## Results and restart

Inputs and results are saved as compressed JSON. The manifest records the
protocol hash, commit, Python/solver versions, CPU and worker count.

Repeating a run skips complete results. A changed manifest is refused.
Failures go into `failures/`; `ABORT` stops new jobs. Wichtig: check the cause
before deleting ABORT and continuing.

Analysis writes ten CSV tables. Primary results need all 15 steps certified.
Secondary results use the common certified prefix. Sequences are averaged
within each base instance before paired comparisons and bootstrapping.

Missing values stay missing. A feasible schedule is not automatically a proven
optimum, and an unproven QO value cannot define the quality gap.

## Current notes

Noch offen: pilot counts and the server rehearsal. The laptop rehearsal passed.

Earlier review item to check: `accept_instance()` used the solver's shortfall
without an oracle check on that solve. The later S0 check does not cover it.

Further decisions and run notes go in [NOTES.md](NOTES.md).
