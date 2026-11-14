# Sequential Schedule Repair

This repository contains the experiment code for my master's thesis. I study
how different repair policies change a volunteer schedule when new information
arrives over several steps. The main question is how much schedule quality,
stability and the distribution of disruption depend on the chosen policy.

The instances are synthetic. I use them for controlled comparisons under the
same conditions, not as a representative sample of real volunteer organisations.
There is no user interface here; the project generates instances, solves them
and writes results for the thesis analysis.

## What the experiment does

First, the generator creates volunteers and shifts. The default planning horizon
is 14 days with two periods per day and five possible roles per period. The main
workforce sizes are 20, 40 and 80. A period-role combination is not necessarily
active, so the number of shifts can differ between instances.

The runner checks coverage and calculates an initial schedule, `S0`. It then
applies 15 perturbations. These can change staffing demand (P1), remove
availability (P2), or change a role preference (P3). Each policy receives the
same instance, starting schedule and event sequence.

At each step, minimum shortfall is solved once. Every policy receives that
coverage cap before applying its own objectives:

| Policy | Repair rule |
| --- | --- |
| QO | Maximise quality. |
| SF | Minimise changed assignment cells, then maximise quality. |
| QL | Maximise quality, then minimise changed cells. |
| WQS | Maximise quality minus `lambda` times changed cells. |
| HWF | Minimise history-weighted changed cells, then maximise quality. |

The runner adds a seeded tie-break stage. This helps make repeated solves
consistent, but different schedules can still have the same rank sum.

The main policy comparison uses chained references: repair distance is measured
against the arm's previous schedule. RQ2 compares SF with an anchored reference
(`S0`) against SF with a chained reference. It removes one assigned availability
pair at step 1 and restores it at step 6, with background events between and
after these steps. This is used to study persistence and lock-in.

## Setup

The project requires Python 3.12 or newer. Dependencies are declared in
`pyproject.toml`; `uv.lock` records the resolved versions used for this snapshot.
The main dependencies are NumPy, pandas and PySCIPOpt/SCIP. Tests use pytest.

From the project root, on macOS or Linux:

```bash
uv sync --locked
source .venv/bin/activate
export PYTHONPATH=src:.
```

Keep `src` on the import path because `oracle.py` is currently a separate module
inside that directory. The commands below assume this environment is active.

Without uv, the declared dependencies can also be installed with pip. This does
not reproduce the exact locked environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export PYTHONPATH=src:.
```

## Start with the tests

For a first check, run the regular tests and then the small pipeline test:

```bash
python -m pytest -m "not slow and not todo"
python -m pytest tests/test_smoke.py -m ""
```

The smoke test generates inputs, runs repairs, writes analysis tables and repeats
two completed jobs in a temporary directory. It needs SCIP. To run all tests,
including the slower distribution and solver checks:

```bash
python -m pytest -m "not todo"
```

For checks without the solver:

```bash
python -m pytest -m "not solver and not slow and not todo"
```

One storage test expects a real Git checkout and reads its commit hash. It fails
in an extracted ZIP without Git history. That is a checkout requirement of the
test, not evidence of a different scheduling result.

I use small hand-calculated examples as the starting point for testing. Their
values are in [tests/fixture_value_sheet.md](tests/fixture_value_sheet.md) and
`src/repair/expected.py`. The brute-force solver checks these examples, and the
MILP is compared with brute force. The Python oracle separately checks hard
constraints and objective values of solved schedules.

## Run the rehearsal

`protocol/dry-run.toml` is the prepared rehearsal configuration: three n=20
instances, 20 sequences per main profile, plus one RQ2 sequence per instance.
This gives 183 jobs. It tests execution and restart behaviour; its results are
not the main study sample.

```bash
python scripts/make_instances.py protocol/dry-run.toml --out results/dry-run --workers 2
python scripts/run_experiment.py protocol/dry-run.toml --out results/dry-run --count
python scripts/run_experiment.py protocol/dry-run.toml --out results/dry-run --workers 2
python scripts/analyse.py results/dry-run --out results/dry-run/analysis
python scripts/rerun_check.py protocol/dry-run.toml results/dry-run --jobs 2
```

The worker count here is a local example. For a resumed run, keep it unchanged.
The longer server rehearsal, including planted failures and monitoring, is
described in [SERVER.md](SERVER.md).

For the smaller M2 integration check, run:

```bash
python scripts/m2_gate.py --size 20
```

This generates one instance, runs one RQ1 and one RQ2 sequence, and checks the
saved results. It uses a separate gate seed root and output directory.

## Pilot and main run

The files have different purposes:

| File | Current use |
| --- | --- |
| `protocol/dry-run.toml` | Execution and monitoring rehearsal. |
| `protocol/pilot.toml` | Pilot template with small placeholder counts. |
| `protocol/main.toml` | Unfrozen main template; counts and calibrated values still need filling. |

The full pilot job list currently refuses to run because it includes robustness
and under-supplied cells without their required values: `robustness_scale` and
`workforce_factor`. Set these from the pilot decisions, or prepare an explicitly
reduced pilot configuration first. Changing `--cell` alone does not bypass the
validation of the protocol's full job list.

Calibration commands write CSV tables and do not edit the protocol:

```bash
python scripts/calibrate.py phi protocol/pilot.toml --out results/pilot
python scripts/calibrate.py intensity protocol/pilot.toml --out results/pilot
python scripts/calibrate.py background protocol/pilot.toml --out results/pilot
```

`phi` selects a tested workforce factor from coverage solves. `intensity`
matches independent targeting to the S0-targeted design using non-inert states
against S0. `background` only reports diagnostics at the default `k=2`; it does
not select a calibrated RQ2 background scale.

Before starting the main experiment, complete
[MAIN_RUN_CHECKLIST.md](MAIN_RUN_CHECKLIST.md), fill and date `main.toml`, and
record the code version. The `--main` flag checks that the protocol is filled
and frozen; it should not be used for trying out unfinished settings.

## Where to read the code

| Location | Responsibility |
| --- | --- |
| `src/repair/model.py` | Data objects, validation and result records. |
| `src/repair/generator.py` | Seeded synthetic volunteers and shifts. |
| `src/repair/perturbations.py` | Event sampling and immutable instance updates. |
| `src/repair/protocol.py` | TOML settings, seeds, arms and job matrix. |
| `src/repair/objectives.py`, `policies.py` | Objective definitions and policy stage order. |
| `src/repair/milp.py` | SCIP variables, constraints and stage solves. |
| `src/oracle.py` | Independent schedule feasibility and score calculations. |
| `src/repair/runner.py` | Acceptance, lexicographic solves and sequential repairs. |
| `src/repair/metrics.py` | Churn, drift, burden, inequality and RQ2 measures. |
| `src/repair/analysis.py` | Instance summaries, paired comparisons and bootstrap intervals. |
| `src/repair/storage.py`, `monitoring.py` | Saved data, restart checks and alert decisions. |
| `scripts/` | Commands that connect these parts. |
| `tests/` and `conftest.py` | Hand examples, reference solver and tests. |

To understand one repair, I would start with `tests/test_slice.py`, then read
`runner.py`, `policies.py` and `milp.py`. The slice test follows the small example
from S0 to the repaired schedule and its churn value.

## Results and reproducibility

Instance files contain the accepted inputs, acceptance log and S0. Sequence
files contain the base instance, events, shared coverage and each arm's stages
and schedules. They are compressed JSON files. The run also writes a manifest
with the protocol hash, code commit, Python and solver versions, CPU and worker
count.

Starting the same run again skips complete results. A changed manifest is
refused instead of mixing runs. Failure details are stored under `failures/`;
an `ABORT` file stops workers from taking new jobs. Diagnose its cause before
removing it and resuming. See SERVER.md for the full procedure.

The analysis command writes ten CSV tables: step, sequence and instance
summaries, RQ2 outcomes, H1-H3 results and burden-sensitivity checks. The primary
analysis uses sequences certified through all 15 steps. Secondary summaries
stop at the common certified prefix. Sequences are averaged within a base
instance before the paired comparisons and bootstrap.

Undefined values stay missing. For example, a quality gap cannot be calculated
from an unproven QO value. A returned feasible schedule and a certified optimum
are different things in the stored results.

## Work still open in this snapshot

- `scripts/plan_precision.py` is unfinished. It iterates the DataFrame from
  `analysis.precision_plan()` as if it contained row objects, and it does not
  yet calculate the H3 precision requirement.
- `runner.accept_instance()` uses the solver's recorded shortfall without an
  oracle check on that acceptance solve. The later S0 check validates its own
  solve, not the earlier acceptance decision.
- The H1 direction and the possible materiality threshold still need the check
  against the frozen exposé noted in `analysis.py` and NOTES.md.
- `DESIGN.md` and `ROADMAP.md` are referenced by the code but are not included in
  this archive. Their references are kept so they can be matched to the original
  project documents.

Decisions and earlier run observations are recorded in [NOTES.md](NOTES.md).
