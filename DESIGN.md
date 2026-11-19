# Implementation Design: Sequential Schedule Repair Study

Revision 3 (22 Oct 2026). Covers the exposé as sent to Dr. Lu on 22 Oct 2026. You write every function body and every test (TDD).
This document fixes **what** each piece does and **how the pieces connect**. Section 11 explains the alternatives I rejected and what changed between revisions.

---

## 1. Design rules

1. **Functional core, solver at the edge.**
   - Everything except `milp.py` and `runner.py` is pure functions over frozen dataclasses.
   - Only `milp.py` imports `pyscipopt`, and only inside its functions, so every other module imports without SCIP.
   - Only `protocol.load_protocol`, `storage.py` and `scripts/` touch the file system. Only `scripts/monitor.py` touches the network (SMTP, health-check ping).
2. **One solve = one fresh model.** `milp.solve_stage` builds a new SCIP model for one objective, with the earlier stage values added as constraints. No SCIP state survives a call: no `freeTransform`, no reoptimization, and no model shared across stages, arms or steps.
3. **The solver is a function argument.** Every runner function that solves takes `solve`.
   - Production passes `milp.solve_stage`.
   - Tests pass `tests/bruteforce.solve_stage`, which enumerates every schedule of a tiny instance.
   - So the whole runner is tested without SCIP, and `milp` is tested against the brute force. The argument is a plain function parameter, not a framework.
4. **A schedule is a `frozenset[(volunteer_id, shift_id)]`.**
   - Distance is `len(a ^ b)`.
   - A shift is cancelled iff it has no assignees (open shifts need at least ⌈d_s/2⌉ ≥ 1 people), so cancellation needs no separate state.
5. **Python oracles check every solver result.** `oracle.py` re-implements the constraints, U and Λ, and `objectives.evaluate` recomputes every stage value. A mismatch raises; it never warns.
6. **All objectives are integer.** λ is an integer, and α is rescaled per stage (α = 0.5 → weights 2 + b). Stage fixing is therefore exact (`≥ value`, no ε).
7. **Perturbations cannot see an arm.** Sequence sampling takes the instance and S₀ only. All arms receive the same list of step instances.
8. **Store raw trajectories, compute everything else afterwards, including the step classification.** A metric bug costs a recomputation, never a rerun.

**Deliberately not used:** class hierarchies, plugin registries, dependency-injection frameworks, persistent solver models, databases, workflow engines, or config frameworks beyond stdlib `tomllib`.

---

## 2. Module map and import rules

```
model ─┬─> protocol        numbers, seeds, experiment matrix (jobs)
       ├─> generator       Table A1 instances
       ├─> perturbations   events, RQ1 / RQ2 / independent sequences
       ├─> oracle          feasibility, U, Λ (independent re-implementation)
       ├─> objectives ──> policies
       ├─> milp            solve_stage: the only pyscipopt user
       ├─> runner          lexicographic loop, acceptance, S₀, coverage, arms (gets `solve`)
       ├─> storage         .json.gz per sequence
       ├─> metrics         post hoc: outcome measures and classification
       ├─> analysis        certified prefix, CIs, H1–H3 (the only pandas user)
       └─> monitoring      pure alert logic (progress deciles, failures, stalls) for scripts/monitor.py

scripts/ wire them: protocol → runner(solve=milp.solve_stage) → storage → metrics → analysis
```

| Module                | May not import                                                              |
| --------------------- | --------------------------------------------------------------------------- |
| `oracle`              | `objectives`, `milp` (it is their independent check)                        |
| `runner`              | `milp` (it receives `solve`)                                                |
| `metrics`, `analysis` | `milp`, `runner`                                                            |
| `monitoring`          | everything except `model` (it gets plain numbers from `scripts/monitor.py`) |
| everything but `milp` | `pyscipopt`                                                                 |

A ten-line test can enforce this table by parsing the imports with `ast`.

---

## 3. File structure

```
repair-study/
├── pyproject.toml            # python >= 3.11; pyscipopt pinned in week 0; markers "solver", "slow"
├── .github/workflows/tests.yml  # CI: every push to dev, every pull request into main
├── DESIGN.md                 # this file
├── ROADMAP.md                # task-level plan, definition of done, traceability
├── SERVER.md                 # runbook: renting, setting up and running on the server
├── NOTES.md                  # (you create it) dated decision log; becomes the pilot report
├── README.md
├── protocol/
│   ├── pilot.toml            # numbers only: main time limit (600 s), small N, own root seed
│   └── main.toml             # numbers only; DATED and frozen after the pilot
├── src/repair/
│   ├── model.py              # Volunteer, Shift, Instance, Schedule, events, Sequence, Arm, records
│   ├── protocol.py           # GeneratorConfig, SolverSettings, Protocol, derive_seed, jobs, shard
│   ├── generator.py          # seeded instance generation (Table A1), under-supplied variant
│   ├── perturbations.py      # apply events; RQ1 / RQ2 / independent-targeting sequences
│   ├── oracle.py             # hard constraints, U, Λ, surplus, mentorship
│   ├── objectives.py         # ObjectiveSpec types + evaluate()
│   ├── policies.py           # Arm -> list[ObjectiveSpec]; references; history weights; tie-break ranks
│   ├── milp.py               # solve_stage: fresh model, one objective, fixed stages, cap, warm start
│   ├── runner.py             # solve_lexicographic, acceptance, S0, shared coverage, run_arm, run_sequence
│   ├── storage.py            # instance files, result files, failure files, ABORT flag, manifest
│   ├── metrics.py            # classification; step / terminal / RQ2 measures
│   ├── analysis.py           # certified prefix, instance table, bootstrap CIs, H1–H3, precision plan
│   └── monitoring.py         # pure alert logic: deciles, failures, stalls, ETA, mail texts
├── scripts/
│   ├── make_instances.py     # accept instances + S0 for a protocol
│   ├── run_experiment.py     # run jobs: restartable, --workers, --shard i/N, failure files, ABORT
│   ├── monitor.py            # e-mail every 10 %, on failure / stall / finish; health-check ping
│   ├── calibrate.py          # phi, robustness intensity, RQ2 background intensity
│   ├── plan_precision.py     # N_I and core-hour projection from pilot output
│   └── analyse.py            # tables, figures, hypothesis verdicts
├── tests/
│   ├── conftest.py           # tiny hand instances (<= 4 volunteers, <= 4 shifts)
│   ├── bruteforce.py         # test-only stage solver with milp.solve_stage's signature
│   ├── builders.py           # test helpers: make_instance, greedy_schedule, spy/scripted solves, trajectories
│   ├── fake_results.py       # planted-effect tables for test_analysis
│   ├── test_<module>.py      # one per module; docstrings list the suggested cases
│   ├── test_slice.py         # week-1 vertical slice
│   ├── test_architecture.py  # import rules of §2, checked with ast
│   └── test_smoke.py         # end-to-end scripts run (solver, slow)
└── results/                  # git-ignored
```

---

## 4. Data model (`model.py`)

| Type             | Fields                                                                                                                      | Notes                                 |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `Volunteer`      | `id, available: frozenset[int], skill: tuple` (per role), `seniority, cap, pref_role: tuple, pref_time: tuple` (per period) | Immutable                             |
| `Shift`          | `id, period, role, demand` (current d_s), `min_skill` (κ_s), `expert` (e_s)                                                 | `demand ≥ 1` always                   |
| `Instance`       | `volunteers, shifts, n_periods, n_roles, size_class` (demand class), `scale` (k), `seed, weights`                           | Events return a **new** Instance      |
| `Schedule`       | `frozenset[tuple[int, int]]`                                                                                                | Type alias                            |
| Events           | `DemandChange` (with `fallback`), `AvailabilityRemoval`, `PreferenceChange`, `AvailabilityRestoration`                      | Union type `Event`                    |
| `Sequence`       | `kind` (rq1 / rq2 / independent), `profile, seed, events, selected` (RQ2 pair)                                              | `digest()` for replay checks          |
| `Arm`            | `policy` (QO / SF / QL / WQS / HWF), `reference` (anchored / chained / None), `lam, alpha`                                  | `name`, e.g. `"HWF-chained-a1"`       |
| `StageRecord`    | `name, status` (optimal / time_limit / no_solution), `value, bound, seconds, retried`                                       | One per solve                         |
| `LexResult`      | `schedule, stages`                                                                                                          | `certified` property                  |
| `Step`           | `t, schedule, stages`                                                                                                       | No classification; metrics derives it |
| `Trajectory`     | `arm, steps, ended_early`                                                                                                   |                                       |
| `CoverageStep`   | `t, lambda_star, record`                                                                                                    | Shared by all arms                    |
| `AcceptanceLog`  | `base_seed, used_seed, draws, rejected, replaced, lambda_star_0`                                                            | Reported by size class (§4.3)         |
| `SequenceResult` | `cell, instance` (full base instance), `acceptance, sequence, s0, s0_stages, coverage, trajectories, manifest`              | What `storage` writes; self-contained |

---

## 5. Solve pipeline

### 5.1 The stage primitive (`milp.solve_stage`)

```
solve_stage(instance, objective, *, fixed, shortfall_cap, start, time_limit, settings)
    -> (schedule | None, StageRecord)
```

- `fixed` lists the earlier stages as `(spec, value)` pairs. Each becomes `expr ≥ value` (max) or `expr ≤ value` (min).
- `start` is handed to SCIP as a primal solution (`addSol`).
- **Distance is linear in x and needs no auxiliary variables:**
  H(x, ref) = Σ*{j∈ref, available} (1 − x_j) + Σ*{j∉ref, available} x_j + |ref ∩ unavailable|.
  - The last term is a constant: a reference cell that is no longer available is always changed.
  - Weighted distance multiplies each term by the volunteer's weight.
- **Auxiliaries are bounded from one side only.**
  - q, z and y make the solver's U-expression a lower bound on the true U(x).
  - u makes the Λ-expression an upper bound on the true Λ(x).
  - Both are tight at a stage that optimises them. So a fixing constraint `U_expr ≥ v` or a cap `Σu ≤ c` stays valid in every later stage, and the oracle recomputes true values from x alone.

### 5.2 The lexicographic loop (`runner.solve_lexicographic`)

```
fixed = []; start = given start
for spec in specs:
    schedule, rec = solve(instance, spec, fixed=fixed, cap, start, τ)
    if schedule is None: retry once with 2τ (rec.retried = True); still None → return LexResult(None, records)
    fixed.append((spec, rec.value))      # incumbent value if the stage timed out
    start = schedule
return LexResult(schedule, records)
```

### 5.3 Per instance (once)

1. `generate_instance(cfg, size_class, seed)` produces a candidate.
2. **Acceptance:** a coverage solve on the candidate.
   - Main classes: accept iff Λ\*₀ = 0. Otherwise redraw, up to 100 draws, then take the next seed from the reserve list.
   - Under-supplied class: accept everything and record Λ\*₀.
3. **S₀** = `solve_lexicographic([MinShortfall, MaxQuality, MaxTieBreak])` with the S₀ time limit. Every stage must be proven optimal; otherwise the instance is replaced from reserve (logged).

### 5.4 Per sequence (shared by all arms)

1. `instances_t = instances_along(instance, sequence.events)` builds instance₁ … instance_T once.
2. **Shared coverage**, for each t: `[MinShortfall]` started from the all-cancelled schedule gives Λ\*\_t, and its solution becomes the warm start for every arm.
   - If this stage times out, its incumbent becomes the cap, and the coverage record says `time_limit`.

### 5.5 Per arm, per step t

| Policy | Stages (under the cap Λ ≤ Λ\*\_t), then `MaxTieBreak`                |
| ------ | -------------------------------------------------------------------- |
| QO     | `MaxQuality`                                                         |
| SF     | `MinDistance(ref)` → `MaxQuality`                                    |
| QL     | `MaxQuality` → `MinDistance(ref)`                                    |
| WQS    | `MaxQualityMinusDistance(ref, λ)`                                    |
| HWF    | `MinDistance(ref, weights = scale·(1 + α·b_{w,t−1}))` → `MaxQuality` |

- **References:** anchored uses S₀; chained uses the arm's own S\_{t−1}. QO has no reference.
- **Warm start:** stage 1 starts from the shared coverage solution. Each later stage starts from the previous stage's schedule, which satisfies all fixings so far.
- **Oracle check** after every step:
  - the schedule is feasible and Λ ≤ cap;
  - every proven stage's value equals `objectives.evaluate`;
  - every unproven stage's value is no worse than recorded.
- **Burden update:** `b ← b + 1[metrics.affected(S_{t−1}, S_t)]`, which HWF needs at t+1.
- **No solution after the retry:** the arm ends with `ended_early=True`, and analysis cuts the sequence for all arms.

### 5.6 What is computed where

| Quantity                                                                                                       | Computed in                   | When                         |
| -------------------------------------------------------------------------------------------------------------- | ----------------------------- | ---------------------------- |
| Λ\*\_t, S₀, S_t, stage records                                                                                 | `runner` with `solve`         | Run time (stored)            |
| Oracle checks                                                                                                  | `runner`                      | Run time (raise, not stored) |
| Classification (binding / relevant / inert / undetermined), binding rates                                      | `metrics.classifications`     | Afterwards                   |
| U*t, g_t, C_t, B_t, A_t, b*{w,t}, G_T, max share, cell variant, per-volunteer burden vectors                   | `metrics`                     | Afterwards                   |
| Burden sensitivity check (§4.6): rank agreement of the two burden counts, H2 contrast with changed-cell burden | `analysis.burden_sensitivity` | Afterwards                   |
| D*i, D_pre, P*{1,6}, lock-in indicator (Sydow et al. criteria; never labelled "path dependence")               | `metrics.rq2_row`             | Afterwards                   |
| Certified prefix, per-instance means, CIs, H1–H3, precision plan                                               | `analysis`                    | Afterwards                   |

`U*_t` is the QO arm's `MaxQuality` stage value at step t. It is valid only where that stage is optimal; otherwise analysis uses bounds.

---

## 6. Storage

- **Instances once:** `results/<protocol>/instances/<size>/<index>.json.gz` (instance, acceptance log, S₀), written by `make_instances.py`. The RQ2 and robustness jobs reuse the main instances. The sensitivity arms run inside the main n = 40 balanced jobs, so their results sit in the same files as the main arms.
- **One file per sequence:** `results/<protocol>/<cell>-n<size>/<instance_seed>/<sequence_seed>.json.gz`, holding the instance, S₀, the coverage steps and all arm trajectories.
- **Schedules** are sorted lists of `[w, s]` pairs; Fractions are `"p/q"` strings.
- **Atomic and restartable:** write to a temp file, then rename. The runner skips any job whose file exists and parses.
- **Manifest** per run: protocol file hash, git commit, Python / SCIP / PySCIPOpt versions, host, start and end time.
- **Size:** tens of kB per sequence, well under 1 GB in total.

---

## 7. Experiment matrix and compute

The TOML holds only numbers. `protocol.jobs(protocol)` builds this matrix in code, in a fixed order:

| Cell            | Instances                                | Sequences                                     | Arms                                                                                                                    |
| --------------- | ---------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `main`          | n ∈ `size_classes`, N_I each             | 3 profiles × N_Q, S₀-targeted                 | QO, SF, QL, WQS(λ), HWF(α), all chained; **n = 40 balanced jobs add** WQS λ ∈ {1, 4} and HWF α ∈ {0.5, 2} (exposé §4.5) |
| `rq2`           | Same instances                           | 1 RQ2 sequence each                           | SF-anchored, SF-chained                                                                                                 |
| `robustness`    | n = 40, N_sec instances                  | Balanced, independent targeting, calibrated k | 5 main arms                                                                                                             |
| `undersupplied` | ⌊40φ⌋ volunteers, demand class 40, N_sec | Balanced, k = 2                               | 5 main arms                                                                                                             |

**Sensitivity grid.** Since 22 Oct, exposé §4.5 states that the other grid values "run as additional arms on the main S₀-targeted sequences of the n = 40 balanced profile, so every setting is compared on identical inputs". So there is no separate cell: those main jobs simply carry nine arms. They share one coverage solve per step, and the comparison with the main λ = 2 and α = 1 stays inside one result file. By default the extra arms run on _all_ main n = 40 balanced sequences, which is what "the main sequences" says. `sensitivity_instances` in the protocol can restrict them to the first m instances, but that is a deviation from the exposé wording and must be recorded in `main.toml` and reported.

**Recommendation: N_Q = 1.** The unit of analysis is the base instance, so extra sequences per instance add little precision for their cost.

**Solves (= model builds) per step:**

- RQ1: 1 (coverage) + QO 2 + SF 3 + QL 3 + WQS 2 + HWF 3 = **14**, i.e. 210 per sequence.
- RQ1 n = 40 balanced, with the sensitivity arms: 14 + 2 × WQS 2 + 2 × HWF 3 = **24**, i.e. 360 per sequence.
- RQ2: 1 + 2 × 3 = 7, i.e. 105 per sequence.

**Budget check:**

    core-seconds ≈ Σ_cells sequences × solves_per_sequence × (mean_solve_time + build_and_presolve_time)

The 1,000 core-hour budget equals 3.6 M core-seconds. A fresh model per stage adds a build-and-presolve cost to every solve. I expect it to be small next to solve times, but I have not measured it; week 0 measures it at n = 80.

**Running:** `run_experiment.py --workers K --shard i/N`. Shard i of N runs `jobs[i::N]`. The same command therefore works on one multi-core machine, on several machines, or as a SLURM array, with no coordination, and restarting it resumes where it stopped.

**Where:** a rented server with dedicated vCPUs, billed by the hour (SERVER.md). The pilot and the main run use the **same server type and the same worker count** (13.13), because certification within τ depends on per-process speed.

**Running and monitoring:**

- **Workers** catch every exception in a job and write `results/<protocol>/failures/<job>.json` (type, message, traceback). A `SystematicError` (`OracleMismatch`, `ManifestMismatch`) or an `AssertionError` is a systematic bug, so it also writes `ABORT`, and no new jobs start; jobs already running finish. A **breaker** catches unclassified bugs: the same error type and message in 5 failure files also writes `ABORT`. Other errors are logged and the run continues. `KeyboardInterrupt` and `SystemExit` are never caught (SCIP's own "userinterrupt" status is re-raised as `KeyboardInterrupt`).
- **Resuming:** the first start writes `results/<protocol>/manifest.json`. A later start with a different protocol hash, commit, clean flag, solver or Python version, CPU model or worker count is refused (`ManifestMismatch`).
- **`scripts/monitor.py`** runs alongside the workers. Every poll (10 minutes) it sends one mail per kind at most: each new 10 % of finished jobs (with an ETA), one batched mail for all new failures, ABORT, no new result for `--stall-hours`, finished. Every start also sends one "monitor started" summary. Its decisions come from the pure `monitoring.decide`; its state in `results/<protocol>/.monitor/state.json` prevents repeated mails after a restart, and the failure and abort flags re-arm once their files are gone.
- **Dead-man's switch:** every poll pings `HEALTHCHECK_URL`. If the server dies, the pings stop and the external service e-mails you. The monitor can't, because it died too.
- **Credentials** come from environment variables only (`~/.repair-study.env`, never committed).

---

## 8. Roadmap

Weeks follow Table 2 of the exposé. Each week ends with a **gate**: a check that must pass before the next week starts. The task-level plan, definition of done and traceability table are in **ROADMAP.md**.

| Week         | Dates        | Work                                                                                                                                                                                                                                                                                                                                            | Gate                                                                                                                                                        |
| ------------ | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 (optional) | 26 Oct–1 Nov | Repo, venv, pinned PySCIPOpt, CI running `pytest -m "not solver"`. **Solver spike** on a throwaway model of realistic size (n = 80, ~84 shifts): a two-stage lexicographic solve with fresh models, a fixing constraint, `addSol` on a fresh model, single thread, fixed seeds. Measure build + presolve time.                                  | Identical results across 3 runs; warm start accepted; build + presolve time per stage known                                                                 |
| 1            | 2–8 Nov      | `model`, `oracle`, hand fixtures in `conftest.py`, `objectives` (MinShortfall, MaxQuality, MaxTieBreak, unweighted MinDistance), `bruteforce.solve_stage`, `runner.solve_lexicographic` + `check_oracles` (with brute force), `milp.solve_stage` for those four specs. **Vertical slice:** hand instance → S₀ → one P2 event → SF repair → C_t. | `milp` = brute force on all fixtures; slice runs with both solvers                                                                                          |
| 2            | 9–15 Nov     | `generator` (Table A1, sub-streams, under-supplied), `perturbations` (P1 fallback, P2, P3, restoration, RQ2 pattern, independent targeting), `protocol` (`derive_seed`, `load_protocol`, `jobs`, `shard`).                                                                                                                                      | Same seed → identical instance and digest; sequence invariants pass; `jobs()` matches the matrix; acceptance rates logged per size                          |
| 3            | 16–22 Nov    | Weighted `MinDistance` and `MaxQualityMinusDistance` in `objectives` and `milp`; `policies`; `runner` (acceptance, S₀, coverage, `run_arm`, `run_sequence`), tested with brute force; `storage`.                                                                                                                                                | Identities pass with brute force (HWF(α=0) ≡ SF, WQS(large λ) ≡ SF, QO = max U); one generated n = 20 sequence runs with SCIP and passes every oracle check |
| 4            | 23–29 Nov    | `metrics` (incl. classification), `analysis` (certified prefix, instance table, bootstrap CIs, H1–H3 verdicts, precision plan), `run_experiment.py` with workers and shards, `make_instances.py`.                                                                                                                                               | Metrics exact on hand trajectories; analysis recovers planted effects in fake results; a killed run resumes                                                 |
| 5            | 30 Nov–6 Dec | **Pilot:** stage times per size, certification rates, σ_d, φ calibration, robustness intensity, RQ2 background check, capacity ratio Σ L_w / Σ d_s, P1 fallback rate. **Fill in, date and commit `protocol/main.toml`.**                                                                                                                        | Protocol committed with its date before any main-run seed is used                                                                                           |
| 6            | 7–13 Dec     | Main run: all four cells (sensitivity arms included in main n = 40)                                                                                                                                                                                                                                                                             | All jobs finished or excluded by the adequacy rule (logged)                                                                                                 |
| 7            | 14–20 Dec    | Analysis, figures, H1–H3 verdicts per size class                                                                                                                                                                                                                                                                                                | Every number in the results chapter is generated by `analyse.py`                                                                                            |
| 8            | 21–27 Dec    | Writing (the methodology chapter is drafted continuously from week 1)                                                                                                                                                                                                                                                                           | Complete draft                                                                                                                                              |

Week 3 is lighter than in revision 1. The runner is tested with the brute force, so it can be finished before any SCIP run of a generated instance.

### If you fall behind

Cut in this order. Each cut keeps a promise made in the exposé, just at lower precision.

1. Halve N_sec (robustness, under-supplied). These cells are secondary and reported with their achieved precision. Restricting the sensitivity arms with `sensitivity_instances` is a further option, but it departs from the §4.5 wording, so record and report it.
2. Drop n = 80 from `size_classes` through the adequacy rule if the pilot shows low certification. The exposé already allows this.
3. If build overhead turns out large at n = 80, add the model-reusing variant from 13.11. Only one function argument in `scripts/` changes.

**Never cut:** oracle checks, the brute-force agreement tests, or the dated protocol.

---

## 9. Test strategy

★ marks tests that protect validity; the rest protect details. The full suggested cases are in each `tests/test_*.py` docstring.

| Needs SCIP        | Tests                                                                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| No                | model, protocol, generator, perturbations, oracle, objectives, policies, runner (brute force), storage, metrics, analysis, monitoring |
| Yes (`-m solver`) | milp, plus a few runner smoke tests on generated instances                                                                            |

- ★ **`milp.solve_stage` = `bruteforce.solve_stage`** on tiny instances, for every spec, with and without a cap and fixed stages, including unavailable reference cells.
- ★ **Oracle agreement:** every solver schedule satisfies `oracle.violations() == []`, and proven stage values equal `objectives.evaluate`.
- ★ **Stage fixing is exact:** a later stage never worsens an earlier stage's value.
- ★ **Replay:** all arms receive the same step instances; the digest depends only on the instance, S₀ and the seed.
- ★ **Policy identities:** HWF(α=0) ≡ SF; WQS with λ above the U range ≡ SF; QO = max U; SF anchored = chained at t = 1.
- ★ **Coverage cap:** every arm attains Λ = Λ\*\_t at every step.
- ★ **Matrix:** `jobs()` has exactly the cells, counts and arms of section 7, including nine arms in the main n = 40 balanced jobs and five everywhere else in `main`.
- ★ **Certified prefix:** cut at the first unproven stage in any arm or in coverage.
- ★ **Acceptance:** under-supplied candidates are never rejected for coverage; main classes always have Λ\*₀ = 0.
- **Generator distributions** (statistical, `slow`): category shares, availability counts per commitment class, shift density ≈ 0.6.
- **Metric edge cases:** Gini at zero burden = 0; max share undefined (`None`); persistence when nothing changed (`None`).

---

## 10. Risks

| Risk                                      | Early signal                        | Mitigation                                                             |
| ----------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------- |
| n = 80 too slow with 14 solves per step   | Week 0 spike, week 5 pilot          | Adequacy rule; change τ only through the protocol, before the main run |
| Build + presolve overhead of fresh models | Week 0 spike                        | Model-reusing variant (13.11), tested against the fresh-model path     |
| `addSol` warm start rejected              | Week 0 spike                        | Run without warm start (slower, still correct)                         |
| Non-determinism across runs               | Week 0, repeated toy solves         | One thread, fixed `randomization/*` seeds, pinned versions             |
| Cancellation binaries weaken the LP       | Pilot stage times                   | Report it; minimum crew is fixed by design, never tuned for speed      |
| Silent lexicographic error                | Oracle check, brute-force tests     | Integer objectives; recomputation after every step                     |
| Server dies or hangs during the main run  | Health-check alert; stall mail      | Restartable jobs and hourly result backups; restart the same command   |
| Timing differs between pilot and main run | Manifest CPU model and worker count | Same server type and `--workers` for both (13.13)                      |

---

## 11. Alternatives considered

| Alternative                                                                            | Decision and reason                                                                                                                                                                                                                                                                                                            |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| One persistent SCIP model per arm, modified after each event, or SCIP's reoptimization | **Rejected.** Fewer builds in principle, but solver state carried between solves is exactly what failed in the earlier warm-start work (basis injection, non-binding perturbations silently keeping solutions). A fresh model makes every solve a function of its inputs: comparable with brute force, reproducible, parallel. |
| One model per arm-step with `freeTransform` between stages (revision 1)                | **Replaced** by `solve_stage`. It couples stages through solver state and can only be tested with SCIP. It remains the fallback optimisation behind the same interface (13.11).                                                                                                                                                |
| One model per step shared by all arms, swapping objectives                             | **Rejected.** Fixing constraints would have to be deleted between arms, and one bookkeeping error leaks one arm's constraints into another's solve.                                                                                                                                                                            |
| CP-SAT or another solver                                                               | **Not proposed.** Table A3 commits to MILP (SCIP), and the exposé already weighs CP-SAT there. A switch would be a protocol change to agree with Dr. Lu; keeping one solver module leaves it possible.                                                                                                                         |
| Policy classes or strategy objects                                                     | **Rejected.** A policy is a list of stage specs, and a function that returns the list is enough.                                                                                                                                                                                                                               |
| Experiment matrix in the TOML (revision 1)                                             | **Moved to code** (`protocol.jobs`). It needed an arm parser, cell references and their tests. The matrix is fixed by the exposé; the TOML only holds the numbers the pilot sets.                                                                                                                                              |
| Classification at run time (revision 1)                                                | **Moved to `metrics`.** It is a function of stored data (rule 8), and the runner gets smaller.                                                                                                                                                                                                                                 |
| `feasibility.py` + `quality.py` (revision 1)                                           | **Merged into `oracle.py`**, so one file carries the independence rule.                                                                                                                                                                                                                                                        |
| SQLite or one results file                                                             | **Rejected.** Parallel workers would write concurrently; per-sequence files make restarts trivial.                                                                                                                                                                                                                             |
| Workflow engine (Snakemake, Make)                                                      | **Rejected.** Five scripts in a line don't need one.                                                                                                                                                                                                                                                                           |

### Changes in revision 3 (22 Oct), following the exposé as sent

| Exposé change                                                                                                                                   | Effect on the code                                                                                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| §4.5: the grid values run as extra arms on the main S₀-targeted n = 40 balanced sequences                                                       | The `sensitivity` cell is removed. Main n = 40 balanced jobs carry nine arms; `sensitivity_instances` is an optional, reportable descope. Open choice 13.2 is closed.         |
| §4.6: the burden sensitivity check repeats the Gini comparison with changed-cell counts (already in the 17 Oct exposé, missing from revision 2) | `analysis.burden_sensitivity`, plus per-volunteer burden vectors in `metrics.terminal_rows`; open choice 13.12                                                                |
| A2: lock-in grounded in Sydow et al.'s identification criteria, self-reinforcement not tested                                                   | Definition unchanged. Docstrings fix the naming: "lock-in indicator", never "path dependence".                                                                                |
| Citations (Mavrotas, Van den Bergh, Arthur and Sydow pages), search protocol, the reason for k = n/20, wording                                  | None                                                                                                                                                                          |
| Not from the exposé: the decision to run on a rented server                                                                                     | `monitoring.py`, `scripts/monitor.py`, failure files and `ABORT` in `storage` / `run_experiment.py`, CPU model and worker count in the manifest, SERVER.md, open choice 13.13 |

### Changes vs. the 6 Oct skeleton

| 6 Oct                                    | Now                                                                  | Why                                                                     |
| ---------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Hard coverage, truncate on infeasibility | Soft coverage + cancellation, shared coverage stage, `shortfall_cap` | WP1 decisions                                                           |
| Policies built inside `milp`             | `objectives` specs + `policies`; `milp` solves one spec              | Policy logic testable without SCIP                                      |
| `perturbations.sample_sequence`          | Plus RQ2 pattern, independent targeting, P1 fallback, restoration    | WP1 / D3 decisions                                                      |
| `feasibility.py`, `quality.py`           | `oracle.py` (adds `shortfall`)                                       | One independent oracle                                                  |
| —                                        | `protocol.jobs`, `analysis.py`, `tests/bruteforce.py`                | Matrix in code; hypotheses and precision plan; solver-free runner tests |

---

## 12. Interface summary

| Module          | Main functions                                                                                                                                                                                                                                                                |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `protocol`      | `load_protocol(path, require_frozen)`, `derive_seed(root, *keys)`, `main_arms`, `rq2_arms`, `sensitivity_arms`, `instance_jobs(p)`, `jobs(p)`, `shard(jobs, i, n)`                                                                                                            |
| `generator`     | `generate_instance(cfg, size_class, seed)`, `n_volunteers_for`, `scale_for`, `capacity_ratio`                                                                                                                                                                                 |
| `perturbations` | `apply_event`, `instances_along`, `sample_p1/p2/p3`, `rq1_sequence(instance, s0, profile, seed, horizon, targeting, scale)`, `rq2_sequence`                                                                                                                                   |
| `oracle`        | `min_crew`, `violations`, `is_feasible`, `cancelled_shifts`, `assignment_utility`, `staffing`, `surplus`, `mentored`, `quality`, `shortfall`                                                                                                                                  |
| `objectives`    | spec dataclasses, `name`, `sense`, `evaluate`, `at_least_as_good`, `lex_compare`                                                                                                                                                                                              |
| `policies`      | `reference_for`, `history_weights`, `policy_stages`, `tiebreak_ranks`, `validate_arm`                                                                                                                                                                                         |
| `milp`          | `solve_stage(instance, objective, *, fixed, shortfall_cap, start, time_limit, settings)`, `solver_versions`                                                                                                                                                                   |
| `runner`        | `solve_lexicographic`, `check_oracles`, `accept_instance`, `initial_schedule`, `shared_coverage`, `run_arm`, `run_sequence`, `non_inert_rate` (all take `solve`)                                                                                                              |
| `storage`       | `instance_path`, `write_instance`, `read_instance`, `result_path`, `to_dict`, `from_dict`, `write_result`, `exists_complete`, `read_results`, `build_manifest(protocol_path, workers)`, `write_failure`, `read_failures`, `request_abort`, `abort_requested`, `count_results` |
| `monitoring`    | `Snapshot`, `MonitorState`, `Alert`, `initial_state`, `deciles_reached`, `eta_seconds`, `is_stalled`, `decide`                                                                                                                                                                |
| `metrics`       | `distance`, `affected`, `changed_cells_by_volunteer`, `burden`, `gini`, `max_share`, `is_binding`, `classify`, `classifications`, `step_rows`, `terminal_rows`, `persistence`, `rq2_row`                                                                                      |
| `analysis`      | `certified_prefix`, `sequence_table`, `instance_table`, `paired_differences`, `bootstrap_ci`, `verdict`, `evaluate_h1/h2/h3`, `burden_sensitivity`, `precision_plan`                                                                                                          |

---

## 13. Open choices

Items 1, 3, 4, 10 and 11 are implementation details. Items 5–9 and 12 are places where the exposé leaves an operational definition open, and 13 is a pilot decision about the hardware. Fix each of those in the pilot report before the main run, and state it in the methodology chapter. Item 2 is settled; its number is kept because the code refers to these numbers.

1. **`pref_time` per period (28 values) or per period type (afternoon / evening)?** The exposé writes $p^{time}_{w,p(s)}$, i.e. per period. The stubs follow that.
2. **Settled (exposé of 22 Oct, §4.5):** the sensitivity arms run as additional arms on the main S₀-targeted n = 40 balanced sequences. See section 7.
3. **CI method:** percentile bootstrap over base instances (10,000 resamples, seeded) for all contrasts, including H1's difference of differences.
4. **Time limit for S₀:** the same τ as repairs, or a larger one. S₀ must be optimal, so a larger limit for S₀ alone loses fewer instances.
5. **"Non-inert events against S₀"** (robustness-intensity calibration, §4.4). Suggested definition: S₀ is binding for the step instance, or U(S₀) < U\*\_t. The calibrated quantity is the scale k of the independent-targeting cell. Since k is an integer, take the k whose rate is closest and report the remaining difference.
6. **Target of the RQ2 background-intensity calibration.** §4.4 says it is "calibrated once, before the main run, for both arms alike", but not to what. Decide before the pilot. Without a stated target, keep k = n/20 and say so.
7. **Persistence $P_{1,6}$:** the reading in `metrics.persistence` is mine. Check it against your Table A2 wording.
8. **"Pooled SD of its outcome" in the precision plan:**
   - For SF vs. QL and HWF vs. SF, use √((sd_a² + sd_b²)/2) of the per-instance outcome.
   - For D*i, which is already a difference, use the pooled SD of the two arms' Σ*{t=6..15} U_t.
   - Also decide whether H1's two outcomes and H2's three profiles count as separate contrasts for "N_I set by the most demanding contrast". The stubs take the maximum over all of them.
9. **P2/P3 with fewer than k eligible targets** (possible late in a sequence at n = 20): either sample all that remain and flag the event, or reuse the P1 fallback logic. Pick one, test it, and report the frequency.
10. **Tie-break weights:** maximising Σ rank_j·x_j with ranks 1..J favours larger schedules among exact ties. Centred ranks (2·rank − J − 1) remove that bias. Both are "a fixed pseudorandom ordering"; pick one and keep it.
11. **Model reuse (only if week 0 shows a large build overhead):**
    - Add `milp.solve_lexicographic`. It keeps one model across the stages of a single call, using `freeTransform` between stages.
    - It has the same inputs and outputs as `runner.solve_lexicographic`. Test it against `runner.solve_lexicographic(solve=milp.solve_stage)`; the two must give identical value tuples.
    - The runner then takes the lexicographic function as its argument instead of the stage function.
    - Rule 2 still holds across calls: no state survives between arms or steps.
12. **Statistic for "compares per-volunteer affected-repair and changed-cell counts" (§4.6).** The exposé doesn't name one. The stub uses the mean Spearman rank correlation per arm, which asks whether both counts rank volunteers the same way. The deciding part of the check is the repeated Gini comparison, which `burden_sensitivity` reports next to H2.
13. **Workers per server and hyperthreads.** A cloud "vCPU" is a hardware thread, so 48 vCPUs are 24 physical cores. Running 48 single-thread solves makes each slower than 24 would, which can change which stages certify within 600 s. Either choice is valid if the pilot and the main run use the same one. Measure in the pilot (one short benchmark at `--workers` = vCPUs and = vCPUs/2: time per solve and certification rate), choose, and record it in the pilot report. The manifest stores CPU model and worker count to prove it.
