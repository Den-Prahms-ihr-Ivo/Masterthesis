# Decision log

## 2026-11-01

- Official thesis start. Repository created.

## 2026-11-13

- **Open choice 13.9 (P2/P3 with fewer than k eligible targets): top up.** Take every
  eligible target, then fill up to k from the independent-targeting set (P2: any currently
  available pair; P3: any (w, r) pair), and flag the event (`fallback=True` on
  `AvailabilityRemoval` / `PreferenceChange`). If even that set is too small: `ValueError`.
  Reason: every event keeps intensity k, consistent with the P1 fallback; the flag lets the
  pilot report how often S0-targeting had to be relaxed. The pinned digest in
  `test_model.py` changed because of the new field (no stored results existed yet).
- **Gate 2.5 (acceptance with SCIP): passed.** `scripts/acceptance_check.py`, pilot.toml
  (root_seed 20261109), 20 base seeds per size class, tau = S0 limit = 60 s,
  max_draws 100. SCIP 10.0.2 / PySCIPOpt 6.3.0, Apple M1, 1 thread; 57 s in total.

  | n  | accepted | draws | rejected (Λ\*₀ > 0) | reserve used | S0 certified | accept s | S0 s |
  | -- | -------- | ----- | ------------------- | ------------ | ------------ | -------- | ---- |
  | 20 | 20       | 21    | 1 (4.8 %)           | 0            | 20/20        | 2.0      | 7.7  |
  | 40 | 20       | 20    | 0                   | 0            | 20/20        | 3.4      | 13.2 |
  | 80 | 20       | 20    | 0                   | 0            | 20/20        | 6.3      | 23.7 |

  Rejections are rare, so the coverage acceptance barely filters the main classes. Not
  yet measured: the under-supplied class (needs a calibrated phi).
- **Gate M2: passed.** `scripts/m2_gate.py`, pilot.toml, n = 20, instance index 0
  (accepted at the first draw, S0 certified). SCIP 10.0.2 / PySCIPOpt 6.3.0, Apple M1,
  tau = 60 s; 37 s in total. One balanced RQ1 sequence through QO, SF, QL, WQS(2),
  HWF(1) (all chained) and one RQ2 sequence through SF-anchored and SF-chained: every
  arm ran all 15 steps, every step certified, no retries, Lambda = Lambda*_t throughout
  (RQ1: Lambda* = 0 at every step; RQ2: Lambda* = 1 at steps 4-5), both result files
  round-trip. Also covered by `test_m2_gate_n20_sequences_with_scip` (slow, solver).
- **Open choice 13.7 (persistence P_{1,6}): closed by Table A2.** A2 defines it as "the
  share of cells changed by the chained arm at step 1 that stay unchanged through step 6
  while the anchored arm differs at step 6", which is the reading `metrics.persistence`
  implements. Also from A2: A_t = sum_w c_{w,t} is a count of affected volunteers, not a
  share (the metrics stub said share; corrected before implementation).
- **Open choice 13.8 (pooled SD in the precision plan): decided for H1 and H2.**
  - The target half-width is 0.1 x pooled SD. For **H1** the pooled SD is taken over the
    per-instance ARM DIFFERENCE at "preference" and at "availability", not over the raw
    SF and QL levels. Reason: QL's first stage is MaxQuality under the same Lambda cap,
    so QL's gap is 0 on certified steps; pooling raw SF and QL gaps would mix in a
    zero-variance arm and shrink the SD by about sqrt(2). For **H2** it is pooled over
    gini(HWF) and gini(SF).
  - Inconsistency to note in the thesis: the D_i rule of 13.8 pools the two arms' raw
    levels (sum_{t=6..15} U_t). That is fine there (neither arm has zero variance), but
    a reader will see the two rules differ; state both reasons.
  - The sample size uses a different SD: N_I = ceil((1.96 * sigma_DiD / half-width)^2),
    with sigma_DiD the SD of the per-instance difference of differences. The same
    instances run both profiles, so sigma_DiD includes their covariance and is not
    sqrt(2) times a per-profile SD (`test_precision_plan_uses_difference_of_differences_sd`).
  - H1 joint verdict: intersection-union rule (support needs both components,
    contradiction needs one), so "supported" needs no multiplicity correction.
  - Still open: the exact H1 direction (`analysis.H1_EXPECTED`, expected +1 for both
    components) and whether the falsification rule has a materiality threshold
    instead of 0; both must be checked against the frozen exposé text.

## 2026-11-14

- **Run control (step 4.4), decided:**
  - **Abort-worthy errors share a base class**, `model.SystematicError`
    (`runner.OracleMismatch`, `storage.ManifestMismatch`). The worker aborts on
    `SystematicError` or `AssertionError`; a new check only needs to subclass it.
  - **Breaker for unclassified bugs:** the same error type and message in
    `BREAKER_THRESHOLD` = 5 failure files triggers the abort as well (e.g. a `KeyError` in
    every job), instead of logging it for days.
  - **What an abort does to running jobs:** they FINISH (no new job starts). That wastes
    at most one job per worker and keeps every file complete. `KeyboardInterrupt` and
    `SystemExit` are never caught; atomic writes leave at most a temp file, and a restart
    redoes the job.
  - **Resuming:** the first start writes `<results>/manifest.json`; a later start with a
    different protocol hash, commit, clean flag, solver or Python version, CPU model or
    worker count is refused (`ManifestMismatch`). Moving the file away is a deliberate,
    logged decision.
- **Seeds:** `derive_seed` is pinned by a golden test. `m2_gate.py` now uses the run's
  scheme (`instance_seeds`, `sequence_seed`) under its own root `gate_root(protocol)`, so
  there is one seed scheme and the gate's instances are disjoint from the run's (tested).
  Consequence: rerunning `m2_gate.py` now draws a different instance than the run logged
  for Gate M2 above (seed 3865029198492689663). That entry stays valid as the record of the
  run it describes. `acceptance_check.py` uses `instance_seeds` and gives the same seeds as
  before.
- **Monitor:** failures are mailed in one batch per poll (keys and error types; each key
  once), because one mail per job could exceed Gmail's ~500 mails/day and delay the abort
  mail. Every monitor start sends one summary mail (results, failures by type, ABORT), so
  nothing that happened while it was down goes unreported. **For SERVER.md:** the stall
  timer starts at the monitor's (re)start, so a run that was already stalled gets a full
  grace period again. (Now in SERVER.md §8.)
- **Gate M3 preparation (rehearsal tooling and two fixes found with it):**
  - **Ctrl-C inside SCIP is an interrupt, not a failure.** SCIP catches SIGINT itself and
    returns status "userinterrupt"; `milp._status` raised `ValueError` for it, so a Ctrl-C
    wrote one identical failure file per interrupted worker (seen locally: 2 files from
    one Ctrl-C with 2 workers). With 48 workers that trips the breaker and writes ABORT.
    Now re-raised as `KeyboardInterrupt`; six Ctrl-Cs at different moments left no
    failure file and no ABORT.
  - **Monitor flags re-arm.** `abort_sent` stayed True for good, so after the abort
    procedure (fix, `rm ABORT`, resume) a second abort would have been silent; the same for
    a job that fails again after a successful rerun. `decide` now sets `abort_sent` and
    `failures_sent` from what is on disk: present = mailed, gone = re-armed. This amends the
    "each key once" rule above: once per failure file.
  - **Planted failures** (`REPAIR_PLANT_FAILURE="<job key>:error|assert,..."`) for the
    rehearsal's failure and abort mails. `run_experiment.py` refuses them with `--main`,
    on a typo, or for a job key that is not selected, and records them in the manifest.
  - **`scripts/rerun_check.py`** for the determinism criterion. It compares exactly the
    inputs, Lambda*_t and every arm's schedules and stage (name, status, value); never
    seconds, bounds, `retried` or the manifest. A coverage or arm stage that hit its time
    limit makes the outcome timing-dependent, so that step and the arm's later steps are
    reported as "not compared", not as a difference.
  - **`protocol/dry-run.toml`**: n = 20, 3 instances, tau = 60 s, main + rq2, 183 jobs.
    Locally (Apple M1, 2 workers) about 24 s per main job and 8 s per rq2 job, so roughly
    18 min on a CCX23. A local shard run with 2 workers, a planted error, a resume and
    `rerun_check` (2 jobs identical) all behaved as SERVER.md §9 expects.
  - **Not runnable yet: `pilot.toml`.** It lists robustness and undersupplied without
    `robustness_scale` / `workforce_factor`, so `jobs()` refuses. To decide before the pilot
    (MAIN_RUN_CHECKLIST.md, D).
