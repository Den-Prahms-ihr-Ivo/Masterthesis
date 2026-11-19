# Notes / decisions

## 2026-11-01

Thesis started. Repository created.

## 2026-11-13

**P2/P3 fallback (13.9)**

Take all eligible targets, then fill up to `k` from the independent set. P2: currently available pairs. P3: any `(w, r)` pair. Set `fallback=True`; if there are still too few, raise `ValueError`.

Same event size as planned, and the pilot can count the fallbacks. Updated the digest in `test_model.py` for the new field. No saved results yet.

**Gate 2.5 passed**

`acceptance_check.py`, `pilot.toml`, root seed 20261109. Twenty base seeds per size, coverage and S0 limits 60 s, `max_draws=100`.

SCIP 10.0.2 / PySCIPOpt 6.3.0, Apple M1, one thread. Total: 57 s.

| n   | Accepted | Draws | Rejected | Reserve used | S0 certified | Acceptance (s) | S0 (s) |
| --- | -------- | ----- | -------- | ------------ | ------------ | -------------- | ------ |
| 20  | 20       | 21    | 1 (4.8%) | 0            | 20/20        | 2.0            | 7.7    |
| 40  | 20       | 20    | 0        | 0            | 20/20        | 3.4            | 13.2   |
| 80  | 20       | 20    | 0        | 0            | 20/20        | 6.3            | 23.7   |

Only one rejection, from positive minimum shortfall. Main instances mostly pass directly. Under-supplied still needs phi.

**Gate M2 passed**

`m2_gate.py`, `pilot.toml`, n=20, instance 0. First draw accepted, S0 certified. Same setup as above, 60 s per stage, 37 s total.

- Balanced RQ1: QO, SF, QL, WQS(2), HWF(1), chained where applicable.
- RQ2: SF-anchored and SF-chained.
- All 15 steps certified, no retries, every repair reached `Lambda*_t`.
- RQ1 shortfall always 0. RQ2 shortfall 1 at steps 4–5.
- Both saved files loaded correctly. Passt erstmal.

Test: `test_m2_gate_n20_sequences_with_scip` (`slow`, `solver`).

**Persistence (13.7)**

Table A2 agrees with `metrics.persistence`: cells changed by chained at step 1 must keep that value through step 6 and differ from anchored at step 6.

Also: `A_t = sum_w c_{w,t}` counts affected volunteers. Not a share. Corrected the stub.

**Precision plan (13.8)**

Target half-width: `0.1 * pooled SD`.

- H1: pool the per-instance arm differences from preference and availability. Raw SF/QL gaps would include QL's constant zero gap and shrink the SD by about `sqrt(2)`.
- H2: pool `gini(HWF)` and `gini(SF)`.
- H3: pool the arms' quality sums over steps 6–15. These are not fixed at zero. Explain why this differs from H1.

For H1: `N_I = ceil((1.96 * sigma_DiD / half_width)^2)`. Use the SD of the actual difference of differences. Same instances in both profiles means covariance matters; `sqrt(2) * profile SD` is not enough.

Test: `test_precision_plan_uses_difference_of_differences_sd`.

H1 needs both components supported; one contradicted component contradicts H1. No multiplicity correction needed for support under this intersection-union rule.

H1 signs and materiality threshold still needed checking. Done on 19 November, below.

## 2026-11-14

**Run control**

- `SystematicError` (`OracleMismatch`, `ManifestMismatch`) and `AssertionError` set ABORT.
- Same error type + message in five failure files also sets ABORT (`BREAKER_THRESHOLD=5`). No point running the same bug for days.
- ABORT stops new jobs. Running jobs finish.
- Ctrl-C and `SystemExit` are not caught. Atomic writes protect finished files; unfinished jobs run again after restart.
- Resume checks `<results>/manifest.json`: protocol hash, commit, clean flag, solver/Python versions, CPU and workers must match. If I move the manifest away deliberately, write down why. Sicher ist sicher.

**Seeds**

Pinned `derive_seed` with fixed test values. M2 now uses the normal seed helpers under `gate_root(protocol)`. Tests check separation from the run seeds.

M2 therefore draws a different instance now. Keep the earlier result for seed 3865029198492689663 as the historical record. Acceptance-check seeds stay the same.

**Monitor**

One combined failure mail per poll, with job keys and error types. Avoid hitting the mail limit before the ABORT mail gets through.

Every start sends a summary. Restart also resets the stall timer, giving another full grace period. Documented in SERVER.md §8.

**M3 preparation**

Found two bugs:

- SCIP returns `userinterrupt` for Ctrl-C. `_status` raised `ValueError`, producing a failure file per worker. Changed to `KeyboardInterrupt`. Six interrupt attempts: no failures or ABORT.
- Monitor flags stayed set after recovery. `decide` now follows the files on disk. Removing a failure/ABORT file allows a later occurrence to trigger mail again.

Added planted failures: `REPAIR_PLANT_FAILURE="<job key>:error"` or `:assert`, comma-separated for several jobs. Logged in the manifest. Refused with `--main`, typos or unselected keys.

`rerun_check.py` compares inputs, coverage, schedules and stage names/statuses/values. Ignore timings, bounds, retry flags and manifests. Time-limited steps and affected later steps are marked “not compared”.

**Dry-run setup**

`dry-run.toml`: n=20, three instances, 60 s per stage, main + RQ2, 183 jobs.

M1, two workers: about 24 s/main job and 8 s/RQ2 job. CCX23 estimate: 18 minutes, not measured yet. Local shard test with error, resume and two identical reruns passed (SERVER.md §9).

Pilot blocked for now: robustness and under-supplied need `robustness_scale` and `workforce_factor`. Decision below on 18 November.

## 2026-11-18

**H3 precision implemented (13.8)**

`rq2_row` and `rq2_table` now include `U_anchored` and `U_chained`, summed over steps 6–15. `D = U_anchored - U_chained`.

`precision_plan(..., rq2=)` uses SD(D) and the pooled SD of the arm sums. Same sample as `evaluate_h3`: eligible primary rows, averaged per instance. H3 now counts towards maximum `N_I`. No eligible RQ2 data for a size means undefined `N_I`, so planning stops. Previously only H1/H2 counted.

**Script fixes**

Fixed against the new tests: `CoverageStep.record`, DataFrame rows, zero-count stages and rejecting phi=1. Tiny SCIP run matched DESIGN.md §7: 210 solves/main sequence, 105/RQ2 sequence.

**Pilot: main + RQ2 only**

Other cells need calibration first. `calibrate.py` can run before or alongside the pilot. Runtime planning uses main jobs with matching size and arm count to estimate the other cells. Including them directly would need provisional protocol values.

Noch offen: time limit (60 vs. 600 s) and counts (currently 2/2). MAIN_RUN_CHECKLIST.md D.

Copied `DESIGN.md` and `ROADMAP.md` from the old project. Updated DESIGN.md §7 running/monitoring notes.

## 2026-11-19

**H1 checked against the sent exposé (§3)**

Preference minus availability should be positive for both SF's excess quality gap and QL's excess churn. So `H1_EXPECTED = {"gap": +1, "churn": +1}` was correct. Exact wording now in `analysis.py`; test: `test_predicted_signs_are_the_exposes`.

95% interval fully in the predicted direction: supported. Fully opposite: contradicted. Otherwise inconclusive. H1 needs both components, using the difference between profiles' mean paired differences. H2 is checked per profile. Code already matched this.

**No materiality threshold**

`theta_m` and its table were only in old drafts. Verdicts use zero. Removed the TODO; covered by `test_verdicts_test_against_zero_only`.

The `0.1 * pooled SD` in §4.7 is the interval half-width target:

`N_I = ceil((1.96 * sigma_d / (0.1 * s_pool))^2)`

Nicht verwechseln: this sets sample size, not which effects are meaningful.

**Pilot limit changed to 600 s**

Matches §4.7 and the main run. At 60 s, pilot timings and certification would describe a different setup. Pilot/main solver settings now match, checked by a test.

Dry run stays at 60 s. Old Gate 2.5 results still describe their original setup.

Only pilot counts remain open: `n_instances`, `n_sequences`.

**Laptop dry run passed**

M1, two workers, tag `dry_run_1`, clean working tree. `dry-run.toml`: 183 jobs, three n=20 instances, 60 s/stage. Planted error at job 10, assertion at job 120, Ctrl-C at roughly 40%, then recovery.

- 15:45–16:36, 51 minutes including stops and both restarts. Not compute time; estimate was 35 minutes of compute with two workers.
- Progress/failure/health-check mails worked (R3, R6).
- Ctrl-C/resume: no new failures (R4). ABORT stopped new jobs (R5).
- Ten analysis tables (R1), identical reruns (R7), planted-effect tests passed (R8).

Laptop done. Still test on the server: Linux worker startup (`fork`/`spawn`) and Ctrl-C inside SCIP (R4), `uv sync --locked`, Gmail port 587, health-check pings and CPU timings.
