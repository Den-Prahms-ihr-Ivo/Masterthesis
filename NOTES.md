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

Noch offen: time limit (60 vs. 600 s) and counts (currently 2/2).

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

- **2026-11-20 Server rehearsal: Gate M3 passed.**
  Hetzner CCX23 (4 vCPUs) in Helsinki, Ubuntu 24.04, CPU "AMD EPYC-Milan Processor",
  4 workers, tag `run-rehearsal-1`, protocol/dry-run.toml (183 jobs, n = 20, 60 s per stage).
  - It took about 30 minutes from the first "monitor started" mail to "finished",
    including the deliberate Ctrl-C, the stop after the planted abort and the restarts.
    This is not a speed measurement.
  - Everything worked as it should:
    - the packages installed on Linux with `uv sync --locked`, and all tests and the
      smoke test passed;
    - the mails reached me from the server (test mail, start, every 10 %, failure,
      abort, finished), and the health-check alert came after the monitor stopped;
    - the planted error in job 10 sent a failure mail, and the run kept going;
    - Ctrl-C at about 40 % and a restart continued where it stopped, without new
      failure files. So stopping inside SCIP also works on Linux, where the workers
      are started differently than on the Mac;
    - the planted assertion in job 120 sent a failure mail and an abort mail, and
      no new jobs started;
    - after removing the plants, jobs 10 and 120 ran again and the run finished
      with no failures;
    - the analysis wrote all 10 tables, a rerun of two jobs gave identical results,
      and the analysis found the planted effects and said "inconclusive" on noise.
  - Lesson for the pilot server: the GitHub key must be loaded into the laptop's
    SSH agent (`ssh-add`) before `ssh -A`, or `git clone` on the server fails with
    "Permission denied (publickey)".
  - Results copied to the laptop; server deleted; health check back to 10 / 30 min.
  - Intensity selection rule changed before the pilot (2026-11-20): compare non-inert rates over
    judged events, not counts. Reason: skipped events grow with k (target 4; k=3: 33; k=4: 40),
    so counts favoured larger k (counts chose 4, judged rates choose 3). Target 285/296 = 96.3 %,
    k=3 255/267 = 95.5 %. robustness_scale = 3.

## 2026-11-20 Calibration and pilot settings

**Workforce factor (phi) for the under-supplied cell: 0.55.**
The rule is: take the largest tested phi at which at least 80 % of 20 raw
candidates cannot be fully staffed (proven positive shortfall). None of the
default values 0.95 to 0.60 met it; the highest was 65 % at 0.60 (24 volunteers).
I extended the grid downward. 0.55 (22 volunteers) reached 95 %, and 0.50 to 0.35
also passed. I also tested 0.58 (23 volunteers), the only volunteer count
in between: 75 %, so it fails. So phi = 0.55. This makes the cell strongly
under-supplied (22 volunteers for the demand of size class 40), which the thesis
has to state.

**What counts as a "non-inert" event (open choice 13.5): decided.**
An event is non-inert if, with only that event applied to the base instance,
S0 breaks a hard rule or is no longer the best schedule (lowest shortfall,
then highest quality). The first version applied all events up to step t and
judged S0 on that. It saturated: almost every step counted for every scale
(targeted 299 of 300, independent 283 to 300 of 300), because after a few
events S0 is simply out of date, whatever the events were. So it could not
tell the scales apart. Some events only make sense after an earlier event
(for example a preference changed back to its base value). They are invalid
on the base instance and are counted as "skipped", never as inert or non-inert.

**Robustness scale (k) for independent targeting: 3.**
The targeted design (k = 2) has 285 non-inert events out of 296 judged (96.3 %).
Independent targeting: k = 1 57.0 %, k = 2 80.1 %, k = 3 95.5 %, k = 4 99.2 %.
k = 3 is closest (0.8 percentage points below the target).

The selection rule was changed before the pilot, after seeing these results:
the scale is now chosen by the non-inert rate over judged events, not by the
count. Reason: skipped events grow with k (targeted 4; k = 3: 33; k = 4: 40). A
skipped event can never count, so raw counts are pushed down at large k and
favour a larger k. On counts the rule picked 4, on judged rates 3. The change
is mechanical, made before any pilot result existed, and is in calibrate.py
with a test.

**RQ2 background: default k = 2, not calibrated (open choice 13.6).**
There is no stated target to calibrate to. Diagnostic only: 200 of 259 judged
background steps were non-inert, 1 skipped.

**Pilot settings (protocol/pilot.toml).**

- Time limit 600 s per stage and for S0, as the exposé (§4.7) fixes, so the
  pilot's timings and certification rates carry over to the main run.
- 10 instances per size class, 1 sequence per instance and profile,
  5 instances for each secondary cell; all four cells; 130 jobs.
- Robustness and under-supplied cells are back in the pilot now that their
  values exist (they were left out on 2026-11-18 only for that reason). The
  pilot checks whether under-supplied instances at phi = 0.55 still get a
  proven-optimal S0 within 600 s.
- Solver seed and threads: the runner does not pass them on, but
  milp.solve_stage's defaults are the same (seed 0, one thread), so the runs
  are seeded and single-threaded as intended. The TOML values stay as they are.
  **Robustness scale (k): 3.** The exposé asks that the expected number of non-inert
  events against S0 match the main design. Some events cannot be judged alone on the
  base instance and are skipped, more of them at larger k (targeted 4; k = 3: 33; k = 4: 40),
  so raw counts underestimate that number at large k. The expected number per
  sequence is therefore estimated as 15 × the non-inert rate over judged events:
  targeted 14.4, k = 1 8.6, k = 2 12.0, k = 3 14.3, k = 4 14.9. Closest: k = 3.
  (Raw counts would have chosen 4; changed before the pilot, in calibrate.py with a test.)

  **Adequacy rule (form fixed 2026-11-20, numbers from the pilot).**
  A size class enters the primary analysis if the main run can be expected to
  deliver the required number of certified base instances for every primary
  outcome: N_I(required) / s ≤ N_I(affordable), where s is the pilot's share of
  sequences certified through each outcome's endpoint (T = 15 for the RQ1
  outcomes, through step 15 for D_i), and N_I(affordable) is what the 1,000
  core-hour budget allows for that size class after the other classes are
  planned. Classes are considered in the order 20, 40, 80. A class that fails
  is reported in the secondary analysis only.

  **RQ2 background scale: fixed at k = 2 (= n/20) before the main run.** The exposé asks
  for it to be calibrated and recorded but names no target, so it is fixed at the
  default; diagnostic: 200 of 259 judged background steps non-inert, 1 skipped.
  (Not to be confused with the robustness scale k = 3 above.)
