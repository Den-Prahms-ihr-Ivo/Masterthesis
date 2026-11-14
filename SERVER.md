# Server Runbook

How to rent the machine, set it up, run the rehearsal, the pilot or the main experiment, and react to alert mails. Commands assume Ubuntu 24.04 on the server and a macOS or Linux terminal on your laptop. Replace `<IP>` throughout.

Before the main run, also work through [MAIN_RUN_CHECKLIST.md](MAIN_RUN_CHECKLIST.md).

---

## 0. Which server

| | Choice | Why |
|---|---|---|
| **Main** | **Hetzner Cloud CCX63**: 48 dedicated vCPUs (AMD EPYC), 188 GB RAM, Falkenstein or Nuremberg | Dedicated vCPUs keep solve times stable, and certification within 600 s depends on that. Hourly billing, many cores at once. |
| Fallback | CCX53: 32 dedicated vCPUs | Use it if your account limit isn't raised in time. Same rule: the pilot and the main run use the same type. |
| Rehearsal | CCX23: 4 dedicated vCPUs | For Gate M3 (§9). Costs cents. |
| Avoid | CPX / CX (shared vCPUs) | Other customers on the same host change your solve times. |

**Cost:**
- CCX23 was listed at about **$0.155/h** on 21 Oct 2026 ([sparecores](https://sparecores.com/server/hcloud/ccx23)). The CCX63 monthly cap has been **€853.49** since the June 2026 price change ([wz-it](https://wz-it.com/en/blog/hetzner-price-increase-june-2026-cpx-ccx-alternatives/)).
- **The hourly rates are estimates; check them in the console before renting.** If the CCX63 hourly price is roughly 12 × CCX23 (an assumption), the 1,000 core-hour main run costs on the order of **$40–80**. You pay for the hours used, capped at the monthly price.
- **A powered-off server is still billed. Only deleting it stops the cost.**

**vCPUs are hardware threads:** 48 vCPUs = 24 physical cores. Whether to run 48 or 24 workers is decided in the pilot (open choice 13.13); keep that choice afterwards. A resumed run must use the same `--workers` value, or the resume check refuses it (§8).

---

## 1. Once: account

1. Create a Hetzner account and a project, e.g. "thesis".
2. **Request a limit increase** (Support → Limits) so you can run one CCX63. This can take days, so do it early.
3. Optional: set a spending alert in the billing settings.

## 2. Once: SSH key

```bash
ssh-keygen -t ed25519 -C "thesis-server"            # on the laptop; accept the default path
cat ~/.ssh/id_ed25519.pub                           # paste into Hetzner → Security → SSH keys
ssh-add -l                                          # your GitHub key must be listed
```

The server clones the repository through agent forwarding (`ssh -A`), so no key or token ever lives on the server. This works for a private repository.

## 3. Once: mail and dead-man's switch

1. **Gmail app password:** turn on 2-step verification, then Google Account → Security → App passwords → create one for "repair-study".
2. **healthchecks.io:** create a free check, enable e-mail alerts, and copy the ping URL. Settings: **period 10 min, grace 30 min** for the pilot and the main run; **period 2 min, grace 3 min** for the rehearsal (§9). The monitor pings once per poll.
3. **`~/.repair-study.env` on the laptop** (`chmod 600`, never commit it):
   ```bash
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=you@gmail.com
   SMTP_PASSWORD=xxxx xxxx xxxx xxxx
   MAIL_TO=you@gmail.com
   HEALTHCHECK_URL=https://hc-ping.com/<uuid>
   ```
4. Test it from the laptop:
   ```bash
   set -a; source ~/.repair-study.env; set +a
   PYTHONPATH=src .venv/bin/python scripts/monitor.py results/test --total 1 --test-mail
   ```

---

## 4. Each time: create and set up the server

**Create:** Hetzner console → Add server → location Falkenstein or Nuremberg → image Ubuntu 24.04 → type (§0) → your SSH key → Create. Note the IP.

**Set up** (about 10 minutes):
```bash
scp ~/.repair-study.env root@<IP>:~                  # from the laptop
ssh -A root@<IP>                                    # -A forwards your GitHub key

apt update && apt -y upgrade
apt -y install git tmux rsync htop curl
chmod 600 ~/.repair-study.env
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env

git clone git@github.com:Den-Prahms-ihr-Ivo/Masterthesis.git && cd Masterthesis
git checkout <ref>              # main run: protocol-frozen; pilot / rehearsal: the commit you test
uv sync --locked                # Python, numpy, pandas, PySCIPOpt exactly as in uv.lock
source .venv/bin/activate
git status --short              # must print nothing: the manifest records git_clean

pytest -m "not slow and not todo"           # includes the SCIP tests: the solver works here
pytest tests/test_smoke.py -m ""            # the whole pipeline once, about 20 s
set -a; source ~/.repair-study.env; set +a
python scripts/monitor.py results/test --total 1 --test-mail   # a mail must arrive
```

If any step fails, stop and fix it before starting a run.

---

## 5. Start a run

The same commands serve all three runs:

| Run | Protocol | Results | Extra flag | Monitor interval |
|---|---|---|---|---|
| Rehearsal | `protocol/dry-run.toml` | `results/dry-run` | | `--interval 60` |
| Pilot | `protocol/pilot.toml` | `results/pilot` | | `--interval 600` |
| Main | `protocol/main.toml` | `results/main` | `--main` | `--interval 600` |

`W` is the worker count from open choice 13.13 (rehearsal: 4). The example uses the main run.

```bash
tmux new -s run                                     # window 0: the experiment
cd ~/Masterthesis && source .venv/bin/activate && set -a && source ~/.repair-study.env && set +a
python scripts/make_instances.py protocol/main.toml --out results/main --workers W
python scripts/run_experiment.py protocol/main.toml --out results/main --main --workers W 2>&1 | tee -a results/main/run.log
```

Open a second window with `Ctrl-b c` for the monitor:
```bash
cd ~/Masterthesis && source .venv/bin/activate && set -a && source ~/.repair-study.env && set +a
python scripts/monitor.py results/main --name main \
  --total $(python scripts/run_experiment.py protocol/main.toml --out results/main --main --count) \
  --interval 600 --stall-hours 3
```

`--count` must use the same protocol, `--cell` and `--shard` flags as the run, or the percentages are wrong.

Every monitor start sends one **"monitor started"** mail: results so far, failures by error type, and whether ABORT is set. If it doesn't arrive, the mail settings are broken.

Detach with `Ctrl-b d`, then log out. **Your laptop can now sleep or shut down.** To look again: `ssh root@<IP>` and `tmux attach -t run`; switch windows with `Ctrl-b 0` / `Ctrl-b 1`.

**Stopping and resuming.** Stop the run with `Ctrl-C` in window 0. This stops every worker, writes no failure files, and running jobs are simply redone later. To resume, start the same command again: complete results are skipped. The first start writes `results/<run>/manifest.json`. Every later start compares it with the current one and refuses (`ManifestMismatch`, exit status 1, ABORT written) if any of these changed: protocol file, git commit, clean working tree, solver versions, Python version, CPU model, worker count.

## 6. Watch and back up

- **Mails** arrive at each 10 %, on failures (one mail per poll listing every new failed job), on abort, on stalls, and when finished. The health check e-mails you if the monitor goes silent.
- **The monitor exits** after the "finished" or "abort" mail, so a health-check alert follows about 40 minutes later (period + grace). That one is expected. Pause the check in healthchecks.io if it bothers you.
- **A quick look:** `htop` (are all vCPUs busy?), `tail -f results/main/run.log`, `ls results/main/failures/`.
- **Backup to the laptop** whenever you are online, e.g. hourly:
  ```bash
  rsync -az root@<IP>:~/Masterthesis/results/ ~/Documents/PROJECTS/masterthesis/Masterthesis/results/
  ```
  The server's disk keeps everything until you delete the server. The backup protects against losing the server.

## 7. Finish

1. Wait for the "finished" mail. Check that `results/main/failures/` is empty, or that every failure in it is explained in `NOTES.md`.
2. Final `rsync` to the laptop. Also archive the results: `tar czf results_main.tar.gz results/main`, and copy the archive off the server.
3. Compare `results/main/manifest.json` with the pilot's: same CPU model, worker count and versions.
4. **Delete the server** in the console. Check the next day that billing has stopped.

---

## 8. When a mail arrives

| Mail | Meaning | Do |
|---|---|---|
| **Monitor started** | The monitor (re)started; summary of the run so far | Check the failure count and the ABORT state. Nothing that happened while the monitor was down goes unreported. |
| **Progress 10 %…90 %** | All fine | Glance at the ETA. If it would overrun the budget, decide now (descope order: DESIGN.md §8). |
| **Failure** (no abort) | Some jobs crashed (memory, disk, a solver error); the run continues | Read the tracebacks in `failures/`. If it's environmental, fix it; failed jobs rerun when you restart the same command later (finished jobs are skipped, a successful rerun deletes the failure file). If it's code, see the next row. |
| **Abort** | A systematic bug, so no new jobs start. Jobs already running finish. Causes: an `AssertionError` or `OracleMismatch`, a `ManifestMismatch`, or the **breaker** (5 failure files with the same error type and message) | 1. Read `results/main/ABORT` and the failure files. Reproduce the job locally as a **regression test**. 2. Fix, test, push. 3. Decide what the fix touches: if it can change schedules, delete the affected cells' results; if it only touches metrics, analysis or scripts, keep everything. 4. On the server: `git pull`. The new commit makes the resume check refuse; if you keep results, move the old manifest aside (`mv results/main/manifest.json results/main/manifest-<date>.json`). 5. Log the fix, what was kept, and why, in `NOTES.md`. 6. `rm results/main/ABORT`, then start the run and the monitor again. |
| **Stall** | No new result for 3 h | `htop`: if the CPUs are busy, a slow instance is probably running (wait, or compare with the pilot's longest sequence time). If they're idle, the process died: restart the command, and it resumes. The stall timer starts again whenever the monitor (re)starts, so a run that was already stalled gets the full 3 h again. |
| **Health-check alert** | Monitor or server silent for longer than period + grace | After a "finished" or "abort" mail this is expected (§6). Otherwise `ssh` in. If the server was rebooted, the tmux session is gone: repeat §5, and both commands resume. If it's unreachable, check the Hetzner status page. |
| **No mails at all** | The monitor isn't running, or the mail settings broke | `tmux attach -t run`, check window 1, and send `--test-mail` again. |

**Never** change `protocol/main.toml` or check out another commit in the middle of the main run. If a code fix is needed, follow the abort row.

---

## 9. Rehearsal: Gate M3

A CCX23 for about an hour, with `protocol/dry-run.toml` (n = 20, 3 instances, τ = 60 s, 183 jobs, about 18 min on 4 workers). Each step names the gate criterion it checks.

**Before renting:** health check set to period 2 min and grace 3 min (§3); the commit to rehearse is pushed and CI is green.

1. **Set up** per §4, with `git checkout` of the rehearsal commit.
2. **Pick two jobs to fail on purpose.**
   ```bash
   python scripts/run_experiment.py protocol/dry-run.toml --out results/dry-run --list | nl | sed -n '10p;120p'
   ```
   Job 10 gets an ordinary error and job 120 an `AssertionError`:
   ```bash
   export REPAIR_PLANT_FAILURE="<key 10>:error,<key 120>:assert"
   ```
   Only `run_experiment.py` reads this variable. It refuses to start if the variable is set with `--main`, contains a typo, or names a job that doesn't exist.
3. **Start** the run (window 0) and the monitor (window 1) per §5, with `W = 4` and `--interval 60`. The monitor is started without the variable.
   → "monitor started" mail. Then the 10 % mails, about every 2 minutes. *(Gate: the 10 % mails arrive.)*
4. **Planted error:** shortly after the start, a failure mail names job 10 with `RuntimeError`, and the run continues. *(Gate: planted exception → failure mail, the run continues.)*
5. **Kill halfway:** at about 40 %, press `Ctrl-C` in window 0. Note `ls results/dry-run/main-n20/*/ | wc -l`. Start the same command again: it reports the complete results as skipped and continues. *(Gate: killed and restarted, the run resumes.)*
6. **Planted assertion:** at about 65 %, a failure mail for job 120 (`AssertionError`) and an abort mail arrive. Running jobs finish, then the run ends with exit status 1. Count the results twice, a few minutes apart: the number stays the same, and `htop` is idle. The monitor exits after the abort mail. *(Gate: failure plus abort mail, no new jobs start.)*
7. **Recover**, as after a real fix:
   ```bash
   unset REPAIR_PLANT_FAILURE
   rm results/dry-run/ABORT
   ```
   Start the run and the monitor again. The "monitor started" mail shows the two failures. The run reruns jobs 10 and 120 (their failure files disappear) and finishes. Then the "finished" mail arrives.
8. **Health check:** the monitor has exited, so within period + grace (≤ 5 min) healthchecks.io sends its alert. *(Gate: stopping the monitor → health-check e-mail within the grace period.)*
9. **Tables:**
   ```bash
   python scripts/analyse.py results/dry-run --out results/dry-run/analysis
   ls results/dry-run/analysis       # all 10 CSVs of analyse.TABLES
   ```
   *(Gate: make_instances → run_experiment → analyse produces every table.)*
10. **Same results on a rerun:**
    ```bash
    python scripts/rerun_check.py protocol/dry-run.toml results/dry-run --jobs 2
    ```
    This reruns the first and the last finished job into `results/dry-run-rerun-*`. It must print "identical" twice and exit with status 0. *(Gate: identical schedules and stage values; timings may differ.)*
11. **Analysis on planted effects:**
    ```bash
    pytest tests/test_analysis.py -m "" -k "planted or noise"
    ```
    *(Gate: the analysis recovers planted effects and says "inconclusive" on noise.)*
12. **Write down** the wall time of the run, the solver seconds per job, and the CPU model from `manifest.json`. These feed the pilot plan.
13. **Back up and delete:** `rsync` (§6), then **delete the server**. Set the health check back to period 10 min and grace 30 min.
14. **Log** "Gate M3 passed" in `NOTES.md` with those numbers. Merge to `main`, tag `m3`.

---

## Cheat sheet

```text
ssh -A root@<IP>                 connect (with GitHub key forwarding)
tmux attach -t run               back into the session
Ctrl-b d / Ctrl-b c / Ctrl-b 0|1 detach / new window / switch window
Ctrl-C (window 0)                stop the run cleanly; the same command resumes it
htop                             CPU load (q to quit)
tail -f results/main/run.log     live log
ls results/main/failures/        failures so far
cat results/main/ABORT           why the run was aborted
rm results/main/ABORT            clear an abort after fixing the cause
rsync -az root@<IP>:~/Masterthesis/results/ ~/Documents/PROJECTS/masterthesis/Masterthesis/results/
```
