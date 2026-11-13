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
