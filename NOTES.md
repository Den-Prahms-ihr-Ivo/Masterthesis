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
