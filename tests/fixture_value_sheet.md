# Fixture Value Sheet

Hand-derived expected values for the four test fixtures. They are the only expected values in the project that are not derived from the code. They were derived on paper. `tests/expected.py` holds the same numbers in code form.

**Formulas (exposé §4.2, weights from Table A1)**

- u_ws = 2·p_role[r(s)] + p_time[p(s)] + k[r(s)]
- Feasible: every cell available; ≤ 1 shift per volunteer and period; ≤ cap shifts per volunteer; every open shift has staffing ≥ ⌈d/2⌉, someone with k ≥ κ, and, if expert-flagged, someone with k = 3
- Λ = Σ_s max(0, d_s − staffing_s)
- Mentor: +1 per shift with a novice (k ≤ 1) and a mentor (σ = 2, k ≥ 2) in that shift's role
- q_s = max(0, staffing_s − d_s), z_s = min(2, q_s); U = Σu + Σmentor + 5·Σz − 10·Σq
- U and Λ are also given for infeasible schedules: the oracle computes them regardless

---

## 1. `single_shift`

n_periods = 1, n_roles = 1

|     | available | skill | seniority | cap | pref_role | pref_time | u(s0)           |
| --- | --------- | ----- | --------- | --- | --------- | --------- | --------------- |
| v0  | {0}       | (1)   | 0         | 10  | (2)       | (1)       | 2·2 + 1 + 1 = 6 |
| v1  | {0}       | (3)   | 2         | 10  | (0)       | (0)       | 0 + 0 + 3 = 3   |

s0: period 0, role 0, d = 1, κ = 1, no expert.

| Schedule          | Violations | Λ   | Σu  | mentor | +5z | −10q | U   |
| ----------------- | ---------- | --- | --- | ------ | --- | ---- | --- |
| {}                | none       | 1   | 0   | 0      | 0   | 0    | 0   |
| {(v0,s0)}         | none       | 0   | 6   | 0      | 0   | 0    | 6   |
| {(v1,s0)}         | none       | 0   | 3   | 0      | 0   | 0    | 3   |
| {(v0,s0),(v1,s0)} | none       | 0   | 9   | 1      | 5   | −10  | 5   |

**Optimum** [MinShortfall, MaxQuality]: {(v0,s0)}, Λ = 0, U = 6.

_Argument:_ every non-empty schedule covers s0 (Λ = 0). Among the three, a single assignee avoids the net surplus cost (−10 + 5 + mentor 1 = −4), and v0 has the higher u.

---

## 2. `min_crew`

n_periods = 1, n_roles = 1. s0: d = 3, so the minimum crew is ⌈3/2⌉ = 2; κ = 1, no expert.

|     | available | skill | seniority | cap | pref_role | pref_time | u(s0)         |
| --- | --------- | ----- | --------- | --- | --------- | --------- | ------------- |
| v0  | {0}       | (2)   | 0         | 10  | (1)       | (1)       | 2 + 1 + 2 = 5 |
| v1  | { }       | (2)   | 0         | 10  | (1)       | (1)       | 5             |

| Schedule          | Violations            | Λ   | U   |
| ----------------- | --------------------- | --- | --- |
| {}                | none                  | 3   | 0   |
| {(v0,s0)}         | min_crew              | 2   | 5   |
| {(v1,s0)}         | unavailable, min_crew | 2   | 5   |
| {(v0,s0),(v1,s0)} | unavailable           | 1   | 10  |

**Optimum:** {}, the only feasible schedule; Λ\* = 3, U = 0.

### Variant `min_crew_both_available` (v1 available in {0})

| Schedule          | Violations | Λ   | U   |
| ----------------- | ---------- | --- | --- |
| {}                | none       | 3   | 0   |
| {(v0,s0)}         | min_crew   | 2   | 5   |
| {(v1,s0)}         | min_crew   | 2   | 5   |
| {(v0,s0),(v1,s0)} | none       | 1   | 10  |

**Optimum:** {(v0,s0),(v1,s0)}, Λ = 1, U = 10.

_Argument:_ only {} and the pair are feasible (a single person is below the minimum crew), and the pair has the lower Λ.

---

## 3. `expert_shift`

n_periods = 1, n_roles = 2. All volunteers: available {0}, seniority 0, cap 10.

|     | skill (r0, r1) | pref_role (r0, r1) | pref_time (p0) | u(s0)         | u(s1)         |
| --- | -------------- | ------------------ | -------------- | ------------- | ------------- |
| v0  | (3, 2)         | (0, 2)             | (1)            | 0 + 1 + 3 = 4 | 4 + 1 + 2 = 7 |
| v1  | (1, 2)         | (0, 0)             | (0)            | 0 + 0 + 1 = 1 | 0 + 0 + 2 = 2 |
| v2  | (0, 1)         | (0, 0)             | (0)            | 0 + 0 + 0 = 0 | 0 + 0 + 1 = 1 |

- s0: period 0, role 0, d = 1, κ = 1, **expert**.
- s1: period 0, role 1, d = 1, κ = 1.

| Schedule          | Violations             | Λ   | U   |
| ----------------- | ---------------------- | --- | --- |
| {}                | none                   | 2   | 0   |
| {(v1,s0)}         | expert                 | 1   | 1   |
| {(v2,s0)}         | skill, expert          | 1   | 0   |
| {(v0,s0),(v0,s1)} | two_shifts_same_period | 0   | 11  |
| {(v0,s1),(v1,s0)} | expert                 | 0   | 8   |
| {(v0,s0)}         | none                   | 1   | 4   |
| {(v0,s1)}         | none                   | 1   | 7   |
| {(v0,s0),(v2,s1)} | none                   | 0   | 5   |
| {(v0,s0),(v1,s1)} | none                   | 0   | 6   |

**Optimum [MaxQuality] alone:** {(v0,s1)}, Λ = 1, U = 7.

**Optimum [MinShortfall, MaxQuality]:** {(v0,s0),(v1,s1)}, Λ = 0, U = 6.

_Argument:_ s0 can open only with v0 (the only volunteer with k = 3 in role 0), and v0 can work only one shift in the single period. Full coverage therefore means v0 on s0 plus the best single person for s1 (v1: u = 2 beats v2: u = 1; a second person would cost a net −5). Without the coverage stage, v0's best cell (s1, u = 7) beats every covered schedule (at most 6), and s0 stays cancelled.

---

## 4. `repair_slice`

n_periods = 2, n_roles = 2. All volunteers: available {0, 1}, seniority 0. Ids: A = v0, B = v1, C = v2.

|     | skill (r0, r1) | pref_role (r0, r1) | pref_time (p0, p1) | cap   | u(s0) | u(s1) | u(s2) |
| --- | -------------- | ------------------ | ------------------ | ----- | ----- | ----- | ----- |
| A   | (2, 0)         | (2, 0)             | (2, 0)             | **1** | 8     | 2     | 6     |
| B   | (1, 2)         | (0, 2)             | (1, 0)             | 10    | 2     | 7     | 1     |
| C   | (1, 1)         | (1, 0)             | (0, 0)             | 10    | 3     | 1     | 3     |

Shifts (all d = 1, κ = 1, no expert): s0 = (p0, r0), s1 = (p0, r1), s2 = (p1, r0). J = 3 · 3 = 9.

### Step 0: S₀

| Schedule                      | Violations | Λ   | U   |
| ----------------------------- | ---------- | --- | --- |
| {(A,s0),(B,s1),(C,s2)}        | none       | 0   | 18  |
| {(A,s2),(B,s1),(C,s0)}        | none       | 0   | 16  |
| {(A,s0),(B,s1),(B,s2)}        | none       | 0   | 16  |
| {(A,s0),(B,s1),(C,s0),(C,s2)} | none       | 0   | 16  |
| {(A,s0),(A,s2),(B,s1)}        | cap        | 0   | 21  |

**S₀ = {(A,s0),(B,s1),(C,s2)}**, Λ = 0, U = 18, unique.

_Argument:_ each shift has d = 1 and any extra person costs a net −5, so the best schedules have exactly one person per shift. A can work one shift only (cap 1); without the cap, A on s0 and s2 would give 21. If A is on s0 (8), s1 is best with B (7) and s2 with C (3, versus B's 1): 18. If A is on s2 (6), s0 gets C (3) and s1 gets B (7): 16. Every other arrangement swaps in lower-u cells.

### Event

`AvailabilityRemoval(pairs=((0, 0),))`: A is no longer available in period 0, so the cells (A,s0) and (A,s1) become unavailable.

- S₀ now violates `unavailable` (cell (A,s0)), so the step is **binding** for every arm.
- Λ\*₁ = 0: B and C can still cover s0 and s1, and A or C can cover s2.

### Step 1: repairs (H = |S ⊕ S₀|)

| Repair                            | Violations  | Λ   | U   | H   | U − 1·H | U − 2·H | U − 4·H |
| --------------------------------- | ----------- | --- | --- | --- | ------- | ------- | ------- |
| S₀ = {(A,s0),(B,s1),(C,s2)}       | unavailable | 0   | 18  | 0   |         |         |         |
| **SF\*** = {(B,s1),(C,s0),(C,s2)} | none        | 0   | 13  | 2   | 11      | **9**   | **5**   |
| **X** = {(A,s2),(B,s1),(C,s0)}    | none        | 0   | 16  | 4   | **12**  | 8       | 0       |
| {(A,s2),(B,s1),(C,s0),(C,s2)}     | none        | 0   | 14  | 3   | 11      | 8       | 2       |

| Arm                              | Chosen repair                                   |
| -------------------------------- | ----------------------------------------------- |
| SF (anchored = chained at t = 1) | SF\* (unique minimum H = 2)                     |
| QL                               | X (unique maximum U = 16)                       |
| QO                               | X                                               |
| WQS λ = 1                        | X (12 vs. 11)                                   |
| WQS λ = 2                        | SF\* (9 vs. 8)                                  |
| WQS λ = 4                        | SF\* (5 vs. 2)                                  |
| HWF α = 1 at t = 1               | SF\* (burden 0 for everyone, so weighted H = H) |

Churn of the SF repair: C₁ = 2 / 9.

_Argument for SF\*:_ (A,s0) must go (H ≥ 1), and s0 then needs a new person in period 0. B is already in period 0 (on s1), A is unavailable, so only C can be added: (C,s0), giving H = 2. Every other repair changes more cells.

_Argument for X:_ with Λ = 0 and one person per shift, s0 and s1 need B and C in some order, and s2 needs A, C or B. B on s1 (7) and C on s0 (3) is the best pairing. Then for s2, A (6) beats C (3) or B (1): U = 16. Adding a second person anywhere costs a net −5.

_All optima are unique_, so the final tie-break stage never decides between them.
