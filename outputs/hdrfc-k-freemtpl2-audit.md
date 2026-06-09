# Paper-Code Audit: HDRFC-K on freMTPL2freq

**Paper:** Combined_Paper.docx — multi-group distributionally robust fair classifier
**Code:** experiment.py
**Date:** 2026-06-09

---

## Audit Summary

| # | Item | Verdict | Severity |
|---|---|---|---|
| 1 | Constraint (7) — robust hinge loss | ✅ Consistent | — |
| 2 | Constraint (8) — `-1` term in fairness | ✅ Consistent | — |
| 3 | Constraints (9)/(10) — dual variable bounds | ✅ Consistent | — |
| 4 | Lambda variable size (full-N declaration) | ⚠️ Wasteful, not incorrect | Low — performance risk |
| 5 | Norm choice (L1 = `cp.norm1`) | ✅ Consistent | — |
| 6 | Fairness over all ordered pairs P_K | ✅ Consistent, intentional asymmetry | — |
| 7 | Zeta value: plan=0.1, script=1.5 | ❌ Plan is wrong; script is correct | **High** — plan value causes infeasibility |
| 8 | Class-weighted objective | ⚠️ Undocumented deviation from Theorem 3.5 | **Medium** — alters theoretical problem |
| 9 | Baseline (ρ=0, ζ=∞) | ✅ Consistent | — |
| 10 | Solver: CLARABEL vs. MOSEK | ⚠️ Documentation mismatch | Low |
| 11 | `max_gap` over ordered pairs | ✅ Consistent (harmless double-count) | — |
| 12 | Age bin edge / comment error | ⚠️ Comment wrong, code correct | Low |
| 13 | Missing ablations / sensitivity analysis | ⚠️ Acknowledged omission | Medium |
| 14 | Reproducibility / seeding | ⚠️ Mostly seeded; two gaps | Low |

---

## Detailed Findings

### 1. Constraint (7) — Robust Hinge Loss ✅

`experiment.py:163`
```python
cp.multiply(-Y, Xw_b) + rho * norm_w <= t - 1
```
Correctly encodes `-y_i(w^T x_i + b) + ρ‖w‖₁ ≤ t_i - 1`. The L∞ Wasserstein robustness term has LP dual representation `ρ‖w‖₁`, which is the L1 norm. Vectorized CVXPY form is correct.

---

### 2. Constraint (8) — Fairness Scalar ✅

`experiment.py:175-179`
```python
cp.sum(lam_kl[Ik]) / len(Ik) + cp.sum(lam_kl[Il]) / len(Il) - 1 <= zeta
```
The `-1` is correct. It encodes H_kl ≤ 1 + ζ. Proposition 3.4 guarantees H_kl ≥ 1, so the constraint is feasible iff ζ ≥ 0. The plan's statement that ζ=0.1 "allows ≤10% pairwise TPR gap" conflates H_kl with TPR gap — these are different quantities.

---

### 3. Constraints (9) and (10) — Dual Variable Bounds ✅

`experiment.py:182-188`
```python
1 + X[Ik] @ w + rho * norm_w + b <= lam_kl[Ik]   # (9) over group k positive class
1 - X[Il] @ w + rho * norm_w - b <= lam_kl[Il]    # (10) over group l positive class
```
The sign asymmetry (`+b` vs `-b`) correctly reflects the directional equal-opportunity structure: group k is the reference, group l is the comparison. Both constraints use the same λ^{kl} variables that appear in the scalar constraint (8), correctly linking the transport structure across constraints.

---

### 4. Lambda Variable Size — Full-N Declaration ⚠️

`experiment.py:148-149`
```python
lam = {pair: cp.Variable(N, name=f"lam_{pair[0]}_{pair[1]}", nonneg=True) for pair in pairs}
```
Each λ^{kl} is declared with `N = N_tr ≈ 16,000` entries. Only positions `Ik` and `Il` are ever constrained; the rest are free nonneg variables set to 0 at optimum. With 20 pairs this adds ~320,000 unused LP variables. Not incorrect but wastes solver time. Fix: declare `cp.Variable(len(Ik) + len(Il))` and use relative indexing.

---

### 5. Norm Choice — L1 ✅

`experiment.py:158`: `norm_w = cp.norm1(w)`

L1 norm is the correct choice for a pure LP formulation. The Wasserstein robustness under L∞ transport cost has dual representation `ρ‖w‖₁`. L2 would require SOCP; the LP formulation requires L1.

---

### 6. Ordered Pairs P_K ✅

`experiment.py:130`: `pairs = list(permutations(AGE_LABELS, 2))` → 20 ordered pairs

Both (k,l) and (l,k) are included. This is intentional: H_kl ≠ H_lk in general. Including both directions enforces symmetric equal-opportunity (neither group's TPR dominates the other's). Correct.

---

### 7. Zeta Value Discrepancy — Plan vs. Script ❌ HIGH

| Location | Value | Feasible? |
|---|---|---|
| `empirical_analysis_plan.md` | ζ = 0.1 | Extremely tight — likely infeasible on this dataset |
| `experiment.py:33` | ζ = 1.5 | Feasible — H_kl ≤ 2.5 |

**The plan value ζ=0.1 is an error.** It would constrain H_kl ≤ 1.1, which may be infeasible given the 95%/5% class imbalance. The script correctly uses ζ=1.5 with the comment "H_kl lower bound is 1 (Prop 3.4), so ζ>1 needed."

The plan's rationale ("Allows ≤10% pairwise TPR gap") is also incorrect — ζ controls H_kl, not TPR gap directly.

**Action required:** Update the plan to ζ=1.5 and correct the rationale to describe what H_kl represents.

---

### 8. Class-Weighted Objective — Undocumented Deviation ⚠️ MEDIUM

`experiment.py:152-156, 191`
```python
w_pos = N / (2 * n_pos)   # ~10x for positive class (5% base rate)
w_neg = N / (2 * n_neg)
sample_weights = np.where(Y == 1, w_pos, w_neg)
# ...
objective = cp.Minimize(cp.sum(cp.multiply(sample_weights, t)) / N)
```

**Theorem 3.5 states:** objective = `(1/N) Σ t_i` (uniform weights).

The script uses inverse-frequency class weights — a standard practice for handling the ~95%/5% label imbalance in freMTPL2freq. Without this weighting, the LP minimizes hinge loss by predicting all-negative, achieving ~0% TPR with high accuracy.

This deviation is pragmatically necessary but is not mentioned in the paper. It changes the optimization problem: theoretical guarantees in Theorem 3.5 apply to the uniform-objective LP. The paper should either (a) incorporate class weights into the theorem statement, or (b) explicitly note this implementation deviation and argue the theorem's qualitative results still hold.

---

### 9. Baseline — Unconstrained SVM ✅

`experiment.py:257-259`
```python
prob_svm, w_svm, b_svm = build_hdrfc_k(
    X_tr, Y_tr, A_tr, I_pos, pairs, rho=0.0, zeta=np.inf
)
```
With ρ=0, the Wasserstein robustness term vanishes. With ζ=∞, constraint (8) is skipped (`if zeta < np.inf` at line 174). This correctly reduces to a class-weighted hinge-loss SVM. Baseline is correctly specified.

---

### 10. Solver: CLARABEL vs. MOSEK ⚠️

**Docstring (line 8):** "A valid MOSEK license (academic licenses are free at mosek.com)"
**Actual solver (lines 260, 287):** `prob_svm.solve(solver=cp.CLARABEL, ...)`

CLARABEL is appropriate for this LP size and is bundled with CVXPY (no license needed). MOSEK would be faster and more numerically stable for larger N. The docstring is misleading. Update docstring to reflect CLARABEL, or add a MOSEK-with-CLARABEL-fallback pattern. The `optimal_inaccurate` handling at lines 267/294 is a good robustness measure for CLARABEL.

---

### 11. max_gap Computation ✅

`experiment.py:215-216`
```python
max_gap = max(abs(tpr[k] - tpr[l]) for (k, l) in pairs ...)
```
Over 20 ordered pairs, each unordered pair appears twice. Since `abs(tpr[k]-tpr[l]) == abs(tpr[l]-tpr[k])`, the max is computed over duplicate values — functionally equivalent to max over 10 unordered pairs. Correct.

---

### 12. Age Bin Edge / Comment Error ⚠️

`experiment.py:39`: `AGE_BINS = [17, 25, 40, 55, 70, 120]` with `right=True`

This creates intervals (17,25], (25,40], etc. The comment says "right-exclusive upper bounds" — this is wrong. With `right=True`, upper bounds ARE included (right-closed intervals).

The code is functionally correct (bin 1 = ages 18–25 as an integer). However:
- Drivers aged exactly 17 are silently excluded (below the first bin left edge). These are dropped by `dropna(subset=["A"])` at line 68 with no warning or count.
- `evaluate()` at line 226 prints `"18-25"` as the label, consistent with the actual interval.

Fix: change comment to "right-closed intervals: (17,25], (25,40], ..." and add a count of dropped rows.

---

### 13. Missing Ablations ⚠️

The script tests exactly one (ρ, ζ) = (0.1, 1.5) configuration. The plan's "Open Items" section acknowledges sweeps over ρ ∈ {0, 0.01, 0.05, 0.1, 0.5} and ζ ∈ {0, 0.05, 0.1, 0.2, 0.5} are planned but not implemented. The plots spec adds a ζ-sweep but not a ρ-sweep.

Without ablations, the paper cannot demonstrate that ρ=0.1 is a meaningful choice. ρ=0.1 in post-standardization space corresponds to 0.1σ perturbations — this implicit scale-dependence is not discussed.

**Minimum for publication:** One ρ sensitivity plot and a ζ-vs-max_gap tradeoff curve. The ζ-sweep in the plots spec is a start.

---

### 14. Reproducibility / Seeding ⚠️

| Operation | Seeded? |
|---|---|
| Stratified subsample (`train_test_split`) | ✅ `random_state=42` |
| Train/test split (`train_test_split`) | ✅ `random_state=42` |
| CVXPY/CLARABEL solver | ✅ (deterministic) |
| `np.random` global state | ❌ Not set |
| OpenML dataset version | ❌ No `data_version` pin |

Two gaps:
1. Add `np.random.seed(RANDOM_STATE)` at top of script as a guard against future code using global state.
2. Add `data_version=1` (or current version) to `fetch_openml` to lock dataset to a specific version.

---

## Critical Actions Required Before Publication

### Action 1 — Fix ζ documentation (HIGH)
In `empirical_analysis_plan.md`: change ζ=0.1 → ζ=1.5, and correct the rationale from "≤10% pairwise TPR gap" to a correct description of H_kl ≤ 1+ζ and why ζ=1.5 was chosen.

### Action 2 — Document class-weighted objective (MEDIUM)
Either:
- Add a remark in the paper noting the inverse-frequency weighting as an implementation choice and explain why it does not affect the qualitative conclusions of Theorem 3.5, OR
- Formally incorporate class weights into the theorem statement.

### Action 3 — Fix lambda variable size (LOW, but important for scalability)
Replace full-N lambda declarations with group-sized declarations to avoid 320k unnecessary LP variables.

### Action 4 — Update solver documentation (LOW)
Change docstring from "MOSEK required" to "CLARABEL (default) or MOSEK" and ensure requirements list reflects actual dependencies.

---

## Sources

- **Code:** `experiment.py` (this repository)
- **Plan:** `empirical_analysis_plan.md` (this repository)
- **Paper:** `Combined_Paper.docx` (this repository — binary, content inferred from plan and code)
- **freMTPL2freq dataset:** https://www.openml.org/d/41214
- **CVXPY documentation:** https://www.cvxpy.org/
- **CLARABEL solver:** https://clarabel.org/
