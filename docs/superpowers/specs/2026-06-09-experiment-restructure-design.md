# Design Spec: Experiment Script Restructure

**Date:** 2026-06-09
**Scope:** Correct `experiment.py` based on paper-code audit findings; extract shared LP core; add ablation sweep script.

---

## Context

A paper-code audit (`outputs/hdrfc-k-freemtpl2-audit.md`) identified 14 items against `experiment.py`. 7 were consistent; the rest ranged from comment errors to a high-severity plan/script mismatch. The restructure addresses all actionable findings and splits the single script into three files with clear responsibilities.

---

## Files

| File | Role |
|---|---|
| `lp_core.py` | Shared LP builder and evaluator — single source of truth for the theoretical formulation |
| `experiment.py` | Single-run validation (baseline vs. HDRFC-K at fixed ρ/ζ); cleaned and imports from `lp_core` |
| `ablation.py` | Full ρ×ζ sweep; saves CSV + two plots to `outputs/` |

---

## `lp_core.py`

### `build_hdrfc_k(X, Y, A, I_pos, pairs, rho, zeta)`

Builds the HDRFC-K LP from Theorem 3.5.

**Audit fix — item 4 (λ variable size):**
λ variables declared with size `len(Ik) + len(Il)` per pair, not full-N. Constraints (8), (9), (10) use relative slice offsets `[0:len(Ik)]` and `[len(Ik):]`. Eliminates ~320k unused LP variables across 20 pairs at N=16k.

**Audit fix — item 8 (class-weighted objective):**
Inverse-frequency class weights are kept (necessary for non-degenerate classification on 5% positive base rate). A docstring comment explicitly documents the deviation from Theorem 3.5's uniform-weight objective and justifies it.

All constraints (7), (8), (9), (10) are unchanged — they were already correct.

### `evaluate(w_val, b_val, X_te, Y_te, A_te, pairs, label)`

Moved from `experiment.py` unchanged. Returns dict of `{accuracy, balanced_accuracy, tpr, max_gap}`.

---

## `experiment.py`

Imports `build_hdrfc_k`, `evaluate` from `lp_core`. All section structure (S1–S7) preserved.

**Audit fixes applied:**

| Item | Fix |
|---|---|
| 10 — solver docstring | Updated: "CLARABEL (default); install mosek for larger N" |
| 12 — age bin comment | Changed to "right-closed intervals: (17,25], (25,40], ..." |
| 12 — silent row drop | After `dropna(subset=["A"])`, prints dropped row count |
| 14 — np.random seed | `np.random.seed(RANDOM_STATE)` added at top |
| 14 — OpenML version pin | `fetch_openml(..., data_version=1)` |

`ZETA = 1.5` already correct in script — no change. Item 7 fix is plan-only (update `empirical_analysis_plan.md` separately).

---

## `ablation.py`

### Sweep grid

| Parameter | Values |
|---|---|
| ρ (Wasserstein radius) | 0, 0.01, 0.05, 0.1, 0.5 |
| ζ (fairness tolerance) | 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0 |
| Total solves | 35 |

ζ values all > 0 (ζ must satisfy H_kl ≤ 1+ζ; Prop 3.4 requires ζ ≥ 0 for feasibility, but ζ < 0.5 risks infeasibility on this dataset given class imbalance).

### Data setup

Same data loading/preprocessing block as `experiment.py`: identical constants (`SAMPLE_N`, `RANDOM_STATE`, `AGE_BINS`, feature columns), same `np.random.seed`, same `fetch_openml(..., data_version=1)` call. Not imported — duplicated once to keep `ablation.py` independently runnable.

### Sweep loop

For each (ρ, ζ):
- Build and solve via CLARABEL
- Print one-line progress: `ρ=0.10 ζ=1.5 → optimal  max_gap=0.1234  (42.3s)`
- On infeasible/error: record `NaN` for metrics, do not raise — sweep continues

### Outputs

**`outputs/ablation_results.csv`**
Columns: `rho, zeta, status, accuracy, bal_acc, max_gap, solve_time`
One row per (ρ, ζ) solve including failed/infeasible runs.

**`outputs/ablation_tpr_gap.png`**
Line plot: max TPR gap (y) vs ζ (x), one line per ρ value. Shows how tightening the fairness constraint reduces TPR disparity.

**`outputs/ablation_pareto.png`**
Scatter plot: balanced accuracy (y) vs max TPR gap (x). Points coloured by ρ, annotated with ζ value. Visualises the accuracy–fairness trade-off across the full grid.

---

## Audit Items NOT addressed in code (plan-level fixes)

| Item | Action |
|---|---|
| 7 — ζ=0.1 in plan | Update `empirical_analysis_plan.md`: change ζ=0.1 → ζ=1.5, correct rationale from "≤10% TPR gap" to correct H_kl description |
| 8 — theorem statement | Paper should add a remark acknowledging the class-weighting implementation choice |

---

## File Layout After Restructure

```
analytics/
├── lp_core.py              # NEW — shared LP builder + evaluator
├── experiment.py           # MODIFIED — imports lp_core, audit fixes applied
├── ablation.py             # NEW — 35-solve sweep, CSV + plots
├── outputs/
│   ├── ablation_results.csv
│   ├── ablation_tpr_gap.png
│   └── ablation_pareto.png
└── empirical_analysis_plan.md   # to be updated separately (item 7)
```

---

## Out of Scope

- Hyperparameter selection via cross-validation (plan open item)
- Binary HDRFC baseline comparison (plan open item)
- L2/SOCP formulation sensitivity (plan open item)
- Pareto frontier curve (covered by `ablation_pareto.png` scatter; a smooth curve requires denser grid)
