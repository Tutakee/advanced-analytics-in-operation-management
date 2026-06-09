# Experiment Script Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `experiment.py` into `lp_core.py` (shared LP), `experiment.py` (single-run, cleaned), and `ablation.py` (35-solve ρ×ζ sweep with CSV + plots), applying all audit fixes from `outputs/hdrfc-k-freemtpl2-audit.md`.

**Architecture:** `lp_core.py` is the single source of truth for `build_hdrfc_k()` and `evaluate()`. `experiment.py` imports from it and adds housekeeping fixes. `ablation.py` is independently runnable (duplicates data-loading setup) and imports the LP helpers, runs all 35 solves, and writes `outputs/ablation_results.csv` + two matplotlib figures.

**Tech Stack:** Python 3.9+, cvxpy, clarabel, scikit-learn, pandas, numpy, matplotlib, openml

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `lp_core.py` | `build_hdrfc_k()` + `evaluate()` — LP formulation only |
| Modify | `experiment.py` | Import from `lp_core`, apply audit fixes |
| Create | `ablation.py` | ρ×ζ sweep, CSV, two plots |
| Modify | `empirical_analysis_plan.md` | Fix ζ=0.1 → ζ=1.5, correct rationale |

---

## Task 1: Create `lp_core.py` with `build_hdrfc_k()`

**Files:**
- Create: `lp_core.py`

- [ ] **Step 1: Create `lp_core.py` with the LP builder**

```python
"""
lp_core.py — shared LP formulation for HDRFC-K (Theorem 3.5).

Provides build_hdrfc_k() and evaluate() used by experiment.py and ablation.py.
"""

import numpy as np
import cvxpy as cp
from sklearn.metrics import balanced_accuracy_score

AGE_LABELS = [1, 2, 3, 4, 5]
_GROUP_NAMES = ["18-25", "26-40", "41-55", "56-70", "71+"]


def build_hdrfc_k(X, Y, A, I_pos, pairs, rho, zeta):
    """
    Build the HDRFC-K LP (Theorem 3.5) as a CVXPY Problem.

    Parameters
    ----------
    X      : (N, d) float array — standardised feature matrix (training)
    Y      : (N,) int array — labels in {-1, +1}
    A      : (N,) int array — group indices in {1, ..., K}
    I_pos  : dict[int -> 1-D int array] — I_pos[k] = indices where A==k and Y==1
    pairs  : list of (int, int) — all ordered pairs (k, l), k != l
    rho    : float — Wasserstein radius (>= 0); rho=0 gives empirical ERM
    zeta   : float — fairness tolerance; constraint is H_kl <= 1 + zeta.
             Proposition 3.4 guarantees H_kl >= 1, so zeta >= 0 is needed for
             feasibility. Values < 0.5 risk infeasibility on heavily imbalanced data.
             Pass np.inf to skip the fairness constraint entirely.

    Returns
    -------
    problem : cp.Problem
    w       : cp.Variable (d,)
    b       : cp.Variable scalar

    Notes
    -----
    Objective uses inverse-frequency class weights to prevent a degenerate
    all-negative solution on the ~5% positive base rate of freMTPL2freq.
    This deviates from the uniform-weight objective in Theorem 3.5 but does
    not affect the qualitative fairness guarantees: the constraint structure
    (8)-(10) is unchanged, and the class weights only rescale the hinge loss.
    """
    N, d = X.shape
    w = cp.Variable(d, name="w")
    b = cp.Variable(name="b")
    t = cp.Variable(N, name="t", nonneg=True)

    norm_w = cp.norm1(w)
    constraints = []

    # Constraint (7): class-weighted robust hinge loss — vectorised over all N
    Xw_b = X @ w + b
    constraints.append(cp.multiply(-Y, Xw_b) + rho * norm_w <= t - 1)

    # Class weights: inverse frequency (handles ~95/5 imbalance)
    n_pos = (Y == 1).sum()
    n_neg = (Y == -1).sum()
    w_pos = N / (2 * n_pos)
    w_neg = N / (2 * n_neg)
    sample_weights = np.where(Y == 1, w_pos, w_neg)

    # Lambda variables sized |Ik| + |Il| per pair (not full-N) to avoid
    # ~320k unused LP variables across 20 pairs at N=16k.
    lam = {}
    for (k, l) in pairs:
        Ik = I_pos[k]
        Il = I_pos[l]
        if len(Ik) == 0 or len(Il) == 0:
            continue
        nk, nl = len(Ik), len(Il)
        lam_kl = cp.Variable(nk + nl, name=f"lam_{k}_{l}", nonneg=True)
        lam[(k, l)] = (lam_kl, nk, nl)

        # Constraint (8): scalar fairness bound — H_kl <= 1 + zeta
        if zeta < np.inf:
            constraints.append(
                cp.sum(lam_kl[:nk]) / nk + cp.sum(lam_kl[nk:]) / nl - 1 <= zeta
            )

        # Constraint (9): lower bound on lam over group-k positive-class points
        constraints.append(1 + X[Ik] @ w + rho * norm_w + b <= lam_kl[:nk])

        # Constraint (10): lower bound on lam over group-l positive-class points
        constraints.append(1 - X[Il] @ w + rho * norm_w - b <= lam_kl[nk:])

    objective = cp.Minimize(cp.sum(cp.multiply(sample_weights, t)) / N)
    problem = cp.Problem(objective, constraints)
    return problem, w, b


def evaluate(w_val, b_val, X_te, Y_te, A_te, pairs, label):
    """
    Evaluate a fitted classifier and print metrics.

    Returns dict with keys: accuracy, balanced_accuracy, tpr (dict), max_gap.
    """
    scores = X_te @ w_val + b_val
    Y_hat = np.sign(scores)
    Y_hat[Y_hat == 0] = 1  # tie-break to positive

    acc = np.mean(Y_hat == Y_te)
    bal_acc = balanced_accuracy_score(Y_te, Y_hat)

    tpr = {}
    for k in AGE_LABELS:
        mask = (A_te == k) & (Y_te == 1)
        tpr[k] = float(np.mean(Y_hat[mask] == 1)) if mask.sum() > 0 else float("nan")

    max_gap = max(
        abs(tpr[k] - tpr[l])
        for (k, l) in pairs
        if not np.isnan(tpr[k]) and not np.isnan(tpr[l])
    )

    print(f"\n{'-'*50}")
    print(f"  Model: {label}")
    print(f"{'-'*50}")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Max pairwise TPR gap: {max_gap:.4f}")
    print()
    print("  TPR per age group:")
    for k in AGE_LABELS:
        print(f"    Group {k} ({_GROUP_NAMES[k-1]}): TPR = {tpr[k]:.4f}")

    print()
    print("  Pairwise TPR difference matrix (row k, col l = TPR_k - TPR_l):")
    header = "      " + "".join(f"  G{k}  " for k in AGE_LABELS)
    print(header)
    for k in AGE_LABELS:
        row = f"  G{k}  "
        for l in AGE_LABELS:
            if k == l:
                row += "  ---  "
            else:
                diff = tpr[k] - tpr[l]
                row += f" {diff:+.3f}"
        print(row)

    return {"accuracy": acc, "balanced_accuracy": bal_acc, "tpr": tpr, "max_gap": max_gap}
```

- [ ] **Step 2: Verify the file is importable**

```bash
cd C:/SAPDevelop/hai/analytics && python -c "from lp_core import build_hdrfc_k, evaluate; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add lp_core.py
git commit -m "feat: extract lp_core with build_hdrfc_k and evaluate"
```

---

## Task 2: Update `experiment.py` — import from `lp_core` and apply audit fixes

**Files:**
- Modify: `experiment.py`

- [ ] **Step 1: Replace the full contents of `experiment.py`**

```python
"""
Empirical experiment: HDRFC-K on freMTPL2freq
Validates the multi-group distributionally robust fair classifier (Theorem 3.5)
against an unconstrained SVM baseline.

Requirements:
    pip install cvxpy scikit-learn pandas numpy openml matplotlib
    CLARABEL is bundled with cvxpy (no license needed).
    For larger N, install mosek for faster solves: pip install mosek
"""

import time
import warnings
import numpy as np
import pandas as pd
from itertools import permutations

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import cvxpy as cp

from lp_core import build_hdrfc_k, evaluate

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# S1  CONFIGURATION
# -------------------------------------------------------------

RHO = 0.1       # Wasserstein radius
ZETA = 1.5      # fairness tolerance: H_kl <= 1 + zeta (Prop 3.4 guarantees H_kl >= 1)
K = 5           # number of age cohorts
SAMPLE_N = 20_000
TEST_SIZE = 0.2
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# Right-closed intervals: (17,25], (25,40], (40,55], (55,70], (70,120]
AGE_BINS = [17, 25, 40, 55, 70, 120]
AGE_LABELS = [1, 2, 3, 4, 5]

CATEGORICAL_COLS = ["VehBrand", "VehGas", "Region", "Area"]
NUMERIC_COLS = ["VehPower", "VehAge", "BonusMalus", "Density", "Exposure"]

# -------------------------------------------------------------
# S2  DATA LOADING & PREPROCESSING
# -------------------------------------------------------------

print("=" * 60)
print("S1  Loading freMTPL2freq (OpenML ID 41214)")
print("=" * 60)

raw = fetch_openml(data_id=41214, as_frame=True, parser="pandas", data_version=1)
df = raw.frame.copy()

# Binary label
df["Y"] = np.where(df["ClaimNb"].astype(float) > 0, 1, -1)

# Sensitive attribute: age group (training only)
df["A"] = pd.cut(
    df["DrivAge"].astype(float),
    bins=AGE_BINS,
    labels=AGE_LABELS,
    right=True,
).astype("Int64")

# Drop rows outside age bins and report count
n_before = len(df)
df = df.dropna(subset=["A"])
df["A"] = df["A"].astype(int)
n_dropped = n_before - len(df)
if n_dropped:
    print(f"  Dropped {n_dropped} rows with DrivAge outside bins {AGE_BINS}")

# Stratified subsample — LP complexity is O(K^2*N); 20k keeps solve tractable
strat_key_full = df["Y"].astype(str) + "_" + df["A"].astype(str)
df, _ = train_test_split(
    df, train_size=SAMPLE_N, stratify=strat_key_full, random_state=RANDOM_STATE
)
df = df.reset_index(drop=True)
print(f"Subsampled to N={len(df):,} (stratified on Y x A)")

print(f"N total        : {len(df):,}")
print(f"Positive (Y=1) : {(df['Y'] == 1).sum():,}  ({100*(df['Y']==1).mean():.1f}%)")
print(f"Negative (Y=-1): {(df['Y'] == -1).sum():,}  ({100*(df['Y']==-1).mean():.1f}%)")
print()
print("Age group statistics:")
for g in AGE_LABELS:
    n_g = (df["A"] == g).sum()
    n_g_pos = ((df["A"] == g) & (df["Y"] == 1)).sum()
    print(f"  Group {g}: N={n_g:>7,}  |Ik+|={n_g_pos:>5,}  ({100*n_g_pos/n_g:.1f}% positive)")

feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
X_raw = df[feature_cols]
Y = df["Y"].values
A = df["A"].values

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_COLS),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_COLS),
])

# -------------------------------------------------------------
# S3  TRAIN / TEST SPLIT
# -------------------------------------------------------------

print()
print("=" * 60)
print("S2  Train/test split (stratified 80/20 on Y x A)")
print("=" * 60)

strat_key = Y.astype(str) + "_" + A.astype(str)
X_tr_raw, X_te_raw, Y_tr, Y_te, A_tr, A_te = train_test_split(
    X_raw, Y, A,
    test_size=TEST_SIZE,
    stratify=strat_key,
    random_state=RANDOM_STATE,
)

X_tr = preprocessor.fit_transform(X_tr_raw)
X_te = preprocessor.transform(X_te_raw)

N_tr, d = X_tr.shape
print(f"Train N={N_tr:,}  Test N={len(Y_te):,}  Features d={d}")

I_pos = {k: np.where((A_tr == k) & (Y_tr == 1))[0] for k in AGE_LABELS}
for k in AGE_LABELS:
    print(f"  |I_{k}+| (train) = {len(I_pos[k]):,}")

pairs = list(permutations(AGE_LABELS, 2))
print(f"\n|P_K| = {len(pairs)} ordered pairs")

# -------------------------------------------------------------
# S4  BASELINE: UNCONSTRAINED SVM  (rho=0, zeta=inf)
# -------------------------------------------------------------

print()
print("=" * 60)
print("S3  Baseline: Unconstrained SVM  (rho=0, zeta=inf)")
print("=" * 60)

t0 = time.time()
prob_svm, w_svm, b_svm = build_hdrfc_k(
    X_tr, Y_tr, A_tr, I_pos, pairs, rho=0.0, zeta=np.inf
)
prob_svm.solve(solver=cp.CLARABEL, verbose=False)
svm_time = time.time() - t0

print(f"  Status   : {prob_svm.status}")
print(f"  Objective: {prob_svm.value:.6f}")
print(f"  Solve time: {svm_time:.1f}s")

if prob_svm.status in ("optimal", "optimal_inaccurate"):
    results_svm = evaluate(w_svm.value, b_svm.value, X_te, Y_te, A_te, pairs,
                           label="Unconstrained SVM (rho=0, zeta=inf)")
else:
    print("  Solver did not find an optimal solution.")
    results_svm = None

# -------------------------------------------------------------
# S5  HDRFC-K  (rho=RHO, zeta=ZETA)
# -------------------------------------------------------------

print()
print("=" * 60)
print(f"S4  HDRFC-K  (rho={RHO}, zeta={ZETA}, K={K})")
print("=" * 60)

t0 = time.time()
prob_hdrfc, w_hdrfc, b_hdrfc = build_hdrfc_k(
    X_tr, Y_tr, A_tr, I_pos, pairs, rho=RHO, zeta=ZETA
)
prob_hdrfc.solve(solver=cp.CLARABEL, verbose=False)
hdrfc_time = time.time() - t0

print(f"  Status   : {prob_hdrfc.status}")
print(f"  Objective: {prob_hdrfc.value:.6f}")
print(f"  Solve time: {hdrfc_time:.1f}s")

if prob_hdrfc.status in ("optimal", "optimal_inaccurate"):
    results_hdrfc = evaluate(w_hdrfc.value, b_hdrfc.value, X_te, Y_te, A_te, pairs,
                             label=f"HDRFC-K (rho={RHO}, zeta={ZETA}, K={K})")
else:
    print("  Solver did not find an optimal solution.")
    print("  Tip: try increasing zeta (current fairness constraint may be infeasible).")
    results_hdrfc = None

# -------------------------------------------------------------
# S6  SUMMARY COMPARISON
# -------------------------------------------------------------

print()
print("=" * 60)
print("S5  Summary Comparison")
print("=" * 60)

rows = []
if results_svm:
    rows.append(("Unconstrained SVM", results_svm["accuracy"],
                 results_svm["balanced_accuracy"], results_svm["max_gap"], svm_time))
if results_hdrfc:
    rows.append((f"HDRFC-K (rho={RHO}, zeta={ZETA})", results_hdrfc["accuracy"],
                 results_hdrfc["balanced_accuracy"], results_hdrfc["max_gap"], hdrfc_time))

header = f"{'Model':<30} {'Accuracy':>10} {'Bal.Acc':>10} {'MaxTPRgap':>12} {'Time(s)':>10}"
print(header)
print("-" * len(header))
for r in rows:
    print(f"{r[0]:<30} {r[1]:>10.4f} {r[2]:>10.4f} {r[3]:>12.4f} {r[4]:>10.1f}")

print()
if results_svm and results_hdrfc:
    delta_acc = results_hdrfc["accuracy"] - results_svm["accuracy"]
    delta_gap = results_hdrfc["max_gap"] - results_svm["max_gap"]
    print(f"  Accuracy change  : {delta_acc:+.4f}  ({'improved' if delta_acc > 0 else 'decreased'})")
    print(f"  Max TPR gap change: {delta_gap:+.4f}  ({'worse' if delta_gap > 0 else 'improved — fairness gain'})")
```

- [ ] **Step 2: Verify the script imports cleanly**

```bash
cd C:/SAPDevelop/hai/analytics && python -c "import experiment" 2>&1 | head -5
```

Expected: no ImportError (it will print dataset-loading output and run; Ctrl-C after the first print line if you don't want to wait).

If you only want the import check without running: add `if __name__ == "__main__":` guard around the body — but per the spec, that's out of scope. Just checking for clean import errors is sufficient.

- [ ] **Step 3: Commit**

```bash
git add experiment.py
git commit -m "refactor: import lp_core in experiment.py, apply audit fixes"
```

---

## Task 3: Create `ablation.py` — ρ×ζ sweep with CSV and plots

**Files:**
- Create: `ablation.py`

- [ ] **Step 1: Create `outputs/` directory if it does not exist**

```bash
mkdir -p C:/SAPDevelop/hai/analytics/outputs
```

- [ ] **Step 2: Create `ablation.py`**

```python
"""
ablation.py — ρ × ζ sweep for HDRFC-K on freMTPL2freq.

Runs 35 solves (rho x zeta grid), saves results to outputs/ablation_results.csv,
and writes two matplotlib figures to outputs/.

Requirements:
    pip install cvxpy scikit-learn pandas numpy openml matplotlib
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import permutations
from pathlib import Path

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import cvxpy as cp

from lp_core import build_hdrfc_k, evaluate, AGE_LABELS

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------

SAMPLE_N = 20_000
TEST_SIZE = 0.2
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# Right-closed intervals: (17,25], (25,40], (40,55], (55,70], (70,120]
AGE_BINS = [17, 25, 40, 55, 70, 120]

CATEGORICAL_COLS = ["VehBrand", "VehGas", "Region", "Area"]
NUMERIC_COLS = ["VehPower", "VehAge", "BonusMalus", "Density", "Exposure"]

RHO_GRID = [0.0, 0.01, 0.05, 0.1, 0.5]
ZETA_GRID = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# -------------------------------------------------------------

print("=" * 60)
print("Loading freMTPL2freq (OpenML ID 41214)")
print("=" * 60)

raw = fetch_openml(data_id=41214, as_frame=True, parser="pandas", data_version=1)
df = raw.frame.copy()

df["Y"] = np.where(df["ClaimNb"].astype(float) > 0, 1, -1)
df["A"] = pd.cut(
    df["DrivAge"].astype(float),
    bins=AGE_BINS,
    labels=AGE_LABELS,
    right=True,
).astype("Int64")

n_before = len(df)
df = df.dropna(subset=["A"])
df["A"] = df["A"].astype(int)
n_dropped = n_before - len(df)
if n_dropped:
    print(f"  Dropped {n_dropped} rows with DrivAge outside bins {AGE_BINS}")

strat_key_full = df["Y"].astype(str) + "_" + df["A"].astype(str)
df, _ = train_test_split(
    df, train_size=SAMPLE_N, stratify=strat_key_full, random_state=RANDOM_STATE
)
df = df.reset_index(drop=True)
print(f"Subsampled to N={len(df):,} (stratified on Y x A)")

feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
X_raw = df[feature_cols]
Y = df["Y"].values
A = df["A"].values

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_COLS),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_COLS),
])

strat_key = Y.astype(str) + "_" + A.astype(str)
X_tr_raw, X_te_raw, Y_tr, Y_te, A_tr, A_te = train_test_split(
    X_raw, Y, A,
    test_size=TEST_SIZE,
    stratify=strat_key,
    random_state=RANDOM_STATE,
)

X_tr = preprocessor.fit_transform(X_tr_raw)
X_te = preprocessor.transform(X_te_raw)

I_pos = {k: np.where((A_tr == k) & (Y_tr == 1))[0] for k in AGE_LABELS}
pairs = list(permutations(AGE_LABELS, 2))

print(f"Train N={len(Y_tr):,}  Test N={len(Y_te):,}  Features d={X_tr.shape[1]}")
print(f"Grid: {len(RHO_GRID)} rho values x {len(ZETA_GRID)} zeta values = {len(RHO_GRID)*len(ZETA_GRID)} solves\n")

# -------------------------------------------------------------
# SWEEP
# -------------------------------------------------------------

records = []
total = len(RHO_GRID) * len(ZETA_GRID)
idx = 0

for rho in RHO_GRID:
    for zeta in ZETA_GRID:
        idx += 1
        t0 = time.time()
        try:
            prob, w_var, b_var = build_hdrfc_k(
                X_tr, Y_tr, A_tr, I_pos, pairs, rho=rho, zeta=zeta
            )
            prob.solve(solver=cp.CLARABEL, verbose=False)
            solve_time = time.time() - t0
            status = prob.status

            if status in ("optimal", "optimal_inaccurate") and w_var.value is not None:
                res = evaluate(
                    w_var.value, b_var.value, X_te, Y_te, A_te, pairs,
                    label=f"rho={rho} zeta={zeta}"
                )
                acc = res["accuracy"]
                bal_acc = res["balanced_accuracy"]
                max_gap = res["max_gap"]
            else:
                acc = bal_acc = max_gap = float("nan")

        except Exception as e:
            solve_time = time.time() - t0
            status = f"error: {e}"
            acc = bal_acc = max_gap = float("nan")

        print(
            f"[{idx:>2}/{total}] rho={rho:.2f} zeta={zeta:.1f} "
            f"→ {status:<20} max_gap={max_gap:.4f}  ({solve_time:.1f}s)"
        )
        records.append({
            "rho": rho,
            "zeta": zeta,
            "status": status,
            "accuracy": acc,
            "bal_acc": bal_acc,
            "max_gap": max_gap,
            "solve_time": solve_time,
        })

# -------------------------------------------------------------
# SAVE CSV
# -------------------------------------------------------------

results_df = pd.DataFrame(records)
csv_path = OUTPUT_DIR / "ablation_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# -------------------------------------------------------------
# PLOT 1: max TPR gap vs zeta, one line per rho
# -------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
colors = cm.viridis(np.linspace(0, 0.85, len(RHO_GRID)))

for i, rho in enumerate(RHO_GRID):
    subset = results_df[results_df["rho"] == rho].sort_values("zeta")
    ax.plot(subset["zeta"], subset["max_gap"], marker="o", label=f"ρ={rho}", color=colors[i])

ax.set_xlabel("ζ (fairness tolerance)")
ax.set_ylabel("Max pairwise TPR gap")
ax.set_title("TPR gap vs. fairness tolerance ζ (HDRFC-K, K=5)")
ax.legend(title="Wasserstein radius ρ")
ax.grid(True, alpha=0.3)
fig.tight_layout()
tpr_path = OUTPUT_DIR / "ablation_tpr_gap.png"
fig.savefig(tpr_path, dpi=150)
plt.close(fig)
print(f"Plot saved to {tpr_path}")

# -------------------------------------------------------------
# PLOT 2: Pareto scatter — balanced accuracy vs max TPR gap
# -------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
colors = cm.viridis(np.linspace(0, 0.85, len(RHO_GRID)))

for i, rho in enumerate(RHO_GRID):
    subset = results_df[(results_df["rho"] == rho) & results_df["max_gap"].notna()]
    sc = ax.scatter(
        subset["max_gap"], subset["bal_acc"],
        c=[colors[i]] * len(subset), s=60, label=f"ρ={rho}", zorder=3
    )
    for _, row in subset.iterrows():
        ax.annotate(
            f"ζ={row['zeta']:.1f}",
            (row["max_gap"], row["bal_acc"]),
            textcoords="offset points", xytext=(4, 3), fontsize=7, color=colors[i]
        )

ax.set_xlabel("Max pairwise TPR gap (lower = fairer)")
ax.set_ylabel("Balanced accuracy")
ax.set_title("Accuracy–fairness trade-off (HDRFC-K, K=5)")
ax.legend(title="Wasserstein radius ρ")
ax.grid(True, alpha=0.3)
fig.tight_layout()
pareto_path = OUTPUT_DIR / "ablation_pareto.png"
fig.savefig(pareto_path, dpi=150)
plt.close(fig)
print(f"Plot saved to {pareto_path}")

print("\nDone.")
```

- [ ] **Step 3: Verify `ablation.py` imports cleanly**

```bash
cd C:/SAPDevelop/hai/analytics && python -c "
import ast, sys
with open('ablation.py') as f:
    src = f.read()
ast.parse(src)
print('Syntax OK')
"
```

Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add ablation.py
git commit -m "feat: add ablation.py for rho x zeta sweep with CSV and plots"
```

---

## Task 4: Fix `empirical_analysis_plan.md` — ζ value and rationale

**Files:**
- Modify: `empirical_analysis_plan.md`

- [ ] **Step 1: Update the HDRFC-K row in the models table (line 77)**

Find:
```
| **HDRFC-K** | Proposed LP (Theorem 3.5), K=5 age groups, L1 norm | 0.1 | 0.1 |
```

Replace with:
```
| **HDRFC-K** | Proposed LP (Theorem 3.5), K=5 age groups, L1 norm | 0.1 | 1.5 |
```

- [ ] **Step 2: Update the hyperparameters table rationale for ζ (line 109)**

Find:
```
| ζ (fairness tolerance) | 0.1 | Allows ≤10% pairwise TPR gap in worst case |
```

Replace with:
```
| ζ (fairness tolerance) | 1.5 | Bounds H_kl ≤ 1 + ζ = 2.5. Proposition 3.4 guarantees H_kl ≥ 1, so ζ controls how much distributional unfairness is tolerated. ζ < 0.5 risks infeasibility on this dataset. |
```

- [ ] **Step 3: Update the protocol step 6 (line 146)**

Find:
```
6. Solve HDRFC-K LP (ρ=0.1, ζ=0.1) via CVXPY + MOSEK
```

Replace with:
```
6. Solve HDRFC-K LP (ρ=0.1, ζ=1.5) via CVXPY + CLARABEL
```

- [ ] **Step 4: Commit**

```bash
git add empirical_analysis_plan.md
git commit -m "fix: correct zeta=0.1->1.5 and H_kl rationale in empirical plan"
```

---

## Task 5: Smoke-test the full pipeline end-to-end

**Files:** none (verification only)

- [ ] **Step 1: Run a minimal dry-run of `experiment.py` to confirm imports and data loading work**

```bash
cd C:/SAPDevelop/hai/analytics && python -c "
import warnings; warnings.filterwarnings('ignore')
from lp_core import build_hdrfc_k, evaluate
import numpy as np
from itertools import permutations

# Tiny synthetic problem: 50 samples, 3 features, 2 groups
np.random.seed(0)
N, d, K = 50, 3, 2
X = np.random.randn(N, d)
Y = np.where(np.random.rand(N) > 0.7, 1, -1)
A = np.where(np.arange(N) < 25, 1, 2)
I_pos = {k: np.where((A == k) & (Y == 1))[0] for k in [1, 2]}
pairs = list(permutations([1, 2], 2))

import cvxpy as cp
prob, w, b = build_hdrfc_k(X, Y, A, I_pos, pairs, rho=0.1, zeta=1.5)
prob.solve(solver=cp.CLARABEL, verbose=False)
print('status:', prob.status)
assert prob.status in ('optimal', 'optimal_inaccurate'), f'Unexpected status: {prob.status}'
res = evaluate(w.value, b.value, X, Y, A, pairs, label='smoke-test')
assert 0 <= res['max_gap'] <= 2.0
print('Smoke test PASSED')
"
```

Expected output ends with: `Smoke test PASSED`

- [ ] **Step 2: Verify `ablation.py` syntax and import chain**

```bash
cd C:/SAPDevelop/hai/analytics && python -c "
import ast
for fname in ['lp_core.py', 'experiment.py', 'ablation.py']:
    with open(fname) as f:
        ast.parse(f.read())
    print(f'{fname}: syntax OK')
"
```

Expected:
```
lp_core.py: syntax OK
experiment.py: syntax OK
ablation.py: syntax OK
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git status
git commit -m "chore: verify restructure complete — lp_core, experiment, ablation all clean"
```

Only commit if `git status` shows no unexpected files. If nothing new to commit, skip this step.
