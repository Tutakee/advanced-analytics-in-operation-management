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

from lp_core import build_hdrfc_k, evaluate, AGE_LABELS

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

CATEGORICAL_COLS = ["VehBrand", "VehGas", "Region", "Area"]
NUMERIC_COLS = ["VehPower", "VehAge", "BonusMalus", "Density", "Exposure"]

# -------------------------------------------------------------
# S2  DATA LOADING & PREPROCESSING
# -------------------------------------------------------------

print("=" * 60)
print("S1  Loading freMTPL2freq (OpenML ID 41214)")
print("=" * 60)

raw = fetch_openml(data_id=41214, as_frame=True, parser="pandas")
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
