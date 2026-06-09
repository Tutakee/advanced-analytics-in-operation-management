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

raw = fetch_openml(data_id=41214, as_frame=True, parser="pandas")
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
            f"-> {status:<20} max_gap={max_gap:.4f}  ({solve_time:.1f}s)"
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
    ax.scatter(
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
