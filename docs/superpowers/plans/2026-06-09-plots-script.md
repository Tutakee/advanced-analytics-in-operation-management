# Plots Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `experiment.py` with a ζ-sweep and results save block, then create `plots.py` that generates 8 publication-ready figures saved to `figures/` with numeric prefixes, and produce a `results/` directory with prefixed output files documenting all run artifacts.

**Architecture:** `experiment.py` is extended to run a ζ-sweep after the main solve and save all results to `results/EXP_results.npz` and a human-readable summary to `results/EXP_summary.txt`. `plots.py` reloads raw data for EDA plots and reads `results/EXP_results.npz` for model result plots, saving all figures to `figures/` with numeric prefixes (`01_` through `08_`).

**Tech Stack:** Python 3.x, cvxpy, numpy, pandas, scikit-learn, openml, matplotlib, seaborn

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `experiment.py` | Modify | Add ζ-sweep loop + save block at end |
| `plots.py` | Create | 8 plot functions + `main()` dispatcher |
| `results/EXP_results.npz` | Generated | Machine-readable results for plots.py |
| `results/EXP_summary.txt` | Generated | Human-readable run summary |
| `figures/01_class_imbalance.png` | Generated | EDA plot |
| `figures/02_age_group_distribution.png` | Generated | EDA plot |
| `figures/03_positive_rate_per_group.png` | Generated | EDA plot |
| `figures/04_feature_correlation.png` | Generated | EDA plot |
| `figures/05_bonusmalus_by_group.png` | Generated | EDA plot |
| `figures/06_pareto_frontier.png` | Generated | Results plot |
| `figures/07_tpr_bar.png` | Generated | Results plot |
| `figures/08_tpr_heatmap.png` | Generated | Results plot |

**Prefix convention:**
- `figures/NN_<name>.png` — plots numbered 01–08
- `results/EXP_<name>.<ext>` — experiment outputs (EXP = experiment identifier prefix)

---

## Task 1: Add ζ-sweep to `experiment.py`

**Files:**
- Modify: `experiment.py` (after S6, before S7 summary section)

- [ ] **Step 1: Add ZETA_SWEEP constant at top of S1 config block**

In `experiment.py`, after the line `ZETA = 1.5`, add:

```python
ZETA_SWEEP = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0, np.inf]
```

- [ ] **Step 2: Add sweep loop after S6 HDRFC-K solve block**

Insert the following section between S6 and S7:

```python
# -------------------------------------------------------------
# S6b  ZETA SWEEP  (Pareto frontier data)
# -------------------------------------------------------------

print()
print("=" * 60)
print("S6b  Zeta sweep for Pareto frontier")
print("=" * 60)

sweep_zetas = []
sweep_acc = []
sweep_gap = []

for zeta_val in ZETA_SWEEP:
    t0 = time.time()
    prob_s, w_s, b_s = build_hdrfc_k(
        X_tr, Y_tr, A_tr, I_pos, pairs, rho=RHO, zeta=zeta_val
    )
    prob_s.solve(solver=cp.CLARABEL, verbose=False)
    elapsed = time.time() - t0

    zeta_label = f"{zeta_val:.2f}" if zeta_val != np.inf else "inf"
    if prob_s.status in ("optimal", "optimal_inaccurate"):
        scores_s = X_te @ w_s.value + b_s.value
        y_hat_s = np.sign(scores_s)
        y_hat_s[y_hat_s == 0] = 1
        acc_s = np.mean(y_hat_s == Y_te)
        tpr_s = {}
        for k in AGE_LABELS:
            mask = (A_te == k) & (Y_te == 1)
            tpr_s[k] = np.mean(y_hat_s[mask] == 1) if mask.sum() > 0 else float("nan")
        gap_s = max(
            abs(tpr_s[k] - tpr_s[l])
            for (k, l) in pairs
            if not np.isnan(tpr_s[k]) and not np.isnan(tpr_s[l])
        )
        sweep_zetas.append(zeta_val if zeta_val != np.inf else 999.0)
        sweep_acc.append(acc_s)
        sweep_gap.append(gap_s)
        print(f"  zeta={zeta_label:>5}  acc={acc_s:.4f}  max_gap={gap_s:.4f}  time={elapsed:.1f}s")
    else:
        print(f"  zeta={zeta_label:>5}  status={prob_s.status} (skipped)")
```

- [ ] **Step 3: Commit**

```bash
git add experiment.py
git commit -m "feat: add zeta sweep loop to experiment.py"
```

---

## Task 2: Add results save block to `experiment.py`

**Files:**
- Modify: `experiment.py` (append after S7 summary section)

- [ ] **Step 1: Create results/ directory and add save block**

Append the following section at the very end of `experiment.py`:

```python
# -------------------------------------------------------------
# S8  SAVE RESULTS
# -------------------------------------------------------------

import os
os.makedirs("results", exist_ok=True)

print()
print("=" * 60)
print("S8  Saving results to results/")
print("=" * 60)

save_dict = {}

if results_svm:
    save_dict["svm_tpr"] = np.array([results_svm["tpr"][k] for k in AGE_LABELS])
    save_dict["svm_acc"] = results_svm["accuracy"]
    save_dict["svm_bal_acc"] = results_svm["balanced_accuracy"]
    save_dict["svm_max_gap"] = results_svm["max_gap"]

if results_hdrfc:
    save_dict["hdrfc_tpr"] = np.array([results_hdrfc["tpr"][k] for k in AGE_LABELS])
    save_dict["hdrfc_acc"] = results_hdrfc["accuracy"]
    save_dict["hdrfc_bal_acc"] = results_hdrfc["balanced_accuracy"]
    save_dict["hdrfc_max_gap"] = results_hdrfc["max_gap"]

if sweep_zetas:
    save_dict["zeta_sweep_zetas"] = np.array(sweep_zetas)
    save_dict["zeta_sweep_acc"] = np.array(sweep_acc)
    save_dict["zeta_sweep_gap"] = np.array(sweep_gap)

np.savez("results/EXP_results.npz", **save_dict)
print("  Saved: results/EXP_results.npz")

# Human-readable summary
with open("results/EXP_summary.txt", "w") as f:
    f.write("HDRFC-K Experiment Summary\n")
    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"RHO={RHO}  ZETA={ZETA}  K={K}  SAMPLE_N={SAMPLE_N}\n\n")
    if results_svm:
        f.write(f"[SVM]   acc={results_svm['accuracy']:.4f}  bal_acc={results_svm['balanced_accuracy']:.4f}  max_gap={results_svm['max_gap']:.4f}  time={svm_time:.1f}s\n")
    if results_hdrfc:
        f.write(f"[HDRFC] acc={results_hdrfc['accuracy']:.4f}  bal_acc={results_hdrfc['balanced_accuracy']:.4f}  max_gap={results_hdrfc['max_gap']:.4f}  time={hdrfc_time:.1f}s\n")
    f.write("\nZeta sweep:\n")
    for z, a, g in zip(sweep_zetas, sweep_acc, sweep_gap):
        z_label = "inf" if z == 999.0 else f"{z:.2f}"
        f.write(f"  zeta={z_label:>5}  acc={a:.4f}  max_gap={g:.4f}\n")

print("  Saved: results/EXP_summary.txt")
```

- [ ] **Step 2: Verify the save block runs**

```bash
python experiment.py
```

Expected: last lines include `Saved: results/EXP_results.npz` and `Saved: results/EXP_summary.txt`. Both files appear in `results/`.

- [ ] **Step 3: Commit**

```bash
git add experiment.py results/EXP_summary.txt
git commit -m "feat: save experiment results to results/ with EXP_ prefix"
```

---

## Task 3: Create `plots.py` — scaffold + data loading

**Files:**
- Create: `plots.py`

- [ ] **Step 1: Create `plots.py` with imports, constants, and data loader**

```python
"""
plots.py — Generate 8 publication figures for HDRFC-K paper.

EDA plots (1-5): reload freMTPL2freq from OpenML.
Results plots (6-8): load results/EXP_results.npz saved by experiment.py.

Outputs saved to figures/ with numeric prefixes (01_ through 08_).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")
sns.set_theme()

FIGURES_DIR = "figures"
RESULTS_FILE = "results/EXP_results.npz"

AGE_BINS = [17, 25, 40, 55, 70, 120]
AGE_LABELS = [1, 2, 3, 4, 5]
AGE_NAMES = ["18-25", "26-40", "41-55", "56-70", "71+"]

CATEGORICAL_COLS = ["VehBrand", "VehGas", "Region", "Area"]
NUMERIC_COLS = ["VehPower", "VehAge", "BonusMalus", "Density", "Exposure"]
SAMPLE_N = 20_000
RANDOM_STATE = 42


def load_data():
    """Load and preprocess freMTPL2freq, return full df and numeric feature DataFrame."""
    from sklearn.model_selection import train_test_split

    raw = fetch_openml(data_id=41214, as_frame=True, parser="pandas")
    df = raw.frame.copy()
    df["Y"] = np.where(df["ClaimNb"].astype(float) > 0, 1, -1)
    df["A"] = pd.cut(
        df["DrivAge"].astype(float),
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
    ).astype(int)
    df = df.dropna(subset=["A"])

    strat_key = df["Y"].astype(str) + "_" + df["A"].astype(str)
    df, _ = train_test_split(
        df, train_size=SAMPLE_N, stratify=strat_key, random_state=RANDOM_STATE
    )
    df = df.reset_index(drop=True)

    X_df = df[NUMERIC_COLS].copy()
    return df, X_df


def load_results():
    """Load results/EXP_results.npz as a dict."""
    data = np.load(RESULTS_FILE, allow_pickle=False)
    return dict(data)


def save_fig(fig, filename):
    """Save figure to figures/ and print confirmation."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
```

- [ ] **Step 2: Commit scaffold**

```bash
git add plots.py
git commit -m "feat: add plots.py scaffold with data loader and save helper"
```

---

## Task 4: EDA plots 1–3 (class imbalance, age distribution, positive rate)

**Files:**
- Modify: `plots.py`

- [ ] **Step 1: Add plot functions 1–3**

Append to `plots.py`:

```python
def plot_class_imbalance(df):
    counts = df["Y"].value_counts().sort_index()
    labels = ["Negative (Y=-1)", "Positive (Y=+1)"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, [counts[-1], counts[1]], color=["steelblue", "tomato"])
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    for i, v in enumerate([counts[-1], counts[1]]):
        ax.text(i, v + 50, f"{v:,}\n({100*v/len(df):.1f}%)", ha="center", fontsize=9)
    save_fig(fig, "01_class_imbalance.png")


def plot_age_group_distribution(df):
    counts = [( df["A"] == k).sum() for k in AGE_LABELS]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(AGE_NAMES, counts, color="steelblue")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Count")
    ax.set_title("Age Group Distribution")
    for i, v in enumerate(counts):
        ax.text(i, v + 30, f"{v:,}", ha="center", fontsize=9)
    save_fig(fig, "02_age_group_distribution.png")


def plot_positive_rate_per_group(df):
    rates = [((df["A"] == k) & (df["Y"] == 1)).sum() / (df["A"] == k).sum()
             for k in AGE_LABELS]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(AGE_NAMES, rates, color="tomato")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("P(Y=1 | A=k)")
    ax.set_title("Positive Rate per Age Group")
    ax.set_ylim(0, max(rates) * 1.2)
    for i, v in enumerate(rates):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    save_fig(fig, "03_positive_rate_per_group.png")
```

- [ ] **Step 2: Commit**

```bash
git add plots.py
git commit -m "feat: add EDA plots 1-3 (class imbalance, age distribution, positive rate)"
```

---

## Task 5: EDA plots 4–5 (feature correlation, BonusMalus by group)

**Files:**
- Modify: `plots.py`

- [ ] **Step 1: Add plot functions 4–5**

Append to `plots.py`:

```python
def plot_feature_correlation(X_df):
    corr = X_df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, linewidths=0.5, ax=ax
    )
    ax.set_title("Feature Correlation Matrix (Numeric Features)")
    save_fig(fig, "04_feature_correlation.png")


def plot_bonusmalus_by_group(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [df.loc[df["A"] == k, "BonusMalus"].values for k in AGE_LABELS]
    ax.violinplot(data, positions=range(1, 6), showmedians=True)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(AGE_NAMES)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("BonusMalus")
    ax.set_title("BonusMalus Distribution by Age Group")
    save_fig(fig, "05_bonusmalus_by_group.png")
```

- [ ] **Step 2: Commit**

```bash
git add plots.py
git commit -m "feat: add EDA plots 4-5 (feature correlation, BonusMalus violin)"
```

---

## Task 6: Results plots 6–8 (Pareto frontier, TPR bar, TPR heatmap)

**Files:**
- Modify: `plots.py`

- [ ] **Step 1: Add plot functions 6–8**

Append to `plots.py`:

```python
def plot_pareto_frontier(results):
    zetas = results["zeta_sweep_zetas"]
    accs = results["zeta_sweep_acc"]
    gaps = results["zeta_sweep_gap"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(gaps, accs, c=np.where(zetas == 999.0, np.inf, zetas),
                    cmap="viridis", s=80, zorder=3)
    for z, g, a in zip(zetas, gaps, accs):
        label = "∞" if z == 999.0 else f"{z:.2f}"
        ax.annotate(f"ζ={label}", (g, a), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)

    if "svm_acc" in results and "svm_max_gap" in results:
        ax.scatter(results["svm_max_gap"], results["svm_acc"],
                   marker="*", s=200, color="red", zorder=4, label="SVM baseline")
        ax.legend()

    ax.set_xlabel("Max Pairwise TPR Gap")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto Frontier: Accuracy vs. Fairness (ζ sweep)")
    plt.colorbar(sc, ax=ax, label="ζ value")
    save_fig(fig, "06_pareto_frontier.png")


def plot_tpr_bar(results):
    svm_tpr = results["svm_tpr"]
    hdrfc_tpr = results["hdrfc_tpr"]
    x = np.arange(len(AGE_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, svm_tpr, width, label="SVM (unconstrained)", color="steelblue")
    ax.bar(x + width/2, hdrfc_tpr, width, label=f"HDRFC-K", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_NAMES)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("TPR")
    ax.set_ylim(0, 1.1)
    ax.set_title("True Positive Rate per Age Group: SVM vs. HDRFC-K")
    ax.legend()
    save_fig(fig, "07_tpr_bar.png")


def plot_tpr_heatmap(results):
    for model_key, label, fname in [
        ("svm_tpr", "SVM (unconstrained)", "08a_tpr_heatmap_svm.png"),
        ("hdrfc_tpr", "HDRFC-K", "08b_tpr_heatmap_hdrfc.png"),
    ]:
        tpr = results[model_key]
        matrix = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                matrix[i, j] = tpr[i] - tpr[j] if i != j else np.nan

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            matrix, annot=True, fmt=".3f", cmap="RdBu", center=0,
            xticklabels=AGE_NAMES, yticklabels=AGE_NAMES,
            linewidths=0.5, ax=ax, mask=np.eye(5, dtype=bool)
        )
        ax.set_title(f"Pairwise TPR Difference Matrix — {label}\n(row k − col l = TPR_k − TPR_l)")
        ax.set_xlabel("Group l")
        ax.set_ylabel("Group k")
        save_fig(fig, fname)
```

Note: plot 8 produces two files (`08a_` for SVM, `08b_` for HDRFC-K) so both models can be compared side-by-side in the paper.

- [ ] **Step 2: Commit**

```bash
git add plots.py
git commit -m "feat: add results plots 6-8 (Pareto frontier, TPR bar, TPR heatmaps)"
```

---

## Task 7: Add `main()` and wire everything together

**Files:**
- Modify: `plots.py`

- [ ] **Step 1: Append `main()` to `plots.py`**

```python
def main():
    print("=" * 60)
    print("Loading freMTPL2freq for EDA plots...")
    print("=" * 60)
    df, X_df = load_data()
    print(f"  N={len(df):,} loaded")

    print()
    print("=" * 60)
    print("EDA Plots (1-5)")
    print("=" * 60)
    plot_class_imbalance(df)
    plot_age_group_distribution(df)
    plot_positive_rate_per_group(df)
    plot_feature_correlation(X_df)
    plot_bonusmalus_by_group(df)

    print()
    print("=" * 60)
    print(f"Loading {RESULTS_FILE} for results plots...")
    print("=" * 60)
    results = load_results()

    print()
    print("=" * 60)
    print("Results Plots (6-8)")
    print("=" * 60)
    plot_pareto_frontier(results)
    plot_tpr_bar(results)
    plot_tpr_heatmap(results)

    print()
    print("All figures saved to figures/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run `plots.py` end-to-end**

```bash
python plots.py
```

Expected output ends with:
```
  Saved: figures/08b_tpr_heatmap_hdrfc.png

All figures saved to figures/
```

Verify all 9 files exist (01–05 EDA, 06 Pareto, 07 TPR bar, 08a SVM heatmap, 08b HDRFC heatmap):

```bash
ls figures/
```

- [ ] **Step 3: Commit**

```bash
git add plots.py figures/
git commit -m "feat: wire main() in plots.py, generate all 8 figures"
```

---

## Task 8: Document outputs with a run README

**Files:**
- Create: `results/EXP_README.md`

- [ ] **Step 1: Create `results/EXP_README.md`**

```markdown
# Experiment Outputs

All files in this directory are generated by `experiment.py`.  
Prefix convention: `EXP_` = experiment output.

| File | Description |
|---|---|
| `EXP_results.npz` | NumPy archive: model metrics, TPR arrays, ζ-sweep data |
| `EXP_summary.txt` | Human-readable run summary (accuracy, balanced accuracy, max TPR gap, solve times) |

## figures/ directory

All files generated by `plots.py`.  
Prefix convention: `NN_` = plot number (01–08).

| File | Description |
|---|---|
| `01_class_imbalance.png` | Class distribution bar chart |
| `02_age_group_distribution.png` | Age group sample counts |
| `03_positive_rate_per_group.png` | P(Y=1 \| A=k) per age group |
| `04_feature_correlation.png` | Numeric feature correlation heatmap |
| `05_bonusmalus_by_group.png` | BonusMalus violin plot by age group |
| `06_pareto_frontier.png` | Accuracy vs. max TPR gap across ζ sweep |
| `07_tpr_bar.png` | TPR per age group: SVM vs. HDRFC-K |
| `08a_tpr_heatmap_svm.png` | Pairwise TPR difference matrix — SVM |
| `08b_tpr_heatmap_hdrfc.png` | Pairwise TPR difference matrix — HDRFC-K |

## Reproducing results

1. Run `python experiment.py` — generates `results/EXP_results.npz` and `results/EXP_summary.txt`
2. Run `python plots.py` — generates all figures in `figures/`
```

- [ ] **Step 2: Commit**

```bash
git add results/EXP_README.md
git commit -m "docs: add EXP_README.md documenting all output files and prefixes"
```

---

## Self-Review

**Spec coverage:**
- ζ-sweep → Task 1 ✓
- `results.npz` save → Task 2 ✓
- All 8 plot functions → Tasks 4, 5, 6 ✓
- `main()` dispatcher → Task 7 ✓
- `figures/` with numeric prefixes → Tasks 4–6 ✓
- `results/` with `EXP_` prefix → Task 2 ✓
- Documentation of outputs → Task 8 ✓

**Placeholder scan:** None found.

**Type consistency:**
- `results` dict keys (`svm_tpr`, `hdrfc_tpr`, `zeta_sweep_zetas`, etc.) defined in Task 2 and consumed in Tasks 6–7 — consistent throughout.
- `save_fig(fig, filename)` defined in Task 3, used identically in Tasks 4–6.
- `AGE_LABELS`, `AGE_NAMES` defined once in Task 3 scaffold, reused in Tasks 4–6 — consistent.
