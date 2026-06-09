# Plots Script Design
**Date:** 2026-06-09  
**Project:** HDRFC-K Empirical Analysis — freMTPL2freq  

---

## Overview

Two files are modified/created:

- **`experiment.py`** — extended with a ζ-sweep loop and a `results.npz` save block at the end
- **`plots.py`** — standalone script, function-per-plot architecture, `main()` dispatcher, saves 8 PNGs to `figures/`

---

## Data Sources

`plots.py` draws from two sources:

| Source | Used for |
|---|---|
| OpenML freMTPL2freq (reloaded) | EDA plots 1–5 |
| `results.npz` (saved by `experiment.py`) | Results plots 6–8 |

---

## Changes to `experiment.py`

### ζ-sweep (Pareto frontier data)

Add a sweep loop after the main HDRFC-K solve:

```python
ZETA_SWEEP = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0, np.inf]
```

For each ζ, solve HDRFC-K and record accuracy and max TPR gap.

### Save block (`results.npz`)

Keys saved:

| Key | Type | Description |
|---|---|---|
| `svm_tpr` | array (5,) | TPR per age group — SVM |
| `hdrfc_tpr` | array (5,) | TPR per age group — HDRFC-K |
| `svm_acc` | scalar | Accuracy — SVM |
| `svm_bal_acc` | scalar | Balanced accuracy — SVM |
| `svm_max_gap` | scalar | Max pairwise TPR gap — SVM |
| `hdrfc_acc` | scalar | Accuracy — HDRFC-K |
| `hdrfc_bal_acc` | scalar | Balanced accuracy — HDRFC-K |
| `hdrfc_max_gap` | scalar | Max pairwise TPR gap — HDRFC-K |
| `zeta_sweep_zetas` | array (8,) | ζ values tested |
| `zeta_sweep_acc` | array (8,) | Accuracy at each ζ |
| `zeta_sweep_gap` | array (8,) | Max TPR gap at each ζ |

---

## `plots.py` Architecture

### Style
- `seaborn` default theme (`sns.set_theme()`)
- All plots saved as PNG to `figures/` (created automatically if missing)

### Functions

| # | Function | Data source | Output |
|---|---|---|---|
| 1 | `plot_class_imbalance(df)` | raw data | `figures/01_class_imbalance.png` |
| 2 | `plot_age_group_distribution(df)` | raw data | `figures/02_age_group_distribution.png` |
| 3 | `plot_positive_rate_per_group(df)` | raw data | `figures/03_positive_rate_per_group.png` |
| 4 | `plot_feature_correlation(X_df)` | raw data | `figures/04_feature_correlation.png` |
| 5 | `plot_bonusmalus_by_group(df)` | raw data | `figures/05_bonusmalus_by_group.png` |
| 6 | `plot_pareto_frontier(results)` | results.npz | `figures/06_pareto_frontier.png` |
| 7 | `plot_tpr_bar(results)` | results.npz | `figures/07_tpr_bar.png` |
| 8 | `plot_tpr_heatmap(results)` | results.npz | `figures/08_tpr_heatmap.png` |

### `main()` flow

1. Load and preprocess freMTPL2freq (same logic as `experiment.py`)
2. Load `results.npz`
3. Call all 8 plot functions in order
4. Print confirmation of saved files

---

## Out of Scope

- PDF export
- Interactive plots
- Hyperparameter sweep beyond ζ
