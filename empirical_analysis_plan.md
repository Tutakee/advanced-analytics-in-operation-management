# Empirical Analysis Plan: HDRFC-K on freMTPL2freq

**Paper:** Fairness in Distributionally Robust Learning  
**Model:** HDRFC-K — multi-group equal-opportunity Wasserstein robust classifier (Theorem 3.5)  
**Dataset:** freMTPL2freq, French Motor Third-Party Liability (OpenML ID 41214, N ≈ 678,013)

---

## 1. Research Question

> How can distributionally robust classification be extended to enforce fairness across multiple age groups in motor insurance pricing?

The experiment validates the tractability and fairness effectiveness of the proposed HDRFC-K LP reformulation compared to an unconstrained SVM baseline.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | OpenML ID 41214 (`freMTPL2freq`) |
| N (policies) | ~678,013 |
| Positive class (Y=+1) | `ClaimNb > 0` (claim occurred) |
| Negative class (Y=−1) | `ClaimNb == 0` (no claim) |
| Sensitive attribute | `Driverage` → K=5 age cohorts |
| Sensitive attribute use | Training only (excluded from feature vector `x` at prediction time) |

### 2.1 Label Construction

```
Y = +1  if ClaimNb > 0
Y = -1  otherwise
```

### 2.2 Feature Matrix

**Included features** (all except sensitive/target/ID):

| Feature | Type | Preprocessing |
|---|---|---|
| `VehPower` | Ordinal int | Standardize |
| `VehAge` | Int | Standardize |
| `BonusMalus` | Int | Standardize |
| `Density` | Int | Standardize |
| `Exposure` | Float | Standardize |
| `VehBrand` | Categorical | One-hot encode |
| `VehGas` | Categorical | One-hot encode |
| `Region` | Categorical | One-hot encode |
| `Area` | Categorical | One-hot encode |

**Excluded:** `IDpol` (identifier), `ClaimNb` (target), `Driverage` (sensitive attribute — training-time only)

### 2.3 Age Group Assignment

| Group | Age Range | A value |
|---|---|---|
| Group 1 | 18–25 | 1 |
| Group 2 | 26–40 | 2 |
| Group 3 | 41–55 | 3 |
| Group 4 | 56–70 | 4 |
| Group 5 | 71+ | 5 |

### 2.4 Train/Test Split

- **Stratified subsample:** 20,000 rows drawn from the full 678k, stratified on `(Y, A)`.  
  Rationale: HDRFC-K LP has O(K²·N) variables and constraints. At N=678k and K=5 that yields ~13.6M λ variables — intractable on a standard machine. N=20k gives ~400k variables and solves in minutes with MOSEK.
- Stratified 80/20 split on `(Y, A)` applied to the subsample
- Ensures each age group × label combination is proportionally represented in both sets

---

## 3. Models

| Model | Description | ρ | ζ |
|---|---|---|---|
| **Unconstrained SVM** | Hinge-loss SVM, no fairness constraint | 0 | ∞ |
| **HDRFC-K** | Proposed LP (Theorem 3.5), K=5 age groups, L1 norm | 0.1 | 1.5 |

Both models use the same feature matrix and train/test split.

### 3.1 HDRFC-K LP Formulation (Theorem 3.5)

Variables: `w ∈ ℝᵈ`, `b ∈ ℝ`, `t ∈ ℝ₊ᴺ`, `λ^{kl} ∈ ℝ₊ᴺ` for all 20 ordered pairs (k,l) ∈ P₅

**Objective:**
```
min  (1/N) Σᵢ tᵢ
```

**Constraints:**
```
(7)  -yᵢ(wᵀxᵢ + b) + ρ‖w‖₁ ≤ tᵢ - 1         ∀i ∈ [N]

(8)  (1/|Ik₊|) Σ_{i ∈ Ik₊} λᵢᵏˡ
   + (1/|Il₊|) Σ_{i ∈ Il₊} λᵢᵏˡ - 1 ≤ ζ      ∀(k,l) ∈ P₅

(9)  1 + wᵀxᵢ + ρ‖w‖₁ + b ≤ λᵢᵏˡ             ∀i ∈ Ik₊, ∀(k,l) ∈ P₅

(10) 1 - wᵀxᵢ + ρ‖w‖₁ - b ≤ λᵢᵏˡ             ∀i ∈ Il₊, ∀(k,l) ∈ P₅
```

**Norm choice:** L1 (`‖w‖₁`) → pure LP (no SOCP), solved with MOSEK via CVXPY.

### 3.2 Hyperparameters (fixed for prototype)

| Parameter | Value | Rationale |
|---|---|---|
| ρ (Wasserstein radius) | 0.1 | Moderate robustness; ρ=0 recovers empirical ERM |
| ζ (fairness tolerance) | 1.5 | Bounds H_kl ≤ 1 + ζ = 2.5. Proposition 3.4 guarantees H_kl ≥ 1, so ζ controls how much distributional unfairness is tolerated. ζ < 0.5 risks infeasibility on this dataset. |
| Norm | L1 | Keeps problem a pure LP; MOSEK handles it efficiently |
| K | 5 | Five age cohorts as defined in Section 3.1 of the paper |

---

## 4. Evaluation Metrics

### 4.1 Predictive Performance

| Metric | Definition |
|---|---|
| Accuracy | `P(Ŷ = Y)` on test set |
| Balanced accuracy | Average of TPR and TNR |

### 4.2 Fairness Metrics

| Metric | Definition |
|---|---|
| TPR per group | `P(Ŷ=1 \| Y=1, A=k)` for k ∈ {1,...,5} |
| Max pairwise TPR gap | `max_{(k,l)} \|TPR_k - TPR_l\|` |
| Pairwise TPR difference matrix | 5×5 matrix of `TPR_k - TPR_l` |

### 4.3 Solver Diagnostics

- Solve status (optimal / infeasible / timeout)
- Wall-clock solve time

---

## 5. Experimental Protocol

1. Load freMTPL2freq from OpenML
2. Construct label Y, assign age groups A, build feature matrix X
3. Print dataset statistics (N, class balance, group sizes)
4. Stratified 80/20 split
5. Solve Unconstrained SVM (ζ=∞, ρ=0)
6. Solve HDRFC-K LP (ρ=0.1, ζ=1.5) via CVXPY + CLARABEL
7. Evaluate both models on test set
8. Print side-by-side comparison table of all metrics
9. Print 5×5 pairwise TPR difference matrix for each model

---

## 6. Expected Outcomes

- HDRFC-K should reduce the max pairwise TPR gap vs. unconstrained SVM (fairness improvement)
- Some accuracy reduction is expected (fairness-accuracy trade-off)
- LP solve should complete in reasonable time given sparse index sets Ik₊
- Solver status should be "optimal"; infeasibility would indicate ζ is set too tight

---

## 7. Open Items (to address in full paper experiments)

- [ ] Hyperparameter sweep: ρ ∈ {0, 0.01, 0.05, 0.1, 0.5}, ζ ∈ {0, 0.05, 0.1, 0.2, 0.5}
- [ ] Binary HDRFC baseline (Wang et al. 2024) for direct comparison
- [ ] L2 norm sensitivity check (SOCP formulation)
- [ ] Pareto frontier plot: accuracy vs. max pairwise TPR gap
- [ ] Cross-validation for ρ selection
