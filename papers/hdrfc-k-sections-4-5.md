# 4. Experiments

We evaluate the multi-group Wasserstein distributionally robust fair classifier (HDRFC-K) of Theorem 3.5 on a real-world insurance pricing dataset, in which the sensitive attribute is driver age partitioned into $K=5$ groups. The experimental design is intentionally conservative: a single representative dataset, a transparent linear-program implementation, and a head-to-head comparison against the unconstrained Wasserstein support-vector machine recovered from the same LP by setting $\rho = 0$ and $\zeta = \infty$. Our objective is to test whether the multi-group equal-opportunity constraints (8)–(10) materially reduce true positive rate (TPR) disparity at acceptable cost to predictive accuracy.

## 4.1 Dataset and preprocessing

We use the **freMTPL2freq** dataset (French Motor Third-Party Liability claim frequency, OpenML ID 41214), a public benchmark in actuarial science containing $678{,}013$ one-year insurance policies. The binary outcome is $y_i = +1$ if at least one claim was filed during the policy year and $y_i = -1$ otherwise. Because the LP in Theorem 3.5 has $\mathcal{O}(N + \sum_{(k,l)} |I_k^+|+|I_l^+|)$ variables and constraints, we draw a stratified subsample of $N = 20{,}000$ policies, stratifying on the joint distribution of $Y$ and the sensitive attribute $A$ to preserve the empirical class-by-group composition. The subsample is split $80\%/20\%$ into a training set ($N_\mathrm{tr} = 16{,}000$) and a held-out test set ($N_\mathrm{te} = 4{,}000$); the positive rate is $5.0\%$ ($1{,}004$ positives out of $20{,}000$).

The sensitive attribute $A_i \in \{1,\ldots,5\}$ is constructed by binning **driver age** into the intervals $[18,25]$, $[26,40]$, $[41,55]$, $[56,70]$, and $[71,\infty)$. Driver age itself is **excluded** from the feature vector $\mathbf{x}_i$ at training and test time, so that the classifier cannot trivially condition on the protected variable. After one-hot encoding the four categorical regressors (vehicle brand, vehicle gas, region, area code) and standardising the five numeric regressors (vehicle power, vehicle age, bonus-malus, density, exposure), the feature dimension is $d = 42$. Table 1 summarises the per-group sample sizes and base rates on the training fold.

**Table 1.** Age-group composition of the training fold ($N_\mathrm{tr} = 16{,}000$).

| Group $k$ | Age range | $N_k$ | $|I_k^+|$ | Positive rate |
|:--:|:--:|--:|--:|--:|
| 1 | 18–25 | 918 | 62 | 6.7% |
| 2 | 26–40 | 5{,}562 | 238 | 4.3% |
| 3 | 41–55 | 5{,}799 | 305 | 5.3% |
| 4 | 56–70 | 2{,}792 | 141 | 5.0% |
| 5 | 71+ | 929 | 58 | 6.3% |

The composition exhibits the classical insurance pattern: the youngest and oldest drivers exhibit the highest claim frequency, while middle-aged drivers (group 2) have the lowest. Because group sizes vary by nearly a factor of seven, naive empirical risk minimisation is dominated by the two large middle groups, motivating the per-group normalisation $1/|I_k^+|$ inside constraint (8).

## 4.2 Models and baselines

We compare two estimators that share the same LP backbone, isolating the effect of the multi-group fairness constraints.

**Unconstrained Wasserstein SVM ($\rho = 0$, $\zeta = \infty$).** Setting $\rho = 0$ removes the distributional-robustness penalty $\rho\|\mathbf{w}\|_1$ from constraints (7), (9), and (10), and sending $\zeta \to \infty$ deactivates the fairness constraint (8). The LP reduces to a class-weighted hinge-loss linear classifier and serves as our predictive-performance baseline.

**HDRFC-K ($\rho = 0.1$, $\zeta = 1.5$).** This is the full multi-group estimator of Theorem 3.5 with Wasserstein radius $\rho = 0.1$ and fairness tolerance $\zeta = 1.5$. By Proposition 3.4, $H_{kl} \geq 1$ for every classifier, so $\zeta = 1.5$ enforces the ratio bound $H_{kl} \leq 2.5$ for all $K(K-1) = 20$ ordered pairs $(k,l) \in \mathcal{P}_K$. The choice $\zeta = 1.5$ is moderate; we leave a systematic sweep over $(\rho, \zeta)$ to future work.

Both estimators use the **L1** norm on the feature space, which preserves the problem as a pure linear program; an L2 ball would lift the formulation to a second-order cone program and is left for future work. Both estimators use **inverse-frequency class weights** $c_i = N/(2N_+)$ for $y_i = +1$ and $c_i = N/(2N_-)$ for $y_i = -1$ in the objective. This deviates from the uniform-weight objective stated in Theorem 3.5 but is necessary in practice because the $5\%$ positive base rate causes the unweighted LP to collapse to the all-negative classifier $\mathbf{w} = \mathbf{0}$, $b < 0$. The class weights enter only the objective; constraints (7)–(10) are unchanged, and all dual and structural results from Section 3 carry over verbatim.

A second implementation choice concerns the dual variables $\boldsymbol{\lambda}^{kl}$. A direct reading of Theorem 3.5 would suggest one variable per training point per pair, yielding $\sum_{(k,l)} N$ entries. Because (9) and (10) involve only indices in $I_k^+$ and $I_l^+$, we instantiate $\boldsymbol{\lambda}^{kl}$ with dimension $|I_k^+| + |I_l^+|$ — this is exact, not an approximation, and reduces the count of fairness-block variables from approximately $336{,}000$ to approximately $63{,}000$ on our training fold.

The LPs are formulated in CVXPY and solved with **CLARABEL**. Both models reach `optimal` solver status. End-to-end solve times are measured on a single workstation and reported with the predictive results; they should be interpreted as orders of magnitude rather than calibrated benchmarks.

## 4.3 Evaluation protocol and metrics

All metrics are computed on the held-out test fold of $4{,}000$ policies, using the trained $(\mathbf{w}, b)$ to assign $\hat{y}_i = \mathrm{sign}(\mathbf{w}^\top \mathbf{x}_i + b)$. We report three families of quantities:

1. **Predictive performance.** Plain classification accuracy $\frac{1}{N_\mathrm{te}}\sum_i \mathbb{1}\{\hat{y}_i = y_i\}$ and balanced accuracy $\tfrac{1}{2}(\mathrm{TPR} + \mathrm{TNR})$, the latter being more informative under the $5\%$ positive base rate.
2. **Per-group equal-opportunity statistics.** The group-conditional true positive rate $\mathrm{TPR}_k = \mathbb{P}(\hat{y} = +1 \mid y = +1, A = k)$ for each $k \in \{1,\ldots,5\}$, the maximum pairwise TPR gap $\Delta := \max_{k,l} |\mathrm{TPR}_k - \mathrm{TPR}_l|$, and the full $K \times K$ pairwise difference matrix $\mathrm{TPR}_k - \mathrm{TPR}_l$ which exposes the directional structure of any residual disparity.
3. **Computational cost.** Wall-clock LP solve time, reported in seconds.

The TPR-based metrics correspond directly to the equal-opportunity criterion that constraints (8)–(10) are designed to enforce; demographic-parity and equalised-odds variants are outside the scope of this paper.

---

# 5. Results and Discussion

## 5.1 Predictive performance

Table 2 summarises the headline comparison. Accuracy decreases by $0.0098$ (from $0.5485$ to $0.5387$) when fairness and robustness are imposed, while balanced accuracy **improves** by $0.0161$ (from $0.5753$ to $0.5914$). The sign reversal between the two metrics is informative: the unconstrained SVM exploits the $5\%$ class imbalance to reach slightly higher raw accuracy by under-predicting positives, whereas HDRFC-K — which is forced by the equal-opportunity constraints to lift TPR in the larger groups — recovers a more balanced operating point. Solve time grows from $3.1$ s to $16.0$ s, a factor of $5.2\times$, consistent with the additional $K(K-1) = 20$ blocks of fairness constraints.

**Table 2.** Test-set comparison on $N_\mathrm{te} = 4{,}000$ held-out policies.

| Model | Accuracy | Balanced acc. | Max TPR gap $\Delta$ | Solve time |
|:--|--:|--:|--:|--:|
| Unconstrained SVM ($\rho{=}0$, $\zeta{=}\infty$) | 0.5485 | 0.5753 | 0.3073 | 3.1 s |
| HDRFC-K ($\rho{=}0.1$, $\zeta{=}1.5$) | 0.5387 | 0.5914 | 0.1232 | 16.0 s |
| Change | $-0.0098$ | $+0.0161$ | $-0.1842\;(-59.9\%)$ | — |

In absolute terms both models leave substantial predictive headroom on this dataset; freMTPL2freq is well known to be a noisy, high-class-imbalance benchmark for which linear classifiers without exposure modelling routinely report balanced accuracies in the high fifties. We therefore interpret the $1.6$ percentage-point gain in balanced accuracy as a side-effect of the class-aware constraint geometry rather than as evidence of a fundamentally stronger classifier. The plain-accuracy decrease of roughly one percentage point is the "fairness tax" in this experiment.

## 5.2 Fairness results

The maximum pairwise TPR gap drops from $\Delta = 0.3073$ under the unconstrained SVM to $\Delta = 0.1232$ under HDRFC-K, a reduction of $59.9\%$. Table 3 reports the full per-group TPR profile.

**Table 3.** Per-group true positive rate on the test fold.

| $k$ | Age | $\mathrm{TPR}_k$ (SVM) | $\mathrm{TPR}_k$ (HDRFC-K) | Change |
|:--:|:--:|--:|--:|--:|
| 1 | 18–25 | 0.8667 | 0.7333 | $-0.1333$ |
| 2 | 26–40 | 0.5593 | 0.6102 | $+0.0508$ |
| 3 | 41–55 | 0.5921 | 0.6711 | $+0.0789$ |
| 4 | 56–70 | 0.6000 | 0.6286 | $+0.0286$ |
| 5 | 71+ | 0.6000 | 0.6667 | $+0.0667$ |

The unconstrained SVM exhibits a pronounced asymmetry: the youngest group (18–25) has $\mathrm{TPR}_1 = 0.8667$ while the four older groups lie in the narrow band $[0.5593, 0.6000]$. The fairness constraints (8) act in the direction one would predict from Theorem 3.5: $\mathrm{TPR}_1$ is **reduced** by $0.1333$ towards the bulk of the distribution, while $\mathrm{TPR}_2,\ldots,\mathrm{TPR}_5$ each **rise** by between $0.0286$ and $0.0789$. After projection, all five group TPRs lie in $[0.6102, 0.7333]$, so no group is left below $0.61$.

The pairwise structure makes the geometry concrete. Under the unconstrained SVM, every pair involving group 1 shows a large positive gap,
$$\mathrm{TPR}_1 - \mathrm{TPR}_2 = +0.307,\quad \mathrm{TPR}_1 - \mathrm{TPR}_3 = +0.275,\quad \mathrm{TPR}_1 - \mathrm{TPR}_4 = +0.267,\quad \mathrm{TPR}_1 - \mathrm{TPR}_5 = +0.267,$$
while all $\binom{4}{2} = 6$ pairs among groups $\{2,3,4,5\}$ lie within $\pm 0.041$. The disparity is therefore essentially a one-versus-rest pattern driven by the youngest drivers. Under HDRFC-K, the analogous pairwise differences contract to
$$\mathrm{TPR}_1 - \mathrm{TPR}_2 = +0.123,\quad \mathrm{TPR}_1 - \mathrm{TPR}_4 = +0.105,\quad \mathrm{TPR}_1 - \mathrm{TPR}_5 = +0.067,\quad \mathrm{TPR}_1 - \mathrm{TPR}_3 = +0.062,\quad \mathrm{TPR}_3 - \mathrm{TPR}_2 = +0.061,$$
with all remaining pairs among groups $\{2,3,4,5\}$ within $\pm 0.057$. The new maximum, $0.123$, again involves group 1 but is now less than half of the largest baseline gap ($0.307$). We note that none of the realised pairwise ratios approach the slack bound $H_{kl} \leq 2.5$ implied by $\zeta = 1.5$; the constraints in (8) are therefore not tight at the optimum, suggesting that tighter $\zeta$ values would still be feasible at this $\rho$.

## 5.3 Discussion

**Accuracy–fairness trade-off.** The empirical trade-off on this benchmark is mild: a $59.9\%$ reduction in maximum TPR gap is purchased at the cost of $0.98$ percentage points of plain accuracy, while balanced accuracy actually improves. This is consistent with the regime described by the LP geometry of Theorem 3.5 when the unconstrained classifier is already near a fair operating point along most pairs and the disparity is concentrated in a small number of groups (here, group 1). In such cases the projection induced by constraint (8) may act on a low-dimensional slice of the decision boundary and the predictive surplus elsewhere could be largely preserved. We caution that this favourable behaviour is **not guaranteed** by the theory; on datasets where the unconstrained optimum is far from any equal-opportunity solution, the price of fairness can be substantially larger.

**Role of $\rho$ and $\zeta$.** The two regularisers play complementary roles. The Wasserstein radius $\rho$ enters the constraints (7), (9), and (10) through the $\rho \|\mathbf{w}\|_1$ term and acts as an L1 penalty that hedges against distributional shift; the fairness tolerance $\zeta$ enters only constraint (8) and controls how close the solution is forced to perfect equal-opportunity ($\zeta = 0$). At our chosen operating point $(\rho, \zeta) = (0.1, 1.5)$ the fairness constraints are slack at the optimum, suggesting that the force shaping the per-group TPR profile may reflect the **interaction** between $\rho$ and $\zeta$ rather than (8) alone — one possible interpretation is that robustness regularisation moves the boundary into a region where group 1 is less over-predicted, though disentangling the contributions of $\rho$ and $\zeta$ would require a controlled ablation. A systematic Pareto sweep over $(\rho, \zeta)$, including the limit $\zeta \to 0$, is the natural next experimental step and would let us trace the full accuracy–fairness frontier predicted by the LP.

**Limitations.** Several caveats apply. First, the results are obtained on a single dataset and a single train/test split; we report point estimates rather than confidence intervals, and any claim about the relative ranking of the two estimators should therefore be regarded as **preliminary** until cross-validated on multiple folds and replicated on additional benchmarks. Second, the inverse-frequency class weighting, while a standard remedy for class imbalance, departs from the uniform-weight objective in Theorem 3.5; although the constraint structure is unchanged, the precise duality argument underlying the LP would need to be re-stated for the weighted case to be fully rigorous. Third, the L1 norm is a modelling choice driven by computational tractability — an L2 (SOCP) variant would be preferable when the feature scaling is heterogeneous, and we have not characterised how much of the observed fairness gain is specific to L1 geometry. Fourth, the sensitive attribute is excluded from $\mathbf{x}_i$ but its proxies (region, area code, bonus-malus) remain, so the residual TPR gap of $0.123$ in group 1 may reflect proxy leakage that the linear-LP framework cannot, by construction, fully eliminate. Fifth, our solve-time figures are wall-clock measurements on a single machine and should not be read as a benchmark of CLARABEL or of LP-based fair learning more broadly. Finally, the experiment uses $K = 5$ groups; the LP scales as $K(K-1)$ in the number of fairness blocks, and the empirical scaling behaviour for larger $K$ remains to be characterised.
