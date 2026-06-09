# Draft Outline: Sections 4 & 5 — HDRFC-K Paper

## Scope
Revised Section 4 (Experiments) and Section 5 (Results & Discussion) for the Combined_Paper on multi-group distributionally robust fair classification.

## Section 4 — Experiments

### 4.1 Dataset
- freMTPL2freq (French MTPL, OpenML 41214, N=678,013)
- Binary label: ClaimNb > 0
- Sensitive attribute: DrivAge → K=5 age cohorts (18-25, 26-40, 41-55, 56-70, 71+)
- Features: 9 variables (5 numeric standardised, 4 categorical one-hot encoded → d=42)
- Subsampling rationale: LP is O(K²N); 20k tractable subsample

### 4.2 Experimental Protocol
- Stratified 80/20 split on (Y, A)
- Baseline: unconstrained class-weighted SVM (ρ=0, ζ=∞)
- Proposed: HDRFC-K (ρ=0.1, ζ=1.5, K=5)
- Solver: CLARABEL via CVXPY
- Implementation note: inverse-frequency class weights added to handle 5% positive base rate

### 4.3 Evaluation Metrics
- Predictive: accuracy, balanced accuracy
- Fairness: per-group TPR, max pairwise TPR gap, 5×5 pairwise difference matrix
- Solver: status, wall-clock time

## Section 5 — Results and Discussion

### 5.1 Quantitative Results
All from experiment.py run 2026-06-09:
- SVM: Acc=0.5485, Bal.Acc=0.5753, MaxTPRgap=0.3073, time=3.1s
- HDRFC-K: Acc=0.5387, Bal.Acc=0.5914, MaxTPRgap=0.1232, time=16.0s
- TPR per group (SVM): G1=0.867, G2=0.559, G3=0.592, G4=0.600, G5=0.600
- TPR per group (HDRFC-K): G1=0.733, G2=0.610, G3=0.671, G4=0.629, G5=0.667

### 5.2 Key Claims
1. HDRFC-K reduces max pairwise TPR gap by 59.9% (0.307 → 0.123)
2. Accuracy cost is minimal: -0.98 pp raw; balanced accuracy improves +1.61 pp
3. Group 1 (18-25) is the fairness bottleneck in the unconstrained model
4. LP is tractable: optimal status, 16s solve time at N=16k train
5. Fairness constraint is binding (not slack)

### 5.3 Discussion
- Fairness-accuracy trade-off interpretation
- Why balanced accuracy improves under the constraint
- Wasserstein robustness role (ρ=0.1)
- Limitations: single ρ/ζ, no cross-validation, no comparison to binary HDRFC

## Key equations to include
- Theorem 3.5 LP (constraints 7-10)
- H_kl definition and Proposition 3.4 bound
- Equal-opportunity definition (TPR parity)
