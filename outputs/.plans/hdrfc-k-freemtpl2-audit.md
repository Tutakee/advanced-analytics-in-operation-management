# Audit Plan: HDRFC-K on freMTPL2freq

**Paper:** Combined_Paper.docx — multi-group distributionally robust fair classifier (Theorem 3.5, HDRFC-K)
**Code:** experiment.py
**Claims to check:**
1. LP formulation matches Theorem 3.5 constraints (7), (8), (9), (10)
2. Wasserstein robustness term uses correct norm
3. Fairness constraint correctly encodes H_kl lower bound (Proposition 3.4)
4. Dataset preprocessing and subsampling match paper description
5. Evaluation metrics (TPR gap, balanced accuracy) match paper's stated metrics
6. Hyperparameter choices (rho, zeta) are consistent with paper guidance
7. Baseline (unconstrained SVM) is correctly specified
8. Solver choice and convergence handling are appropriate
