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
    for (k, l) in pairs:
        Ik = I_pos[k]
        Il = I_pos[l]
        if len(Ik) == 0 or len(Il) == 0:
            continue
        nk, nl = len(Ik), len(Il)
        lam_kl = cp.Variable(nk + nl, name=f"lam_{k}_{l}", nonneg=True)

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


def compute_hkl(w_val, b_val, X_tr, A_tr, I_pos, pairs, rho):
    """
    Compute realized H_kl values on the training set for all ordered pairs.

    H_kl = (1/|Ik+|) sum_{i in Ik+} max(0, 1 + w'xi + rho||w||1 + b)
         + (1/|Il+|) sum_{i in Il+} max(0, 1 - w'xi + rho||w||1 - b)
         - 1

    Returns dict (k, l) -> H_kl value and prints constraint slack table.
    """
    norm_w = np.sum(np.abs(w_val))
    Xw_b = X_tr @ w_val + b_val

    hkl = {}
    for (k, l) in pairs:
        Ik = I_pos[k]
        Il = I_pos[l]
        if len(Ik) == 0 or len(Il) == 0:
            hkl[(k, l)] = float("nan")
            continue
        term_k = np.mean(np.maximum(0.0, 1.0 + Xw_b[Ik] + rho * norm_w))
        term_l = np.mean(np.maximum(0.0, 1.0 - Xw_b[Il] + rho * norm_w))
        hkl[(k, l)] = term_k + term_l - 1.0
    return hkl


def print_hkl_table(hkl, zeta, pairs):
    """Print H_kl values, constraint LHS, and slack for all ordered pairs."""
    groups = sorted(set(k for p in pairs for k in p))
    print("\n  Realized H_kl values (constraint (8): H_kl <= 1 + zeta = {:.4f}):".format(1 + zeta))
    print(f"  {'Pair (k,l)':<12} {'H_kl':>10} {'LHS (H_kl)':>12} {'Slack':>10} {'Binding?':>10}")
    print("  " + "-" * 56)
    bound = 1.0 + zeta if zeta < np.inf else float("inf")
    for (k, l) in sorted(pairs):
        v = hkl.get((k, l), float("nan"))
        if np.isnan(v):
            print(f"  ({k},{l}){'':8} {'nan':>10} {'nan':>12} {'nan':>10}")
            continue
        slack = bound - v
        binding = "YES" if slack < 1e-4 else "no"
        print(f"  ({k},{l}){'':8} {v:>10.5f} {v:>12.5f} {slack:>10.5f} {binding:>10}")
    max_hkl = max((v for v in hkl.values() if not np.isnan(v)), default=float("nan"))
    print(f"\n  Max H_kl = {max_hkl:.5f}  (bound = {bound:.4f})")


def evaluate(w_val, b_val, X_te, Y_te, A_te, pairs, label,
             X_tr=None, A_tr=None, I_pos=None, rho=0.0, zeta=np.inf):
    """
    Evaluate a fitted classifier and print metrics.

    Returns dict with keys: accuracy, balanced_accuracy, tpr (dict), max_gap,
    hkl (dict of realized H_kl values, or None if training data not supplied).
    """
    scores = X_te @ w_val + b_val
    Y_hat = np.sign(scores)
    Y_hat[Y_hat == 0] = 1  # tie-break to positive

    acc = np.mean(Y_hat == Y_te)
    bal_acc = balanced_accuracy_score(Y_te, Y_hat)

    tpr = {}
    groups = sorted(set(k for p in pairs for k in p))
    for k in groups:
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
    for k in groups:
        print(f"    Group {k} ({_GROUP_NAMES[k-1]}): TPR = {tpr[k]:.4f}")

    print()
    print("  Pairwise TPR difference matrix (row k, col l = TPR_k - TPR_l):")
    header = "      " + "".join(f"  G{k}  " for k in groups)
    print(header)
    for k in groups:
        row = f"  G{k}  "
        for l in groups:
            if k == l:
                row += "  ---  "
            else:
                diff = tpr[k] - tpr[l]
                row += f" {diff:+.3f}"
        print(row)

    hkl = None
    if X_tr is not None and I_pos is not None:
        hkl = compute_hkl(w_val, b_val, X_tr, A_tr, I_pos, pairs, rho)
        print_hkl_table(hkl, zeta, pairs)

    return {"accuracy": acc, "balanced_accuracy": bal_acc, "tpr": tpr,
            "max_gap": max_gap, "hkl": hkl}
