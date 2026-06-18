# Review of Empirical Analysis Plan and Revised Paper

**Reviewed artifacts**

- `empirical_analysis_plan.md`
- `papers/Combined_Paper_Revised.docx`
- `papers/hdrfc-k-sections-4-5.md`
- `experiment.py`
- `lp_core.py`
- `ablation.py`
- `outputs/ablation_results.csv`

## Overall Assessment

The revised paper presents a clear multi-group extension and a useful initial
experiment, but further revisions are needed before submission. The most
important issue is that the headline experiment does not isolate the effect of
the fairness constraints. The completed ablation also contains important
evidence that is not yet incorporated into the paper.

## Priority Findings

### 1. Critical: The Claimed Fairness Effect Is Not Isolated

The headline comparison changes both hyperparameters:

- Baseline: `(rho=0, zeta=infinity)`
- HDRFC-K: `(rho=0.1, zeta=1.5)`

Consequently, the comparison cannot attribute the reported TPR-gap reduction
specifically to the multi-group fairness constraints.

The ablation strengthens this concern. For `rho=0.1`, the results at
`zeta=1.5`, `2.0`, and `5.0` are identical:

| rho | zeta | Accuracy | Balanced accuracy | Max TPR gap |
|---:|---:|---:|---:|---:|
| 0.1 | 1.5 | 0.53875 | 0.59145 | 0.12316 |
| 0.1 | 2.0 | 0.53875 | 0.59145 | 0.12316 |
| 0.1 | 5.0 | 0.53875 | 0.59145 | 0.12316 |

This suggests that the fairness constraint is nonbinding at the selected
operating point and that robustness regularization may drive the reported
59.9% TPR-gap reduction.

**Required improvement:** report a controlled comparison containing:

- Unconstrained baseline: `(rho=0, zeta=infinity)`
- Robust-only model: `(rho=0.1, zeta=infinity)`
- Fair-only model: `(rho=0, selected zeta)`
- Full HDRFC-K model: `(rho=0.1, selected zeta)`

### 2. Critical: The Paper Does Not Incorporate the Completed Ablation

The paper states that a systematic parameter sweep is future work, but the
repository already contains a 35-setting `rho x zeta` sweep and two plots.

The sweep reveals important behavior:

- `zeta=0.5` and `zeta=0.8` are infeasible for every tested `rho`.
- `zeta=1.0` frequently produces an all-negative classifier with accuracy
  `0.95`, balanced accuracy `0.50`, and TPR gap `0`.
- `rho=0.5` also produces the all-negative classifier across the feasible
  tested settings.

A zero TPR gap is not a useful fairness outcome when the classifier rejects
every positive case.

**Required improvement:** incorporate the ablation table and plots into the
paper and explicitly distinguish meaningful fairness improvements from
degenerate zero-gap solutions.

### 3. High: Constraint Slack Is Not Actually Measured

The paper concludes that constraint (8) is slack because the realized TPR
differences do not approach the bound on `H_kl`. However, TPR differences and
the hinge unfairness quantity `H_kl` are different metrics.

The current implementation does not report:

- Realized `H_kl` values
- Constraint (8) left-hand sides
- Constraint slack
- Dual values

**Required improvement:** calculate and report the realized left-hand side and
slack for every ordered-pair fairness constraint before describing the
constraints as binding or nonbinding.

### 4. High: Results Are Statistically Fragile

The evaluation uses a single train/test split. The youngest group has only
about 15 positive test examples, so its reported TPR change corresponds to
approximately two fewer correctly classified positive cases. The maximum TPR
gap and its percentage reduction may therefore vary substantially across
splits.

**Required improvement:** run repeated stratified splits or cross-validation
and report uncertainty intervals for:

- Accuracy and balanced accuracy
- Each group TPR
- Maximum pairwise TPR gap

### 5. High: Section 6 Is Empty

The revised DOCX contains the `6. Conclusion` heading followed immediately by
the references.

**Required improvement:** add a conclusion that summarizes:

- The multi-group theoretical contribution
- The controlled empirical findings
- The limitations of the current evidence
- The most important next experiments

## Theory and Wording Issues

### Mixed-Integer Versus Linear Program

The introduction describes the contribution as a "mixed integer program
reformulation," while Theorem 3.5 presents HDRFC-K as a linear program. The
introduction should say "linear-program reformulation."

### Weighted Objective Claim

The experiment uses inverse-frequency class weights, while Theorem 3.5 uses a
uniform objective. The paper states that all dual and structural results carry
over "verbatim." This is too strong without a formal argument.

**Improvement:** restate the theorem for positive sample weights or describe
the weighted objective as an implementation-level deviation without claiming
that every theoretical result transfers verbatim.

### Ordered-Pair Redundancy

The statement that all ordered-pair equal-opportunity equalities are "not
redundant" is inaccurate. Exact TPR equality does not require every ordered
pair. Both directions may be necessary for the directional hinge surrogate,
which is the more precise justification the paper should give.

### Scalability Claim

The paper claims that the full dataset with approximately 678,000 observations
remains tractable, while the experiment uses a 20,000-row subsample because
the full problem is considered intractable on the available machine.

**Improvement:** qualify the scalability claim and report estimated full-data
memory and solve requirements.

### Insurance Pricing Terminology

The experiment predicts whether a claim occurs. It does not directly estimate
claim frequency, severity, premiums, or prices. Calling it an insurance
"pricing" experiment overstates what is evaluated.

**Improvement:** use "insurance claim-risk classification" unless an actual
pricing model is added.

## Empirical Analysis Plan Updates

The plan should be synchronized with the completed implementation:

- Use CLARABEL consistently instead of mixing MOSEK and CLARABEL descriptions.
- Replace obsolete full-size lambda complexity estimates with the optimized
  group-sized lambda formulation.
- Mark the hyperparameter sweep and Pareto plots as completed.
- Add the inverse-frequency class-weighted objective.
- Add controlled robust-only and fair-only baselines.
- Add repeated-split or cross-validation uncertainty analysis.
- Add reporting of fairness-constraint values and slack.
- Pin the OpenML dataset version for reproducibility.

## Presentation and Document Cleanup

- Fix duplicate table numbering. The document restarts at Table 1 and Table 2
  in later sections.
- Order the reference list consistently, preferably alphabetically.
- Reconcile the Zhang et al. in-text citation year with the reference-list
  year.
- Add a complete conclusion.
- Review the baseline implementation: when `zeta=infinity`, constraint (8) is
  skipped, but lambda variables and constraints (9)-(10) are still created.
  Removing the unused fairness block would provide a cleaner baseline and
  solve-time comparison.

## Recommended Revision Sequence

1. Add controlled unconstrained, robust-only, fair-only, and full-model runs.
2. Report `H_kl` values and fairness-constraint slack.
3. Run repeated splits and report uncertainty.
4. Integrate the completed ablation results and discuss degenerate solutions.
5. Revise claims about fairness causality, tractability, and theory transfer.
6. Add the conclusion and complete document-formatting cleanup.

## Review Limitation

The DOCX content and structure were inspected. Full visual rendering could not
be completed because the available DOCX renderer dependencies were missing and
Word automation was unavailable in the execution environment.
