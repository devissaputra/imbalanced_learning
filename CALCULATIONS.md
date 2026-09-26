# Calculation guide

## Question and evidence

How does class weighting change minority-class recovery?

Wisconsin Diagnostic Breast Cancer observations; retain 212 malignant cases and sample 35 benign cases.

**Status:** RECORDED SMALL-SAMPLE BENCHMARK | see validation scope.

## Design

Stratified 70/30 split; ordinary and weighted logistic regression; weighted random forest.

## Calculation and interpretation

`Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2PR/(P+R).`

Class 1 is benign in this constructed experiment. Average precision summarizes ranking; threshold metrics describe one operating point. The minority test sample is small, and the deliberately altered prevalence is not clinical prevalence.

## Evidence table

Selected recorded values (units and context shown). Full precision below is for traceability, not a claim of measurement precision.

| Quantity | Value | Unit / meaning | JSON path |
|---|---:|---|---|
| logistic | 0.9090909090909091 | minority F1 ↑ | `models.logistic.f1` |
| balanced_logistic | 0.9565217391304348 | minority F1 ↑ | `models.balanced_logistic.f1` |
| balanced_rf | 1.0 | minority F1 ↑ | `models.balanced_rf.f1` |

Source: [results/metrics.json](results/metrics.json). Values resolve directly from this file when figures are regenerated.

This controlled experiment uses real Wisconsin Diagnostic Breast Cancer observations to compare ordinary logistic regression, class-weighted logistic regression, and a weighted random forest. It deliberately makes benign cases the minority class, then reports average precision, recall, F1, and balanced accuracy on one stratified split. The recorded perfect forest score is retained alongside the small-sample limitation: it demonstrates the behavior of this setup, not clinical reliability.

## Verification performed in this review

The existing suite requires unavailable dependencies; no full-suite pass is claimed. The complete data/model experiment was not rerun in this review.

The figure-generation check verifies agreement between the selected source values and SVGs. It does not validate the raw dataset, fitted model, identification assumptions, or external generalization.

```bash
python scripts/build_review_figures.py
python scripts/build_review_figures.py --check
```

## Implementation map

Follow these functions to inspect each transformation. Validation helpers and private functions remain visible in the linked modules.

| Function | Purpose / documented behavior |
|---|---|
| [`controlled_imbalance`](src/run_experiment.py#L31) | Inspect the explicit implementation and its callers. |
| [`build_models`](src/run_experiment.py#L46) | Inspect the explicit implementation and its callers. |
| [`classification_metrics`](src/run_experiment.py#L79) | Inspect the explicit implementation and its callers. |
| [`run_experiment`](src/run_experiment.py#L92) | Inspect the explicit implementation and its callers. |
| [`main`](src/run_experiment.py#L158) | Inspect the explicit implementation and its callers. |

## What remains before a stronger research claim

Class 1 is benign in this constructed experiment. Average precision summarizes ranking; threshold metrics describe one operating point. The minority test sample is small, and the deliberately altered prevalence is not clinical prevalence. A successful software test is not validation of a scientific construct. New experiments should state their split unit, comparator, outcome, uncertainty procedure and failure criteria before examining final test results.
