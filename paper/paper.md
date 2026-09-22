# Learning from Imbalanced Real Data: Scientific-Style Technical Report

**Status:** reproducible portfolio report, not peer reviewed.  
**Difficulty:** ★★★  
**Dataset:** Wisconsin Diagnostic Breast Cancer dataset

## Abstract
This project studies a concrete AI Engineering problem using a real public dataset and a fully inspectable pipeline. The project focuses on imbalanced learning, precision-recall, class weighting, robust evaluation. Its central engineering goal is to make data preparation, model fitting, evaluation, and limitations reproducible rather than treating the model as a black box.

## 1. Research objective
Study class weighting under a controlled rare-positive sampling scenario derived only from real observations.

## 2. Data
The dataset is **Wisconsin Diagnostic Breast Cancer dataset**. Provenance and the original reference are documented in [`DATA.md`](../DATA.md).

## 3. Method
The implemented pipeline is:
1. Load real data
2. Controlled imbalance
3. Scale
4. Weighted classifiers
5. PR evaluation

## 4. Evaluation
**Primary metric(s):** Average Precision / F1.  
**Validation design:** stratified hold-out.  
The experiment saves machine-readable metrics and visual diagnostics so claims can be traced to an executable run.

## 5. Results
Generated metrics:
```json
{
  "logistic": {
    "average_precision": 0.9924242424242424,
    "f1": 0.9090909090909091
  },
  "balanced_logistic": {
    "average_precision": 0.9924242424242424,
    "f1": 0.9565217391304348
  },
  "balanced_rf": {
    "average_precision": 1.0,
    "f1": 1.0
  }
}
```

## 6. Limitations and validity
Key concern: rare-class instability. Benchmark performance on one dataset does not imply universal performance. The project is intended to demonstrate research engineering discipline and to provide a base for stronger comparative studies.

## 7. Reproducibility
Run `python src/run_experiment.py` from the repository root after installing `requirements.txt`.

## 8. Next research extension
Add repeated cross-validation or temporal/external validation, stronger baselines, hyperparameter sensitivity, confidence intervals, and a domain-specific error analysis.

## References
- Dataset/reference page: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
