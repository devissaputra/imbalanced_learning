# Learning from Imbalanced Data

## Abstract

This study examines how class weighting changes classification behaviour under a controlled minority-class setting. Real observations from the Wisconsin Diagnostic Breast Cancer benchmark are used; no synthetic patient rows are generated. All class-0 cases are retained and 35 class-1 cases are selected with seed 42.

## Method

The controlled sample contains 247 observations. A stratified 70/30 train/test split is used. Three models are compared: ordinary logistic regression, class-weighted logistic regression, and a class-weighted Random Forest. Evaluation reports Average Precision, F1, precision, recall, and balanced accuracy.

## Recorded results

| Model | Avg. Precision | F1 | Precision | Recall | Balanced Acc. |
|---|---:|---:|---:|---:|---:|
| Logistic | 0.9924 | 0.9091 | 0.9091 | 0.9091 | 0.9467 |
| Balanced logistic | 0.9924 | 0.9565 | 0.9167 | 1.0000 | 0.9922 |
| Balanced Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Interpretation

Class weighting improves minority recall for logistic regression in this split. The perfect Random Forest score is not treated as proof of a perfect model because the minority test sample is small and the imbalance was deliberately constructed.

## Limitations

A stronger study would repeat the controlled subsampling across many seeds, report confidence intervals, separate threshold selection from final evaluation, and validate on external data.

## Responsible use

This is a machine-learning benchmark exercise, not a medical model.
