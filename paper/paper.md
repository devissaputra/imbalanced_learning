# Learning from Imbalanced Data

## Question

How much does class weighting change performance when one class is deliberately made much less common?

## Data

I start with the Wisconsin Diagnostic Breast Cancer dataset. I keep all 212 observations from class 0 and a seeded subset of 35 observations from class 1.

Every row is still a real observation. The imbalance is created only by selecting a subset of the original data.

This is an experimental setup, not an estimate of real medical prevalence.

## Method

I use a stratified 70/30 train/test split and compare three models:

- standard logistic regression;
- logistic regression with balanced class weights;
- Random Forest with balanced class weights and 350 trees.

The logistic models use standardized features.

I evaluate Average Precision and F1 because both are more informative than raw accuracy for this setup.

## Results

| Model | Average Precision | F1 |
|---|---:|---:|
| Logistic regression | 0.9924 | 0.9091 |
| Balanced logistic regression | 0.9924 | 0.9565 |
| Balanced Random Forest | 1.0000 | 1.0000 |

## Interpretation

Class weighting improved the logistic model's F1 score in this split. The Random Forest reached perfect scores on the small held-out set.

I do not interpret that perfect result as evidence that the model will generalize perfectly. The sample is small and the imbalance was constructed for the experiment.

## Limitations

The study uses one dataset, one seeded sampling design, and one hold-out split. The target labels also come from a medical dataset, so the exercise should not be treated as a clinical decision model.

A stronger follow-up would repeat the subsampling many times, report uncertainty, compare threshold choices, and test resampling methods alongside class weighting.

## Reproduce

```bash
pip install -r requirements.txt
python src/run_experiment.py
```
