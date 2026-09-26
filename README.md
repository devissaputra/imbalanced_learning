# Learning from Imbalanced Data

This controlled experiment uses real Wisconsin Diagnostic Breast Cancer observations to compare ordinary logistic regression, class-weighted logistic regression, and a weighted random forest. It deliberately makes benign cases the minority class, then reports average precision, recall, F1, and balanced accuracy on one stratified split. The recorded perfect forest score is retained alongside the small-sample limitation: it demonstrates the behavior of this setup, not clinical reliability.

## Start here

- [Calculations, evidence and verification scope](CALCULATIONS.md)
- [Figure sources and exact numerical paths](docs/figure_spec.json)
- [Working paper](paper/paper.md)
- [Data and provenance](DATA.md)

![Study question, data, design and interpretation](assets/review_overview.svg)

![Defined calculation and source-linked evidence](assets/review_calculations.svg)

**Review scope:** The existing suite requires unavailable dependencies; no full-suite pass is claimed. The complete data/model experiment was not rerun in this review.

## Detailed project documentation

[![CI](https://github.com/devissaputra/imbalanced_learning/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/imbalanced_learning/actions/workflows/ci.yml)


**Category:** AI Engineering

A controlled experiment on how class imbalance changes model behaviour and why **accuracy alone is a poor summary when the minority class matters**.

## Question

> What changes when the same real observations are modeled with ordinary logistic regression, class-weighted logistic regression, and a class-weighted Random Forest?

The study uses real observations from scikit-learn's Wisconsin Diagnostic Breast Cancer dataset and creates a deterministic minority-class scenario. Here, class 1 is benign and is deliberately downsampled; it does **not** invent synthetic patients.

## Controlled data setup

- all 212 class-0 (malignant) observations are retained
- 35 class-1 (benign) observations are selected with seed 42
- total controlled sample: 247
- stratified 70/30 train/test split

This setup is a methodological exercise. It is not a claim about real disease prevalence.

## Models


1. standard logistic regression
2. logistic regression with `class_weight="balanced"`
3. Random Forest with `class_weight="balanced"`

Logistic models are scaled inside scikit-learn pipelines.

## Why several metrics matter


The repository reports:

- **Average Precision** for precision-recall ranking quality
- **F1** for the precision/recall trade-off at threshold 0.5
- **Precision**
- **Recall**
- **Balanced accuracy**

Those metrics answer different questions. A model can rank cases well but still use a poor operating threshold.

## Recorded results

| Model | Avg. Precision | F1 | Precision | Recall | Balanced Acc. |
|---|---:|---:|---:|---:|---:|
| Logistic | 0.9924 | 0.9091 | 0.9091 | 0.9091 | 0.9467 |
| Balanced logistic | 0.9924 | 0.9565 | 0.9167 | **1.0000** | 0.9922 |
| Balanced Random Forest | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |


The perfect Random Forest result should be read cautiously. The held-out minority sample is very small and the imbalance was deliberately constructed. This is precisely why the repository keeps the limitations visible instead of presenting the score as deployment evidence.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

Generated metrics and figures are written under `results/`.

## Test

```bash
pip install pytest
pytest
```

Tests verify deterministic minority sampling, metric ranges, and output structure.

## What changed under reweighting

The balanced logistic model recovers the minority class more reliably on this split without sacrificing much precision. The Random Forest reaches a perfect held-out score, but that result is exactly where caution is needed: the minority test set is small and the class distribution was deliberately constructed for this experiment.

The repository is designed to make that tension visible. A headline accuracy number is not enough when the rare class is the one we care about.

## What I would test next

I would repeat the experiment across multiple random seeds, add confidence intervals, choose operating thresholds on a separate validation set, and test the same methods on a naturally imbalanced dataset. Nothing here is a medical decision rule; see [ETHICS.md](ETHICS.md).
