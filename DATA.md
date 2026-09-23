# Data

The starting point is scikit-learn's Wisconsin Diagnostic Breast Cancer dataset.

Source documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

For this experiment I create a controlled imbalance from the original observations:

- all 212 class-0 rows are kept;
- 35 class-1 rows are selected with a seeded random sample;
- no synthetic rows are generated.

The resulting dataset is intentionally artificial in its class balance. It is useful for studying class weighting, but it should not be interpreted as a realistic estimate of medical prevalence.
