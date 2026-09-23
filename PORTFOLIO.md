# Learning from Imbalanced Data

**Focus:** class imbalance, decision thresholds, and minority-class evaluation.

I construct a deterministic rare-class setting from real Wisconsin Diagnostic Breast Cancer observations and compare ordinary logistic regression, class-weighted logistic regression, and a class-weighted Random Forest.

The repository reports Average Precision, F1, precision, recall, and balanced accuracy rather than hiding behind raw accuracy. Class weighting raises logistic-regression recall from 0.9091 to 1.0000 in the recorded split. The Random Forest reaches perfect held-out metrics, but the repository explicitly treats that result as fragile because the test minority sample is small.

The project now includes import-safe experiment code, behavioural tests, GitHub Actions CI, reproducibility notes, and responsible-use documentation.
