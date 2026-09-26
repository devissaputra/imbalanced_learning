# Learning from Imbalanced Data

This controlled experiment uses real Wisconsin Diagnostic Breast Cancer observations to compare ordinary logistic regression, class-weighted logistic regression, and a weighted random forest. It deliberately makes benign cases the minority class, then reports average precision, recall, F1, and balanced accuracy on one stratified split. The recorded perfect forest score is retained alongside the small-sample limitation: it demonstrates the behavior of this setup, not clinical reliability.

See [CALCULATIONS.md](CALCULATIONS.md) for evidence and verification scope.
