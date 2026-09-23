# Portfolio Summary

## Learning from Imbalanced Data

I use real observations from the Wisconsin Diagnostic Breast Cancer dataset to create a controlled imbalance and compare ordinary and class-weighted classifiers.

The main point is not to chase the highest accuracy. I compare Average Precision and F1 so the less common class remains visible in the evaluation.

### Images

![Project overview](assets/01_cover.svg)

![Processing pipeline](assets/02_data_pipeline.svg)

![Imbalance and model strategy](assets/03_data_or_model.svg)

![Evaluation summary](assets/04_evaluation_or_results.svg)

**Key result:** class weighting improved logistic-regression F1 from 0.9091 to 0.9565 in the recorded split.
