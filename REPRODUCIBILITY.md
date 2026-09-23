# Reproducing the Experiment

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The script uses seed 42 for the class-1 subsample, the train/test split, and the Random Forest.

The controlled dataset contains 212 class-0 observations and 35 class-1 observations. The same seed is important because changing the selected minority examples can change the results noticeably.

Metrics are saved to `results/metrics.json`, and the class-balance and precision-recall plots are regenerated in `assets/`.

If you compare results across machines, record your Python and scikit-learn versions.
