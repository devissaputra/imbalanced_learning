# Reproducing the experiment

Use Python 3.11 or another current Python 3 release compatible with the dependency ranges.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The controlled minority sample, split, logistic models, and Random Forest all use seed 42. The controlled dataset contains 212 class-0 observations and 35 class-1 observations.

Outputs:

- `results/metrics.json`
- `results/figures/class_balance.png`
- `results/figures/precision_recall.png`

Run tests with:

```bash
pip install pytest
pytest
```

GitHub Actions runs the same tests on every push and pull request. Exact floating-point values may move slightly across library versions, so committed metrics are treated as a recorded run.
