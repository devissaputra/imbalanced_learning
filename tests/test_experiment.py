from pathlib import Path

from src.run_experiment import controlled_imbalance, run_experiment


def test_controlled_imbalance_is_deterministic():
    X1, y1 = controlled_imbalance()
    X2, y2 = controlled_imbalance()

    assert X1.equals(X2)
    assert y1.equals(y2)
    assert y1.value_counts().sort_index().to_dict() == {0: 212, 1: 35}


def test_experiment_metrics_are_valid(tmp_path):
    result = run_experiment(tmp_path, make_plots=False)

    assert set(result["models"]) == {
        "logistic",
        "balanced_logistic",
        "balanced_rf",
    }

    for metrics in result["models"].values():
        for value in metrics.values():
            assert 0.0 <= value <= 1.0

    assert (tmp_path / "metrics.json").exists()


def test_repository_structure():
    root = Path(__file__).resolve().parents[1]
    for relative_path in [
        "README.md",
        "DATA.md",
        "ETHICS.md",
        "REPRODUCIBILITY.md",
        "CITATION.cff",
        "src/run_experiment.py",
        "paper/paper.md",
        "assets/01_cover.svg",
        "assets/02_data_pipeline.svg",
        "assets/03_data_or_model.svg",
        "assets/04_evaluation_or_results.svg",
        ".github/workflows/ci.yml",
    ]:
        assert (root / relative_path).exists(), relative_path
