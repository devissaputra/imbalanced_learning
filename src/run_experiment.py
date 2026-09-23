from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42


def controlled_imbalance(seed: int = SEED):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    rng = np.random.default_rng(seed)

    class_zero = np.flatnonzero(y.to_numpy() == 0)
    class_one = np.flatnonzero(y.to_numpy() == 1)
    minority_count = max(35, len(class_zero) // 6)
    selected_one = rng.choice(class_one, size=minority_count, replace=False)
    keep = np.r_[class_zero, selected_one]

    X = X.iloc[keep].reset_index(drop=True)
    y = y.iloc[keep].reset_index(drop=True)
    return X, y


def build_models(seed: int = SEED):
    return {
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=3000, random_state=seed),
                ),
            ]
        ),
        "balanced_logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "balanced_rf": RandomForestClassifier(
            n_estimators=350,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def classification_metrics(y_true, probability, threshold: float = 0.5):
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, probability)),
        "f1": float(f1_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
    }


def run_experiment(
    results_dir: str | Path = "results",
    seed: int = SEED,
    make_plots: bool = True,
):
    X, y = controlled_imbalance(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    results = {
        "seed": int(seed),
        "class_counts": {
            str(key): int(value)
            for key, value in y.value_counts().sort_index().items()
        },
        "models": {},
    }
    curves = {}

    for name, model in build_models(seed).items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        results["models"][name] = classification_metrics(y_test, probability)
        precision, recall, _ = precision_recall_curve(y_test, probability)
        curves[name] = (precision, recall)

    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    if make_plots:
        figures = output / "figures"
        figures.mkdir(parents=True, exist_ok=True)

        counts = y.value_counts().sort_index()
        plt.figure(figsize=(7, 5))
        plt.bar([str(index) for index in counts.index], counts.values)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("Controlled class imbalance")
        plt.tight_layout()
        plt.savefig(figures / "class_balance.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 5))
        for name, (precision, recall) in curves.items():
            plt.plot(recall, precision, label=name)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-recall comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "precision_recall.png", dpi=150)
        plt.close()

    return results


def main() -> None:
    print(json.dumps(run_experiment(), indent=2))


if __name__ == "__main__":
    main()
