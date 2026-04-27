"""Experiment result reporting and repeated evaluation helpers."""

from datetime import datetime
from pathlib import Path
import socket

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm


def save_grid_search_results(
    cv_results: dict[str, np.ndarray],
    experiment_name: str,
    experiment_description: str | None = None,
    config: dict | None = None,
) -> Path:
    """Save grid-search metrics and configuration.

    Parameters
    ----------
    cv_results : dict
        ``GridSearchCV.cv_results_`` dictionary.
    experiment_name : str
        Prefix used for the run directory name.
    experiment_description : str or None, default=None
        Optional text written to the summary file.
    config : dict or None, default=None
        Optional experiment configuration written to the summary file.

    Returns
    -------
    pathlib.Path
        Directory containing the saved result files.
    """
    df = pd.DataFrame(cv_results)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("results") / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(run_dir / "gridsearch.csv", index=False)

    with (run_dir / "gridsearch.txt").open("w", encoding="utf-8") as f:
        if "mean_test_uar" in cv_results:
            f.write(f"Best UAR: {np.nanmax(cv_results['mean_test_uar'])}\n")
        if "mean_test_accuracy" in cv_results:
            f.write(f"Best ACC: {np.nanmax(cv_results['mean_test_accuracy'])}\n")
        f.write(f"Machine: {socket.gethostname()}\n")

        if experiment_description is not None:
            f.write(experiment_description)

        if config is not None:
            f.write("\n\nConfiguration:\n")
            for key, value in config.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for inner_key, inner_value in value.items():
                        f.write(f"\t{inner_key}: {inner_value}\n")
                else:
                    f.write(f"{key}: {value}\n")

    return run_dir


def repeated_cv_evaluation(
    best_estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 10,
    n_repeats: int = 1000,
) -> pd.DataFrame:
    """Evaluate an estimator with repeated stratified group CV.

    Parameters
    ----------
    best_estimator : estimator
        Estimator to clone and evaluate in each fold.
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Target labels.
    groups : numpy.ndarray
        Group labels used by ``StratifiedGroupKFold``.
    n_splits : int, default=10
        Number of folds per repeat.
    n_repeats : int, default=1000
        Number of repeated CV rounds.

    Returns
    -------
    pandas.DataFrame
        Per-fold metrics with columns ``repeat``, ``fold``, ``UAR``,
        ``accuracy``, and ``AUC``.
    """
    rows = []
    for repeat in tqdm(range(n_repeats), desc="Repeated CV"):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            estimator = clone(best_estimator)
            estimator.fit(X[train_idx], y[train_idx])
            predicted = estimator.predict(X[test_idx])
            probabilities = estimator.predict_proba(X[test_idx])[:, 1]

            rows.append(
                (
                    repeat,
                    fold,
                    balanced_accuracy_score(y[test_idx], predicted),
                    accuracy_score(y[test_idx], predicted),
                    roc_auc_score(y[test_idx], probabilities),
                )
            )

    return pd.DataFrame(rows, columns=["repeat", "fold", "UAR", "accuracy", "AUC"])
