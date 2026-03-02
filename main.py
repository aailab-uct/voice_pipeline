"""Main entry point for minimal voice classification pipeline."""

from typing import Any
from datetime import datetime
from importlib import import_module
from pathlib import Path
import numpy as np
import argparse
import socket
import time

import pandas as pd
import matplotlib.pyplot as plt
import sys
from contextlib import contextmanager
from tqdm import tqdm

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin

from feature_extraction import (
    MFCCExtractor,
    DropSignalColumn,
    TimeDomainFeaturesExtractor,
    ZCRExtractor,
    SpectralFeaturesExtractor,
)
from dataset_filtration import build_dataset

import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


class ReplaceNaN(BaseEstimator, TransformerMixin):
    """Replaces NaN, +Inf, -Inf with 0.0"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_pipeline(random_state: int = 42) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("mfcc", MFCCExtractor(enabled=True, sr=50000, n_mfcc=20)),
            ("timedomain", TimeDomainFeaturesExtractor(
                sr=50000,
                enabled=True,
                sex_col=1,
                f0_range_m=(75.0, 300.0),
                f0_range_f=(100.0, 600.0),
            )),
            ("zcr", ZCRExtractor(sr=50000, enabled=True)),
            ("spectral", SpectralFeaturesExtractor(sr=50000, enabled=True)),
            ("drop_sig", DropSignalColumn()),
            ("replace_nan", ReplaceNaN()),
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    )


def log_results(
    cv_results: dict[str, np.ndarray],
    experiment_name: str,
    experiment_description: str | None = None,
    config: dict | None = None,
) -> Path:
    df = pd.DataFrame(cv_results)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("results") / f"{experiment_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "gridsearch.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    with (run_dir / "gridsearch.txt").open("w", encoding="utf-8") as f:
        if "mean_test_bal_acc" in cv_results:
            f.write(f"Best BAL_ACC: {np.nanmax(cv_results['mean_test_bal_acc'])}\n")
        if "mean_test_acc" in cv_results:
            f.write(f"Best ACC: {np.nanmax(cv_results['mean_test_acc'])}\n")
        f.write(f"Machine: {socket.gethostname()}\n")

        if experiment_description is not None:
            f.write(experiment_description)

        if config is not None:
            f.write("\n\nConfiguration:\n")
            for k, v in config.items():
                f.write(f"{k}: {v}\n")

    return run_dir


@contextmanager
def tqdm_from_verbose(total, desc="GridSearch", unit="fit"):
    pbar = tqdm(total=total, desc=desc, unit=unit)
    old_stdout = sys.stdout

    class StdoutProxy:
        def write(self, s):
            old_stdout.write(s)
            if "[CV] END" in s:
                pbar.update(1)

        def flush(self):
            old_stdout.flush()

    sys.stdout = StdoutProxy()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        pbar.close()


def main(config: dict[str, Any] | None = None):
    if config is None:
        config = {
            "task": "dataset_origin",  # "pathology" or "dataset_origin"
            "param_grid": [
                {
                    "clf__strategy": ["most_frequent"],
                }
            ],
        }

    task = config.get("task", "pathology")

    if task == "dataset_origin":
        X, _y_path, groups, _pathologies, dataset_names = build_dataset(
            excluded_pathologies=config.get("excluded_pathologies", set()),
            include_age=config.get("include_age", True),
            include_sex=config.get("include_sex", True),
            allowed_vowels=config.get("allowed_vowels", {"a"}),
            allowed_intonations=config.get("allowed_intonations", {"n"}),
            selected_datasets=config.get("selected_datasets", {"SVD", "VOICED"}),
            selected_fs=config.get("selected_fs", 50000),
            drop_nan_f0=config.get("drop_nan_f0", True),
            return_dataset_names=True,
        )
        y_raw = np.asarray(dataset_names)
    else:
        X, y, groups, _pathologies = build_dataset(
            excluded_pathologies=config.get("excluded_pathologies", set()),
            include_age=config.get("include_age", True),
            include_sex=config.get("include_sex", True),
            allowed_vowels=config.get("allowed_vowels", {"a"}),
            allowed_intonations=config.get("allowed_intonations", {"n"}),
            selected_datasets=config.get("selected_datasets", {"SVD", "VOICED"}),
            selected_fs=config.get("selected_fs", 50000),
            drop_nan_f0=config.get("drop_nan_f0", True),
        )
        y_raw = np.asarray(y)

    groups = LabelEncoder().fit_transform(np.asarray(groups))
    y = LabelEncoder().fit_transform(y_raw)

    pipe = build_pipeline(random_state=config.get("random_state", 42))
    cv = config.get("cv", StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.get("random_state", 42)))

    param_grid = config.get("param_grid", [{"clf__strategy": ["most_frequent"]}])

    scoring = {
        "acc": make_scorer(accuracy_score),
        "bal_acc": make_scorer(balanced_accuracy_score),
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit="bal_acc",
        verbose=2,
        error_score="raise",
    )

    n_candidates = len(list(ParameterGrid(param_grid)))
    n_folds = cv.get_n_splits(X, y, groups)
    total = n_candidates * n_folds

    start_time = time.time()
    with tqdm_from_verbose(total=total):
        gs.fit(X, y, groups=groups)
    end_time = time.time()

    print("Fit time:", end_time - start_time)
    print("Best params:", gs.best_params_)
    print("Best bal_acc:", gs.best_score_)

    run_dir = log_results(
        gs.cv_results_,
        config.get("experiment_name", "exp"),
        experiment_description=config.get("experiment_description"),
        config=config,
    )

    # jen ať je vidět kam se to uložilo
    (run_dir / "done.txt").write_text("OK\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run minimal voice pipeline.")
    parser.add_argument(
        "config_file",
        nargs="?",
        type=str,
        default="config.py",
        help="Path to python file to load configuration from (default: config.py)",
    )
    _ = parser.parse_args()

    parsed_config = import_module("config_experimental").config
    print(parsed_config)
    main(parsed_config)