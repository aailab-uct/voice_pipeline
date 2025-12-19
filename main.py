from typing import Any
from datetime import datetime
from importlib import import_module
from pathlib import Path
import numpy as np
import argparse
import socket
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd

from feature_extraction import (
    MFCCExtractor,
    DropSignalColumn,
    TimeDomainFeaturesExtractor,
    ZCRExtractor,
    SpectralFeaturesExtractor,
)
from signal_augmenters import RawAugmenter
from feature_augmenters import FeatureAugmenter
from oversampling import RandomCopySampler, SMOTESampler
from dataset_filtration import build_dataset


# ==========
# 1) Build imblearn pipeline
# ==========


def build_pipeline(
    pipeline_params: dict | None = None, class_weight: dict | None | str = None, random_state: int = 42
) -> ImbPipeline:
    """
    imblearn Pipeline:

    RawAugment (waveform) ->
    MFCCExtractor ->
    StandardScaler ->
    FeatureAugment (optional jitter) ->
    SMOTE (train-only resampling) ->
    LogisticRegression
    """
    if pipeline_params is None:
        pipeline_params = {}

    # raise NotImplementedError("This function is under development.")
    pipe = ImbPipeline(
        steps=[
            ("raw_aug", RawAugmenter(enabled=True, noise_std=0.005, random_state=random_state)),
            ("mfcc", MFCCExtractor(sr=50000, n_mfcc=20)),
            ("timedomain", TimeDomainFeaturesExtractor(sr=50000, enabled=True)),
            ("zcr", ZCRExtractor(sr=50000, enabled=True)),
            ("spectral", SpectralFeaturesExtractor(sr=50000, enabled=True)),
            ("drop_sig", DropSignalColumn()),
            ("scaler", StandardScaler()),
            ("feat_aug", FeatureAugmenter(enabled=True, std_factor=0.02, random_state=random_state)),
            ("resampler", SMOTESampler(enabled=True, random_state=random_state, artificial_only=True)),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
            # ("clf", LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=random_state)),
        ]
    )
    return pipe


def make_synthetic_audio_dataset(n_records=60, sr=16000, random_state=0):
    """
    Create a toy dataset:
    - X: each sample is one "recording" and demographic features
    - y: binary classification (healthy/pathological)
    - groups: speaker IDs (1 group per sample)
    """
    rng = np.random.RandomState(random_state)

    X = []
    y = []
    groups = []

    for i in range(n_records):
        label = rng.randint(0, 2)  # binary classes 0 / 1 - healthy/pathological
        sex = rng.choice([0, 1])  # sex feature: 0/1 - male/female
        length = rng.randint(int(0.5 * sr), int(1.5 * sr))  # 0.5–1.5 s
        base = 0.0 if label == 0 else 0.3  # class-dependent mean shift
        age = rng.randint(20, 60)  # random age between 20 and 60
        sig = base + 0.1 * rng.randn(length).astype(np.float32)
        X.append([sig, sex, age])
        y.append(label)
        groups.append(i)  # each recording has its own group ID

    X = np.array(X, dtype=object)
    y = np.array(y)
    groups = np.array(groups)

    return X, y, groups


def log_results(
    cv_results: dict[str, np.ndarray],
    experiment_name: str,
    experiment_description: str | None = None,
    config: dict | None = None,
):
    """
    Log grid search results to a CSV file.
    """
    df = pd.DataFrame(cv_results)
    results_filename = Path(f"results/{experiment_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv")
    print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    results_filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_filename, index=False)

    description_filename = results_filename.with_suffix(".txt")
    with description_filename.open("w") as f:
        # write machine name
        f.write(f"Best UAR: {cv_results["mean_test_uar"].max()}\n")
        f.write(f"Machine: {socket.gethostname()}\n")
        if experiment_description is not None:
            f.write(experiment_description)
        if config is not None:
            f.write("\n\nConfiguration:\n")
            for k, v in config.items():
                if isinstance(v, dict):
                    f.write(f"{k}:\n")
                    for kk, vv in v.items():
                        f.write(f"\t{kk}: {vv}\n")
                f.write(f"{k}: {v}\n")


# ==========
# 3) Main entry point
# ==========
def main(config: dict[str, Any] | None = None):
    """
    Run the pipeline.
    """
    if config is None:
        config = {
            "param_grid": {
                "raw_aug__enabled": [True, False],
                "feat_aug__enabled": [True, False],
                "feat_aug__std_factor": [0.01, 0.02, 0.05],
            },
        }
    X, y, groups = build_dataset(
        excluded_pathologies=config.get("excluded_pathologies", {"Sängerstimme", "Gesangsstimme"}),
        include_age=config.get("include_age", True),
        include_sex=config.get("include_sex", True),
        allowed_vowels=config.get("allowed_vowels", {"a"}),
        allowed_intonations=config.get("allowed_intonations", {"n"}),
        selected_datasets=config.get("selected_datasets", {"SVD"}),
        selected_fs=config.get("selected_fs", 50000),
        drop_nan_f0=config.get("drop_nan_f0", True),
    )

    # 1.1) Encode groups to number
    le = LabelEncoder()
    groups = le.fit_transform(groups)

    # 2) Build pipeline
    class_weight = config.get("class_weight", "balanced")
    assert class_weight in (
        "balanced",
        "balanced_dataset",
        None,
    ), "class_weight must be 'balanced', 'balanced_dataset' or None"
    if class_weight == "balanced_dataset":  # means the weights are computed based on the dataset distribution
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        class_weight = {cls: total / count for cls, count in zip(unique, counts)}
    pipe = build_pipeline(
        config.get("pipeline", {}), class_weight=class_weight, random_state=config.get("random_state", 42)
    )

    # 3) prepare CV
    cv = config.get("cv", StratifiedGroupKFold(n_splits=5))

    # 4) prepare grid search
    param_grid = config["param_grid"]

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring={
            "uar": make_scorer(balanced_accuracy_score),
            "accuracy": make_scorer(accuracy_score),
            "TP": make_scorer(lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]),
            "TN": make_scorer(lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]),
            "FP": make_scorer(lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]),
            "FN": make_scorer(lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]),
        },
        refit=False,
        verbose=2,
        error_score="raise",
        # n_jobs=-1,
    )

    start_time = time.time()
    grid_search.fit(X, y, groups=groups)
    end_time = time.time()
    print("Fit time:", end_time - start_time)
    print("Size of training set:", len(y))
    for i in range(1, 5):
        print(pipe.steps[i][1]._cached_extraction.cache_info())

    log_results(
        grid_search.cv_results_,
        config.get("experiment_name", "exp"),
        experiment_description=config.get("experiment_description"),
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run voice pathology classification pipeline.")
    parser.add_argument(
        "config_file",
        nargs="?",
        type=str,
        default="config.py",
        help="Path to python file to load configuration from (default: config.py)",
    )
    try:
        parsed_config = import_module("config").config
    except AttributeError:
        parsed_config = None
    except ImportError:
        parsed_config = None
    main(parsed_config)
