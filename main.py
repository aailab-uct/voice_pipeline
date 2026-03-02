"""Main entry point for voice pathology classification pipeline."""

from typing import Any
from datetime import datetime
from importlib import import_module
from pathlib import Path
import numpy as np
import argparse
import socket
import time

from sklearn.base import clone
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import sys
from contextlib import contextmanager
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid


from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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
    OpenSmileExtractor
)
from signal_augmenters import RawAugmenter
from feature_augmenters import FeatureAugmenter
from oversampling import SMOTESampler, VAESampler
from dataset_filtration import build_dataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV


import warnings

# shouldnt have impact on calculations
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# ==========
# 1) Build imblearn pipeline
# ==========

class ReplaceNaN(BaseEstimator, TransformerMixin):
    """
    Replaces NaN, +Inf, -Inf with 0.0
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def build_pipeline(pipeline_params: dict | None = None, random_state: int = 42) -> ImbPipeline:
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

    pipe = ImbPipeline(
        steps=[
            ("raw_aug", RawAugmenter(enabled=False, noise_std=0.005, random_state=random_state)),
            ("mfcc", MFCCExtractor(enabled=True, sr=50000, n_mfcc=20)),
            ("timedomain", TimeDomainFeaturesExtractor(sr=50000, enabled=True, sex_col=1,
                                                       f0_range_m=(75.0, 300.0),
                                                       f0_range_f=(100.0, 600.0),)),
            ("zcr", ZCRExtractor(sr=50000, enabled=True)),
            ("spectral", SpectralFeaturesExtractor(sr=50000, enabled=True)),
            ("opensmile", OpenSmileExtractor(
                enabled=False,
                sr=50000,
                feature_set="GeMAPS", # GeMAPS, ComParE_2016
                feature_level="functionals",
            )),
            ("drop_sig", DropSignalColumn()),
            ("replace_nan", ReplaceNaN()),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=mutual_info_classif, k=20)),
            # ("selector", RFECV(
            #     estimator=RandomForestClassifier(
            #         n_estimators=300,
            #         random_state=42,
            #         n_jobs=-1,
            #         class_weight="balanced",
            #     ),
            #     step=0.1,
            #     cv=5,
            #     scoring="balanced_accuracy",
            #     n_jobs=-1,
            # )),

            # ("selector", SelectFromModel(
            #     estimator=RandomForestClassifier(
            #         n_estimators=300,
            #         random_state=42,
            #         n_jobs=-1,
            #         class_weight="balanced",
            #     ),
            #     threshold="median",
            # )),
            ("feat_aug", FeatureAugmenter(enabled=True, std_factor=0.02, random_state=random_state)),
            # artificial only means that real samples are discarded
            # sampling strategi set how much there will be samples in the end, {0:1000, 1:1000} for each class
            (
                "resampler",
                #SMOTESampler(enabled=True, random_state=random_state, artificial_only=True, sampling_strategy="auto"),
                VAESampler(enabled=True, random_state=random_state, artificial_only=True, sampling_strategy="auto"),
            ),
            # Placeholder classifier.
            # The real classifier (e.g. KNN or KAN) is injected later by GridSearchCV
            # via the "clf" parameter in param_grid.
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    )
    # take settings from config if config is available
    # if config is not available or is empty, tak settings of pipe from above
    if pipeline_params:
        # --- handle selector enable/disable safely ---
        params = dict(pipeline_params)

        selector_enabled = params.pop("selector__enabled", True)

        if not selector_enabled:
            # Switch selector to passthrough
            pipe.set_params(selector="passthrough")
            # Remove any selector__* params (e.g., selector__k), otherwise sklearn will crash
            for k in list(params.keys()):
                if k.startswith("selector__"):
                    params.pop(k, None)

        pipe.set_params(**params)

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
) -> Path:

    df = pd.DataFrame(cv_results)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("results") / f"{experiment_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results_csv = run_dir / "gridsearch.csv"
    df.to_csv(results_csv, index=False)

    results_txt = run_dir / "gridsearch.txt"
    with results_txt.open("w") as f:
        if "mean_test_uar" in cv_results:
            f.write(f"Best UAR: {np.nanmax(cv_results['mean_test_uar'])}\n")
        if "mean_test_accuracy" in cv_results:
            f.write(f"Best ACC: {np.nanmax(cv_results['mean_test_accuracy'])}\n")
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
                else:
                    f.write(f"{k}: {v}\n")

    return run_dir

def per_pathology_uar(best_estimator, X, y, groups, pathologies, cv, healthy_name="healthy"):
    y_true_all, y_pred_all, p_all = [], [], []

    for tr, te in cv.split(X, y, groups):
        est = clone(best_estimator)
        est.fit(X[tr], y[tr])
        yp = est.predict(X[te])

        y_true_all.append(y[te])
        y_pred_all.append(yp)
        p_all.append(pathologies[te])

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    p_all = np.concatenate(p_all)

    rows = []
    for p in np.unique(p_all):
        if p.lower() == healthy_name:
            continue

        m = (p_all == p) | (np.char.lower(p_all) == healthy_name)

        yt = y_true_all[m]
        yp = y_pred_all[m]

        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())

        r0, r1 = recall_score(yt, yp, labels=[0, 1], average=None, zero_division=0)
        uar = 0.5 * (r0 + r1)

        rows.append((
            p, uar,
            tp, fn, fp, tn,
            int((p_all[m] == p).sum()),
            int((np.char.lower(p_all[m]) == healthy_name).sum())
        ))

    df = pd.DataFrame(rows, columns=[
        "pathology", "UAR",
        "TP", "FN", "FP", "TN",
        "n_pathology", "n_healthy"
    ])

    return df.sort_values("UAR", ascending=False)

def plot_uar_by_pathology(dfp, savepath=None):
    d = dfp.sort_values("UAR", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))

    plt.figure(figsize=(8, max(4, 0.3 * len(d))))
    plt.barh(y, d["UAR"].values)

    plt.yticks(y, d["pathology"].values, fontsize=7)

    # přidáme počet patologických vzorků
    for i, (uar, n_pat) in enumerate(zip(d["UAR"], d["n_pathology"])):
        plt.text(
            min(0.99, uar + 0.01),
            i,
            f"{uar:.2f} (n={int(n_pat)})",
            va="center",
            fontsize=7
        )

    plt.xlim(0, 1)
    plt.xlabel("UAR")
    plt.title("UAR per pathology")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)

    # plt.show()

@contextmanager
def tqdm_from_verbose(total, desc="GridSearch", unit="fit"):
    """
    Help functions for tqdm progress bar for grid search.
    It scann output for gridsearch verbose and count
    them and by that update progress ar
    """
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

# ==========
# 3) Main entry point
# ==========
def main(config: dict[str, Any] | None = None):
    """
    Run the pipeline.
    """
    if config is None:
        config = {
            "param_grid": [
                {
                    "clf": [KNeighborsClassifier()],
                    "clf__n_neighbors": [3, 5],
                    "raw_aug__enabled": [True, False],
                    "feat_aug__enabled": [True, False],
                    "feat_aug__std_factor": [0.01, 0.02, 0.05],
                }
            ]
        }
    X, y, groups, pathologies = build_dataset(
        excluded_pathologies=config.get(
            "excluded_pathologies",
            {
                "Sängerstimme",
                "Gesangsstimme",
                "Sigmatismus",
                "Balbuties",
                "Dysodie",
                "Poltersyndrom",
                "Non-fluency-Syndrom",
            },
        ),
        include_age=config.get("include_age", True),
        include_sex=config.get("include_sex", True),
        allowed_vowels=config.get("allowed_vowels", {"a"}),
        allowed_intonations=config.get("allowed_intonations", {"n"}),
        selected_datasets=config.get("selected_datasets", {"SVD", "VOICED"}),
        selected_fs=config.get("selected_fs", 50000),
        drop_nan_f0=config.get("drop_nan_f0", True),
    )

    # 1.1) Encode groups to number
    le = LabelEncoder()
    groups = le.fit_transform(groups)

    # 2) Build pipeline
    pipe = build_pipeline(config.get("pipeline", {}), random_state=config.get("random_state", 42))

    # 3) prepare CV
    cv = config.get("cv", StratifiedGroupKFold(n_splits=5))

    # 4) prepare grid search
    param_grid = config["param_grid"]

    scoring_mode = config.get("scoring_mode", "multimetric")

    if scoring_mode == "multimetric":
        scoring = {
            "uar": make_scorer(balanced_accuracy_score),
            "accuracy": make_scorer(accuracy_score),
            "TP": make_scorer(lambda yt, yp: confusion_matrix(yt, yp)[1, 1]),
            "TN": make_scorer(lambda yt, yp: confusion_matrix(yt, yp)[0, 0]),
            "FP": make_scorer(lambda yt, yp: confusion_matrix(yt, yp)[0, 1]),
            "FN": make_scorer(lambda yt, yp: confusion_matrix(yt, yp)[1, 0]),
        }
        refit = config.get("refit_metric", False)
    elif scoring_mode == "accuracy_only":
        scoring = "accuracy"
        refit = True
    elif scoring_mode == "uar_only":
        scoring = "balanced_accuracy"
        refit = True
    else:
        raise ValueError(f"Unknown scoring_mode: {scoring_mode}")

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=refit,
        verbose=2,
        error_score="raise",
        # n_jobs=-1, # causes issues with caching as the forked processes copy the cache.
    )

    # calculate number of iterations for tqdm in gridsearch
    n_candidates = len(list(ParameterGrid(param_grid)))
    n_folds = cv.get_n_splits(X, y, groups)
    total = n_candidates * n_folds

    start_time = time.time()
    with tqdm_from_verbose(total=total):
        grid_search.fit(X, y, groups=groups)
    end_time = time.time()
    print("Fit time:", end_time - start_time)
    print("Size of training set:", len(y))

    run_dir = log_results(
        grid_search.cv_results_,
        config.get("experiment_name", "exp"),
        experiment_description=config.get("experiment_description"),
        config=config,
    )

    best = grid_search.best_estimator_
    dfp = per_pathology_uar(best, X, y, groups, pathologies, cv)
    # dfp.to_csv("per_pathology_results.csv", index=False)

    # uložit všechny
    dfp.to_csv(run_dir / "per_pathology_UAR_all.csv", index=False)
    plot_uar_by_pathology(dfp, run_dir / "uar_all.png")

    # jen slabší než většina
    threshold = dfp["UAR"].median()
    dfp_weak = dfp[dfp["UAR"] < threshold]

    dfp_weak.to_csv(run_dir / "per_pathology_UAR_weak.csv", index=False)
    plot_uar_by_pathology(dfp_weak, run_dir / "uar_weak.png")

    # plot_uar_by_pathology(dfp, "uar_per_pathology.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run voice pathology classification pipeline.")
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
