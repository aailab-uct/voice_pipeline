"""Main entry point for voice pathology classification pipeline."""

from typing import Any
from datetime import datetime
from importlib import import_module
from pathlib import Path
import numpy as np
import argparse
import socket
import time

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, ParameterGrid
from sklearn.base import clone
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
from visualize import run_visualizations

from feature_extraction import (
    MFCCExtractor,
    DropSignalColumn,
    TimeDomainFeaturesExtractor,
    ZCRExtractor,
    SpectralFeaturesExtractor,
    OpenSmileExtractor,
)
from signal_augmenters import RawAugmenter, WaveformSynthesizer
from feature_augmenters import FeatureAugmenter
from oversampling import SMOTESampler
from dataset_filtration import build_dataset
from sklearn.ensemble import RandomForestClassifier
from utils import (tqdm_from_verbose, ReplaceNaN, log_cv_class_distribution)

import warnings

# shouldnt have impact on calculations
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# ==========
# 1) Build imblearn pipeline
# ==========
def build_pipeline(pipeline_params: dict | None = None, random_state: int = 42) -> ImbPipeline:
    """
    Build an imblearn pipeline for voice pathology classification.

    Pipeline steps (in order):
        raw_aug      – waveform augmentation; modifies every sample (train + test)
        synthesizer  – waveform synthesis; adds n_samples new rows (train only)
        mfcc         – MFCC feature extraction
        timedomain   – time-domain features (F0, jitter, shimmer, HNR, …)
        zcr          – zero-crossing rate
        spectral     – spectral features (flatness, rolloff, centroid, contrast)
        opensmile    – OpenSMILE feature extraction (disabled by default)
        drop_sig     – drops raw signal column after feature extraction
        replace_nan  – replaces NaN / ±Inf with 0.0
        scaler       – StandardScaler
        selector     – optional feature selection (default: passthrough)
        feat_aug     – feature-space Gaussian jitter (train + test)
        resampler    – SMOTE / VAE oversampling (train only)
        clf          – classifier (default: RandomForestClassifier)

    Synthetic audio safety
    ----------------------
    WaveformSynthesizer inherits from imblearn BaseSampler, so ImbPipeline
    calls its fit_resample() only on the TRAIN fold.  Synthetic waveforms
    are therefore NEVER present in the test fold and NEVER influence the
    evaluation metrics.

    Cache invalidation
    ------------------
    Feature extractors cache results keyed on the raw waveform array.
    When raw_aug or synthesizer is enabled, the waveforms change each fit,
    so the cache would return stale features.  build_pipeline detects this
    and sets cache_disabled=True on all extractors automatically.

    Parameters
    ----------
    pipeline_params : dict, optional
        Flat sklearn-style parameter dict passed to ``pipe.set_params()``.
        Special keys (handled before set_params, not forwarded):
            ``selector__instance``    – feature selector object
            ``selector__enabled``     – if False, selector → passthrough
            ``clf__instance``         – classifier object
            ``synthesizer__instance`` – WaveformSynthesizer object
    random_state : int, optional
        Random state forwarded to all stochastic steps. Default: 42.

    Returns
    -------
    ImbPipeline
        Configured but unfitted pipeline.
    """
    if pipeline_params is None:
        pipeline_params = {}

    # If either waveform augmenter is turned on, feature caches must be
    # disabled — otherwise extractors would return stale cached features
    # computed from the un-augmented originals.
    _raw_aug_on = pipeline_params.get("raw_aug__enabled", False)
    _synth_on = (
        pipeline_params.get("synthesizer__enabled", False)
        and pipeline_params.get("synthesizer__n_samples", 0) > 0
    )
    _disable_cache = bool(_raw_aug_on or _synth_on)

    pipe = ImbPipeline(
        steps=[
            # --- waveform augmentation (before feature extraction) ---
            ("raw_aug",     RawAugmenter(random_state=random_state)),
            # WaveformSynthesizer is an imblearn Sampler: ImbPipeline will
            # call fit_resample() on it, which runs only on the train fold.
            # Placed as a direct step (like resampler) so all parameters
            # are accessible via synthesizer__* in param_grid.
            ("synthesizer", WaveformSynthesizer(enabled=False, n_samples=0,
                                                random_state=random_state)),
            # --- feature extraction ---
            # cache_disabled is set to True when any waveform augmentation
            # is active so the cache never serves stale features.
            ("mfcc",        MFCCExtractor(cache_disabled=_disable_cache)),
            ("timedomain",  TimeDomainFeaturesExtractor(cache_disabled=_disable_cache)),
            ("zcr",         ZCRExtractor(cache_disabled=_disable_cache)),
            ("spectral",    SpectralFeaturesExtractor(cache_disabled=_disable_cache)),
            ("opensmile",   OpenSmileExtractor(cache_disabled=_disable_cache)),
            # --- post-extraction ---
            ("drop_sig",    DropSignalColumn()),
            ("replace_nan", ReplaceNaN()),
            ("scaler",      StandardScaler()),
            ("selector",    pipeline_params.get("selector__instance", "passthrough")),
            ("feat_aug",    FeatureAugmenter(random_state=random_state)),
            ("resampler",   SMOTESampler(random_state=random_state)),
            ("clf",         pipeline_params.get(
                                "clf__instance",
                                RandomForestClassifier(random_state=random_state, n_jobs=-1),
                            )),
        ]
    )

    if pipeline_params:
        params = dict(pipeline_params)

        # --- keys handled at construction time, must not reach set_params ---
        selector_enabled = params.pop("selector__enabled", True)
        params.pop("selector__instance", None)
        params.pop("clf__instance", None)

        if not selector_enabled:
            pipe.set_params(selector="passthrough")
            for k in list(params.keys()):
                if k.startswith("selector__"):
                    params.pop(k)

        pipe.set_params(**params)

    return pipe


def log_results(
    cv_results: dict[str, np.ndarray],
    experiment_name: str,
    experiment_description: str | None = None,
    config: dict | None = None,
) -> Path:
    """
    Save cross-validation results to a timestamped directory.

    Creates a new directory under ``results/<experiment_name>_<timestamp>/``
    containing two files:
    - ``gridsearch.csv`` – full cv_results as a CSV
    - ``gridsearch.txt`` – human-readable summary with best UAR/accuracy,
      hostname, optional description, and configuration dump

    Parameters
    ----------
    cv_results : dict[str, np.ndarray]
        Output of ``GridSearchCV.cv_results_``.
    experiment_name : str
        Used as a prefix for the results directory name.
    experiment_description : str, optional
        Free-form text appended to the summary file.
    config : dict, optional
        Experiment configuration written to the summary file.

    Returns
    -------
    Path
        Path to the created run directory.
    """
    df = pd.DataFrame(cv_results)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("results") / f"{experiment_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results_csv = run_dir / "gridsearch.csv"
    df.to_csv(results_csv, index=False)

    results_txt = run_dir / "gridsearch.txt"
    with results_txt.open("w") as f:
        if "mean_test_uar" in cv_results:
            f.write(f"Best UAR: {np.nanmax(cv_results["mean_test_uar"])}\n")
        if "mean_test_accuracy" in cv_results:
            f.write(f"Best ACC: {np.nanmax(cv_results["mean_test_accuracy"])}\n")
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


def repeated_cv_evaluation(
    best_estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 10,
    n_repeats: int = 1000,
) -> pd.DataFrame:
    """
    Repeated stratified group k-fold evaluation of a fitted estimator.

    Each repeat uses a different random_state for the CV split, producing
    n_repeats × n_splits rows in the output.

    Returns
    -------
    pd.DataFrame with columns: repeat, fold, UAR, accuracy
    """
    rows = []
    for repeat in tqdm(range(n_repeats), desc="Repeated CV"):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
            est = clone(best_estimator)
            est.fit(X[tr], y[tr])
            yp = est.predict(X[te])
            y_proba = est.predict_proba(X[te])

            uar = balanced_accuracy_score(y[te], yp)
            acc = accuracy_score(y[te], yp)
            if y_proba.shape[1] == 2:
                auc = roc_auc_score(y[te], y_proba[:, 1])
            else:
                auc = roc_auc_score(y[te], y_proba, multi_class="ovr", average="macro")
            rows.append((repeat, fold, uar, acc, auc))

    return pd.DataFrame(rows, columns=["repeat", "fold", "UAR", "accuracy", "AUC"])


# ==========
# 3) Main entry point
# ==========
def main(config: dict[str, Any] | None = None):
    """
    Run the pipeline.
    """
    label_mode = config.get("label_mode", "dataset")
    X, y, groups, pathologies, datasets = build_dataset(
        excluded_pathologies=config["excluded_pathologies"],
        include_age=config["include_age"],
        include_sex=config["include_sex"],
        allowed_vowels=config["allowed_vowels"],
        allowed_intonations=config["allowed_intonations"],
        selected_datasets=config["selected_datasets"],
        selected_fs=config["selected_fs"],
        drop_nan_f0=config.get("drop_nan_f0", True),
        label_mode=label_mode,
    )

    # 1.1) Encode groups to number
    groups = LabelEncoder().fit_transform(groups)

    # 2) Build pipeline
    pipe = build_pipeline(config.get("pipeline", {}), random_state=config.get("random_state", 42))

    # 3) prepare CV
    cv = config.get("cv", StratifiedGroupKFold(n_splits=5))

    # 4) prepare grid search
    param_grid = config["param_grid"]

    scoring_mode = config.get("scoring_mode", "multimetric")

    if scoring_mode == "multimetric":
        if label_mode == "dataset":
            scoring = {
                "uar": make_scorer(balanced_accuracy_score),
                "accuracy": make_scorer(accuracy_score),
                "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba",
                                        multi_class="ovr", average="macro"),
            }
        else:
            scoring = {
                "uar": make_scorer(balanced_accuracy_score),
                "accuracy": make_scorer(accuracy_score),
                "TP": make_scorer(lambda yt, yp: confusion_matrix(yt, yp)[1, 1]),
                "TN": make_scorer(lambda yt, yp: confusion_matrix(yt, yp)[0, 0]),
                "FP": make_scorer(lambda yt, yp: confusion_matrix(yt, yp)[0, 1]),
                "FN": make_scorer(lambda yt, yp: confusion_matrix(yt, yp)[1, 0]),
                "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
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
        n_jobs=None,  # -1 causes issues with caching as the forked processes copy the cache.
    )

    # calculate number of iterations for tqdm in gridsearch
    n_candidates = len(list(ParameterGrid(param_grid)))
    n_folds = cv.get_n_splits(X, y, groups)
    total_training_steps = n_candidates * n_folds

    start_time = time.time()
    with tqdm_from_verbose(total=total_training_steps):
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

    log_cv_class_distribution(y, groups, cv, run_dir / "cv_distribution.txt")

    run_visualizations(config, grid_search.best_estimator_, X, y, groups, pathologies, datasets, cv, run_dir)

    if config.get("n_repeats", 0) > 0:
        n_splits = config.get("cv").get_n_splits() if hasattr(config.get("cv"), "get_n_splits") else 10
        df_repeated = repeated_cv_evaluation(
            grid_search.best_estimator_,
            X,
            y,
            groups,
            n_splits=n_splits,
            n_repeats=config.get("n_repeats"),
        )
        df_repeated.to_csv(run_dir / "repeated_cv.csv", index=False)
        print(f"Repeated CV UAR: {df_repeated["UAR"].mean():.4f} ± {df_repeated["UAR"].std():.4f}")


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