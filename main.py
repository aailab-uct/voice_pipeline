# main.py
"""Main entry point for minimal voice classification pipeline."""

from typing import Any
from datetime import datetime
from importlib import import_module
from pathlib import Path
import numpy as np
import argparse
import socket
import time
import sys
from contextlib import contextmanager

from sklearn.base import clone
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from feature_extraction import (
    MFCCExtractor,
    DropSignalColumn,
    TimeDomainFeaturesExtractor,
    ZCRExtractor,
    SpectralFeaturesExtractor,
    AudioPreprocessor,
)
from dataset_filtration import build_dataset

import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# Při paralelním grid searchi (n_jobs != 1) vypneme joblib cache v feature_extraction,
# protože na Windows způsobuje PermissionError při souběžném čtení .pkl souborů.
import os
os.environ["DISABLE_FEATURE_CACHE"] = "1"


# ============================================================
# Helpers
# ============================================================

class ReplaceNaN(BaseEstimator, TransformerMixin):
    """Replaces NaN, +Inf, -Inf with 0.0"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class FeatureNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalizace příznaků před StandardScalerem.

    Metody (aplikují se v pořadí, pokud jsou zapnuty):
        use_rms — RMS normalizace: každý příznak se škáluje tak aby jeho
                  RMS na trénovacích datech bylo 1.0. Srovnává energetické
                  úrovně příznaků napříč datasety.

    Params:
        enabled:  zapnutí/vypnutí celého stepu
        use_rms:  True/False
    """
    def __init__(self, enabled: bool = True, use_rms: bool = True):
        self.enabled = enabled
        self.use_rms = use_rms

    def fit(self, X, y=None):
        if not self.enabled:
            return self

        X = np.asarray(X, dtype=np.float64)

        if self.use_rms:
            self.rms_ref_ = np.sqrt(np.mean(X ** 2, axis=0))
            # kde je RMS nulové, neškalujeme (=1.0)
            self.rms_ref_ = np.where(self.rms_ref_ > 1e-8, self.rms_ref_, 1.0)

        return self

    def transform(self, X, y=None):
        if not self.enabled:
            return X

        X = np.asarray(X, dtype=np.float64)

        if self.use_rms and hasattr(self, "rms_ref_"):
            rms_current = np.sqrt(np.mean(X ** 2, axis=0))
            scale = np.where(rms_current > 1e-8, self.rms_ref_ / rms_current, 1.0)
            X = X * scale

        return X.astype(np.float32)


def _sanitize_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def _transform_until_scaler(pipeline: ImbPipeline, X: np.ndarray) -> np.ndarray:
    X_tmp = X
    for step_name, step in pipeline.steps:
        if not hasattr(step, "transform"):
            continue
        X_tmp = step.transform(X_tmp)
        if step_name == "scaler":
            break
    return np.asarray(X_tmp, dtype=np.float32)


def _feature_names_after_drop_sig(pipeline: ImbPipeline, include_age: bool, include_sex: bool) -> list[str]:
    names: list[str] = ["signal"]
    if include_age:
        names.append("age")
    if include_sex:
        names.append("sex")

    for step_name in ("mfcc", "timedomain", "zcr", "spectral"):
        step = pipeline.named_steps.get(step_name)
        if step is None:
            continue
        if hasattr(step, "enabled") and not getattr(step, "enabled"):
            continue
        if hasattr(step, "appended_feature_names"):
            names += list(step.appended_feature_names())

    drop = pipeline.named_steps.get("drop_sig")
    if drop is not None and hasattr(drop, "get_feature_names_out"):
        out = drop.get_feature_names_out(names)
        if out is not None:
            return [str(x) for x in list(out)]
    return names[1:]


def save_feature_histograms(
    X_scaled: np.ndarray,
    feature_names: list[str],
    dataset_names: np.ndarray,
    out_dir: Path,
    bins: int = 60,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_values = [str(x) for x in np.unique(dataset_names)]
    for j, fname in enumerate(feature_names):
        plt.figure(figsize=(8, 4.5))
        for ds in ds_values:
            mask = (dataset_names.astype(str) == ds)
            if not np.any(mask):
                continue
            plt.hist(X_scaled[mask, j], bins=bins, alpha=0.5, label=ds)

        plt.title(fname)
        plt.xlabel("scaled value")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        fpath = out_dir / f"{j:03d}_{_sanitize_filename(fname)}.png"
        plt.savefig(fpath, dpi=150)
        plt.close()


def run_domain_feature_importance(
    X: np.ndarray,
    dataset_names: np.ndarray,
    pipeline: ImbPipeline,
    feature_names: list[str],
    out_dir: Path,
    random_state: int = 42,
) -> None:
    X_transformed = _transform_until_scaler(pipeline, X)
    y_domain = LabelEncoder().fit_transform(np.asarray(dataset_names))

    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_transformed, y_domain)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    df = pd.DataFrame({
        "rank": np.arange(1, len(feature_names) + 1),
        "feature": [feature_names[i] for i in indices],
        "importance": importances[indices],
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "feature_importance_dataset_rf.csv", index=False)

    plt.figure(figsize=(12, 8))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in indices],
        rotation=90,
        fontsize=7,
    )
    plt.title("Domain feature importance (RF)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance_dataset_rf.png", dpi=150)
    plt.close()

    print("\nTop 10 domain features:")
    for i in range(min(10, len(feature_names))):
        print(f"  {i+1:2d}. {feature_names[indices[i]]:<25s} {importances[indices[i]]:.4f}")


# ============================================================
# Pipeline
# ============================================================

def build_pipeline(random_state: int = 42) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("audio_prep", AudioPreprocessor(
                enabled=True,
                use_rir_normalization=False,
                use_wiener=False,
                use_spectral_subtraction=True,
                sr=48000,
            )),
            ("mfcc", MFCCExtractor(enabled=False, sr=48000, n_mfcc=20)),
            ("timedomain", TimeDomainFeaturesExtractor(
                sr=48000,
                enabled=True,
                use_skewness=False,
                use_zcr=False,
                use_entropy=False,
                sex_col=1,
                f0_range_m=(75.0, 300.0),
                f0_range_f=(100.0, 600.0),
            )),
            ("zcr", ZCRExtractor(sr=48000, enabled=False)),
            ("spectral", SpectralFeaturesExtractor(sr=48000, enabled=True, use_flatness=False)),
            ("drop_sig", DropSignalColumn()),
            ("replace_nan", ReplaceNaN()),
            ("feat_norm", FeatureNormalizer(enabled=True, use_rms=True)),
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    )


# ============================================================
# Logging
# ============================================================

def log_results(
    cv_results: dict[str, np.ndarray],
    experiment_name: str,
    experiment_description: str | None = None,
    config: dict | None = None,
) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("results") / f"{experiment_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(cv_results).to_csv(run_dir / "gridsearch.csv", index=False)

    with (run_dir / "gridsearch.txt").open("w", encoding="utf-8") as f:
        if "mean_test_bal_acc" in cv_results:
            f.write(f"Best BAL_ACC: {np.nanmax(cv_results['mean_test_bal_acc'])}\n")
        if "mean_test_acc" in cv_results:
            f.write(f"Best ACC:     {np.nanmax(cv_results['mean_test_acc'])}\n")
        if experiment_description:
            f.write(f"Description: {experiment_description}\n")
        if config is not None:
            f.write("\nConfig:\n")
            f.write(repr(config))
            f.write("\n")
        f.write("\n\nSystem:\n")
        f.write(f"Host: {socket.gethostname()}\n")
        f.write(f"Python: {sys.version}\n")

    return run_dir


# ============================================================
# tqdm progress bar pro GridSearchCV
# ============================================================

@contextmanager
def tqdm_from_verbose(total: int, desc: str = "GridSearch"):
    """Zobrazí tqdm progress bar sledováním verbose výstupu GridSearchCV."""
    pbar = tqdm(total=total, desc=desc, unit="fit")
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


# ============================================================
# Main
# ============================================================

def main(config: dict[str, Any]) -> None:
    task = config.get("task", "pathology")

    X, y_bin, groups, _pathologies, dataset_names = build_dataset(
        excluded_pathologies=config.get("excluded_pathologies", set()),
        include_age=config.get("include_age", True),
        include_sex=config.get("include_sex", True),
        allowed_vowels=config.get("allowed_vowels", {"a"}),
        allowed_intonations=config.get("allowed_intonations", {"n"}),
        selected_datasets=config.get("selected_datasets", {"SVD", "VOICED"}),
        selected_fs=config.get("selected_fs", 22000),
        drop_nan_f0=config.get("drop_nan_f0", True),
        return_dataset_names=True,
    )

    y_raw = np.asarray(dataset_names) if task == "dataset_origin" else np.asarray(y_bin)
    groups = LabelEncoder().fit_transform(np.asarray(groups))
    y = LabelEncoder().fit_transform(y_raw)

    pipe = build_pipeline(random_state=config.get("random_state", 42))
    cv = config.get("cv", StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.get("random_state", 42)))

    param_grid = config.get("param_grid", [{}])

    scoring = {
        "bal_acc": make_scorer(balanced_accuracy_score),
        "acc": make_scorer(accuracy_score),
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit="bal_acc",
        verbose=2,
        error_score="raise",
        n_jobs=-1,
    )

    n_candidates = len(list(ParameterGrid(param_grid)))
    n_folds = cv.get_n_splits(X, y, groups)
    total = n_candidates * n_folds

    print(f"\nGrid search: {n_candidates} kombinací × {n_folds} foldů = {total} fitů")

    start_time = time.time()
    with tqdm_from_verbose(total=total):
        grid_search.fit(X, y, groups=groups)
    end_time = time.time()

    print(f"\nFit time: {end_time - start_time:.1f}s")
    print(f"Nejlepší bal_acc: {grid_search.best_score_:.4f}")
    print(f"Nejlepší parametry: {grid_search.best_params_}")

    run_dir = log_results(
        grid_search.cv_results_,
        config.get("experiment_name", "exp"),
        experiment_description=config.get("experiment_description"),
        config=config,
    )

    print(f"\nVýsledky uloženy do: {run_dir}")

    # Post-processing pro nejlepší estimator
    best_estimator_ = grid_search.best_estimator_
    try:
        feat_names = _feature_names_after_drop_sig(
            best_estimator_,
            include_age=config.get("include_age", True),
            include_sex=config.get("include_sex", True),
        )
        run_domain_feature_importance(
            X=X,
            dataset_names=dataset_names,
            pipeline=best_estimator_,
            feature_names=feat_names,
            out_dir=run_dir,
            random_state=config.get("random_state", 42),
        )
        X_scaled = _transform_until_scaler(best_estimator_, X)
        save_feature_histograms(
            X_scaled,
            feat_names,
            dataset_names,
            out_dir=run_dir / "feature_histograms_scaled",
            bins=config.get("hist_bins", 60),
        )
    except Exception as e:
        print("Post-processing failed:", repr(e))

    (run_dir / "done.txt").write_text("OK\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run minimal voice pipeline.")
    parser.add_argument(
        "config_file",
        nargs="?",
        type=str,
        default="config_experimental",
        help="Python module path to config dict, e.g. configs.exp_0",
    )
    args = parser.parse_args()

    mod = import_module(args.config_file)
    cfg = getattr(mod, "config")

    main(cfg)