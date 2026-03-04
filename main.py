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


@contextmanager
def tqdm_from_verbose(total: int):
    pbar = tqdm(total=total)

    orig = GridSearchCV._run_search  # type: ignore[attr-defined]

    def _wrapped(self, evaluate_candidates):
        def _eval(candidates):
            pbar.update(len(candidates))
            return evaluate_candidates(candidates)
        return orig(self, _eval)

    GridSearchCV._run_search = _wrapped  # type: ignore[attr-defined]
    try:
        yield
    finally:
        GridSearchCV._run_search = orig  # type: ignore[attr-defined]
        pbar.close()


def main(config: dict[str, Any]) -> None:
    task = config.get("task", "pathology")

    X, y_bin, groups, _pathologies, dataset_names = build_dataset(
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

    y_raw = np.asarray(dataset_names) if task == "dataset_origin" else np.asarray(y_bin)

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

    # feature histograms (scaled; per dataset)
    try:
        X_scaled = _transform_until_scaler(gs.best_estimator_, X)
        feat_names = _feature_names_after_drop_sig(
            gs.best_estimator_,
            include_age=config.get("include_age", True),
            include_sex=config.get("include_sex", True),
        )
        save_feature_histograms(
            X_scaled,
            feat_names,
            dataset_names,
            out_dir=run_dir / "feature_histograms_scaled",
            bins=config.get("hist_bins", 60),
        )
    except Exception as e:
        print("Histogram export failed:", repr(e))

    # jen ať je vidět kam se to uložilo
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