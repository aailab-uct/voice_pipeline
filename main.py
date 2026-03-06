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

from sklearn.base import clone
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

class DomainNormalizer(BaseEstimator, TransformerMixin):
    """
    Kombinuje RMS normalizaci a skutečný dvoudoménový CORAL.
    Pokud jsou předány dataset_names, počítá kovarianci zvlášť pro každou doménu (S→T).
    Bez dataset_names fallback na whitening na gaussovské rozložení.
    """
    def __init__(
        self,
        enabled: bool = True,
        rms_enabled: bool = True,
        coral_enabled: bool = True,
        rms_features: list = None,
        reg: float = 1e-5,
        source_domain: str = None,   # název zdrojové domény, None = první abecedně
        target_domain: str = None,   # název cílové domény, None = druhá abecedně
    ):
        self.enabled = enabled
        self.rms_enabled = rms_enabled
        self.coral_enabled = coral_enabled
        self.rms_features = rms_features
        self.reg = reg
        self.source_domain = source_domain
        self.target_domain = target_domain

    def fit(self, X, y=None, dataset_names=None):
        if not self.enabled:
            return self

        X = np.asarray(X, dtype=np.float64)

        if self.rms_enabled:
            cols = self.rms_features if self.rms_features is not None \
                   else np.arange(X.shape[1])
            self.rms_ref_ = np.sqrt(np.mean(X[:, cols] ** 2, axis=0))
            self.rms_ref_ = np.where(self.rms_ref_ > 1e-8, self.rms_ref_, 1.0)

        if self.coral_enabled:
            if dataset_names is not None:
                # skutečný CORAL: S → T
                domains = np.asarray(dataset_names)
                unique = sorted(np.unique(domains))
                src = self.source_domain if self.source_domain is not None else unique[0]
                tgt = self.target_domain if self.target_domain is not None else unique[1]

                mask_s = domains == src
                mask_t = domains == tgt

                C_S = np.cov(X[mask_s], rowvar=False) + self.reg * np.eye(X.shape[1])
                C_T = np.cov(X[mask_t], rowvar=False) + self.reg * np.eye(X.shape[1])

                # whitening ze source
                eigvals_s, eigvecs_s = np.linalg.eigh(C_S)
                eigvals_s = np.maximum(eigvals_s, 1e-10)
                self.whiten_ = eigvecs_s @ np.diag(1.0 / np.sqrt(eigvals_s)) @ eigvecs_s.T

                # re-coloring z target
                eigvals_t, eigvecs_t = np.linalg.eigh(C_T)
                eigvals_t = np.maximum(eigvals_t, 1e-10)
                self.color_ = eigvecs_t @ np.diag(np.sqrt(eigvals_t)) @ eigvecs_t.T

                self._used_two_domain_coral = True
            else:
                # fallback: whitening na gaussovské rozložení
                C = np.cov(X, rowvar=False) + self.reg * np.eye(X.shape[1])
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                self.whiten_ = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                self.color_ = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                self._used_two_domain_coral = False

        return self

    def transform(self, X, y=None):
        if not self.enabled:
            return X

        X = np.asarray(X, dtype=np.float64)

        if self.rms_enabled and hasattr(self, "rms_ref_"):
            cols = self.rms_features if self.rms_features is not None \
                   else np.arange(X.shape[1])
            rms_current = np.sqrt(np.mean(X[:, cols] ** 2, axis=0))
            scale = np.where(rms_current > 1e-8, self.rms_ref_ / rms_current, 1.0)
            X[:, cols] = X[:, cols] * scale

        if self.coral_enabled and hasattr(self, "whiten_"):
            X = X @ self.whiten_ @ self.color_

        return X

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


from sklearn.ensemble import RandomForestClassifier

def build_pipeline(random_state: int = 42) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("mfcc", MFCCExtractor(enabled=True, sr=50000, n_mfcc=20)),
            ("timedomain", TimeDomainFeaturesExtractor(
                sr=50000,
                enabled=True,
                use_skewness=False,
                sex_col=1,
                f0_range_m=(75.0, 300.0),
                f0_range_f=(100.0, 600.0),
            )),
            ("zcr", ZCRExtractor(sr=50000, enabled=True)),
            ("spectral", SpectralFeaturesExtractor(sr=50000, enabled=True, use_flatness=False)),
            ("drop_sig", DropSignalColumn()),
            ("replace_nan", ReplaceNaN()),
            ("domain_norm", DomainNormalizer(
                enabled=True,
                rms_enabled=True,
                coral_enabled=True,
                source_domain="SVD",   # uprav podle svých názvů
                target_domain="AVFAD",
            )),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state)),
        ]
    )


def log_results(
    cv_results: dict[str, np.ndarray],
    experiment_name: str,
    experiment_description: str | None = None,
    config: dict | None = None,
) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("results") / f"{experiment_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # fold-level výsledky jako tabulka
    fold_df = pd.DataFrame({
        "fold": cv_results.get("fold_acc", np.array([])) * 0 + np.arange(len(cv_results.get("fold_acc", []))),
        "acc": cv_results.get("fold_acc", np.array([])),
        "bal_acc": cv_results.get("fold_bal_acc", np.array([])),
    })
    fold_df.to_csv(run_dir / "cv_results.csv", index=False)

    with (run_dir / "gridsearch.txt").open("w", encoding="utf-8") as f:
        if "mean_test_bal_acc" in cv_results:
            f.write(f"Best BAL_ACC: {np.nanmax(cv_results['mean_test_bal_acc'])}\n")
        if "mean_test_acc" in cv_results:
            f.write(f"Mean ACC:     {np.nanmax(cv_results['mean_test_acc'])}\n")
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

def run_domain_feature_importance(
    X: np.ndarray,
    dataset_names: np.ndarray,
    pipeline: ImbPipeline,
    feature_names: list[str],
    out_dir: Path,
    random_state: int = 42,
) -> None:
    from sklearn.ensemble import RandomForestClassifier

    # transformuj příznaky stejnou pipeline (bez clf)
    X_transformed = _transform_until_scaler(pipeline, X)

    y_domain = LabelEncoder().fit_transform(np.asarray(dataset_names))

    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_transformed, y_domain)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # uložit jako CSV
    df = pd.DataFrame({
        "rank": np.arange(1, len(feature_names) + 1),
        "feature": [feature_names[i] for i in indices],
        "importance": importances[indices],
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "feature_importance_dataset_rf.csv", index=False)

    # uložit jako graf
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

    df = pd.DataFrame({
        "dataset": dataset_names,
        "label": y_bin,
        "sex": [row[1] for row in X],
    })

    print(df.groupby(["dataset", "label", "sex"]).size().unstack(fill_value=0))
    print("\nCelkem:")
    print(df.groupby(["dataset", "sex"]).size().unstack(fill_value=0))

    y_raw = np.asarray(dataset_names) if task == "dataset_origin" else np.asarray(y_bin)

    groups = LabelEncoder().fit_transform(np.asarray(groups))
    y = LabelEncoder().fit_transform(y_raw)

    pipe = build_pipeline(random_state=config.get("random_state", 42))
    cv = config.get("cv", StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.get("random_state", 42)))

    fold_results = []
    best_estimator_ = None
    start_time = time.time()

    for fold_i, (train_idx, test_idx) in enumerate(
        tqdm(list(cv.split(X, y, groups)), desc="CV folds")
    ):
        pipe_fold = clone(pipe)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        ds_train = dataset_names[train_idx]

        # fit krok po kroku aby domain_norm dostal dataset_names
        X_tmp = X_train
        for step_name, step in pipe_fold.steps[:-1]:  # vše kromě clf
            if step_name == "domain_norm":
                step.fit(X_tmp, y_train, dataset_names=ds_train)
            else:
                step.fit(X_tmp, y_train)
            X_tmp = step.transform(X_tmp)

        # fit klasifikátoru
        pipe_fold.steps[-1][1].fit(X_tmp, y_train)

        # predikce
        X_tmp_test = X_test
        for step_name, step in pipe_fold.steps[:-1]:
            X_tmp_test = step.transform(X_tmp_test)
        y_pred = pipe_fold.steps[-1][1].predict(X_tmp_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        fold_results.append({"fold": fold_i, "acc": acc, "bal_acc": bal_acc})
        print(f"  Fold {fold_i}: acc={acc:.4f}  bal_acc={bal_acc:.4f}")

        if best_estimator_ is None:
            best_estimator_ = pipe_fold

    end_time = time.time()

    mean_acc = np.mean([r["acc"] for r in fold_results])
    mean_bal_acc = np.mean([r["bal_acc"] for r in fold_results])

    print(f"\nFit time: {end_time - start_time:.1f}s")
    print(f"Mean acc:     {mean_acc:.4f}")
    print(f"Mean bal_acc: {mean_bal_acc:.4f}")

    cv_results = {
        "mean_test_acc": np.array([mean_acc]),
        "mean_test_bal_acc": np.array([mean_bal_acc]),
        "fold_acc": np.array([r["acc"] for r in fold_results]),
        "fold_bal_acc": np.array([r["bal_acc"] for r in fold_results]),
    }

    run_dir = log_results(
        cv_results,
        config.get("experiment_name", "exp"),
        experiment_description=config.get("experiment_description"),
        config=config,
    )

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