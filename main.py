"""Run a voice pathology classification experiment.

This root script is the readable experiment protocol. It loads a config,
loads data, builds the pipeline, runs cross-validation, and saves outputs.
Lower-level signal processing and model components live in ``voice_pipeline``.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
import argparse
import time

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from voice_pipeline.dataset_loading import build_dataset
from voice_pipeline.pipeline import build_pipeline
from voice_pipeline.reporting import repeated_cv_evaluation, save_grid_search_results
from voice_pipeline.utils import log_cv_class_distribution, tqdm_from_verbose
from voice_pipeline.visualization import run_visualizations


def load_config(config_path: str) -> dict:
    """Load an experiment configuration.

    Parameters
    ----------
    config_path : str
        Path to a Python file that defines a top-level ``config`` dictionary.

    Returns
    -------
    dict
        Experiment configuration.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ImportError
        If the file cannot be loaded as a Python module.
    AttributeError
        If the module does not define ``config``.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    spec = spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config file: {path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "config"):
        raise AttributeError(f"Config file {path} must define a variable named 'config'.")

    return module.config


def build_scoring(scoring_mode: str):
    """Create the scoring scheme used by ``GridSearchCV``.

    Parameters
    ----------
    scoring_mode : str
        One of ``"multimetric"``, ``"accuracy_only"``, or ``"uar_only"``.

    Returns
    -------
    dict or str
        A sklearn-compatible scoring object.

    Raises
    ------
    ValueError
        If ``scoring_mode`` is unknown.
    """
    if scoring_mode == "multimetric":
        return {
            "uar": make_scorer(balanced_accuracy_score),
            "accuracy": make_scorer(accuracy_score),
            "TP": make_scorer(lambda yt, yp: confusion_matrix(yt, yp)[1, 1]),
            "TN": make_scorer(lambda yt, yp: confusion_matrix(yt, yp)[0, 0]),
            "FP": make_scorer(lambda yt, yp: confusion_matrix(yt, yp)[0, 1]),
            "FN": make_scorer(lambda yt, yp: confusion_matrix(yt, yp)[1, 0]),
            "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
        }
    if scoring_mode == "accuracy_only":
        return "accuracy"
    if scoring_mode == "uar_only":
        return "balanced_accuracy"
    raise ValueError(f"Unknown scoring_mode: {scoring_mode}")


def choose_refit_metric(config: dict[str, Any], scoring) -> str | bool:
    """Choose which model ``GridSearchCV`` should refit.

    Parameters
    ----------
    config : dict
        Experiment configuration.
    scoring : dict or str
        Scoring specification returned by :func:`build_scoring`.

    Returns
    -------
    str or bool
        Metric name for multi-metric scoring, otherwise ``True``.
    """
    if isinstance(scoring, dict):
        return config.get("refit_metric", "uar")
    return True


def run_experiment(config: dict[str, Any]):
    """Run one configured experiment.

    Parameters
    ----------
    config : dict
        Experiment configuration.

    Returns
    -------
    tuple
        Fitted ``GridSearchCV`` instance and the output directory path.
    """

    # 1) Load the selected dataset rows and raw waveforms.
    X, y, groups, pathologies, datasets = build_dataset(
        excluded_pathologies=config["excluded_pathologies"],
        include_age=config["include_age"],
        include_sex=config["include_sex"],
        allowed_vowels=config["allowed_vowels"],
        allowed_intonations=config["allowed_intonations"],
        selected_datasets=config["selected_datasets"],
        selected_fs=config["selected_fs"],
        drop_nan_f0=config.get("drop_nan_f0", True),
    )
    groups = LabelEncoder().fit_transform(groups)

    # 2) Build the sklearn/imblearn pipeline from the config.
    pipe = build_pipeline(config.get("pipeline", {}), random_state=config.get("random_state", 42))

    # 3) Define cross-validation, scoring, and hyperparameter search.
    cv = config.get("cv", StratifiedGroupKFold(n_splits=5))
    param_grid = config["param_grid"]
    scoring = build_scoring(config.get("scoring_mode", "multimetric"))

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=choose_refit_metric(config, scoring),
        verbose=2,
        error_score="raise",
        n_jobs=None,
    )

    # 4) Fit all candidate models.
    n_candidates = len(list(ParameterGrid(param_grid)))
    n_folds = cv.get_n_splits(X, y, groups)

    start_time = time.time()
    with tqdm_from_verbose(total=n_candidates * n_folds):
        grid_search.fit(X, y, groups=groups)

    print("Fit time:", time.time() - start_time)
    print("Size of training set:", len(y))

    # 5) Save numerical results and diagnostic summaries.
    run_dir = save_grid_search_results(
        grid_search.cv_results_,
        config.get("experiment_name", "exp"),
        experiment_description=config.get("experiment_description"),
        config=config,
    )
    log_cv_class_distribution(y, groups, cv, run_dir / "cv_distribution.txt")

    # 6) Save optional plots and repeated-CV checks.
    if hasattr(grid_search, "best_estimator_"):
        run_visualizations(config, grid_search.best_estimator_, X, y, groups, pathologies, datasets, cv, run_dir)

    if config.get("n_repeats", 0) > 0 and hasattr(grid_search, "best_estimator_"):
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
        print(f"Repeated CV UAR: {df_repeated["UAR"].mean():.4f} +/- {df_repeated["UAR"].std():.4f}")

    return grid_search, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run voice pathology classification pipeline.")
    parser.add_argument(
        "config_file",
        nargs="?",
        type=str,
        default="config_experimental.py",
        help="Path to Python config file (default: config_experimental.py)",
    )
    args = parser.parse_args()

    config = load_config(args.config_file)
    print(config)
    run_experiment(config)


if __name__ == "__main__":
    main()
