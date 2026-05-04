"""Run a voice pathology classification experiment.

This root script is the readable experiment protocol. It loads a config,
loads data, builds the pipeline, runs cross-validation, and saves outputs.
Lower-level signal processing and model components live in ``voice_pipeline``.
"""

import yaml
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
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.preprocessing import LabelEncoder

from voice_pipeline.dataset_loading import build_dataset
from voice_pipeline.pipeline import build_pipeline
from voice_pipeline.reporting import repeated_cv_evaluation, save_grid_search_results
from voice_pipeline.utils import log_cv_class_distribution, tqdm_from_verbose
from voice_pipeline.registry import instantiate
from voice_pipeline.visualization import run_visualizations

_NON_PIPELINE_SECTIONS = {"experiment", "dataset", "search", "cross_validation", "visualization"}


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


def load_yaml_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load an experiment configuration from a YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to the ``.yaml`` config file. Defaults to ``"config.yaml"``.

    Returns
    -------
    dict[str, Any]
        Hierarchical configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def extract_param_grid(config: dict[str, Any]) -> list[dict]:
    """Extract a sklearn-style param_grid from list values in the YAML config.

    Any parameter set to a list in the config is treated as a search dimension.
    Scalar values are ignored — they are fixed parameters handled by the pipeline
    directly.

    Parameters
    ----------
    config : dict
        Hierarchical config dict as returned by :func:`load_yaml_config`.

    Returns
    -------
    list[dict]
        A one-element list containing the param_grid dict, as expected by
        ``GridSearchCV``. Empty dict if no list values are found.
    """
    params = {}

    for section, values in config.items():
        if section in _NON_PIPELINE_SECTIONS or not isinstance(values, dict):
            continue

        if section == "features":
            # each feature subsection maps to its own pipeline step name
            for extractor, extractor_params in values.items():
                if not isinstance(extractor_params, dict):
                    continue
                for key, value in extractor_params.items():
                    if isinstance(value, list):
                        params[f"{extractor}__{key}"] = value
        else:
            # top-level pipeline step — section name is the step name
            for key, value in values.items():
                if isinstance(value, list):
                    params[f"{section}__{key}"] = value

    return [params]


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


def choose_refit_metric(cv_cfg: dict[str, Any], scoring) -> str | bool:
    """Choose which model ``GridSearchCV`` should refit.

    Parameters
    ----------
    cv_cfg : dict
        Cross-validation section of the YAML config.
    scoring : dict or str
        Scoring specification returned by :func:`build_scoring`.

    Returns
    -------
    str or bool
        Metric name for multi-metric scoring, otherwise ``True``.
    """
    if isinstance(scoring, dict):
        return cv_cfg.get("refit_metric", "uar")
    return True


def run_experiment(config: dict[str, Any]):
    """Run one configured experiment.

    Parameters
    ----------
    config : dict
        Hierarchical config dict as returned by ``load_yaml_config()``.

    Returns
    -------
    tuple
        Fitted ``GridSearchCV`` instance and the output directory path.
    """
    dataset_cfg = config.get("dataset", {})
    exp_cfg     = config.get("experiment", {})
    cv_cfg      = config.get("cross_validation", {})

    # 1) Load the selected dataset rows and raw waveforms.
    X, y, groups, pathologies, datasets = build_dataset(
        excluded_pathologies=set(dataset_cfg.get("excluded_pathologies", [])),
        include_age=dataset_cfg.get("include_age", True),
        include_sex=dataset_cfg.get("include_sex", True),
        allowed_vowels=set(dataset_cfg.get("vowels", ["a"])),
        allowed_intonations=set(dataset_cfg.get("intonations", ["n"])),
        selected_datasets=set(dataset_cfg.get("selected", ["SVD"])),
        selected_fs=dataset_cfg.get("fs", 48000),
        drop_nan_f0=dataset_cfg.get("drop_nan_f0", True),
        drop_children=dataset_cfg.get("drop_children", True),
    )
    groups = LabelEncoder().fit_transform(groups)

    # 2) Build the sklearn/imblearn pipeline from the YAML config.
    pipe = build_pipeline(config, random_state=42)
    print(pipe)

    # 3) Define cross-validation, scoring, and hyperparameter search.
    cv = instantiate({
        "type": cv_cfg.get("type", "StratifiedGroupKFold"),
        "n_splits": cv_cfg.get("n_splits", 5),
    })
    param_grid = extract_param_grid(config)
    scoring    = build_scoring(cv_cfg.get("scoring_mode", "multimetric"))

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=choose_refit_metric(cv_cfg, scoring),
        verbose=2,
        error_score="raise",
        n_jobs=None,
    )

    # 4) Fit all candidate models.
    n_candidates = len(list(ParameterGrid(param_grid)))
    n_folds      = cv.get_n_splits(X, y, groups)

    start_time = time.time()
    with tqdm_from_verbose(total=n_candidates * n_folds):
        grid_search.fit(X, y, groups=groups)

    print("Fit time:", time.time() - start_time)
    print("Size of training set:", len(y))

    # 5) Save numerical results and diagnostic summaries.
    run_dir = save_grid_search_results(
        grid_search.cv_results_,
        exp_cfg.get("name", "exp"),
        experiment_description=exp_cfg.get("description"),
        config=config,
    )
    log_cv_class_distribution(y, groups, cv, run_dir / "cv_distribution.txt")

    # 6) Save optional plots and repeated-CV checks.
    if hasattr(grid_search, "best_estimator_"):
        run_visualizations(config, grid_search.best_estimator_, X, y, groups, pathologies, datasets, cv, run_dir)

    n_repeats = cv_cfg.get("n_repeats", 0)
    if n_repeats > 0 and hasattr(grid_search, "best_estimator_"):
        df_repeated = repeated_cv_evaluation(
            grid_search.best_estimator_,
            X, y, groups,
            n_splits=cv_cfg.get("n_splits", 5),
            n_repeats=n_repeats,
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
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config_file)
    run_experiment(config)


if __name__ == "__main__":
    main()
