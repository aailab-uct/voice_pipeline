"""Utility functions and classes for the project."""

from tqdm import tqdm
import sys
from contextlib import contextmanager
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import math


class ReplaceNaN(BaseEstimator, TransformerMixin):
    """Replace invalid numeric values with zero.

    NaN, positive infinity, and negative infinity are converted to ``0.0``.
    """

    def fit(self, X, y=None):
        del X, y  # unused
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@contextmanager
def tqdm_from_verbose(total, desc="GridSearch", unit="fit"):
    """Display a tqdm progress bar for ``GridSearchCV``.

    Intercepts stdout and increments the progress bar each time
    GridSearchCV's verbose output emits a ``[CV] END`` line.

    Parameters
    ----------
    total : int
        Expected number of fits (n_candidates * n_folds).
    desc : str, default="GridSearch"
        Label shown in the progress bar.
    unit : str, default="fit"
        Unit label for each step.

    Example
    -------
    >>> with tqdm_from_verbose(total=n_candidates * n_folds):
    ...     grid_search.fit(X, y, groups=groups)
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


def _floor_power_of_2(n: int) -> int:
    """Round ``n`` down to the nearest power of 2.

    Parameters
    ----------
    n : int
        Input value. Values below 1 are clamped to 1.

    Returns
    -------
    int
        Largest power of 2 that is less than or equal to n.

    Examples
    --------
    >>> _floor_power_of_2(2048)
    2048
    >>> _floor_power_of_2(1966)
    1024
    >>> _floor_power_of_2(655)
    512
    """
    return 2 ** int(math.floor(math.log2(max(n, 1))))


def log_cv_class_distribution(
    y: np.ndarray,
    groups: np.ndarray,
    cv,
    output_path: Path,
    label_names: dict | None = None,
) -> None:
    """Save per-fold class distributions to a text file.

    For each CV fold the function writes how many healthy and pathological
    samples fall into the train and test split, together with per-split
    totals and the overall dataset summary.

    Parameters
    ----------
    y : numpy.ndarray
        Binary label array (0 = healthy, 1 = pathological).
    groups : numpy.ndarray
        Speaker / group identifiers forwarded to the CV splitter.
    cv : cross-validator
        Any sklearn CV object with a ``split(X, y, groups)`` method.
    output_path : pathlib.Path
        File path where the report is written.
    label_names : dict or None, default=None
        Mapping ``{encoded_value: display_name}``.

    Example
    -------
    >>> log_cv_class_distribution(y, groups, cv, run_dir / "cv_distribution.txt")
    """
    if label_names is None:
        label_names = {0: "healthy", 1: "pathological"}

    healthy_label = next(k for k, v in label_names.items() if v == "healthy")
    patho_label = next(k for k, v in label_names.items() if v == "pathological")

    lines = []
    header = f"{"Fold":<6} {"Split":<8} {"Healthy":>10} {"Pathological":>14} {"Total":>8} {"Ratio H:P":>12}"
    lines.append(header)
    lines.append("-" * len(header))

    # Dummy X: cv.split only needs y and groups.
    X_dummy = np.zeros((len(y), 1))

    for fold, (tr, te) in enumerate(cv.split(X_dummy, y, groups)):
        for split_name, idx in [("train", tr), ("test", te)]:
            n_healthy = int((y[idx] == healthy_label).sum())
            n_patho = int((y[idx] == patho_label).sum())
            total = len(idx)
            ratio = f"{n_healthy}:{n_patho}"
            lines.append(f"{fold:<6} {split_name:<8} {n_healthy:>10} {n_patho:>14} {total:>8} {ratio:>12}")

    # overall summary
    lines.append("-" * len(header))
    n_healthy_total = int((y == healthy_label).sum())
    n_patho_total = int((y == patho_label).sum())
    lines.append(
        f"{"ALL":<6} {"total":<8} {n_healthy_total:>10} {n_patho_total:>14}"
        f" {len(y):>8} {n_healthy_total}:{n_patho_total:>1}"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
