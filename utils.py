from tqdm import tqdm
import sys
from contextlib import contextmanager
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import math

@contextmanager
def tqdm_from_verbose(total, desc="GridSearch", unit="fit"):
    """
    Context manager that displays a tqdm progress bar for GridSearchCV.

    Intercepts stdout and increments the progress bar each time
    GridSearchCV's verbose output emits a ``[CV] END`` line.

    Parameters
    ----------
    total : int
        Expected number of fits (n_candidates * n_folds).
    desc : str, optional
        Label shown in the progress bar. Default: "GridSearch".
    unit : str, optional
        Unit label for each step. Default: "fit".

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


class ReplaceNaN(BaseEstimator, TransformerMixin):
    """
    Replaces NaN, +Inf, -Inf with 0.0
    """

    def fit(self, X, y=None):
        del X, y  # unused
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def _floor_power_of_2(n: int) -> int:
    """
    Round n down to the nearest power of 2.

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