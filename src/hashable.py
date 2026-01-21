import numpy as np


class HashableArray:
    """Hashable wrapper for numpy arrays (content-based, pickle-friendly)."""

    def __init__(self, array: np.ndarray):
        a = np.asarray(array)
        self.array = np.ascontiguousarray(a)

    def __hash__(self) -> int:
        a = self.array
        return hash((a.dtype.str, a.shape, a.tobytes()))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HashableArray) and np.array_equal(self.array, other.array)


class HashableDict(dict):
    """Hashable dict suitable for **kwargs unpacking."""

    def __hash__(self) -> int:
        # assumes keys/values are hashable (true for your params: sr, n_mfcc, fmin, fmax, ...)
        return hash(tuple(sorted(self.items())))