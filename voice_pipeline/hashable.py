"""Hashable wrappers used by the feature cache."""

import numpy as np
from xxhash import xxh128


class HashableArray:
    """Wrap a NumPy array so it can be used as a dictionary key.

    Parameters
    ----------
    array : numpy.ndarray
        Array to hash by its byte representation.
    """

    def __init__(self, array: np.ndarray):
        self.array = array
        self.hash = xxh128(array.tobytes()).intdigest()

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.hash == other.hash


class HashableDict(dict):
    """Dictionary that hashes its key-value items.

    Notes
    -----
    Values must themselves be hashable.
    """

    def __hash__(self):
        return hash(frozenset(self.items()))
