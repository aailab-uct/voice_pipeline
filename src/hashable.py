"""A module providing a hashable wrapper around numpy arrays for caching purposes."""

import numpy as np
from xxhash import xxh128


class HashableArray:
    """This is a wrapper around a numpy array to make it hashable for caching purposes."""

    def __init__(self, array: np.ndarray):
        self.array = array
        self.hash = xxh128(array.tobytes()).intdigest()

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.hash == other.hash


class HashableDict(dict):
    """A dictionary that is hashable by cinverting to  its items."""

    def __hash__(self):
        return hash(frozenset(self.items()))
