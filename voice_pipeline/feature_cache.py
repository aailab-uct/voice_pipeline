"""Disk-backed cache for expensive feature extraction."""

from pathlib import Path
import pickle
import gzip


class FeatureCache:
    """Cache extracted feature vectors by extractor name and parameters."""

    def __init__(self):
        """Load an existing feature cache or create an empty one."""
        self.cache_file = Path("feature_cache.pkl.gz")
        if self.cache_file.exists():
            with gzip.open(self.cache_file, "rb") as f:
                self.cache = pickle.load(f)
                print(f"Cache size: {len(self.cache)} items.")

        else:
            self.cache = {}

    def flush_to_disk(self):
        """Write the current cache to disk.

        Notes
        -----
        This method is intended to be registered with ``atexit.register``.
        """
        with gzip.open(self.cache_file, "wb") as f:
            pickle.dump(self.cache, f)
        print("Feature cache flushed to disk.")
        print(f"Cache size: {len(self.cache)} items.")

    def cache_method(self, func):
        """Wrap an extraction method with cache lookup.

        Parameters
        ----------
        func : callable
            Class method that computes features from a hashable waveform and
            hashable parameter dictionary.

        Returns
        -------
        callable
            Wrapped method that returns cached results when available.
        """

        def wrapper(cls, x, kwargs):
            name = cls.__name__
            key = (name, x, kwargs)
            if key in self.cache:
                return self.cache[key]
            result = func(cls, x, kwargs)
            self.cache[key] = result
            return result

        return wrapper

    def invalidate(self, cls_name=None):
        """Remove cached entries.

        Parameters
        ----------
        cls_name : str or None, default=None
            Extractor class name to invalidate. ``None`` clears the full cache.
        """
        if cls_name is None:
            self.cache.clear()
        else:
            keys_to_delete = [key for key in self.cache.keys() if key[0] == cls_name]
            for key in keys_to_delete:
                del self.cache[key]
