"""Module for feature caching functionality."""

from pathlib import Path
import pickle
import gzip


class FeatureCache:
    """Class for caching extracted features to disk."""

    def __init__(self):
        """Initialize the feature cache."""
        self.hits = 0
        self.misses = 0
        self.cache_file = Path("feature_cache.pkl.gz")
        if self.cache_file.exists():
            with gzip.open(self.cache_file, "rb") as f:
                self.cache = pickle.load(f)
        else:
            self.cache = {}

    def flush_to_disk(self):
        """Flush the cache to disk. This should be called at program exit - `atexit.register` is recommended."""
        with gzip.open(self.cache_file, "wb") as f:
            pickle.dump(self.cache, f)
        print("Feature cache flushed to disk.")
        print(f"Cache size: {len(self.cache)} items.")

    def cache_method(self, func):
        """Decorator to cache method results based on class name, input, and kwargs."""

        def wrapper(cls, x, kwargs):
            key = (cls.__name__, x.hash, kwargs)
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            result = func(cls, x, kwargs)
            self.cache[key] = result
            return result

        return wrapper

    def print_stats(self, label: str = "") -> None:
        total = self.hits + self.misses
        ratio = self.hits / total if total else 0.0
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}Cache: {len(self.cache)} items | hits {self.hits} / misses {self.misses} ({ratio:.0%})")

    def invalidate(self, cls_name=None):
        """Invalidate cache entries for a specific class or all entries if cls_name is None."""
        if cls_name is None:
            self.cache.clear()
        else:
            keys_to_delete = [key for key in self.cache.keys() if key[0] == cls_name]
            for key in keys_to_delete:
                del self.cache[key]
