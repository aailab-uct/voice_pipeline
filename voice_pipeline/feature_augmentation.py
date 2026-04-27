"""Feature-level augmentation transformers."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """Add small Gaussian jitter to continuous feature columns.

    The augmenter is intended for training data. It estimates feature standard
    deviations during ``fit`` and perturbs non-integer columns during
    ``fit_transform``.
    """

    def __init__(
        self,
        enabled: bool = True,
        std_factor: float = 0.01,
        skip_integer_features: bool = True,
        random_state: None | int = None,
    ) -> None:
        """Initialize a feature augmenter.

        Parameters
        ----------
        enabled : bool, default=True
            Whether to apply augmentation.
        std_factor : float, default=0.01
            Factor multiplied by each feature standard deviation to set the
            noise standard deviation.
        skip_integer_features : bool, default=True
            Whether to leave integer-like columns unchanged.
        random_state : int or None, default=None
            Random seed for reproducible noise.
        """
        super().__init__()
        self.enabled = enabled
        self.std_factor = std_factor
        self.random_state = random_state
        self.skip_integer_features = skip_integer_features
        self.mask_ = None
        self.X_std_ = None
        self.rng_ = None

    def fit(self, X, y):
        del y  # unused
        X_array = np.asarray(X, dtype=np.float32)

        if not self.skip_integer_features:
            mask = np.ones(X_array.shape[1], dtype=bool)
        else:
            mask = np.sum(np.abs(np.remainder(X_array, 1)), axis=0) != 0

        self.mask_ = mask
        self.X_std_ = np.std(X_array[:, mask], axis=0, ddof=1).astype(np.float32)
        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def fit_transform(
        self,
        X,
        y=None,
        **fit_params,
    ):
        del fit_params  # unused

        if not self.enabled:
            return X

        self.fit(X, y)
        if self.mask_ is None or self.X_std_ is None or self.rng_ is None:
            raise RuntimeError("FeatureAugmenter failed to initialize fitted attributes.")

        noise_std = self.std_factor * (self.X_std_ + 1e-8)
        noise = self.rng_.normal(
            loc=0.0,
            scale=noise_std,
            size=X[:, self.mask_].shape,
        )

        X[:, self.mask_] += noise.astype(np.float32)
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)
