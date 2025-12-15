"""Feature-level augmenters."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """
    Feature-level augmentation: small Gaussian jitter relative to per-feature std.

    Note: This is a standard transformer, so it is applied to both
    train and test folds.



    """
    def __init__(self,
                 enabled: bool=False,
                 std_factor: float=0.01,
                 skip_integer_features: bool=True,
                 random_state: None|int=None):
        """
        Docstring for __init__

        Parameters
        -----------
        enabled: bool, default False
            whether to apply augmentation - useful to toggle during grid search
        std_factor: float, default 0.01
            factor multiplied by feature std to get noise std
        skip_integer_features: bool, default True
          whether to skip integer features (e.g., categorical). Default: True.
        random_state: int or None, default None
            random seed for reproducibility
        """
        super().__init__()
        self.enabled = enabled
        self.std_factor = std_factor
        self.random_state = random_state
        self.skip_integer_features = skip_integer_features

    def fit(self, X, y=None):
        if self.skip_integer_features:
            X = np.asarray(X, dtype=np.float32) # select all features as float
            self.mask = np.ones(X.shape[1], dtype=bool) # no masking
        else: # apply only to non-integer features (to avoid messing up categorical features)
            self.mask = np.sum( # sum of fractional parts
                np.abs( # in absolute value
                    np.remainder(X, 1) # computed as a remainder of division by 1
                ), axis=0) != 0 # should not be zero, as that would mean this is an integer feature
            X = np.asarray(X[:,self.mask], dtype=np.float32) # select only non-integer features
        self.X_std_ = np.std(X, axis=0, ddof=1) # compute std only on selected features
        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if not self.enabled:
            return X
        noise_std = self.std_factor * (self.X_std_ + 1e-8)
        noise = self.rng_.normal(loc=0.0, scale=noise_std, size=X.shape)
        X[:,self.mask] += noise # apply noise only to selected features
        return X
