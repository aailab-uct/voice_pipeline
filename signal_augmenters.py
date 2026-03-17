"""Signal (waveform)-level augmenters."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RawAugmenter(BaseEstimator, TransformerMixin):
    """
    Waveform-level augmentation:
    - random gain
    - additive Gaussian noise

    Shape-preserving: does NOT change number of samples.
    """

    def __init__(
        self,
        enabled=True,
        noise_std=0.005,
        gain_min=0.8,
        gain_max=1.2,
        random_state=None,
    ):
        """Initialize the augmenter.

        :param enabled: Whether to apply augmentation. If False, the transform will return the input unchanged.
        :param noise_std: Standard deviation of the additive Gaussian noise.
        :param gain_min: Minimum gain factor for random gain.
        :param gain_max: Maximum gain factor for random gain.
        :param random_state: Random seed for reproducibility.
        """
        self.enabled = enabled
        self.noise_std = noise_std
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the augmenter. This is a no-op since there are no learnable parameters.
        We just initialize the random state here."""
        del X, y  # unused
        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def transform(self, X):
        """Apply augmentation to the input data X.
        The input X is expected to be a 2D array of shape (n_samples, 1+n_features),
        where the first column contains the raw waveforms as numpy arrays."""
        if not self.enabled:
            # Keep as object array to allow variable-length signals
            return X

        X_out = []
        for row in X:
            x = row[0]  # the first column is the waveform
            x = np.asarray(x, dtype=np.float32)

            # random gain
            gain = self.rng_.uniform(self.gain_min, self.gain_max)
            x_aug = x * gain

            # additive noise
            noise = self.rng_.normal(0.0, self.noise_std, size=x.shape)
            x_aug = x_aug + noise

            row[0] = x_aug.astype(np.float32)

            X_out.append(row)

        return np.array(X_out, dtype=object)
