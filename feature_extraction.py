"""Feature extraction transformers."""

import numpy as np
import librosa

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Base class for feature extractors.
    """

    def __init__(self, enabled: bool):
        """
        Base class for feature extractors.

        :param enabled: Whether to enable this extractor
        :type enabled: bool
        """
        super().__init__()
        self.enabled = enabled

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """
        Fit method. Only checks input shape.

        :param X: Input data array, first column must be waveform
        :type X: np.ndarray
        :param y: Target labels (optional)
        :type y: np.ndarray | None
        :raises: AssertionError
        """
        assert X[0, 0].ndim == 1, "First column must be waveform"
        assert X.shape[0] > 0, "Input X must have at least one sample"
        assert X.shape[0] == (
            y.shape[0] if y is not None else X.shape[0]
        ), "X and y must have the same number of samples"
        return self

    def _extract_feature(self, x: np.ndarray) -> np.ndarray:
        """
        Extract features from a single waveform.
        To be implemented by subclasses.
        """
        raise NotImplementedError("_extract_feature must be implemented by subclasses")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform method: extracts features and appends them to input data.

        :param X: Input data array, first column must be waveform
        :type X: np.ndarray
        :return: Transformed data array with extracted features appended
        :rtype: ndarray
        """
        if not self.enabled:
            return X
        # extract features for each sample but use mapping for efficiency
        X_out = []
        for row in X:
            x = row[0]  # the first column is the waveform
            x = np.asarray(x, dtype=np.float32)
            features = self._extract_feature(x)
            new_row = np.concatenate((row, features))
            X_out.append(new_row)

        return np.array(np.vstack(X_out), dtype=object)


class DropSignalColumn(BaseEstimator, TransformerMixin):
    """
    Drops the first column (waveform) from the input data.
    This should be used after feature extraction.
    """

    def fit(self, X, y=None):
        assert X[0, 0].ndim == 1, "First column must be waveform"
        if y is not None:
            assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        return self

    def transform(self, X):
        return np.asarray(X[:, 1:], dtype=np.float32)


class MFCCExtractor(FeatureExtractor):
    """
    Extracts MFCC features: mean and std over time for each coefficient.
    Output: 2 * n_mfcc features per sample.
    """

    def __init__(self, sr=16000, n_mfcc=20, enabled=True):
        super().__init__(enabled=enabled)
        self.sr = sr
        self.n_mfcc = n_mfcc

    def _extract_feature(self, x: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=x, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        features = np.concatenate((mfcc_mean, mfcc_std))
        return features
