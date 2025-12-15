"""Feature extraction transformers."""
import numpy as np
import librosa

from sklearn.base import BaseEstimator, TransformerMixin


class DropSignalColumn(BaseEstimator, TransformerMixin):
    """
    Drops the first column (waveform) from the input data.
    This should be used after feature extraction.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        assert X[0,0].ndim == 1, "First column must be waveform"
        return self

    def transform(self, X):
        return np.asarray(X[:,1:], dtype=np.float32)

class MFCCExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts MFCC features: mean and std over time for each coefficient.
    Output: 2 * n_mfcc features per sample.
    """
    def __init__(self, sr=16000, n_mfcc=20, pass_through=True):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.pass_through = pass_through

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for row in X:
            x = row[0]  # the first column is the waveform
            x = np.asarray(x, dtype=np.float32)
            mfcc = librosa.feature.mfcc(y=x, sr=self.sr, n_mfcc=self.n_mfcc)
            mfcc_mean = mfcc.mean(axis=1)
            mfcc_std = mfcc.std(axis=1)
            if self.pass_through:
                feat = np.concatenate([row, mfcc_mean, mfcc_std])
            else:
                feat = np.concatenate([row[1:], mfcc_mean, mfcc_std])
            feats.append(feat)
        if self.pass_through:
            return np.array(feats, dtype=object)
        return np.vstack(feats)
