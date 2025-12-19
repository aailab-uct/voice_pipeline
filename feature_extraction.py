"""Feature extraction transformers."""
import numpy as np
import librosa
from pathlib import Path
import parselmouth
from parselmouth.praat import call
from scipy.stats import skew
import spkit as sp

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Base class for feature extractors.
    """

    def __init__(self, enabled: bool, cache: bool | str = False):
        """
        Base class for feature extractors.

        :param enabled: Whether to enable this extractor
        :type enabled: bool
        """
        super().__init__()
        self.enabled = enabled
        if isinstance(cache, str):
            self.cache_path = Path(cache)
        elif cache:
            self.cache_path = Path("feature_cache")
        else:
            self.cache_path = None

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


class TimeDomainFeaturesExtractor(FeatureExtractor):
    """
    Output order:
    [diff_pitch, mean_f0, stdev_f0, hnr, jitter, shimmer,
     zero_crossing_rate, shannon_entropy, skewness]
    """
    def __init__(
        self,
        fmin=60.0,
        fmax=250.0,
        use_diff_pitch=True,
        use_f0=True,
        use_hnr=True,
        use_jitter=True,
        use_shimmer=True,
        use_zcr=True,
        use_entropy=True,
        use_skewness=True,
        enabled=True,
    ):
        super().__init__(enabled=enabled)
        self.fmin = fmin
        self.fmax = fmax
        self.use_diff_pitch = use_diff_pitch
        self.use_f0 = use_f0
        self.use_hnr = use_hnr
        self.use_jitter = use_jitter
        self.use_shimmer = use_shimmer
        self.use_zcr = use_zcr
        self.use_entropy = use_entropy
        self.use_skewness = use_skewness

    def _extract_feature(self, x: np.ndarray) -> np.ndarray:
        sound = parselmouth.Sound(x)

        pitch = call(sound, "To Pitch", 0.0, self.fmin, self.fmax)
        point_process = call(
            sound, "To PointProcess (periodic, cc)", self.fmin, self.fmax
        )

        pitch_data = pitch.selected_array["frequency"]
        pitch_data[pitch_data == 0] = np.nan

        diff_pitch = 0.0
        mean_f0 = 0.0
        stdev_f0 = 0.0
        hnr = 0.0
        jitter = 0.0
        shimmer = 0.0
        zcr = 0.0
        entropy = 0.0
        skewness = 0.0

        if self.use_diff_pitch and np.any(~np.isnan(pitch_data)):
            diff_pitch = (np.nanmax(pitch_data) - np.nanmin(pitch_data)) / np.nanmin(pitch_data)

        if self.use_f0:
            mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
            stdev_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")

        if self.use_hnr:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, self.fmin, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)

        if self.use_jitter:
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

        if self.use_shimmer:
            shimmer = call(
                [sound, point_process],
                "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )

        if self.use_zcr:
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=x)[0])

        if self.use_entropy:
            entropy = sp.entropy(x, alpha=1)

        if self.use_skewness:
            skewness = skew(x)

        return np.array(
            [diff_pitch, mean_f0, stdev_f0, hnr, jitter, shimmer, zcr, entropy, skewness],
            dtype=np.float32,
        )




class ZCRExtractor(FeatureExtractor):
    """
    Extracts Zero Crossing Rate feature.
    Output: 1 feature per sample (mean ZCR).
    """
    def __init__(self, enabled=True):
        super().__init__(enabled=enabled)

    def _extract_feature(self, x: np.ndarray) -> np.ndarray:
        zcr = librosa.feature.zero_crossing_rate(y=x)[0]
        zcr_mean = np.mean(zcr)
        return np.array([zcr_mean], dtype=np.float32)

class SpectralFeaturesExtractor(FeatureExtractor):
    """
    Extracts spectral features with per-feature enable switches.
    """
    def __init__(
        self,
        sr=16000,
        use_flatness=True,
        use_rolloff=True,
        use_centroid=True,
        use_contrast=True,
        enabled=True,
    ):
        super().__init__(enabled=enabled)
        self.sr = sr
        self.use_flatness = use_flatness
        self.use_rolloff = use_rolloff
        self.use_centroid = use_centroid
        self.use_contrast = use_contrast

    def _extract_feature(self, x: np.ndarray) -> np.ndarray:
        # default values (keep fixed dimension)
        flatness = 0.0
        rolloff = 0.0
        centroid = 0.0
        contrast = np.zeros(6, dtype=np.float32)

        # spectral flatness
        if self.use_flatness:
            flatness = np.mean(
                librosa.feature.spectral_flatness(y=x),
                axis=1
            )[0]

        # spectral rolloff
        if self.use_rolloff:
            rolloff = np.mean(
                librosa.feature.spectral_rolloff(y=x, sr=self.sr),
                axis=1
            )[0]

        # spectral centroid
        if self.use_centroid:
            centroid = np.mean(
                librosa.feature.spectral_centroid(y=x, sr=self.sr),
                axis=1
            )[0]

        # spectral contrast (6 bands by default)
        if self.use_contrast:
            contrast = np.mean(
                librosa.feature.spectral_contrast(y=x, sr=self.sr),
                axis=1
            )

        features = np.concatenate(
            ([flatness, rolloff, centroid], contrast)
        )

        return features.astype(np.float32)
