"""Feature extraction transformers."""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.stats import skew
import spkit as sp
from sklearn.base import BaseEstimator, TransformerMixin
import atexit

from src.feature_caching import FeatureCache
from src.hashable import HashableArray, HashableDict

cache = FeatureCache()  # Initialize the feature cache
atexit.register(cache.flush_to_disk)  # Ensure cache is flushed to disk on program exit


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Base class for feature extractors.

    All classes inheriting from this must implement:
    - __init__(self, enabled: bool, cache_disabled: bool = False)
    - static method `_extract_full_features(x: np.ndarray, **params) -> np.ndarray`
    - method `extract_feature(x: np.ndarray) -> np.ndarray`
    The `_extract_full_features` method should extract all possible features, regardless of enabled switches.
    The `extract_feature` method should return only the enabled features, and can use caching via `cached_extraction`.
    """

    def __init__(self, enabled: bool, cache_disabled: bool = False):
        """
        Base class for feature extractors.

        :param enabled: Whether to enable this extractor
        :type enabled: bool
        :param cache_disabled: Whether to disable caching, should be used if raw augmentations are applied
        :type cache_disabled: bool
        """
        super().__init__()
        self.enabled = enabled
        self.cache_disabled = cache_disabled

    @staticmethod
    def _extract_full_features(x: np.ndarray, **params) -> np.ndarray:
        """
        Static method to extract features from a single waveform.
        To be implemented by subclasses. This should extract all possible features, regardless of enabled switches.
        """
        raise NotImplementedError("_extract_full_features must be implemented by subclasses")

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        """
        Extract features from a single waveform.

        To be implemented by subclasses. Only enabled features should be returned. If caching is possible, subclasses
        should use `cached_extraction` for extraction of all features.
        """
        raise NotImplementedError("extract_feature must be implemented by subclasses")

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
        X_out = []
        for row in X:
            x = row[0]  # the first column is the waveform
            features = self.extract_feature(x)
            new_row = np.concatenate((row, features))
            X_out.append(new_row)

        return np.array(np.vstack(X_out), dtype=object)

    def cached_extraction(self, x: np.ndarray, **params) -> np.ndarray:
        """
        Wrapper for cached extraction of all features from a single waveform.
        :param x: Waveform array
        :param params: Additional parameters for extraction
        """
        if self.cache_disabled:
            return self._extract_full_features(x, **params)
        hashable_x = HashableArray(x)
        hashable_params = HashableDict(params)
        return self._cached_extraction(hashable_x, hashable_params)

    @classmethod
    @cache.cache_method
    def _cached_extraction(cls, x: HashableArray, params: HashableDict) -> np.ndarray:
        """
        Cached extraction of all features from a single waveform.
        :param x: Hashable array wrapper around waveform
        """
        return cls._extract_full_features(x.array, **params)


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

    def __init__(self, sr, n_mfcc=20, enabled=True, use_std=True, cache_disabled=False):
        """
        Extracts MFCC features: mean and std over time for each coefficient.

        :param sr: Sampling rate
        :param n_mfcc: Number of MFCC coefficients to extract
        :param enabled: Whether the extractor is enabled
        :param use_std: Whether to use standard deviation of MFCCs
        :param cache_disabled: Whether to disable caching
        """
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.use_std = use_std

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        features = self.cached_extraction(x, sr=self.sr, n_mfcc=self.n_mfcc)
        if not self.use_std:
            features = features[: self.n_mfcc]
        return features

    @staticmethod
    def _extract_full_features(x: np.ndarray, sr: int, n_mfcc: int) -> np.ndarray:  # type: ignore[override]
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        features = np.concatenate((mfcc_mean, mfcc_std))
        return features


class TimeDomainFeaturesExtractor(FeatureExtractor):
    """
    Extracts time-domain features with per-feature enable switches.

    8 features returned in following order:
        - diff_pitch
        - mean_f0
        - stdev_f0
        - hnr
        - jitter
        - shimmer
        - zero_crossing_rate
        - shannon_entropy
        - skewness
    """

    def __init__(
        self,
        sr,
        fmin=60.0,
        fmax=250.0,
        enabled=True,
        cache_disabled=False,
        use_diff_pitch=True,
        use_f0=True,
        use_stdf0=True,
        use_hnr=True,
        use_jitter=True,
        use_shimmer=True,
        use_zcr=True,
        use_entropy=True,
        use_skewness=True,
    ):
        """
        Extracts time-domain features with per-feature enable switches.

        8 features returned in following order:
            - diff_pitch
            - mean_f0
            - stdev_f0
            - hnr
            - jitter
            - shimmer
            - zero_crossing_rate
            - shannon_entropy
            - skewness

        :param sr:  Sampling rate
        :param fmin: Minimum pitch frequency for pitch extraction (praat)
        :param fmax: Maximum pitch frequency for pitch extraction (praat)
        :param enabled: Whether the extractor is enabled
        :param cache_disabled: Whether to disable caching
        :param use_diff_pitch: Whether to use diff_pitch feature
        :param use_f0: Whether to use mean_f0 feature
        :param use_stdf0: Whether to use stdev_f0 feature
        :param use_hnr: Whether to use hnr feature
        :param use_jitter: Whether to use jitter feature
        :param use_shimmer: Whether to use shimmer feature
        :param use_zcr: Whether to use zero_crossing_rate feature
        :param use_entropy: Whether to use shannon_entropy feature
        :param use_skewness: Whether to use skewness feature
        """
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.use_diff_pitch = use_diff_pitch
        self.use_f0 = use_f0
        self.use_stdf0 = use_stdf0
        self.use_hnr = use_hnr
        self.use_jitter = use_jitter
        self.use_shimmer = use_shimmer
        self.use_zcr = use_zcr
        self.use_entropy = use_entropy
        self.use_skewness = use_skewness

    @staticmethod
    def _extract_full_features(x: np.ndarray, fmin: int, fmax: int, sr: int) -> np.ndarray:  # type: ignore[override]
        sound = parselmouth.Sound(x, sampling_frequency=sr)
        pitch = call(sound, "To Pitch", 0.0, fmin, fmax)
        point_process = call(sound, "To PointProcess (periodic, cc)", fmin, fmax)

        mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
        stdev_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")

        pitch_data = pitch.selected_array["frequency"]
        pitch_data[pitch_data == 0] = np.nan
        diff_pitch = (np.nanmax(pitch_data) - np.nanmin(pitch_data)) / np.nanmin(pitch_data)

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, fmin, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

        shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        entropy = sp.entropy(x, alpha=1)

        skewness = skew(x)

        features = np.array(
            [mean_f0, stdev_f0, diff_pitch, hnr, jitter, shimmer, entropy, skewness],
            dtype=np.float32,
        )
        #  replace nan with 0
        features = np.nan_to_num(features, nan=0.0)

        return features

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        features = self.cached_extraction(
            x,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
        )

        mask = [
            self.use_f0,
            self.use_stdf0,
            self.use_diff_pitch,
            self.use_hnr,
            self.use_jitter,
            self.use_shimmer,
            self.use_entropy,
            self.use_skewness,
        ]

        return features[np.array(mask)]


class ZCRExtractor(FeatureExtractor):
    """
    Extracts Zero Crossing Rate feature.
    Output: 1 feature per sample (mean ZCR).
    """

    def __init__(self, sr, use_mean=True, use_std=True, enabled=True, cache_disabled=False):
        """
        Extracts Zero Crossing Rate feature.
        Output: 1 feature per sample (mean ZCR).

        :param sr: Sampling rate
        :param use_mean: Whether to use mean ZCR feature
        :param use_std: Whether to use std ZCR feature
        :param enabled: Whether the extractor is enabled
        :param cache_disabled: Whether caching is disabled
        """
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.use_mean = use_mean
        self.use_std = use_std

    @staticmethod
    def _extract_full_features(x: np.ndarray, sr) -> np.ndarray:  # type: ignore[override]
        frame_length = 2048 * sr // 50000
        hop_length = 512 * sr // 50000
        zcr = librosa.feature.zero_crossing_rate(y=x, frame_length=frame_length, hop_length=hop_length)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        return np.array([zcr_mean, zcr_std], dtype=np.float32)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        features = self.cached_extraction(x, sr=self.sr)
        mask = [self.use_mean, self.use_std]
        return features[mask]


class SpectralFeaturesExtractor(FeatureExtractor):
    """
    Extracts spectral features with per-feature enable switches.
    """

    def __init__(
        self,
        sr,
        use_flatness=True,
        use_rolloff=True,
        use_centroid=True,
        use_contrast=True,
        enabled=True,
        cache_disabled=False,
    ):
        """
        Extracts spectral features with per-feature enable switches.
        :param sr: Sampling rate
        :param use_flatness: Whether to use spectral flatness feature
        :param use_rolloff: Whether to use spectral rolloff feature
        :param use_centroid: Whether to use spectral centroid feature
        :param use_contrast: Whether to use spectral contrast features
        :param enabled: Whether the extractor is enabled
        :param cache_disabled: Whether caching is disabled
        """
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.use_flatness = use_flatness
        self.use_rolloff = use_rolloff
        self.use_centroid = use_centroid
        self.use_contrast = use_contrast

    @staticmethod
    def _extract_full_features(x: np.ndarray, sr) -> np.ndarray:  # type: ignore[override]
        n_fft = 2048 * sr // 50000
        hop_length = 512 * sr // 50000

        # spectral flatness
        flatness = np.mean(librosa.feature.spectral_flatness(y=x, n_fft=n_fft, hop_length=hop_length), axis=1)[0]

        # spectral rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)[0]

        # spectral centroid
        centroid = np.mean(librosa.feature.spectral_centroid(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)[0]

        # spectral contrast (6 bands by default)
        contrast = np.mean(librosa.feature.spectral_contrast(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)

        features = np.concatenate(([flatness, rolloff, centroid], contrast))

        # features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


        return features.astype(np.float32)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        features = self.cached_extraction(x, sr=self.sr)
        mask = [
            self.use_flatness,
            self.use_rolloff,
            self.use_centroid,
        ] + [
            self.use_contrast
        ] * len(features[3:])

        return features[np.array(mask)].astype(np.float32)
