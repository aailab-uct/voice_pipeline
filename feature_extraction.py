"""Feature extraction transformers."""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.stats import skew
import spkit as sp
from sklearn.base import BaseEstimator, TransformerMixin
import atexit
from utils import (_floor_power_of_2)
import opensmile

from src.feature_caching import FeatureCache
from src.hashable import HashableArray, HashableDict

cache = FeatureCache()  # Initialize the feature cache
atexit.register(cache.flush_to_disk)  # Ensure cache is flushed to disk on program exit


# ============================================================
# Sex normalization (to m/f)
# ============================================================

def normalize_sex_mf(value):
    """
    Normalize sex/gender to 'm' or 'f'.
    Returns None if unknown.
    Accepts also 'w' (women) used in some SVD-related metadata.
    We used mapping f->1 m->0 in the past, so we accept them for backward compatibility, but they are not recommended.
    """
    if value is None:
        return None

    s = str(value).strip().lower()

    if s in {"m", "male", "man", "0"}:
        return "m"

    if s in {"f", "female", "woman", "w", "1"}:
        return "f"

    return None


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

    def __init__(self, sr=16000, n_mfcc=20, enabled=True, use_std=True, cache_disabled=False):
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
        sr=16000,
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
        sex_col=1,
        f0_range_m=(75.0, 300.0),
        f0_range_f=(100.0, 600.0),
    ):
        """
        Extracts time-domain features with per-feature enable switches.

        :param sr:  Sampling rate
        :param fmin: Default minimum pitch frequency (fallback if sex unknown)
        :param fmax: Default maximum pitch frequency (fallback if sex unknown)
        :param enabled: Whether the extractor is enabled
        :param cache_disabled: Whether to disable caching
        :param sex_col: Column index in X rows containing sex/gender
        :param f0_range_m: (fmin, fmax) pitch range for males
        :param f0_range_f: (fmin, fmax) pitch range for females
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
        self.sex_col = sex_col
        self.f0_range_m = f0_range_m
        self.f0_range_f = f0_range_f

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

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Override transform to use sex-dependent pitch range.

        Assumes:
          - waveform is in column 0
          - sex/gender is in column `sex_col`

        If sex is unknown/missing, falls back to self.fmin/self.fmax.
        """
        if not self.enabled:
            return X

        X_out = []
        for row in X:
            x = row[0]

            sex_val = row[self.sex_col] if len(row) > self.sex_col else None
            sex_mf = normalize_sex_mf(sex_val)

            if sex_mf == "m":
                fmin, fmax = self.f0_range_m
            elif sex_mf == "f":
                fmin, fmax = self.f0_range_f
            else:
                fmin, fmax = self.fmin, self.fmax

            features = self.cached_extraction(
                x,
                fmin=fmin,
                fmax=fmax,
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

            new_row = np.concatenate((row, features[np.array(mask)]))
            X_out.append(new_row)

        return np.array(np.vstack(X_out), dtype=object)


class ZCRExtractor(FeatureExtractor):
    """
    Extracts Zero Crossing Rate (ZCR) features.
    Output: up to 2 features per sample (mean and/or std of ZCR).

    Frame size and hop length are derived automatically from the sampling rate
    to preserve the same time resolution as the reference 50 kHz configuration.
    """

    def __init__(self, sr=16000, use_mean=True, use_std=True, enabled=True, cache_disabled=False):
        """
        Parameters
        ----------
        sr : int, optional
            Sampling rate of the input signal. Default: 16000.
        use_mean : bool, optional
            Whether to include mean ZCR in the output. Default: True.
        use_std : bool, optional
            Whether to include std of ZCR in the output. Default: True.
        enabled : bool, optional
            Whether the extractor is enabled. Default: True.
        cache_disabled : bool, optional
            Whether to disable caching. Should be True when raw augmentation
            is applied. Default: False.
        """
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.use_mean = use_mean
        self.use_std = use_std

    @staticmethod
    def _extract_full_features(x: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract mean and std of ZCR from a waveform.

        Frame length and hop length are scaled proportionally from the
        reference values (frame=2048, hop=512 at 50 kHz) and floored
        to the nearest power of 2.

        Parameters
        ----------
        x : np.ndarray
            1D waveform array.
        sr : int
            Sampling rate.

        Returns
        -------
        np.ndarray
            Array of shape (2,): [zcr_mean, zcr_std], dtype float32.
        """
        n_fft = _floor_power_of_2(2048 * sr // 50000)
        hop_length = _floor_power_of_2(512 * sr // 50000)
        zcr = librosa.feature.zero_crossing_rate(y=x, frame_length=n_fft, hop_length=hop_length)[0]
        return np.array([np.mean(zcr), np.std(zcr)], dtype=np.float32)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        """
        Extract enabled ZCR features from a single waveform.

        Parameters
        ----------
        x : np.ndarray
            1D waveform array.

        Returns
        -------
        np.ndarray
            Array of shape (k,) where k = sum([use_mean, use_std]).
        """
        features = self.cached_extraction(x, sr=self.sr)
        return features[[self.use_mean, self.use_std]]


class SpectralFeaturesExtractor(FeatureExtractor):
    """
    Extracts spectral features with per-feature enable switches.

    Features (in order): spectral flatness, rolloff, centroid, contrast (7 bands).
    Frame size and hop length are derived automatically from the sampling rate.
    """

    def __init__(self, sr=16000, use_flatness=True, use_rolloff=True,
                 use_centroid=True, use_contrast=True, enabled=True, cache_disabled=False):
        """
        Parameters
        ----------
        sr : int, optional
            Sampling rate of the input signal. Default: 16000.
        use_flatness : bool, optional
            Whether to include spectral flatness. Default: True.
        use_rolloff : bool, optional
            Whether to include spectral rolloff. Default: True.
        use_centroid : bool, optional
            Whether to include spectral centroid. Default: True.
        use_contrast : bool, optional
            Whether to include spectral contrast (7 sub-band values). Default: True.
        enabled : bool, optional
            Whether the extractor is enabled. Default: True.
        cache_disabled : bool, optional
            Whether to disable caching. Should be True when raw augmentation
            is applied. Default: False.
        """
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.use_flatness = use_flatness
        self.use_rolloff = use_rolloff
        self.use_centroid = use_centroid
        self.use_contrast = use_contrast

    @staticmethod
    def _extract_full_features(x: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract all spectral features from a waveform.

        Frame length and hop length are scaled proportionally from the
        reference values (n_fft=2048, hop=512 at 50 kHz) and floored
        to the nearest power of 2.

        Parameters
        ----------
        x : np.ndarray
            1D waveform array.
        sr : int
            Sampling rate.

        Returns
        -------
        np.ndarray
            Array of shape (10,): [flatness, rolloff, centroid, contrast×7], dtype float32.
        """
        n_fft = _floor_power_of_2(2048 * sr // 50000)
        hop_length = _floor_power_of_2(512 * sr // 50000)
        flatness = np.mean(librosa.feature.spectral_flatness(y=x, n_fft=n_fft, hop_length=hop_length), axis=1)[0]
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)[0]
        centroid = np.mean(librosa.feature.spectral_centroid(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)[0]
        contrast = np.mean(librosa.feature.spectral_contrast(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)
        return np.concatenate(([flatness, rolloff, centroid], contrast)).astype(np.float32)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        """
        Extract enabled spectral features from a single waveform.

        Parameters
        ----------
        x : np.ndarray
            1D waveform array.

        Returns
        -------
        np.ndarray
            Array of shape (k,) where k depends on enabled feature switches,
            dtype float32.
        """
        features = self.cached_extraction(x, sr=self.sr)
        mask = ([self.use_flatness,
                self.use_rolloff,
                self.use_centroid] +
                [self.use_contrast] * len(features[3:]))
        return features[np.array(mask)].astype(np.float32)

# ============================================================
# OpenSMILE extractor
# ============================================================

_SMILE_SINGLETONS = {}  # (feature_set, feature_level) -> opensmile.Smile instance


def _get_smile(feature_set: str, feature_level: str):
    """
    Lazy singleton factory for opensmile.Smile to avoid re-instantiation per sample.
    """
    key = (feature_set, feature_level)
    if key in _SMILE_SINGLETONS:
        return _SMILE_SINGLETONS[key]

    fs = getattr(opensmile.FeatureSet, feature_set)
    fl = getattr(opensmile.FeatureLevel, feature_level)

    smile = opensmile.Smile(feature_set=fs, feature_level=fl)
    _SMILE_SINGLETONS[key] = smile
    return smile


class OpenSmileExtractor(FeatureExtractor):
    """
    Extracts OpenSMILE feature set (e.g., ComParE_2016, eGeMAPSv02) from waveform.

    Notes:
      - Feature sets and levels are defined by opensmile-python (audEERING). :contentReference[oaicite:1]{index=1}
      - ComParE_2016 'functionals' can be very large (thousands of features). :contentReference[oaicite:2]{index=2}
    """

    def __init__(
        self,
        sr: int = 16000,
        feature_set: str = "ComParE_2016",
        feature_level: str = "functionals",
        enabled: bool = False,
        cache_disabled: bool = False,
    ):
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.feature_set = feature_set
        self.feature_level = feature_level

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        # cached_extraction returns ALL features (fixed order), as float32 vector
        return self.cached_extraction(
            x,
            sr=self.sr,
            feature_set=self.feature_set,
            feature_level=self.feature_level,
        )

    @staticmethod
    def _extract_full_features(  # type: ignore[override]
        x: np.ndarray,
        sr: int,
        feature_set: str,
        feature_level: str,
    ) -> np.ndarray:
        # opensmile expects 1D signal
        x = np.asarray(x).reshape(-1)

        smile = _get_smile(feature_set=feature_set, feature_level=feature_level)

        # opensmile returns a pandas.DataFrame for process_signal(); for functionals it's typically 1 row.
        df = smile.process_signal(x, sr)

        # Flatten in stable column order
        vec = df.iloc[0].to_numpy(dtype=np.float32, copy=False)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return vec
