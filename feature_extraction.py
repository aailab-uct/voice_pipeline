# feature_extraction.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.stats as sp
from joblib import Memory

import librosa
import parselmouth
from parselmouth.praat import call

from sklearn.base import BaseEstimator, TransformerMixin


# ------------------------------------------------------------
# Small hashing helpers (optional, but kept)
# ------------------------------------------------------------

@dataclass(frozen=True)
class HashableArray:
    array: np.ndarray

    def __hash__(self) -> int:
        h = hashlib.sha1(self.array.tobytes()).hexdigest()
        return int(h[:16], 16)


@dataclass(frozen=True)
class HashableDict:
    d: dict[str, Any]

    def __hash__(self) -> int:
        items = tuple(sorted(self.d.items()))
        h = hashlib.sha1(repr(items).encode("utf-8")).hexdigest()
        return int(h[:16], 16)


# ------------------------------------------------------------
# Cache setup (joblib compatible)
# ------------------------------------------------------------

cache = Memory(location="cache", verbose=0)


# ============================================================
# Feature extractor base
# ============================================================

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Base class for feature extractors.

    Subclasses must implement:
    - staticmethod `_extract_full_features(x: np.ndarray, **params) -> np.ndarray`
    - `extract_feature(x: np.ndarray) -> np.ndarray`
    - `appended_feature_names() -> list[str]`
    """

    def __init__(self, enabled: bool, cache_disabled: bool = False):
        self.enabled = enabled
        self.cache_disabled = cache_disabled
        self._cached_full = None  # lazy init of joblib cached function

    def fit(self, X, y=None):
        assert X[0, 0].ndim == 1, "First column must be waveform"
        if y is not None:
            assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return X
        X_out = []
        for row in X:
            x = row[0]  # waveform
            feats = self.extract_feature(x)
            X_out.append(np.concatenate((row, feats)))
        return np.array(np.vstack(X_out), dtype=object)

    def cached_extraction(self, x: np.ndarray, **params) -> np.ndarray:
        """
        Joblib-cache compatible cached extraction.
        (Fixes: Memory.cache_method not present in older joblib.)
        """
        if self.cache_disabled:
            return self._extract_full_features(x, **params)

        # lazy init of cached function per instance
        if self._cached_full is None:
            # cache the extractor's full feature function
            self._cached_full = cache.cache(self._extract_full_features)

        return self._cached_full(x, **params)

    def appended_feature_names(self) -> list[str]:
        raise NotImplementedError

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.appended_feature_names(), dtype=object)
        return np.array(list(input_features) + self.appended_feature_names(), dtype=object)


# ============================================================
# Post extraction utilities
# ============================================================

class DropSignalColumn(BaseEstimator, TransformerMixin):
    """Drops the first column (waveform) after feature extraction."""

    def fit(self, X, y=None):
        assert X[0, 0].ndim == 1, "First column must be waveform"
        if y is not None:
            assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        return self

    def transform(self, X):
        return np.asarray(X[:, 1:], dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        feats = list(input_features)
        if len(feats) > 0:
            feats = feats[1:]  # remove "signal"
        return np.array(feats, dtype=object)


# ============================================================
# MFCC
# ============================================================

class MFCCExtractor(FeatureExtractor):
    """MFCC mean + std over time."""

    def __init__(self, sr, n_mfcc=20, enabled=True, use_std=True, cache_disabled=False):
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.use_std = use_std

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        feats = self.cached_extraction(x, sr=self.sr, n_mfcc=self.n_mfcc)
        if not self.use_std:
            feats = feats[: self.n_mfcc]
        return feats

    @staticmethod
    def _extract_full_features(x: np.ndarray, sr: int, n_mfcc: int) -> np.ndarray:  # type: ignore[override]
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        return np.concatenate((mfcc_mean, mfcc_std)).astype(np.float32)

    def appended_feature_names(self) -> list[str]:
        names = [f"mfcc_{i:02d}_mean" for i in range(1, self.n_mfcc + 1)]
        if self.use_std:
            names += [f"mfcc_{i:02d}_std" for i in range(1, self.n_mfcc + 1)]
        return names


# ============================================================
# Time-domain (Praat + basic stats)
# ============================================================

class TimeDomainFeaturesExtractor(FeatureExtractor):
    """
    Cached full order:
      mean_f0, stdev_f0, diff_pitch, hnr, jitter, shimmer, entropy, skewness
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
        use_zcr=True,      # kept for backward compat; not used in vector
        use_entropy=True,
        use_skewness=True,
        sex_col=1,
        f0_range_m=(75.0, 300.0),
        f0_range_f=(100.0, 600.0),
    ):
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
    def _entropy_from_signal(x: np.ndarray, bins: int = 128) -> float:
        # robust entropy: histogram of amplitudes
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return 0.0
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist = hist.astype(np.float64)
        hist = hist[hist > 0]
        if hist.size == 0:
            return 0.0
        return float(sp.entropy(hist))

    @staticmethod
    def _extract_full_features(x: np.ndarray, fmin: float, fmax: float, sr: int) -> np.ndarray:  # type: ignore[override]
        sound = parselmouth.Sound(x, sampling_frequency=sr)
        pitch = call(sound, "To Pitch", 0.0, fmin, fmax)
        point_process = call(sound, "To PointProcess (periodic, cc)", fmin, fmax)

        mean_f0 = float(call(pitch, "Get mean", 0, 0, "Hertz"))
        stdev_f0 = float(call(pitch, "Get standard deviation", 0, 0, "Hertz"))

        pitch_data = pitch.selected_array["frequency"].astype(np.float64, copy=True)
        pitch_data[pitch_data == 0] = np.nan
        if np.all(np.isnan(pitch_data)) or np.nanmin(pitch_data) == 0:
            diff_pitch = np.nan
        else:
            diff_pitch = float((np.nanmax(pitch_data) - np.nanmin(pitch_data)) / np.nanmin(pitch_data))

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, fmin, 0.1, 1.0)
        hnr = float(call(harmonicity, "Get mean", 0, 0))

        jitter = float(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        shimmer = float(call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

        entropy = TimeDomainFeaturesExtractor._entropy_from_signal(x)
        skewness = float(sp.skew(np.asarray(x, dtype=np.float64), bias=False))

        return np.array([mean_f0, stdev_f0, diff_pitch, hnr, jitter, shimmer, entropy, skewness], dtype=np.float32)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        feats = self.cached_extraction(x, fmin=self.fmin, fmax=self.fmax, sr=self.sr)
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
        return feats[np.array(mask)]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return X

        X_out = []
        for row in X:
            x = row[0]
            sex = row[self.sex_col] if self.sex_col is not None and self.sex_col < len(row) else np.nan

            if sex == 1:
                fmin, fmax = self.f0_range_f
            else:
                fmin, fmax = self.f0_range_m

            self.fmin = float(fmin)
            self.fmax = float(fmax)

            feats = self.extract_feature(x)
            X_out.append(np.concatenate((row, feats)))

        return np.array(np.vstack(X_out), dtype=object)

    def appended_feature_names(self) -> list[str]:
        all_names = ["mean_f0", "stdev_f0", "diff_pitch", "hnr", "jitter", "shimmer", "entropy", "skewness"]
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
        return [n for n, keep in zip(all_names, mask) if keep]


# ============================================================
# ZCR
# ============================================================

class ZCRExtractor(FeatureExtractor):
    """ZCR mean + std."""

    def __init__(self, sr, enabled=True, use_mean=True, use_std=True, cache_disabled=False):
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.use_mean = use_mean
        self.use_std = use_std

    @staticmethod
    def _extract_full_features(x: np.ndarray, sr: int) -> np.ndarray:  # type: ignore[override]
        n_fft = 2048 * sr // 50000
        hop_length = 512 * sr // 50000
        zcr = librosa.feature.zero_crossing_rate(y=x, frame_length=n_fft, hop_length=hop_length)
        return np.array([float(zcr.mean()), float(zcr.std())], dtype=np.float32)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        feats = self.cached_extraction(x, sr=self.sr)
        mask = [self.use_mean, self.use_std]
        return feats[np.array(mask)]

    def appended_feature_names(self) -> list[str]:
        all_names = ["zcr_mean", "zcr_std"]
        mask = [self.use_mean, self.use_std]
        return [n for n, keep in zip(all_names, mask) if keep]


# ============================================================
# Spectral
# ============================================================

class SpectralFeaturesExtractor(FeatureExtractor):
    """Spectral flatness, rolloff, centroid, contrast (7 bands)."""

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
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.use_flatness = use_flatness
        self.use_rolloff = use_rolloff
        self.use_centroid = use_centroid
        self.use_contrast = use_contrast

    @staticmethod
    def _extract_full_features(x: np.ndarray, sr: int) -> np.ndarray:  # type: ignore[override]
        n_fft = 2048 * sr // 50000
        hop_length = 512 * sr // 50000

        flatness = float(np.mean(librosa.feature.spectral_flatness(y=x, n_fft=n_fft, hop_length=hop_length), axis=1)[0])
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)[0])
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)[0])
        contrast = np.mean(librosa.feature.spectral_contrast(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1).astype(np.float32)

        return np.concatenate(([flatness, rolloff, centroid], contrast)).astype(np.float32)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        feats = self.cached_extraction(x, sr=self.sr)
        mask = [self.use_flatness, self.use_rolloff, self.use_centroid] + ([self.use_contrast] * 7)
        return feats[np.array(mask)]

    def appended_feature_names(self) -> list[str]:
        contrast_names = [f"contrast_{i}" for i in range(7)]
        all_names = ["flatness", "rolloff", "centroid"] + contrast_names
        mask = [self.use_flatness, self.use_rolloff, self.use_centroid] + ([self.use_contrast] * 7)
        return [n for n, keep in zip(all_names, mask) if keep]