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
import warnings
warnings.filterwarnings("ignore", message="n_fft=.*is too large for input signal")

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

# Cache setup — při paralelním grid searchi (n_jobs != 1) na Windows
# způsobuje joblib cache PermissionError kvůli file lockingu.
# Cache je užitečná pouze při single-process spuštění.
import os
_CACHE_DIR = "cache" if os.environ.get("DISABLE_FEATURE_CACHE", "0") == "0" else None
cache = Memory(location=_CACHE_DIR, verbose=0)


import scipy.signal as scipy_signal

class AudioPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessing surového audio signálu pro redukci domain shiftu.
    Musí být první krok pipeline před feature extraktory.

    Metody se aplikují vždy v tomto pevném pořadí (pokud jsou zapnuty):
        1. rir_normalization    — kompenzace akustiky místnosti (cepstrální dekonvoluce)
        2. wiener               — Wiener filtr pro redukci šumu
        3. spectral_subtraction — odečtení odhadnutého spektra šumu

    Params:
        enabled:                    zapnutí/vypnutí celého preprocessingu
        use_rir_normalization:      True/False
        use_wiener:                 True/False
        use_spectral_subtraction:   True/False
        sr:                         vzorkovací kmitočet
        noise_frames:               počet úvodních framů pro odhad šumu
        wiener_noise_std:           odhadnutý std šumu pro Wiener filtr
        rir_ir_len_ms:              délka odhadované impulsní odezvy v ms
    """
    def __init__(
        self,
        enabled: bool = True,
        use_rir_normalization: bool = False,
        use_wiener: bool = False,
        use_spectral_subtraction: bool = True,
        sr: int = 50000,
        noise_frames: int = 10,
        wiener_noise_std: float = 0.005,
        rir_ir_len_ms: int = 50,
    ):
        self.enabled = enabled
        self.use_rir_normalization = use_rir_normalization
        self.use_wiener = use_wiener
        self.use_spectral_subtraction = use_spectral_subtraction
        self.sr = sr
        self.noise_frames = noise_frames
        self.wiener_noise_std = wiener_noise_std
        self.rir_ir_len_ms = rir_ir_len_ms

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not self.enabled:
            return X

        X_out = []
        for row in X:
            signal = np.asarray(row[0], dtype=np.float32)

            # pevné pořadí: RIR → Wiener → Spectral subtraction
            if self.use_rir_normalization:
                signal = self._rir_normalization(signal)
            if self.use_wiener:
                signal = self._wiener(signal)
            if self.use_spectral_subtraction:
                signal = self._spectral_subtraction(signal)

            new_row = row.copy()
            new_row[0] = signal
            X_out.append(new_row)

        return np.array(X_out, dtype=object)

    def _spectral_subtraction(self, x: np.ndarray) -> np.ndarray:
        n_fft = 2048 * self.sr // 50000
        hop = 512 * self.sr // 50000

        # padding pokud je signál kratší než n_fft
        orig_len = len(x)
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)), mode="reflect")

        # STFT
        f, t, stft = scipy_signal.stft(x, fs=self.sr, nperseg=n_fft, noverlap=n_fft - hop)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # odhad šumu z prvních N framů
        noise_est = np.mean(magnitude[:, :self.noise_frames], axis=1, keepdims=True)

        # odečtení šumu s half-wave rectification
        magnitude_clean = np.maximum(magnitude - noise_est, 0.0)

        # rekonstrukce signálu
        stft_clean = magnitude_clean * np.exp(1j * phase)
        _, x_clean = scipy_signal.istft(stft_clean, fs=self.sr, nperseg=n_fft, noverlap=n_fft - hop)

        # zarovnání na původní délku
        x_clean = x_clean[:orig_len].astype(np.float32)
        return x_clean

    def _wiener(self, x: np.ndarray) -> np.ndarray:
        n_fft = 2048 * self.sr // 50000
        hop = 512 * self.sr // 50000

        # padding pokud je signál kratší než n_fft
        orig_len = len(x)
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)), mode="reflect")

        f, t, stft = scipy_signal.stft(x, fs=self.sr, nperseg=n_fft, noverlap=n_fft - hop)
        power = np.abs(stft) ** 2

        # odhad výkonu šumu z prvních N framů
        noise_power = np.mean(power[:, :self.noise_frames], axis=1, keepdims=True)
        noise_power = np.maximum(noise_power, 1e-10)

        # Wiener filtr: H = (P_signal) / (P_signal + P_noise)
        wiener_gain = np.maximum(power - noise_power, 0.0) / (power + noise_power + 1e-10)
        stft_clean = wiener_gain * stft

        _, x_clean = scipy_signal.istft(stft_clean, fs=self.sr, nperseg=n_fft, noverlap=n_fft - hop)
        x_clean = x_clean[:orig_len].astype(np.float32)
        return x_clean

    def _rir_normalization(self, x: np.ndarray) -> np.ndarray:
        ir_len = int(self.rir_ir_len_ms * self.sr / 1000)

        # odhad impulsní odezvy pomocí cepstra
        # log spektrum → ifft → okénkování na délku IR → fft → dekonvoluce
        n_fft = max(2048 * self.sr // 50000, ir_len * 2)

        X_freq = np.fft.rfft(x, n=n_fft)
        log_mag = np.log(np.abs(X_freq) + 1e-10)
        cepstrum = np.fft.irfft(log_mag)

        # okénkování — ponech jen pozdní část (odezva místnosti)
        win = np.zeros_like(cepstrum)
        win[ir_len:ir_len * 2] = 1.0
        rir_est = np.real(np.fft.rfft(cepstrum * win, n=n_fft))

        # dekonvoluce: X_clean = X / RIR v frekvenční doméně
        rir_freq = np.fft.rfft(np.fft.irfft(rir_est)[:ir_len], n=n_fft)
        rir_power = np.abs(rir_freq) ** 2
        # regularizovaná dekonvoluce
        gain = np.conj(rir_freq) / (rir_power + 1e-6 * np.max(rir_power))
        x_clean = np.fft.irfft(X_freq * gain, n=n_fft)

        x_clean = x_clean[:len(x)].astype(np.float32)
        # normalizace amplitudy
        if np.max(np.abs(x_clean)) > 1e-8:
            x_clean = x_clean / np.max(np.abs(x_clean)) * np.max(np.abs(x))
        return x_clean


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
        # Praat vyžaduje: fmin >= sr / n_samples
        # Po TrimSilence může být signál krátký — dynamicky zvýšíme fmin pokud je potřeba
        praat_min = sr / len(x) if len(x) > 0 else fmin
        fmin = max(fmin, praat_min * 1.05)  # +5% bezpečnostní rezerva

        try:
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

        except Exception:
            # Krátká nebo problematická nahrávka — vrátíme NaN, ReplaceNaN step je nahradí 0
            return np.full(8, np.nan, dtype=np.float32)

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