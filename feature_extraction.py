"""Feature extraction transformers."""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.stats import skew, kurtosis as sp_kurtosis
from scipy.signal import lfilter
import spkit as sp
from sklearn.base import BaseEstimator, TransformerMixin
import atexit
import nolds
from utils import (_floor_power_of_2)
import opensmile

from src.feature_caching import FeatureCache
from src.hashable import HashableArray, HashableDict

cache = FeatureCache()
atexit.register(cache.flush_to_disk)


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


# ============================================================
# Base class
# ============================================================

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
        super().__init__()
        self.enabled = enabled
        self.cache_disabled = cache_disabled

    @staticmethod
    def _extract_full_features(x: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError("_extract_full_features must be implemented by subclasses")

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("extract_feature must be implemented by subclasses")

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        assert X[0, 0].ndim == 1, "First column must be waveform"
        assert X.shape[0] > 0, "Input X must have at least one sample"
        assert X.shape[0] == (
            y.shape[0] if y is not None else X.shape[0]
        ), "X and y must have the same number of samples"
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return X
        X_out = []
        for row in X:
            x = row[0]
            features = self.extract_feature(x)
            new_row = np.concatenate((row, features))
            X_out.append(new_row)
        return np.array(np.vstack(X_out), dtype=object)

    def cached_extraction(self, x: np.ndarray, **params) -> np.ndarray:
        if self.cache_disabled:
            return self._extract_full_features(x, **params)
        hashable_x = HashableArray(x)
        hashable_params = HashableDict(params)
        return self._cached_extraction(hashable_x, hashable_params)

    @classmethod
    @cache.cache_method
    def _cached_extraction(cls, x: HashableArray, params: HashableDict) -> np.ndarray:
        return cls._extract_full_features(x.array, **params)


# ============================================================
# Utility: drop signal column
# ============================================================

class DropSignalColumn(BaseEstimator, TransformerMixin):
    """Drops the first column (waveform) from the input data."""

    def fit(self, X, y=None):
        assert X[0, 0].ndim == 1, "First column must be waveform"
        if y is not None:
            assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        return self

    def transform(self, X):
        return np.asarray(X[:, 1:], dtype=np.float32)


# ============================================================
# Original extractors
# ============================================================

class MFCCExtractor(FeatureExtractor):
    """
    Extracts MFCC features: mean and std over time for each coefficient.
    Output: 2 * n_mfcc features per sample.
    """

    def __init__(self, sr=16000, n_mfcc=20, enabled=True, use_std=True, cache_disabled=False):
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
        return np.concatenate((mfcc_mean, mfcc_std))


class TimeDomainFeaturesExtractor(FeatureExtractor):
    """
    Extracts time-domain features with per-feature enable switches.

    Features returned in following order:
        mean_f0, stdev_f0, diff_pitch, hnr, jitter, shimmer, entropy, skewness
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
        return np.nan_to_num(features, nan=0.0)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        features = self.cached_extraction(x, fmin=self.fmin, fmax=self.fmax, sr=self.sr)
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
            features = self.cached_extraction(x, fmin=fmin, fmax=fmax, sr=self.sr)
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
            X_out.append(np.concatenate((row, features[np.array(mask)])))
        return np.array(np.vstack(X_out), dtype=object)


class ZCRExtractor(FeatureExtractor):
    """
    Extracts Zero Crossing Rate (ZCR) features.
    Output: up to 2 features per sample (mean and/or std of ZCR).
    """

    def __init__(self, sr=16000, use_mean=True, use_std=True, enabled=True, cache_disabled=False):
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.use_mean = use_mean
        self.use_std = use_std

    @staticmethod
    def _extract_full_features(x: np.ndarray, sr: int) -> np.ndarray:
        n_fft = _floor_power_of_2(2048 * sr // 50000)
        hop_length = _floor_power_of_2(512 * sr // 50000)
        zcr = librosa.feature.zero_crossing_rate(y=x, frame_length=n_fft, hop_length=hop_length)[0]
        return np.array([np.mean(zcr), np.std(zcr)], dtype=np.float32)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        features = self.cached_extraction(x, sr=self.sr)
        return features[[self.use_mean, self.use_std]]


class SpectralFeaturesExtractor(FeatureExtractor):
    """
    Extracts spectral features with per-feature enable switches.
    Features (in order): spectral flatness, rolloff, centroid, contrast (7 bands).
    """

    def __init__(self, sr=16000, use_flatness=True, use_rolloff=True,
                 use_centroid=True, use_contrast=True, enabled=True, cache_disabled=False):
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.use_flatness = use_flatness
        self.use_rolloff = use_rolloff
        self.use_centroid = use_centroid
        self.use_contrast = use_contrast

    @staticmethod
    def _extract_full_features(x: np.ndarray, sr: int) -> np.ndarray:
        n_fft = _floor_power_of_2(2048 * sr // 50000)
        hop_length = _floor_power_of_2(512 * sr // 50000)
        flatness = np.mean(librosa.feature.spectral_flatness(y=x, n_fft=n_fft, hop_length=hop_length), axis=1)[0]
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)[0]
        centroid = np.mean(librosa.feature.spectral_centroid(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)[0]
        contrast = np.mean(librosa.feature.spectral_contrast(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=1)
        return np.concatenate(([flatness, rolloff, centroid], contrast)).astype(np.float32)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        features = self.cached_extraction(x, sr=self.sr)
        mask = ([self.use_flatness, self.use_rolloff, self.use_centroid] +
                [self.use_contrast] * len(features[3:]))
        return features[np.array(mask)].astype(np.float32)


# ============================================================
# OpenSMILE extractor
# ============================================================

_SMILE_SINGLETONS = {}


def _get_smile(feature_set: str, feature_level: str):
    key = (feature_set, feature_level)
    if key in _SMILE_SINGLETONS:
        return _SMILE_SINGLETONS[key]
    fs = getattr(opensmile.FeatureSet, feature_set)
    fl = getattr(opensmile.FeatureLevel, feature_level)
    smile = opensmile.Smile(feature_set=fs, feature_level=fl)
    _SMILE_SINGLETONS[key] = smile
    return smile


class OpenSmileExtractor(FeatureExtractor):
    """Extracts OpenSMILE feature set (e.g., ComParE_2016, eGeMAPSv02) from waveform."""

    def __init__(self, sr=16000, feature_set="ComParE_2016", feature_level="functionals",
                 enabled=False, cache_disabled=False):
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.feature_set = feature_set
        self.feature_level = feature_level

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        return self.cached_extraction(x, sr=self.sr, feature_set=self.feature_set,
                                      feature_level=self.feature_level)

    @staticmethod
    def _extract_full_features(x: np.ndarray, sr: int, feature_set: str,
                                feature_level: str) -> np.ndarray:  # type: ignore[override]
        x = np.asarray(x).reshape(-1)
        smile = _get_smile(feature_set=feature_set, feature_level=feature_level)
        df = smile.process_signal(x, sr)
        vec = df.iloc[0].to_numpy(dtype=np.float32, copy=False)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


# ============================================================
# Shared helpers for robust extractors
# ============================================================

def _normalise(x: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance normalisation."""
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-10:
        return x - mu
    return (x - mu) / sigma


def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """
    Solve Yule-Walker equations via Levinson-Durbin recursion.

    Parameters
    ----------
    r     : autocorrelation sequence r[0..order]
    order : LP model order

    Returns
    -------
    a : LP coefficients a[1..order]  (a[0]=1 implicit)
    """
    a = np.zeros(order)
    e = r[0]
    if e == 0:
        return a
    for i in range(order):
        if i == 0:
            lam = -r[1] / e
        else:
            lam = -(r[i + 1] + np.dot(a[:i], r[i:0:-1])) / e
        a_new = a.copy()
        if i > 0:
            a_new[:i] = a[:i] + lam * a[i - 1::-1]
        a_new[i] = lam
        a = a_new
        e *= 1.0 - lam ** 2
        if e <= 1e-12:
            break
    return -a


def _lp_to_area(a: np.ndarray) -> np.ndarray:
    """
    Convert LP coefficients to vocal-tract area function
    via the Schur (lattice) recursion.

    Returns
    -------
    areas : shape (len(a)+1,) - relative tube cross-sections
    """
    k_all = []
    coef = a.copy().astype(float)
    order = len(coef)
    for i in range(order - 1, -1, -1):
        ki = coef[i]
        k_all.append(ki)
        denom = 1.0 - ki ** 2
        if abs(denom) < 1e-12:
            break
        if i > 0:
            coef[:i] = (coef[:i] - ki * coef[i - 1::-1]) / denom
    areas = np.ones(len(k_all) + 1)
    for j, ki in enumerate(k_all):
        areas[j + 1] = areas[j] * (1.0 - ki) / (1.0 + ki + 1e-12)
    return areas


def _rqa_features(x: np.ndarray, m: int = 3, tau: int = 2,
                  eps_frac: float = 0.10, lmin: int = 2) -> np.ndarray:
    """
    Compute RQA features (RR, DET, LAM) from a 1-D normalised signal.

    Parameters
    ----------
    x        : signal (normalised internally)
    m        : embedding dimension
    tau      : time delay
    eps_frac : threshold as fraction of mean pairwise distance
    lmin     : minimum line length for DET / LAM

    Returns
    -------
    np.ndarray([RR, DET, LAM], dtype=float32)
    """
    x = _normalise(x)
    N = len(x)
    M = N - (m - 1) * tau
    if M <= lmin:
        return np.zeros(3, dtype=np.float32)

    X = np.stack([x[i: i + (m - 1) * tau + 1: tau] for i in range(M)])
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    eps = eps_frac * dist.mean()
    R = (dist < eps).astype(np.int8)
    np.fill_diagonal(R, 0)

    total = R.sum()
    if total == 0:
        return np.zeros(3, dtype=np.float32)

    rr = total / (M * M)

    # DET - diagonal lines >= lmin
    det_pts = 0
    for d in range(-(M - 1), M):
        diag = np.diag(R, d)
        run, in_line = 0, False
        for v in diag:
            if v:
                run += 1; in_line = True
            else:
                if in_line and run >= lmin:
                    det_pts += run
                run = 0; in_line = False
        if in_line and run >= lmin:
            det_pts += run
    det = det_pts / total

    # LAM - vertical lines >= lmin
    lam_pts = 0
    for col in range(M):
        col_vec = R[:, col]
        run, in_line = 0, False
        for v in col_vec:
            if v:
                run += 1; in_line = True
            else:
                if in_line and run >= lmin:
                    lam_pts += run
                run = 0; in_line = False
        if in_line and run >= lmin:
            lam_pts += run
    lam = lam_pts / total

    return np.array([rr, det, lam], dtype=np.float32)


def _rpde(x: np.ndarray, m: int = 4, tau: int = 1,
          eps_frac: float = 0.10, T_max: int = 200) -> float:
    """
    Recurrence Period Density Entropy (Little et al., 2007).

    Returns normalised entropy H in [0, 1].
    """
    x = _normalise(x)
    N = len(x)
    M = N - (m - 1) * tau
    if M < 20:
        return 0.0

    X = np.stack([x[i: i + (m - 1) * tau + 1: tau] for i in range(M)])
    ref = X[M // 2]
    dist = np.linalg.norm(X - ref, axis=1)
    eps = eps_frac * dist.mean()

    inside = dist < eps
    periods = []
    last_in = -1
    for i, v in enumerate(inside):
        if v:
            if last_in >= 0:
                p = i - last_in
                if 1 <= p <= T_max:
                    periods.append(p)
            last_in = i

    if len(periods) < 5:
        return 0.0

    counts = np.bincount(periods, minlength=T_max + 1)[1: T_max + 1].astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(np.clip(H / np.log(T_max), 0.0, 1.0))


# ============================================================
# New robust extractor 1: F0DerivativesExtractor
# ============================================================

class F0DerivativesExtractor(FeatureExtractor):
    """
    Robust F0-derivative features for sustained-vowel analysis.

    Features (in order):
        f0_iqr         - interquartile range of F0 (Hz)
        f0_range       - max - min of voiced F0 frames (Hz)
        ppe            - Pitch Period Entropy (normalised to [0, 1])
        nan_rate       - fraction of unvoiced (NaN/zero) F0 frames
        pitch_diff_std - std of relative cycle-to-cycle F0 change

    All five are device-independent: they depend on the relative
    distribution of F0 values, not on absolute signal amplitude.

    Parameters
    ----------
    sr          : sampling rate (Hz)
    fmin, fmax  : fallback F0 search range when sex is unknown
    f0_range_m  : (fmin, fmax) for male speakers
    f0_range_f  : (fmin, fmax) for female speakers
    sex_col     : column index in X rows containing sex/gender label
    use_*       : per-feature enable switches
    """

    def __init__(
        self,
        sr: int = 16000,
        fmin: float = 60.0,
        fmax: float = 400.0,
        f0_range_m: tuple = (75.0, 300.0),
        f0_range_f: tuple = (100.0, 600.0),
        sex_col: int = 1,
        enabled: bool = True,
        cache_disabled: bool = False,
        use_f0_iqr: bool = True,
        use_f0_range: bool = True,
        use_ppe: bool = True,
        use_nan_rate: bool = True,
        use_pitch_diff_std: bool = True,
    ):
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.f0_range_m = f0_range_m
        self.f0_range_f = f0_range_f
        self.sex_col = sex_col
        self.use_f0_iqr = use_f0_iqr
        self.use_f0_range = use_f0_range
        self.use_ppe = use_ppe
        self.use_nan_rate = use_nan_rate
        self.use_pitch_diff_std = use_pitch_diff_std

    @staticmethod
    def _extract_full_features(  # type: ignore[override]
        x: np.ndarray, fmin: float, fmax: float, sr: int,
    ) -> np.ndarray:
        sound = parselmouth.Sound(x, sampling_frequency=sr)
        pitch = call(sound, "To Pitch", 0.0, fmin, fmax)
        f0_arr = pitch.selected_array["frequency"].copy()  # 0 = unvoiced

        nan_rate = float(np.mean(f0_arr == 0))
        voiced = f0_arr[f0_arr > 0]

        if len(voiced) < 4:
            return np.zeros(5, dtype=np.float32)

        f0_iqr = float(np.percentile(voiced, 75) - np.percentile(voiced, 25))
        f0_range = float(voiced.max() - voiced.min())

        # PPE - normalised entropy of the distribution of pitch periods
        periods = 1.0 / voiced
        periods_norm = periods / periods.mean()
        hist, _ = np.histogram(periods_norm, bins=50, range=(0.5, 2.0))
        hist = hist.astype(float)
        total = hist.sum()
        if total > 0:
            p = hist / total
            p = p[p > 0]
            ppe = float(-np.sum(p * np.log2(p)) / np.log2(len(hist)))
        else:
            ppe = 0.0

        # Relative cycle-to-cycle pitch change
        pitch_diff = np.diff(voiced) / (voiced[:-1] + 1e-6)
        pitch_diff_std = float(np.std(pitch_diff))

        features = np.array(
            [f0_iqr, f0_range, ppe, nan_rate, pitch_diff_std],
            dtype=np.float32,
        )
        return np.nan_to_num(features, nan=0.0)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        features = self.cached_extraction(x, fmin=self.fmin, fmax=self.fmax, sr=self.sr)
        mask = np.array([
            self.use_f0_iqr, self.use_f0_range, self.use_ppe,
            self.use_nan_rate, self.use_pitch_diff_std,
        ])
        return features[mask]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Sex-aware override."""
        if not self.enabled:
            return X
        X_out = []
        for row in X:
            x = row[0]
            sex_mf = normalize_sex_mf(row[self.sex_col] if len(row) > self.sex_col else None)
            if sex_mf == "m":
                fmin, fmax = self.f0_range_m
            elif sex_mf == "f":
                fmin, fmax = self.f0_range_f
            else:
                fmin, fmax = self.fmin, self.fmax
            features = self.cached_extraction(x, fmin=fmin, fmax=fmax, sr=self.sr)
            mask = np.array([
                self.use_f0_iqr, self.use_f0_range, self.use_ppe,
                self.use_nan_rate, self.use_pitch_diff_std,
            ])
            X_out.append(np.concatenate((row, features[mask])))
        return np.array(np.vstack(X_out), dtype=object)


# ============================================================
# New robust extractor 2: NonlinearExtractor
# ============================================================

class NonlinearExtractor(FeatureExtractor):
    """
    Nonlinear-dynamics features for sustained-vowel analysis.

    Features (in order):
        rqa_rr   - Recurrence Rate
        rqa_det  - Determinism
        rqa_lam  - Laminarity
        rpde     - Recurrence Period Density Entropy
        dfa      - Detrended Fluctuation Analysis exponent
        sampen   - Sample Entropy (m=2)

    All features are computed after normalising the signal to
    zero-mean / unit-variance, making them gain-independent.

    The signal is internally downsampled to 8 kHz before computation
    to keep cost tractable; this preserves all voice-relevant content.
    RQA is run on a 1000-sample segment to avoid O(N²) blowup.

    Parameters
    ----------
    rqa_m, rqa_tau, rqa_eps_frac, rqa_lmin
        Embedding and threshold parameters for RQA.
    rpde_m, rpde_tau, rpde_eps_frac, rpde_T_max
        Parameters for RPDE.
    sampen_m, sampen_r
        Template length and tolerance multiplier for sample entropy.
    use_*
        Per-feature enable switches.
    """

    def __init__(
        self,
        sr: int = 16000,
        rqa_m: int = 3,
        rqa_tau: int = 2,
        rqa_eps_frac: float = 0.10,
        rqa_lmin: int = 2,
        rpde_m: int = 4,
        rpde_tau: int = 1,
        rpde_eps_frac: float = 0.10,
        rpde_T_max: int = 200,
        sampen_m: int = 2,
        sampen_r: float = 0.2,
        enabled: bool = True,
        cache_disabled: bool = False,
        use_rqa_rr: bool = True,
        use_rqa_det: bool = True,
        use_rqa_lam: bool = True,
        use_rpde: bool = True,
        use_dfa: bool = True,
        use_sampen: bool = True,
    ):
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.rqa_m = rqa_m
        self.rqa_tau = rqa_tau
        self.rqa_eps_frac = rqa_eps_frac
        self.rqa_lmin = rqa_lmin
        self.rpde_m = rpde_m
        self.rpde_tau = rpde_tau
        self.rpde_eps_frac = rpde_eps_frac
        self.rpde_T_max = rpde_T_max
        self.sampen_m = sampen_m
        self.sampen_r = sampen_r
        self.use_rqa_rr = use_rqa_rr
        self.use_rqa_det = use_rqa_det
        self.use_rqa_lam = use_rqa_lam
        self.use_rpde = use_rpde
        self.use_dfa = use_dfa
        self.use_sampen = use_sampen

    @staticmethod
    def _extract_full_features(  # type: ignore[override]
        x: np.ndarray,
        sr: int,
        rqa_m: int, rqa_tau: int, rqa_eps_frac: float, rqa_lmin: int,
        rpde_m: int, rpde_tau: int, rpde_eps_frac: float, rpde_T_max: int,
        sampen_m: int, sampen_r: float,
    ) -> np.ndarray:
        # Downsample to 8 kHz for speed, then normalise
        step = max(1, sr // 8000)
        xs = _normalise(x[::step])

        # RQA on a short segment (O(N²) cost)
        seg_len = min(len(xs), 1000)
        rr, det, lam = _rqa_features(xs[:seg_len], m=rqa_m, tau=rqa_tau,
                                      eps_frac=rqa_eps_frac, lmin=rqa_lmin)

        rpde_val = _rpde(xs, m=rpde_m, tau=rpde_tau,
                         eps_frac=rpde_eps_frac, T_max=rpde_T_max)

        try:
            dfa_val = float(nolds.dfa(xs))
        except Exception:
            dfa_val = 0.0

        try:
            sampen_val = float(nolds.sampen(xs, emb_dim=sampen_m,
                                            tolerance=sampen_r * np.std(xs)))
        except Exception:
            sampen_val = 0.0

        features = np.array([rr, det, lam, rpde_val, dfa_val, sampen_val],
                             dtype=np.float32)
        return np.nan_to_num(features, nan=0.0)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        features = self.cached_extraction(
            x,
            sr=self.sr,
            rqa_m=self.rqa_m, rqa_tau=self.rqa_tau,
            rqa_eps_frac=self.rqa_eps_frac, rqa_lmin=self.rqa_lmin,
            rpde_m=self.rpde_m, rpde_tau=self.rpde_tau,
            rpde_eps_frac=self.rpde_eps_frac, rpde_T_max=self.rpde_T_max,
            sampen_m=self.sampen_m, sampen_r=self.sampen_r,
        )
        mask = np.array([
            self.use_rqa_rr, self.use_rqa_det, self.use_rqa_lam,
            self.use_rpde, self.use_dfa, self.use_sampen,
        ])
        return features[mask]


# ============================================================
# New robust extractor 3: VocalTractExtractor
# ============================================================

class VocalTractExtractor(FeatureExtractor):
    """
    Vocal-tract area-function features for sustained-vowel analysis.

    The vocal tract is modelled as p cylindrical tubes via linear
    prediction (LP).  For healthy phonation the LP coefficients -
    and thus the area function - are stable across frames.
    Pathological phonation introduces frame-to-frame variability
    captured by the statistics below.

    Each frame is normalised to unit energy before LP analysis so
    results are gain-independent and robust across microphones.

    Features (total = 3 * (lp_order + 1)):
        vt_var_k        - variance of area_k across frames
        vt_ratio_var_k  - var(area_k) / mean(area_k)          (coefficient of variation)
        vt_kurtosis_k   - excess kurtosis of area_k across frames

    Parameters
    ----------
    lp_order  : LP model order (default 12, standard for 8 kHz speech)
    frame_ms  : analysis frame length in milliseconds
    hop_ms    : frame hop in milliseconds
    use_var, use_ratio_var, use_kurtosis
        Per-feature-group enable switches.
    """

    def __init__(
        self,
        sr: int = 16000,
        lp_order: int = 12,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        enabled: bool = True,
        cache_disabled: bool = False,
        use_var: bool = True,
        use_ratio_var: bool = True,
        use_kurtosis: bool = True,
    ):
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.lp_order = lp_order
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.use_var = use_var
        self.use_ratio_var = use_ratio_var
        self.use_kurtosis = use_kurtosis

    @staticmethod
    def _extract_full_features(  # type: ignore[override]
        x: np.ndarray, sr: int, lp_order: int, frame_ms: float, hop_ms: float,
    ) -> np.ndarray:
        frame_len = int(frame_ms * sr / 1000)
        hop_len = int(hop_ms * sr / 1000)
        n_tubes = lp_order + 1
        window = np.hamming(frame_len)
        areas_over_time = []

        for start in range(0, len(x) - frame_len + 1, hop_len):
            frame = x[start: start + frame_len] * window
            energy = np.sqrt(np.mean(frame ** 2))
            if energy < 1e-8:
                continue
            frame = frame / energy   # unit-energy normalisation

            r = np.correlate(frame, frame, mode="full")[frame_len - 1:]
            r = r[:lp_order + 1]
            if r[0] == 0:
                continue

            a = _levinson_durbin(r, lp_order)
            areas = _lp_to_area(a)
            if len(areas) == n_tubes:
                areas_over_time.append(areas)

        if len(areas_over_time) < 4:
            return np.zeros(3 * n_tubes, dtype=np.float32)

        A = np.array(areas_over_time)           # (n_frames, n_tubes)
        var = np.var(A, axis=0)
        mean = np.mean(A, axis=0)
        ratio_var = var / (np.abs(mean) + 1e-8)
        kurt = sp_kurtosis(A, axis=0, fisher=True)

        features = np.concatenate([var, ratio_var, kurt]).astype(np.float32)
        return np.nan_to_num(features, nan=0.0)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        all_feats = self.cached_extraction(
            x, sr=self.sr, lp_order=self.lp_order,
            frame_ms=self.frame_ms, hop_ms=self.hop_ms,
        )
        n = self.lp_order + 1
        parts = []
        if self.use_var:
            parts.append(all_feats[:n])
        if self.use_ratio_var:
            parts.append(all_feats[n: 2 * n])
        if self.use_kurtosis:
            parts.append(all_feats[2 * n: 3 * n])
        if not parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(parts)
