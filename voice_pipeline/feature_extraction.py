"""Feature extraction transformers."""

import numpy as np
import librosa
import parselmouth
from typing import Literal
from parselmouth.praat import call
from scipy.stats import skew, kurtosis as sp_kurtosis
import spkit as sp
from sklearn.base import BaseEstimator, TransformerMixin
import atexit
import nolds
from voice_pipeline.utils import _floor_power_of_2
import opensmile

from voice_pipeline.feature_cache import FeatureCache
from voice_pipeline.hashable import HashableArray, HashableDict

cache = FeatureCache()
atexit.register(cache.flush_to_disk)


# ============================================================
# Helper functions
# ============================================================

# Normalize sex labels

def normalize_sex(value) -> str:
    """Normalize a sex label to ``"m"`` or ``"f"``.

    Parameters
    ----------
    value : object
        Original sex or gender label.

    Returns
    -------
    str
        ``"m"`` for male labels, ``"f"`` for female labels

    Notes
    -----
    The labels ``"0"`` and ``"1"`` are accepted for backward compatibility.
    """
    if value is None:
        raise ValueError(f"None as sex label")

    s = str(value).strip().lower()

    if s in {"m", "male", "man", "0", 0}:
        return "m"
    elif s in {"f", "female", "woman", "w", "1", 1}:
        return "f"
    else:
        raise ValueError(f"Unknown sex label: {value}")


# sex-specific F0 ranges

F0_RANGES = {
    "m": (75.0, 300.0),
    "f": (100.0, 600.0),
}


def f0_range_for_sex(sex: Literal['m', 'f'] | None) -> tuple[float, float]:
    """Get the F0 search range for a given sex.

    Parameters
    ----------
    sex : str or None
        The sex (normalized to "m" or "f") for which to get the F0 range.
        If None, the default range (minimal lower bound and maximal upper bound) is returned.

    Returns
    -------
    tuple[float, float]
        The f0 search range for the given sex.
    """
    if sex is None:
        return (min(r[0] for r in F0_RANGES.values()), # minimum of all lower bounds
                max(r[1] for r in F0_RANGES.values())) # maximum of all upper bounds
    if not sex in F0_RANGES.keys():
        raise ValueError(f"Unknown sex label: {sex}")
    return F0_RANGES[sex]


# ============================================================
# Base class
# ============================================================


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Base class for waveform feature extractors.

    Subclasses extract features from the waveform stored in column 0 and append
    the selected feature values to each row.

    Parameters
    ----------
    enabled : bool
        Whether this extractor should append features.
    cache_disabled : bool, default=False
        Whether to bypass the feature cache.

    Notes
    -----
    Subclasses must implement ``_extract_full_features`` and
    ``extract_feature``.
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
    """Drop the waveform column after feature extraction.

    The first column of ``X`` must contain the waveform. All remaining columns
    are converted to ``float32`` features.
    """

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
    """Extract Mel-frequency cepstral coefficient features.

    Parameters
    ----------
    sr : int, default=16000
        Sampling rate in Hz.
    n_mfcc : int, default=20
        Number of MFCC coefficients.
    enabled : bool, default=True
        Whether this extractor should append features.
    use_std : bool, default=True
        Whether to include coefficient standard deviations in addition to
        coefficient means.
    cache_disabled : bool, default=False
        Whether to bypass the feature cache.

    Notes
    -----
    The full feature vector contains ``2 * n_mfcc`` values: means followed by
    standard deviations.
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
    """Extract time-domain voice features.

    The available features are mean F0, F0 standard deviation, relative pitch
    range, harmonics-to-noise ratio, jitter, shimmer, entropy, and skewness.

    Parameters
    ----------
    sr : int, default=16000
        Sampling rate in Hz.
    fmin, fmax : float, default=60.0, 250.0
        Fallback F0 search range when sex is unknown.
    enabled : bool, default=True
        Whether this extractor should append features.
    cache_disabled : bool, default=False
        Whether to bypass the feature cache.
    use_diff_pitch, use_f0, use_stdf0, use_hnr, use_jitter, use_shimmer, use_zcr, use_entropy, use_skewness : bool
        Per-feature switches.
    sex_col : int, default=1
        Column index containing the sex label.
    f0_range_m, f0_range_f : tuple of float
        Sex-specific F0 search ranges.
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
        sex_col=None,
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
        # suppress numpy warning when all frames are unvoiced (all-NaN slice) — result is NaN, handled by nan_to_num below
        with np.errstate(all="ignore"):
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
            sex = normalize_sex(row[self.sex_col]) if self.sex_col is not None else None
            fmin, fmax = f0_range_for_sex(sex)
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
    """Extract zero-crossing rate features.

    Parameters
    ----------
    sr : int, default=16000
        Sampling rate in Hz.
    use_mean, use_std : bool, default=True
        Whether to include the mean and standard deviation of the frame-wise
        zero-crossing rate.
    enabled : bool, default=True
        Whether this extractor should append features.
    cache_disabled : bool, default=False
        Whether to bypass the feature cache.
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
    """Extract spectral summary features.

    Parameters
    ----------
    sr : int, default=16000
        Sampling rate in Hz.
    use_flatness, use_rolloff, use_centroid, use_contrast : bool, default=True
        Per-feature switches.
    enabled : bool, default=True
        Whether this extractor should append features.
    cache_disabled : bool, default=False
        Whether to bypass the feature cache.

    Notes
    -----
    The full feature vector contains spectral flatness, rolloff, centroid, and
    spectral contrast bands.
    """

    def __init__(
        self,
        sr=16000,
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
        mask = [self.use_flatness, self.use_rolloff, self.use_centroid] + [self.use_contrast] * len(features[3:])
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
    """Extract an OpenSMILE feature set from a waveform.

    Parameters
    ----------
    sr : int, default=16000
        Sampling rate in Hz.
    feature_set : str, default="ComParE_2016"
        OpenSMILE feature set name.
    feature_level : str, default="functionals"
        OpenSMILE feature level name.
    enabled : bool, default=False
        Whether this extractor should append features.
    cache_disabled : bool, default=False
        Whether to bypass the feature cache.
    """

    def __init__(
        self, sr=16000, feature_set="ComParE_2016", feature_level="functionals", enabled=False, cache_disabled=False
    ):
        super().__init__(enabled=enabled, cache_disabled=cache_disabled)
        self.sr = sr
        self.feature_set = feature_set
        self.feature_level = feature_level

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        return self.cached_extraction(x, sr=self.sr, feature_set=self.feature_set, feature_level=self.feature_level)

    @staticmethod
    def _extract_full_features(
        x: np.ndarray, sr: int, feature_set: str, feature_level: str
    ) -> np.ndarray:  # type: ignore[override]
        x = np.asarray(x).reshape(-1)
        smile = _get_smile(feature_set=feature_set, feature_level=feature_level)
        df = smile.process_signal(x, sr)
        vec = df.iloc[0].to_numpy(dtype=np.float32, copy=False)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


# ============================================================
# Shared helpers for robust extractors
# ============================================================


def _normalise(x: np.ndarray) -> np.ndarray:
    """Normalize a signal to zero mean and unit variance.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal.

    Returns
    -------
    numpy.ndarray
        Normalized signal. If the standard deviation is nearly zero, only the
        mean is removed.
    """
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-10:
        return x - mu
    return (x - mu) / sigma


def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """Solve Yule-Walker equations with Levinson-Durbin recursion.

    Parameters
    ----------
    r : numpy.ndarray
        Autocorrelation sequence from lag 0 through ``order``.
    order : int
        Linear prediction model order.

    Returns
    -------
    numpy.ndarray
        Linear prediction coefficients. The leading coefficient 1 is implicit.
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
            a_new[:i] = a[:i] + lam * a[i - 1 :: -1]
        a_new[i] = lam
        a = a_new
        e *= 1.0 - lam**2
        if e <= 1e-12:
            break
    return -a


def _lp_to_area(a: np.ndarray) -> np.ndarray:
    """Convert linear prediction coefficients to vocal-tract areas.

    Parameters
    ----------
    a : numpy.ndarray
        Linear prediction coefficients.

    Returns
    -------
    numpy.ndarray
        Relative tube cross-sections with length ``len(a) + 1``.
    """
    k_all = []
    coef = a.copy().astype(float)
    order = len(coef)
    for i in range(order - 1, -1, -1):
        ki = coef[i]
        k_all.append(ki)
        denom = 1.0 - ki**2
        if abs(denom) < 1e-12:
            break
        if i > 0:
            coef[:i] = (coef[:i] - ki * coef[i - 1 :: -1]) / denom
    areas = np.ones(len(k_all) + 1)
    for j, ki in enumerate(k_all):
        areas[j + 1] = areas[j] * (1.0 - ki) / (1.0 + ki + 1e-12)
    return areas


def _rqa_features(x: np.ndarray, m: int = 3, tau: int = 2, eps_frac: float = 0.10, lmin: int = 2) -> np.ndarray:
    """Compute recurrence quantification analysis features.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional signal. It is normalized internally.
    m : int, default=3
        Embedding dimension.
    tau : int, default=2
        Time delay.
    eps_frac : float, default=0.10
        Recurrence threshold as a fraction of mean pairwise distance.
    lmin : int, default=2
        Minimum line length used for determinism and laminarity.

    Returns
    -------
    numpy.ndarray
        Three values: recurrence rate, determinism, and laminarity.
    """
    x = _normalise(x)
    # pylint: disable=invalid-name
    N = len(x)
    M = N - (m - 1) * tau
    if M <= lmin:
        return np.zeros(3, dtype=np.float32)

    X = np.stack([x[i : i + (m - 1) * tau + 1 : tau] for i in range(M)])
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))

    eps = eps_frac * dist.mean()
    R = (dist < eps).astype(np.int8)
    np.fill_diagonal(R, 0)
    # pylint: enable=invalid-name

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
                run += 1
                in_line = True
            else:
                if in_line and run >= lmin:
                    det_pts += run
                run = 0
                in_line = False
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
                run += 1
                in_line = True
            else:
                if in_line and run >= lmin:
                    lam_pts += run
                run = 0
                in_line = False
        if in_line and run >= lmin:
            lam_pts += run
    lam = lam_pts / total

    return np.array([rr, det, lam], dtype=np.float32)


def _rpde(x: np.ndarray, m: int = 4, tau: int = 1, eps_frac: float = 0.10, T_max: int = 200) -> float:
    """Compute recurrence period density entropy.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional signal. It is normalized internally.
    m : int, default=4
        Embedding dimension.
    tau : int, default=1
        Time delay.
    eps_frac : float, default=0.10
        Recurrence threshold as a fraction of mean distance.
    T_max : int, default=200
        Maximum recurrence period considered.

    Returns
    -------
    float
        Normalized entropy in the interval [0, 1].
    """
    x = _normalise(x)
    N = len(x)
    M = N - (m - 1) * tau
    if M < 20:
        return 0.0

    X = np.stack([x[i : i + (m - 1) * tau + 1 : tau] for i in range(M)])
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

    counts = np.bincount(periods, minlength=T_max + 1)[1 : T_max + 1].astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(np.clip(H / np.log(T_max), 0.0, 1.0))


# ============================================================
# New robust extractor 1: F0DerivativesExtractor
# ============================================================


class F0DerivativesExtractor(FeatureExtractor):
    """Extract robust F0-derived features for sustained vowels.

    The available features are F0 interquartile range, F0 range, pitch period
    entropy, unvoiced-frame rate, and the standard deviation of relative
    cycle-to-cycle F0 change. All features are independent of absolute signal
    amplitude.

    Parameters
    ----------
    sr : int, default=16000
        Sampling rate in Hz.
    fmin, fmax : float, default=60.0, 400.0
        Fallback F0 search range when sex is unknown.
    f0_range_m, f0_range_f : tuple of float
        Sex-specific F0 search ranges.
    sex_col : int, default=1
        Column index containing the sex label.
    enabled : bool, default=True
        Whether this extractor should append features.
    cache_disabled : bool, default=False
        Whether to bypass the feature cache.
    use_f0_iqr, use_f0_range, use_ppe, use_nan_rate, use_pitch_diff_std : bool
        Per-feature switches.
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
        x: np.ndarray,
        fmin: float,
        fmax: float,
        sr: int,
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
        mask = np.array(
            [
                self.use_f0_iqr,
                self.use_f0_range,
                self.use_ppe,
                self.use_nan_rate,
                self.use_pitch_diff_std,
            ]
        )
        return features[mask]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Append F0-derived features using sex-specific F0 ranges.

        Parameters
        ----------
        X : numpy.ndarray
            Input rows. Column 0 must contain the waveform.

        Returns
        -------
        numpy.ndarray
            Rows with selected F0-derived features appended.
        """
        if not self.enabled:
            return X
        X_out = []
        for row in X:
            x = row[0]
            sex_mf = normalize_sex(row[self.sex_col] if len(row) > self.sex_col else None)
            if sex_mf == "m":
                fmin, fmax = self.f0_range_m
            elif sex_mf == "f":
                fmin, fmax = self.f0_range_f
            else:
                fmin, fmax = self.fmin, self.fmax
            features = self.cached_extraction(x, fmin=fmin, fmax=fmax, sr=self.sr)
            mask = np.array(
                [
                    self.use_f0_iqr,
                    self.use_f0_range,
                    self.use_ppe,
                    self.use_nan_rate,
                    self.use_pitch_diff_std,
                ]
            )
            X_out.append(np.concatenate((row, features[mask])))
        return np.array(np.vstack(X_out), dtype=object)


# ============================================================
# New robust extractor 2: NonlinearExtractor
# ============================================================


class NonlinearExtractor(FeatureExtractor):
    """Extract nonlinear dynamics features for sustained vowels.

    Available features are recurrence rate, determinism, laminarity,
    recurrence period density entropy, detrended fluctuation analysis exponent,
    and sample entropy.

    The signal is normalized before feature computation, making the features
    gain-independent. RQA is run on at most 1000 samples to keep runtime
    tractable.

    Parameters
    ----------
    sr : int, default=16000
        Sampling rate in Hz.
    rqa_m, rqa_tau, rqa_eps_frac, rqa_lmin : int or float
        Embedding and threshold parameters for RQA.
    rpde_m, rpde_tau, rpde_eps_frac, rpde_T_max : int or float
        Parameters for RPDE.
    sampen_m, sampen_r : int or float
        Template length and tolerance multiplier for sample entropy.
    enabled : bool, default=True
        Whether this extractor should append features.
    cache_disabled : bool, default=False
        Whether to bypass the feature cache.
    use_rqa_rr, use_rqa_det, use_rqa_lam, use_rpde, use_dfa, use_sampen : bool
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
        rqa_m: int,
        rqa_tau: int,
        rqa_eps_frac: float,
        rqa_lmin: int,
        rpde_m: int,
        rpde_tau: int,
        rpde_eps_frac: float,
        rpde_T_max: int,
        sampen_m: int,
        sampen_r: float,
    ) -> np.ndarray:
        # Downsample to 8 kHz for speed, then normalise
        step = max(1, sr // 8000)
        xs = _normalise(x[::step])

        # RQA on a short segment (O(N^2) cost)
        seg_len = min(len(xs), 1000)
        rr, det, lam = _rqa_features(xs[:seg_len], m=rqa_m, tau=rqa_tau, eps_frac=rqa_eps_frac, lmin=rqa_lmin)

        rpde_val = _rpde(xs, m=rpde_m, tau=rpde_tau, eps_frac=rpde_eps_frac, T_max=rpde_T_max)

        try:
            dfa_val = float(nolds.dfa(xs))
        except Exception:
            dfa_val = 0.0

        try:
            sampen_val = float(nolds.sampen(xs, emb_dim=sampen_m, tolerance=sampen_r * np.std(xs)))
        except Exception:
            sampen_val = 0.0

        features = np.array([rr, det, lam, rpde_val, dfa_val, sampen_val], dtype=np.float32)
        return np.nan_to_num(features, nan=0.0)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        features = self.cached_extraction(
            x,
            sr=self.sr,
            rqa_m=self.rqa_m,
            rqa_tau=self.rqa_tau,
            rqa_eps_frac=self.rqa_eps_frac,
            rqa_lmin=self.rqa_lmin,
            rpde_m=self.rpde_m,
            rpde_tau=self.rpde_tau,
            rpde_eps_frac=self.rpde_eps_frac,
            rpde_T_max=self.rpde_T_max,
            sampen_m=self.sampen_m,
            sampen_r=self.sampen_r,
        )
        mask = np.array(
            [
                self.use_rqa_rr,
                self.use_rqa_det,
                self.use_rqa_lam,
                self.use_rpde,
                self.use_dfa,
                self.use_sampen,
            ]
        )
        return features[mask]


# ============================================================
# New robust extractor 3: VocalTractExtractor
# ============================================================


class VocalTractExtractor(FeatureExtractor):
    """Extract vocal-tract area-function features.

    The vocal tract is modelled as cylindrical tubes via linear prediction.
    Frame-to-frame variability in the area function is summarized with
    variance, normalized variance, and kurtosis.

    Parameters
    ----------
    sr : int, default=16000
        Sampling rate in Hz.
    lp_order : int, default=12
        Linear prediction model order.
    frame_ms : float, default=25.0
        Analysis frame length in milliseconds.
    hop_ms : float, default=10.0
        Frame hop in milliseconds.
    enabled : bool, default=True
        Whether this extractor should append features.
    cache_disabled : bool, default=False
        Whether to bypass the feature cache.
    use_var, use_ratio_var, use_kurtosis : bool, default=True
        Per-feature-group switches.

    Notes
    -----
    The full feature vector has ``3 * (lp_order + 1)`` values.
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
        x: np.ndarray,
        sr: int,
        lp_order: int,
        frame_ms: float,
        hop_ms: float,
    ) -> np.ndarray:
        frame_len = int(frame_ms * sr / 1000)
        hop_len = int(hop_ms * sr / 1000)
        n_tubes = lp_order + 1
        window = np.hamming(frame_len)
        areas_over_time = []

        for start in range(0, len(x) - frame_len + 1, hop_len):
            frame = x[start : start + frame_len] * window
            energy = np.sqrt(np.mean(frame**2))
            if energy < 1e-8:
                continue
            frame = frame / energy  # unit-energy normalisation

            r = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
            r = r[: lp_order + 1]
            if r[0] == 0:
                continue

            a = _levinson_durbin(r, lp_order)
            areas = _lp_to_area(a)
            if len(areas) == n_tubes:
                areas_over_time.append(areas)

        if len(areas_over_time) < 4:
            return np.zeros(3 * n_tubes, dtype=np.float32)

        areas_over_time = np.array(areas_over_time)  # (n_frames, n_tubes)
        var = np.var(areas_over_time, axis=0)
        mean = np.mean(areas_over_time, axis=0)
        ratio_var = var / (np.abs(mean) + 1e-8)
        kurt = sp_kurtosis(areas_over_time, axis=0, fisher=True)

        features = np.concatenate([var, ratio_var, kurt]).astype(np.float32)
        return np.nan_to_num(features, nan=0.0)

    def extract_feature(self, x: np.ndarray) -> np.ndarray:
        all_feats = self.cached_extraction(
            x,
            sr=self.sr,
            lp_order=self.lp_order,
            frame_ms=self.frame_ms,
            hop_ms=self.hop_ms,
        )
        n = self.lp_order + 1
        parts = []
        if self.use_var:
            parts.append(all_feats[:n])
        if self.use_ratio_var:
            parts.append(all_feats[n : 2 * n])
        if self.use_kurtosis:
            parts.append(all_feats[2 * n : 3 * n])
        if not parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(parts)
