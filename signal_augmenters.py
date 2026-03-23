"""Signal (waveform)-level augmenters.

Two public classes
------------------
RawAugmenter
    sklearn Transformer — modifies every sample during transform().
    Applied to both train AND test folds (shape-preserving).
    Place before feature extractors.

WaveformSynthesizer
    imblearn Sampler — generates n_samples brand-new rows by augmenting
    existing waveforms.  Called by ImbPipeline only on the TRAIN fold,
    so synthetic audio NEVER reaches the classifier as test data.
    Place before feature extractors (after RawAugmenter).

Available augmentation techniques (shared by both classes)
----------------------------------------------------------
Waveform domain:
    noise           Additive Gaussian white noise.
    gain            Random amplitude scaling (uniform range).
    polarity        Polarity inversion — flip sign of the whole signal.
                    Applied randomly with 50 % probability.
    time_crop       Random crop to [crop_min_frac ... 1.0] of the original
                    length, zero-padded back to original length.

Pitch / tempo:
    pitch_shift     Shift pitch by a random number of semitones without
                    changing duration.  Requires librosa.
    time_stretch    Speed up / slow down without changing pitch.
                    Requires librosa.

Environment / codec:
    reverb          Convolve with a synthetic exponential room impulse
                    response (RIR) to simulate room acoustics.
    codec           mu-law encode -> quantise to 8 bit -> decode.
                    Simulates telephone / low-bitrate codec artefacts.

librosa is imported lazily — the module loads without it, but a clear
ImportError is raised if pitch_shift or time_stretch are enabled and
librosa is not installed.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.base import BaseSampler

_librosa = None


def _get_librosa():
    global _librosa
    if _librosa is None:
        try:
            import librosa as _lib
            _librosa = _lib
        except ImportError as exc:
            raise ImportError(
                "librosa is required for pitch_shift / time_stretch. "
                "Install it with:  pip install librosa"
            ) from exc
    return _librosa


# ---------------------------------------------------------------------------
# Core augmentation logic (shared by both classes)
# ---------------------------------------------------------------------------

def _augment_waveform(
    x: np.ndarray,
    sr: int,
    rng: np.random.RandomState,
    noise: bool,
    gain: bool,
    polarity: bool,
    time_crop: bool,
    pitch_shift: bool,
    time_stretch: bool,
    reverb: bool,
    codec: bool,
    noise_std: float,
    gain_min: float,
    gain_max: float,
    crop_min_frac: float,
    pitch_semitones_range: tuple[float, float],
    stretch_rate_range: tuple[float, float],
    reverb_decay: float,
    n_augmentations: int | None,
) -> np.ndarray:
    """Apply all enabled augmentations to a single 1-D waveform."""

    x = np.asarray(x, dtype=np.float32).copy()
    original_len = len(x)

    # If n_augmentations is set, randomly pick that many techniques from
    # the enabled ones and disable the rest for this sample.
    # n_augmentations=None means apply all enabled techniques (original behaviour).
    if n_augmentations is not None:
        enabled_techniques = [k for k, v in {
            "noise": noise, "gain": gain, "polarity": polarity,
            "time_crop": time_crop, "pitch_shift": pitch_shift,
            "time_stretch": time_stretch, "reverb": reverb, "codec": codec,
        }.items() if v]
        n_pick = min(n_augmentations, len(enabled_techniques))
        chosen = set(rng.choice(enabled_techniques, size=n_pick, replace=False))
        noise       = "noise"       in chosen
        gain        = "gain"        in chosen
        polarity    = "polarity"    in chosen
        time_crop   = "time_crop"   in chosen
        pitch_shift = "pitch_shift" in chosen
        time_stretch = "time_stretch" in chosen
        reverb      = "reverb"      in chosen
        codec       = "codec"       in chosen

    # 1. Random gain
    if gain:
        x = x * rng.uniform(gain_min, gain_max)

    # 2. Additive Gaussian noise
    if noise:
        x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)

    # 3. Polarity inversion (50 % chance)
    if polarity and rng.rand() < 0.5:
        x = -x

    # 4. Time crop — keep a random fraction, zero-pad back to original length
    if time_crop:
        keep = int(original_len * rng.uniform(crop_min_frac, 1.0))
        start = rng.randint(0, max(1, original_len - keep))
        buf = np.zeros(original_len, dtype=np.float32)
        buf[:keep] = x[start: start + keep]
        x = buf

    # 5. Pitch shift — shift by random semitones, duration unchanged
    if pitch_shift:
        lib = _get_librosa()
        n_steps = rng.uniform(*pitch_semitones_range)
        x = lib.effects.pitch_shift(x.astype(np.float64), sr=sr, n_steps=n_steps).astype(np.float32)

    # 6. Time stretch — speed up/slow down, pitch unchanged
    if time_stretch:
        lib = _get_librosa()
        rate = rng.uniform(*stretch_rate_range)
        x = lib.effects.time_stretch(x.astype(np.float64), rate=rate).astype(np.float32)

    # Restore original length after librosa ops (may change array size)
    if pitch_shift or time_stretch:
        if len(x) > original_len:
            x = x[:original_len]
        elif len(x) < original_len:
            x = np.pad(x, (0, original_len - len(x)))

    # 7. Reverb — convolve with synthetic exponential RIR
    if reverb:
        rir_len = int(sr * 0.3)
        t = np.arange(rir_len, dtype=np.float32) / sr
        rir = np.exp(-reverb_decay * t)
        rir /= np.sum(np.abs(rir)) + 1e-8
        x = np.convolve(x, rir, mode="full")[:original_len].astype(np.float32)

    # 8. Codec simulation — mu-law encode -> 8-bit quantise -> decode
    if codec:
        mu = 255.0
        xc = np.clip(x, -1.0, 1.0)
        enc = np.sign(xc) * np.log1p(mu * np.abs(xc)) / np.log1p(mu)
        q = np.round(enc * 127).astype(np.int8).astype(np.float32) / 127.0
        x = (np.sign(q) * (1.0 / mu) * (np.power(1.0 + mu, np.abs(q)) - 1.0)).astype(np.float32)

    return x


# ---------------------------------------------------------------------------
# Shared parameter defaults (single source of truth)
# ---------------------------------------------------------------------------

_AUG_DEFAULTS = dict(
    sr=50_000,
    noise=True,
    gain=True,
    polarity=False,
    time_crop=False,
    pitch_shift=False,
    time_stretch=False,
    reverb=False,
    codec=False,
    noise_std=0.005,
    gain_min=0.8,
    gain_max=1.2,
    crop_min_frac=0.8,
    pitch_semitones_range=(-2.0, 2.0),
    stretch_rate_range=(0.9, 1.1),
    reverb_decay=10.0,
    n_augmentations=None,  # None = apply all enabled techniques
)


def _aug_kwargs(self) -> dict:
    """Collect augmentation parameters from an instance into a dict for _augment_waveform."""
    return dict(
        sr=self.sr,
        noise=self.noise,
        gain=self.gain,
        polarity=self.polarity,
        time_crop=self.time_crop,
        pitch_shift=self.pitch_shift,
        time_stretch=self.time_stretch,
        reverb=self.reverb,
        codec=self.codec,
        noise_std=self.noise_std,
        gain_min=self.gain_min,
        gain_max=self.gain_max,
        crop_min_frac=self.crop_min_frac,
        pitch_semitones_range=self.pitch_semitones_range,
        stretch_rate_range=self.stretch_rate_range,
        reverb_decay=self.reverb_decay,
        n_augmentations=self.n_augmentations,
    )


# ---------------------------------------------------------------------------
# RawAugmenter — sklearn Transformer, shape-preserving
# ---------------------------------------------------------------------------

class RawAugmenter(BaseEstimator, TransformerMixin):
    """Waveform-level augmentation applied to every sample in transform().

    This is a standard sklearn Transformer, so it is called on BOTH the
    train and test folds.  It does NOT change the number of rows — it only
    modifies the waveform in column 0 of each row in-place.

    Fully backwards-compatible with the original two-parameter version
    (noise_std, gain_min, gain_max still work unchanged).

    Parameters
    ----------
    enabled : bool
        Master switch.  When False returns X unchanged.
    sr : int
        Sample rate.  Required for pitch_shift, time_stretch, and reverb.
    noise, gain, polarity, time_crop, pitch_shift, time_stretch,
    reverb, codec : bool
        Per-technique enable flags.  See module docstring for details.
    noise_std : float
        Std of additive Gaussian noise.
    gain_min, gain_max : float
        Uniform range for amplitude scaling.
    crop_min_frac : float
        Minimum fraction of signal retained by time_crop.
    pitch_semitones_range : (float, float)
        (min, max) semitones for pitch_shift.
    stretch_rate_range : (float, float)
        (min, max) rate for time_stretch (>1 faster, <1 slower).
    reverb_decay : float
        Exponential decay coefficient for the synthetic RIR.
    n_augmentations : int or None
        If set, randomly pick this many techniques from the enabled ones
        for each sample.  None = apply all enabled techniques (default).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        enabled: bool = True,
        sr: int = _AUG_DEFAULTS["sr"],
        noise: bool = _AUG_DEFAULTS["noise"],
        gain: bool = _AUG_DEFAULTS["gain"],
        polarity: bool = _AUG_DEFAULTS["polarity"],
        time_crop: bool = _AUG_DEFAULTS["time_crop"],
        pitch_shift: bool = _AUG_DEFAULTS["pitch_shift"],
        time_stretch: bool = _AUG_DEFAULTS["time_stretch"],
        reverb: bool = _AUG_DEFAULTS["reverb"],
        codec: bool = _AUG_DEFAULTS["codec"],
        noise_std: float = _AUG_DEFAULTS["noise_std"],
        gain_min: float = _AUG_DEFAULTS["gain_min"],
        gain_max: float = _AUG_DEFAULTS["gain_max"],
        crop_min_frac: float = _AUG_DEFAULTS["crop_min_frac"],
        pitch_semitones_range: tuple[float, float] = _AUG_DEFAULTS["pitch_semitones_range"],
        stretch_rate_range: tuple[float, float] = _AUG_DEFAULTS["stretch_rate_range"],
        reverb_decay: float = _AUG_DEFAULTS["reverb_decay"],
        n_augmentations: int | None = _AUG_DEFAULTS["n_augmentations"],
        random_state: int | None = None,
    ):
        self.enabled = enabled
        self.sr = sr
        self.noise = noise
        self.gain = gain
        self.polarity = polarity
        self.time_crop = time_crop
        self.pitch_shift = pitch_shift
        self.time_stretch = time_stretch
        self.reverb = reverb
        self.codec = codec
        self.noise_std = noise_std
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.crop_min_frac = crop_min_frac
        self.pitch_semitones_range = pitch_semitones_range
        self.stretch_rate_range = stretch_rate_range
        self.reverb_decay = reverb_decay
        self.n_augmentations = n_augmentations
        self.random_state = random_state

    def fit(self, X, y=None):
        del X, y
        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply augmentation to every row.  Column 0 is the waveform."""
        if not self.enabled:
            return X
        kwargs = _aug_kwargs(self)
        X_out = []
        for row in X:
            row = list(row)
            row[0] = _augment_waveform(x=row[0], rng=self.rng_, **kwargs)
            X_out.append(row)
        return np.array(X_out, dtype=object)


# ---------------------------------------------------------------------------
# WaveformSynthesizer — imblearn Sampler, generates new rows
# ---------------------------------------------------------------------------

class WaveformSynthesizer(BaseSampler):
    """Imblearn-compatible sampler that synthesises new waveform rows.

    Because this class inherits from BaseSampler and is placed inside an
    ImbPipeline, its fit_resample() is called ONLY on the training fold.
    Synthetic audio therefore NEVER reaches the test fold or the
    classifier as evaluation data.

    The synthesizer picks existing waveforms at random (with replacement),
    augments each one, and appends the result as a new row.  Demographic
    columns (age, sex, ...) are copied unchanged from the source row.

    n_samples new rows are distributed proportionally to existing class
    counts so the class balance is preserved.

    Parameters
    ----------
    enabled : bool
        Master switch.  When False returns X, y unchanged.
    n_samples : int
        Total number of NEW rows to add across all classes.
        Set to 0 to effectively disable without changing enabled flag.
    sr : int
        Sample rate.  Required for pitch_shift, time_stretch, and reverb.
    noise, gain, polarity, time_crop, pitch_shift, time_stretch,
    reverb, codec : bool
        Per-technique flags — same semantics as RawAugmenter.
    n_augmentations : int or None
        If set, randomly pick this many techniques from the enabled ones
        for each synthetic sample.  None = apply all enabled techniques.
    (all other parameters identical to RawAugmenter)
    random_state : int or None
        Seed for reproducibility.
    """

    _sampling_type = "over-sampling"

    def __init__(
        self,
        enabled: bool = False,
        n_samples: int = 0,
        sr: int = _AUG_DEFAULTS["sr"],
        noise: bool = _AUG_DEFAULTS["noise"],
        gain: bool = _AUG_DEFAULTS["gain"],
        polarity: bool = _AUG_DEFAULTS["polarity"],
        time_crop: bool = _AUG_DEFAULTS["time_crop"],
        pitch_shift: bool = _AUG_DEFAULTS["pitch_shift"],
        time_stretch: bool = _AUG_DEFAULTS["time_stretch"],
        reverb: bool = _AUG_DEFAULTS["reverb"],
        codec: bool = _AUG_DEFAULTS["codec"],
        noise_std: float = _AUG_DEFAULTS["noise_std"],
        gain_min: float = _AUG_DEFAULTS["gain_min"],
        gain_max: float = _AUG_DEFAULTS["gain_max"],
        crop_min_frac: float = _AUG_DEFAULTS["crop_min_frac"],
        pitch_semitones_range: tuple[float, float] = _AUG_DEFAULTS["pitch_semitones_range"],
        stretch_rate_range: tuple[float, float] = _AUG_DEFAULTS["stretch_rate_range"],
        reverb_decay: float = _AUG_DEFAULTS["reverb_decay"],
        n_augmentations: int | None = _AUG_DEFAULTS["n_augmentations"],
        random_state: int | None = None,
    ):
        super().__init__()
        self.enabled = enabled
        self.n_samples = n_samples
        self.sr = sr
        self.noise = noise
        self.gain = gain
        self.polarity = polarity
        self.time_crop = time_crop
        self.pitch_shift = pitch_shift
        self.time_stretch = time_stretch
        self.reverb = reverb
        self.codec = codec
        self.noise_std = noise_std
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.crop_min_frac = crop_min_frac
        self.pitch_semitones_range = pitch_semitones_range
        self.stretch_rate_range = stretch_rate_range
        self.reverb_decay = reverb_decay
        self.n_augmentations = n_augmentations
        self.random_state = random_state

    # ImbPipeline calls this from fit_resample() — only on the train fold.
    def _fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.enabled or self.n_samples <= 0:
            return X, y

        rng = np.random.RandomState(self.random_state)
        kwargs = _aug_kwargs(self)

        classes, counts = np.unique(y, return_counts=True)
        ratios = counts / counts.sum()
        n_per_class = np.round(ratios * self.n_samples).astype(int)
        # Fix rounding drift so total == n_samples exactly
        n_per_class[np.argmax(counts)] += self.n_samples - n_per_class.sum()

        X_new, y_new = [], []
        for cls, n_gen in zip(classes, n_per_class):
            if n_gen <= 0:
                continue
            src_idx = np.where(y == cls)[0]
            chosen = rng.choice(src_idx, size=n_gen, replace=True)
            for idx in chosen:
                row = list(X[idx])
                row[0] = _augment_waveform(x=row[0], rng=rng, **kwargs)
                X_new.append(row)
                y_new.append(cls)

        X_syn = np.array(X_new, dtype=object)
        y_syn = np.array(y_new, dtype=y.dtype)

        X_out = np.concatenate([X, X_syn], axis=0)
        y_out = np.concatenate([y, y_syn], axis=0)

        perm = rng.permutation(len(y_out))
        return X_out[perm], y_out[perm]
