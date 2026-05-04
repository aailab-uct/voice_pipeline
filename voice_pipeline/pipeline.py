"""Build the machine-learning pipeline used in experiments."""

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from voice_pipeline.feature_augmentation import FeatureAugmenter
from voice_pipeline.feature_extraction import (
    DropSignalColumn,
    F0DerivativesExtractor,
    MFCCExtractor,
    NonlinearExtractor,
    OpenSmileExtractor,
    SpectralFeaturesExtractor,
    TimeDomainFeaturesExtractor,
    VocalTractExtractor,
    ZCRExtractor,
)
from voice_pipeline.registry import instantiate
from voice_pipeline.sampling import SMOTESampler
from voice_pipeline.signal_augmentation import RawAugmenter, WaveformSynthesizer
from voice_pipeline.utils import ReplaceNaN


def _scalars(section: dict | None) -> dict:
    """Return only scalar (non-list) values from a config section.

    List values are search dimensions for GridSearch/Optuna and must not
    be passed to constructors — they would cause a TypeError.
    """
    if not section:
        return {}
    return {k: v for k, v in section.items() if not isinstance(v, list)}


def build_pipeline(config: dict, random_state: int = 42) -> ImbPipeline:
    """Create the audio-to-classification pipeline from a YAML config dict.

    The raw waveform is always the first column of ``X``. The pipeline then:
    augments waveforms, extracts acoustic features, removes the waveform column,
    cleans invalid numbers, scales features, optionally selects features,
    optionally resamples classes, and finally fits a classifier.

    A pipeline step is active when its key is present in ``config``. Absent
    keys fall back to disabled defaults. List values in ``config`` are ignored
    here — they are search dimensions handled by :func:`extract_param_grid`.

    Parameters
    ----------
    config : dict
        Hierarchical config dict as returned by ``load_yaml_config()``.
    random_state : int, default=42
        Seed passed to all stochastic pipeline components.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Unfitted pipeline ready for ``fit`` or ``GridSearchCV``.
    """
    # pull top-level sections used across multiple steps
    dataset_cfg  = config.get("dataset", {})
    features_cfg = config.get("features", {})

    # sampling rate is shared by all audio-processing steps
    fs = dataset_cfg.get("fs", 48000)

    # sex column index: column 1 holds the sex label when include_sex is True;
    # None tells extractors to skip sex-dependent F0 range selection
    include_sex = dataset_cfg.get("include_sex", True)
    sex_col     = 1 if include_sex else None

    # feature cache must be disabled whenever waveforms are modified during
    # training (raw_aug or synthesizer active), because cached keys would
    # collide between original and augmented versions of the same recording
    raw_aug_cfg    = config.get("raw_aug")
    synth_cfg      = config.get("synthesizer")
    raw_aug_active = raw_aug_cfg is not None
    # synthesizer is active only when n_samples > 0; presence alone is not enough
    synth_active   = synth_cfg is not None and _scalars(synth_cfg).get("n_samples", 0) > 0
    disable_cache  = raw_aug_active or synth_active

    # -------------------------------------------------------------------------
    # Waveform augmentation
    # -------------------------------------------------------------------------

    # RawAugmenter modifies every existing waveform in-place (train AND test);
    # it does not change the number of rows
    raw_aug = RawAugmenter(
        enabled=raw_aug_active,
        sr=fs,
        random_state=random_state,
        **_scalars(raw_aug_cfg),
    )

    # WaveformSynthesizer generates new rows from existing ones;
    # as an imblearn sampler it runs only on the training fold —
    # synthetic audio never reaches evaluation
    synthesizer = WaveformSynthesizer(
        enabled=synth_active,
        sr=fs,
        random_state=random_state,
        **_scalars(synth_cfg),
    )

    # -------------------------------------------------------------------------
    # Feature extractors
    # Presence of the key in features_cfg enables the extractor.
    # _scalars() forwards only the scalar params; lists stay in param_grid.
    # -------------------------------------------------------------------------

    # Mel-frequency cepstral coefficients
    mfcc = MFCCExtractor(
        enabled="mfcc" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("mfcc")),
    )

    # Classic time-domain voice quality features (F0, HNR, jitter, shimmer …)
    # sex_col is needed for sex-specific F0 range selection
    timedomain = TimeDomainFeaturesExtractor(
        enabled="timedomain" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        sex_col=sex_col,
        **_scalars(features_cfg.get("timedomain")),
    )

    # Zero-crossing rate (frame-wise mean and std)
    zcr = ZCRExtractor(
        enabled="zcr" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("zcr")),
    )

    # Spectral summary features (centroid, rolloff, flatness, contrast)
    spectral = SpectralFeaturesExtractor(
        enabled="spectral" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("spectral")),
    )

    # Robust F0-derived features (IQR, range, PPE, unvoiced rate, pitch diff std)
    f0deriv = F0DerivativesExtractor(
        enabled="f0deriv" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("f0deriv")),
    )

    # Nonlinear dynamics features (RQA, RPDE, DFA, sample entropy)
    nonlinear = NonlinearExtractor(
        enabled="nonlinear" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("nonlinear")),
    )

    # Vocal tract area-function features derived from LPC analysis
    vocaltract = VocalTractExtractor(
        enabled="vocaltract" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("vocaltract")),
    )

    # OpenSMILE feature set (GeMAPS, ComParE …)
    opensmile = OpenSmileExtractor(
        enabled="opensmile" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("opensmile")),
    )

    # -------------------------------------------------------------------------
    # Post-extraction processing
    # -------------------------------------------------------------------------

    # feature selector: instantiated from registry if present, else skipped
    selector_cfg = features_cfg.get("selector")
    selector = instantiate(_scalars(selector_cfg)) if selector_cfg else "passthrough"

    # feature augmentation: adds small Gaussian jitter to feature columns
    # (train only, applied after extraction, before resampling)
    feat_aug_cfg = features_cfg.get("feat_aug")
    feat_aug = FeatureAugmenter(
        enabled="feat_aug" in features_cfg,
        random_state=random_state,
        **_scalars(feat_aug_cfg),
    )

    # -------------------------------------------------------------------------
    # Resampling (train fold only via imblearn)
    # -------------------------------------------------------------------------

    # resampler balances class distribution; falls back to a disabled SMOTESampler
    # when the section is absent so the pipeline step is always present
    resampler_cfg = config.get("resampler")
    resampler = (
        instantiate({**_scalars(resampler_cfg), "random_state": random_state})
        if resampler_cfg
        else SMOTESampler(enabled=False, random_state=random_state)
    )

    # -------------------------------------------------------------------------
    # Classifier
    # -------------------------------------------------------------------------

    # strip "type" before passing to constructor; list values are search dims
    clf_cfg    = config.get("clf", {})
    clf_params = {k: v for k, v in _scalars(clf_cfg).items() if k != "type"}
    clf = instantiate({
        "type": clf_cfg.get("type", "RandomForestClassifier"),
        "random_state": random_state,
        "n_jobs": -1,
        **clf_params,
    })

    # -------------------------------------------------------------------------
    # Assemble and return
    # drop_sig, replace_nan, scaler are always present and have no config params
    # -------------------------------------------------------------------------

    return ImbPipeline(steps=[
        ("raw_aug",     raw_aug),
        ("synthesizer", synthesizer),
        ("mfcc",        mfcc),
        ("timedomain",  timedomain),
        ("zcr",         zcr),
        ("spectral",    spectral),
        ("f0deriv",     f0deriv),
        ("nonlinear",   nonlinear),
        ("vocaltract",  vocaltract),
        ("opensmile",   opensmile),
        ("drop_sig",    DropSignalColumn()),
        ("replace_nan", ReplaceNaN()),
        ("scaler",      StandardScaler()),
        ("selector",    selector),
        ("feat_aug",    feat_aug),
        ("resampler",   resampler),
        ("clf",         clf),
    ])
