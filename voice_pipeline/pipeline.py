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
    """Return only scalar (non-list) values from a config section."""
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
    dataset_cfg  = config.get("dataset", {})
    features_cfg = config.get("features", {})
    fs           = dataset_cfg.get("fs", 48000)
    include_sex  = dataset_cfg.get("include_sex", True)
    sex_col      = 1 if include_sex else None

    # disable feature cache when waveforms change between fit and transform
    raw_aug_cfg = config.get("raw_aug")
    synth_cfg   = config.get("synthesizer")
    raw_aug_active  = raw_aug_cfg is not None
    synth_active    = synth_cfg is not None and _scalars(synth_cfg).get("n_samples", 0) > 0
    disable_cache   = raw_aug_active or synth_active

    # --- waveform augmentation ---
    raw_aug = RawAugmenter(
        enabled=raw_aug_active,
        sr=fs,
        random_state=random_state,
        **_scalars(raw_aug_cfg),
    )

    synthesizer = WaveformSynthesizer(
        enabled=synth_active,
        sr=fs,
        random_state=random_state,
        **_scalars(synth_cfg),
    )

    # --- feature extractors ---
    mfcc = MFCCExtractor(
        enabled="mfcc" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("mfcc")),
    )

    timedomain = TimeDomainFeaturesExtractor(
        enabled="timedomain" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        sex_col=sex_col,
        **_scalars(features_cfg.get("timedomain")),
    )

    zcr = ZCRExtractor(
        enabled="zcr" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("zcr")),
    )

    spectral = SpectralFeaturesExtractor(
        enabled="spectral" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("spectral")),
    )

    f0deriv = F0DerivativesExtractor(
        enabled="f0deriv" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("f0deriv")),
    )

    nonlinear = NonlinearExtractor(
        enabled="nonlinear" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("nonlinear")),
    )

    vocaltract = VocalTractExtractor(
        enabled="vocaltract" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("vocaltract")),
    )

    opensmile = OpenSmileExtractor(
        enabled="opensmile" in features_cfg,
        sr=fs,
        cache_disabled=disable_cache,
        **_scalars(features_cfg.get("opensmile")),
    )

    # --- feature selection ---
    selector_cfg = features_cfg.get("selector")
    selector = instantiate(_scalars(selector_cfg)) if selector_cfg else "passthrough"

    # --- feature augmentation ---
    feat_aug_cfg = features_cfg.get("feat_aug")
    feat_aug = FeatureAugmenter(
        enabled="feat_aug" in features_cfg,
        random_state=random_state,
        **_scalars(feat_aug_cfg),
    )

    # --- resampling ---
    resampler_cfg = config.get("resampler")
    resampler = (
        instantiate({**_scalars(resampler_cfg), "random_state": random_state})
        if resampler_cfg
        else SMOTESampler(enabled=False, random_state=random_state)
    )

    # --- classifier ---
    clf_cfg = config.get("clf", {})
    clf_params = {k: v for k, v in _scalars(clf_cfg).items() if k != "type"}
    clf = instantiate({
        "type": clf_cfg.get("type", "RandomForestClassifier"),
        "random_state": random_state,
        "n_jobs": -1,
        **clf_params,
    })

    return ImbPipeline(steps=[
        ("raw_aug",    raw_aug),
        ("synthesizer", synthesizer),
        ("mfcc",       mfcc),
        ("timedomain", timedomain),
        ("zcr",        zcr),
        ("spectral",   spectral),
        ("f0deriv",    f0deriv),
        ("nonlinear",  nonlinear),
        ("vocaltract", vocaltract),
        ("opensmile",  opensmile),
        ("drop_sig",   DropSignalColumn()),
        ("replace_nan", ReplaceNaN()),
        ("scaler",     StandardScaler()),
        ("selector",   selector),
        ("feat_aug",   feat_aug),
        ("resampler",  resampler),
        ("clf",        clf),
    ])
