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
from voice_pipeline.sampling import SMOTESampler
from voice_pipeline.signal_augmentation import RawAugmenter, WaveformSynthesizer
from voice_pipeline.utils import ReplaceNaN


def build_pipeline(pipeline_params: dict | None = None, random_state: int = 42) -> ImbPipeline:
    """Create the audio-to-classification pipeline.

    The raw waveform is always the first column of ``X``. The pipeline then:
    augments waveforms, extracts acoustic features, removes the waveform column,
    cleans invalid numbers, scales features, optionally selects/features augments,
    optionally resamples classes, and finally fits a classifier.

    Synthetic waveforms and feature resampling are implemented as imblearn
    samplers, so they are applied only to training folds during cross-validation.

    Parameters
    ----------
    pipeline_params : dict or None, default=None
        Sklearn-style parameter dictionary. Keys such as ``mfcc__enabled`` and
        ``clf__max_depth`` are forwarded to pipeline steps.
    random_state : int, default=42
        Seed passed to stochastic pipeline components.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Unfitted pipeline ready for ``fit`` or ``GridSearchCV``.
    """
    if pipeline_params is None:
        pipeline_params = {}

    raw_augmentation_enabled = pipeline_params.get("raw_aug__enabled", False)
    synthesis_enabled = (
        pipeline_params.get("synthesizer__enabled", False) and pipeline_params.get("synthesizer__n_samples", 0) > 0
    )
    disable_feature_cache = bool(raw_augmentation_enabled or synthesis_enabled)

    pipe = ImbPipeline(
        steps=[
            ("raw_aug", RawAugmenter(random_state=random_state)),
            ("synthesizer", WaveformSynthesizer(enabled=False, n_samples=0, random_state=random_state)),
            ("mfcc", MFCCExtractor(cache_disabled=disable_feature_cache)),
            ("timedomain", TimeDomainFeaturesExtractor(cache_disabled=disable_feature_cache)),
            ("zcr", ZCRExtractor(cache_disabled=disable_feature_cache)),
            ("spectral", SpectralFeaturesExtractor(cache_disabled=disable_feature_cache)),
            ("f0deriv", F0DerivativesExtractor(cache_disabled=disable_feature_cache)),
            ("nonlinear", NonlinearExtractor(cache_disabled=disable_feature_cache)),
            ("vocaltract", VocalTractExtractor(cache_disabled=disable_feature_cache)),
            ("opensmile", OpenSmileExtractor(cache_disabled=disable_feature_cache)),
            ("drop_sig", DropSignalColumn()),
            ("replace_nan", ReplaceNaN()),
            ("scaler", StandardScaler()),
            ("selector", pipeline_params.get("selector__instance", "passthrough")),
            ("feat_aug", FeatureAugmenter(random_state=random_state)),
            ("resampler", SMOTESampler(random_state=random_state)),
            (
                "clf",
                pipeline_params.get(
                    "clf__instance",
                    RandomForestClassifier(random_state=random_state, n_jobs=-1),
                ),
            ),
        ]
    )

    if pipeline_params:
        params = dict(pipeline_params)

        selector_enabled = params.pop("selector__enabled", True)
        params.pop("selector__instance", None)
        params.pop("clf__instance", None)

        if not selector_enabled:
            pipe.set_params(selector="passthrough")
            for key in list(params.keys()):
                if key.startswith("selector__"):
                    params.pop(key)

        pipe.set_params(**params)

    return pipe
