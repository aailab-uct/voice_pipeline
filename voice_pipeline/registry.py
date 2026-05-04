"""Registry of instantiable classes for YAML-based configuration."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFECV,
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.model_selection import StratifiedGroupKFold

from voice_pipeline.classifiers import KANClassifier
from voice_pipeline.sampling import SMOTESampler, VAESampler

REGISTRY: dict[str, type] = {
    # cross-validation
    "StratifiedGroupKFold": StratifiedGroupKFold,
    # classifiers
    "RandomForestClassifier": RandomForestClassifier,
    "KANClassifier": KANClassifier,
    # resamplers
    "SMOTESampler": SMOTESampler,
    "VAESampler": VAESampler,
    # feature selectors
    "SelectKBest": SelectKBest,
    "SelectFromModel": SelectFromModel,
    "RFECV": RFECV,
    # score functions
    "mutual_info_classif": mutual_info_classif,
    "f_classif": f_classif,
}


def instantiate(spec: dict) -> object:
    """Instantiate a registered class from a config spec dict.

    Parameters
    ----------
    spec : dict
        Must contain a ``"type"`` key matching an entry in :data:`REGISTRY`.
        Remaining keys are passed as keyword arguments to the constructor.
        If a value is itself a dict with a ``"type"`` key, it is instantiated
        recursively first.

    Returns
    -------
    object
        Instance of the class identified by ``spec["type"]``.

    Raises
    ------
    KeyError
        If ``"type"`` is missing or not found in :data:`REGISTRY`.
    """
    # avoid mutating the caller's dict
    spec = spec.copy()
    # look up the class and remove "type" so it isn't passed to the constructor
    cls = REGISTRY[spec.pop("type")]
    resolved = {
        # recursively instantiate nested specs, pass everything else as-is
        k: instantiate(v) if isinstance(v, dict) and "type" in v else v
        for k, v in spec.items()
    }
    # construct and return the instance
    return cls(**resolved)
