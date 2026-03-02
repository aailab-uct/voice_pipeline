"""Experiment 3: No SMOTE, class_weight='balanced', selection by balanced accuracy (UAR)"""

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

from custom_classifiers import KANClassifier

sampling_rate = 48000
# resampler_seeds = sorted(set(range(0, 105, 1)) | set(range(100, 1100, 20)))

config = {
    "experiment_name": "afthermath_only_svd",
    "experiment_description": "first try afther repair of nsp to wav conversion error",

    "param_grid": [
        {
            "clf": [RandomForestClassifier(random_state=42, n_jobs=-1)],
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None],
            "clf__class_weight": [None, "balanced"]
        },
    ],

    "pipeline": {
        "raw_aug__noise_std": 0.005,
        "raw_aug__random_state": 42,
        "raw_aug__enabled": False,
        "feat_aug__enabled": False,
        "feat_aug__std_factor": 0.01,

        "mfcc__sr": sampling_rate,
        "mfcc__n_mfcc": 20,
        "mfcc__enabled": False,

        "timedomain__enabled": True,
        "timedomain__sr": sampling_rate,
        "timedomain__use_zcr": False,
        "timedomain__use_entropy": False,
        "timedomain__use_skewness": False,
        "timedomain__use_diff_pitch": False,

        "zcr__enabled": False,
        "zcr__sr": sampling_rate,

        "spectral__enabled": True,
        "spectral__sr": sampling_rate,

        "feat_aug__random_state": 42,

        # SMOTE OFF
        "resampler__enabled": True,
        "resampler__artificial_only": False,
        "resampler__sampling_strategy": "auto",
        "resampler__random_state": 42,

        "selector__enabled": False,
        "selector__k": 20,
    },

    "cv": StratifiedGroupKFold(n_splits=10),
    "scoring_mode": "multimetric",
    "refit_metric": "uar",  # vybírá podle balanced accuracy

    "excluded_pathologies": {
        "Stuttering",
        "Singing voice",
        "Dysodia",
        "Sigmatism",
        "Non-fluency-Syndrom",
        "Mutatio"
    },
    "include_age": True,
    "include_sex": True,
    "allowed_vowels": {"a"},
    "allowed_intonations": {"n"},
    "selected_datasets": {"SVD"},   # "SVD" "AVFAD"
    "selected_fs": sampling_rate,
}

if __name__ == "__main__":
    raise NotImplementedError("This is a configuration file. It is not meant to be run directly.")
