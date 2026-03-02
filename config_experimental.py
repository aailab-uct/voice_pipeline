"""Dataset origin classification (minimal config)."""

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

sampling_rate = 48000

config = {
    "task": "dataset_origin",
    "experiment_name": "dataset_origin_rf",
    "experiment_description": "classify which dataset a recording comes from",

    "param_grid": [
        {
            "clf": [RandomForestClassifier(random_state=42, n_jobs=-1)],
            "clf__n_estimators": [300],
            "clf__max_depth": [None],
            "clf__min_samples_leaf": [1],
            "clf__max_features": ["sqrt"],
            "clf__min_samples_split": [2],
            "clf__class_weight": [None],
        },
    ],

    "cv": StratifiedGroupKFold(n_splits=10),

    "excluded_pathologies": {
        "Stuttering",
        "Singing voice",
        "Dysodia",
        "Sigmatism",
        "Non-fluency-Syndrom",
        "Mutatio"
    },

    "include_age": False,
    "include_sex": True,
    "allowed_vowels": {"a"},
    "allowed_intonations": {"n"},

    # musí být víc než 1 dataset
    "selected_datasets": {"SVD", "AVFAD"},
    "selected_fs": sampling_rate,
}

if __name__ == "__main__":
    raise NotImplementedError("This is a configuration file. It is not meant to be run directly.")