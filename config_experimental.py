"""Dataset origin classification (minimal config)."""

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

sampling_rate = 48000

config = {
    "task": "dataset_origin",   # pathology   dataset_origin
    "experiment_name": "domain_shift_experiment_alpha",
    "experiment_description": (
        "Grid search over audio preprocessing methods (spectral_subtraction, wiener, rir_normalization, none) "
        "combined with extended RF hyperparameters."
    ),

    "param_grid": [
        {
            # --- Audio preprocessing ---
            # "audio_prep__enabled": [True,False],
            "audio_prep__use_rir_normalization": [True, False],
            "audio_prep__use_wiener": [True, False],
            "audio_prep__use_spectral_subtraction": [True, False],

            # --- Silenci trim ---
            "trim_silence__top_db": [15.0, 20.0],

            # --- Features preprocessing ---
            # "feat_norm__enabled": [True, False],
            "feat_norm__use_rms": [True, False],

            # --- Random Forest ---
            "clf": [RandomForestClassifier(random_state=42, n_jobs=-1)],
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 20, 40],
            "clf__min_samples_leaf": [1, 3],
            "clf__max_features": ["sqrt", "log2"],
            "clf__min_samples_split": [2, 5],
            "clf__class_weight": [None, "balanced"],
        },
    ],

    "cv": StratifiedGroupKFold(n_splits=5),

    "excluded_pathologies": {
        "Stuttering",
        "Singing voice",
        "Dysodia",
        "Sigmatism",
        "Non-fluency-Syndrom",
        "Mutatio",
    },

    "include_age": False,
    "include_sex": True,
    "allowed_vowels": {"a"},
    "allowed_intonations": {"n"},
    "trim_silence__trim_start": True,

    # musí být víc než 1 dataset
    "selected_datasets": {"SVD", "AVFAD", "VOICED", "PVQD"},
    "selected_fs": sampling_rate,
}

if __name__ == "__main__":
    raise NotImplementedError("This is a configuration file. It is not meant to be run directly.")