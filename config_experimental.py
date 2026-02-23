"""Experiment 3: No SMOTE, class_weight='balanced', selection by balanced accuracy (UAR)"""

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

from custom_classifiers import KANClassifier

sampling_rate = 48000
resampler_seeds = sorted(set(range(0, 105, 1)) | set(range(100, 1100, 20)))

config = {
    "experiment_name": "AVFAD_SVD_experiment",
    "experiment_description": "combined experiment with both datasets without oversampling, with unificated patologies across AVFAD and SVD",

    "param_grid": [
        {
            "clf": [RandomForestClassifier(random_state=42, n_jobs=-1)],
            "clf__n_estimators": [300],
            "clf__max_depth": [None],
            "clf__min_samples_split": [2],
            "clf__min_samples_leaf": [1],
            "clf__max_features": ["sqrt"],
            # "clf__class_weight": ["balanced"],  # váhy tříd

            "raw_aug__enabled": [False],
            "feat_aug__enabled": [False],
            "feat_aug__std_factor": [0.01],
            "resampler__random_state": [42],  # může zůstat, ale nebude použito
        },
    ],

    "pipeline": {
        "raw_aug__noise_std": 0.005,
        "raw_aug__random_state": 42,

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
        # "Sängerstimme",
        # "Gesangsstimme",
        # "Sigmatismus",
        # "Balbuties",
        # "Dysodie",
        # "Poltersyndrom",
        # "Balbuties, Laryngitis"
        # "Non-fluency-Syndrom",
        # "hyperkinetic dysphonia (Prolapse)",
        # "hypokinetic dysphonia (glottic insufficiency)",
        # "hyperkinetic dysphonia (Polyps)",
        # "hypokinetic dysphonia (Vocal fold paralysis)",
        # "hyperkinetic dysphonia (Reinke's edema)",
        # "hyperkinetic dysphonia (nodule)",
        # "hyperkinetic dysphonia (Nodule)",
        # "hypokinetic dysphonia (Extraglottic air leak)",
        # "hypokinetic dysphonia (Adduction deficit)",
        # "hypokinetic dysphonia (Glottic insufficiency)",
        # "hyperkinetic dysphonia (Vocal fold paralysis)",
        # "hyperkinetic dysphonia (Vocal fold nodules)",
        # "hypokinetic dysphonia (Presbiphonia)",
        # "hyperkinetic dysphonia (Adduction deficit)",
        # "hyperkinetic dysphonia (Cordite)",
        # "hypokinetic dysphonia (Spasmodic dysphonia)",
        # "hypokinetic dysphonia (Bilateral vocal fold)",
        # "hypokinetic dysphonia (Laryngitis)",
        # "hyperkinetic dysphonia (polyps)",
        # "hypokinetic dysphonia (Dysphonia by chordal groove)",
        # "hyperkinetic dysphonia (Vocal fold prolapse)",
        # "hypokinetic dysphonia (Conversion dysphonia)",
        # "hyperkinetic dysphonia  (Rigid vocal fold)",
    },
    "include_age": True,
    "include_sex": True,
    "allowed_vowels": {"a"},
    "allowed_intonations": {"n"},
    "selected_datasets": {"AVFAD", "SVD"},   # "SVD"
    "selected_fs": sampling_rate,
}

if __name__ == "__main__":
    raise NotImplementedError("This is a configuration file. It is not meant to be run directly.")
