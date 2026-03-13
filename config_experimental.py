"""Experiment 3: No SMOTE, class_weight='balanced', selection by balanced accuracy (UAR)"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel, RFECV
from sklearn.model_selection import StratifiedGroupKFold
from oversampling import SMOTESampler, VAESampler

from custom_classifiers import KANClassifier

sampling_rate = 48000

config = {
    # ------------------------------------------------------------------
    # Experiment metadata
    # ------------------------------------------------------------------
    "experiment_name": "SMOTE_experiment_without_smote",
    "experiment_description": "article experiment",

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    "selected_datasets": {"SVD"},       # "SVD" | "AVFAD" "VOICED" "PVQD"
    "selected_fs": sampling_rate,
    "allowed_vowels": {"a"},
    "allowed_intonations": {"n"},
    "include_age": True,
    "include_sex": True,
    "excluded_pathologies": {
        "Sängerstimme",
        "Poltersyndrom",
        "Stuttering",
        "Singing voice",
        "Singing voice ; Vocal fold nodules",
        "Dysphonia ; Mutatio",
        "Dysodia",
        "Sigmatism",
        "Non-fluency-Syndrom",
        "Mutatio",
        "Stuttering ; Laryngitis",
        "Dysodia ; Hyperfunctional dysphonia"
    },

    # ------------------------------------------------------------------
    # Cross-validation & scoring
    # ------------------------------------------------------------------
    "cv": StratifiedGroupKFold(n_splits=10),
    "scoring_mode": "multimetric",
    "refit_metric": "uar",
    "n_repeats": 1000,          # 0 = vypnuto, jinak počet opakování repeated CV
    "plot_uar_per_pathology": True,
    "plot_feature_histograms": True,

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    "pipeline": {
        # Signal augmentation (before feature extraction)
        "raw_aug__enabled": False,
        "raw_aug__noise_std": 0.005,
        "raw_aug__random_state": 42,

        # Feature extractors
        "mfcc__enabled": False,
        "mfcc__sr": sampling_rate,
        "mfcc__n_mfcc": 20,

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

        "opensmile__enabled": False,
        "opensmile__sr": sampling_rate,
        "opensmile__feature_set": "GeMAPS",       # "GeMAPS" | "ComParE_2016"
        "opensmile__feature_level": "functionals",

        # Feature augmentation (after extraction, before resampling)
        "feat_aug__enabled": False,
        "feat_aug__std_factor": 0.01,
        "feat_aug__random_state": 42,

        # Feature selection
        "selector__enabled": False,
        "selector__instance": SelectKBest(score_func=mutual_info_classif, k=20),
        # "selector__instance": SelectFromModel(
        #     estimator=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
        #     threshold="median",
        # ),
        # "selector__instance": RFECV(
        #     estimator=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
        #     step=0.1, cv=5, scoring="balanced_accuracy", n_jobs=-1,
        # ),

        # Resampling
        "resampler": SMOTESampler(enabled=False, random_state=42, artificial_only=False, sampling_strategy="auto"),
        # "resampler": VAESampler(enabled=True, random_state=42, artificial_only=False, sampling_strategy="auto"),

        # Classifier
        "clf": RandomForestClassifier(random_state=42, n_jobs=-1),
    },

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------
    "param_grid": [
        {
            "clf__n_estimators": [200, 300, 400],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__max_features": ["sqrt", 0.3],
            "clf__class_weight": [None, "balanced"],
        },
    ],
}

if __name__ == "__main__":
    raise NotImplementedError("This is a configuration file. It is not meant to be run directly.")