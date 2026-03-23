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
    "experiment_name": "test_of_new_preprocesing",
    "experiment_description": "experiment",

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    "selected_datasets": {"SVD", "AVFAD"},       # "SVD" | "AVFAD" "VOICED" "PVQD"
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
    "n_repeats": 5,          # 0 = vypnuto, jinak počet opakování repeated CV
    "plot_uar_per_pathology": True,
    "plot_feature_histograms": True,

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    "pipeline": {
        # --- RawAugmenter -----------------------------------------------
        # Modifies every existing waveform (train AND test).
        # Shape-preserving — does not add new rows.
        #
        # Techniques (all default False except noise + gain):
        #   noise         additive Gaussian white noise
        #   gain          random amplitude scaling
        #   polarity      flip signal sign (50 % probability)
        #   time_crop     random crop + zero-pad to original length
        #   pitch_shift   semitone shift, duration unchanged  [needs librosa]
        #   time_stretch  speed change, pitch unchanged        [needs librosa]
        #   reverb        convolve with synthetic room impulse response
        #   codec         μ-law encode→decode (telephone codec simulation)
        "raw_aug__enabled": False,
        "raw_aug__sr": sampling_rate,
        "raw_aug__noise": True,
        "raw_aug__gain": True,
        "raw_aug__polarity": False,
        "raw_aug__time_crop": False,
        "raw_aug__pitch_shift": False,      # slow — enable only when needed
        "raw_aug__time_stretch": False,     # slow — enable only when needed
        "raw_aug__reverb": False,
        "raw_aug__codec": False,
        "raw_aug__noise_std": 0.005,
        "raw_aug__gain_min": 0.8,
        "raw_aug__gain_max": 1.2,
        "raw_aug__crop_min_frac": 0.8,
        "raw_aug__pitch_semitones_range": (-2.0, 2.0),
        "raw_aug__stretch_rate_range": (0.9, 1.1),
        "raw_aug__reverb_decay": 10.0,
        "raw_aug__random_state": 42,

        # --- WaveformSynthesizer ----------------------------------------
        # Generates n_samples NEW waveform rows from existing ones.
        # Runs ONLY on the train fold (imblearn BaseSampler) — synthetic
        # audio NEVER enters the test fold or evaluation metrics.
        # Must stay before feature extractors so synthetic waveforms pass
        # through the same feature pipeline as real recordings.
        #
        # All synthesizer__* keys are directly accessible in param_grid,
        # e.g.:  "synthesizer__n_samples": [100, 300]
        #        "synthesizer__n_augmentations": [1, 2, 3]
        #        "synthesizer__noise": [True, False]
        #
        # n_augmentations: how many techniques to randomly pick per sample.
        #   None  = apply all enabled techniques to every sample (default)
        #   1     = pick 1 random technique per sample
        #   2     = pick 2 random techniques per sample   <-- RandAugment style
        "synthesizer__enabled": False,
        # "synthesizer__n_samples": 0,        # total new rows across all classes
        "synthesizer__sr": sampling_rate,
        # "synthesizer__n_augmentations": None,
        "synthesizer__noise": True,
        "synthesizer__gain": True,
        "synthesizer__polarity": True,
        "synthesizer__time_crop": True,
        "synthesizer__pitch_shift": True,  # slow — enable only when needed
        "synthesizer__time_stretch": True, # slow — enable only when needed
        "synthesizer__reverb": True,
        "synthesizer__codec": True,
        "synthesizer__noise_std": 0.005,
        "synthesizer__gain_min": 0.8,
        "synthesizer__gain_max": 1.2,
        "synthesizer__random_state": 42,

        # --- Feature extractors -----------------------------------------
        # Note: cache_disabled is set automatically by build_pipeline when
        # raw_aug or synthesizer is enabled — no need to set it here.
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

        # --- Feature augmentation (after extraction, before resampling) --
        "feat_aug__enabled": False,
        "feat_aug__std_factor": 0.01,
        "feat_aug__random_state": 42,

        # --- Feature selection ------------------------------------------
        "selector__enabled": False,
        "selector__instance": SelectKBest(score_func=mutual_info_classif, k=5),
        # "selector__instance": SelectFromModel(
        #     estimator=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
        #     threshold="median",
        # ),
        # "selector__instance": RFECV(
        #     estimator=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
        #     step=0.1, cv=5, scoring="balanced_accuracy", n_jobs=-1,
        # ),

        # --- Resampling (train fold only) --------------------------------
        "resampler": SMOTESampler(enabled=False, random_state=42, artificial_only=False, sampling_strategy="auto"),
        # "resampler": VAESampler(enabled=True, random_state=42, artificial_only=False, sampling_strategy="auto"),

        # --- Classifier --------------------------------------------------
        "clf__instance": RandomForestClassifier(random_state=42, n_jobs=-1),
    },

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------
    "param_grid": [
        {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None],
            "clf__class_weight": [None, "balanced"],

            # "synthesizer__n_samples": [200, 300],
            # "synthesizer__n_augmentations": [1, 2, 3],
            # "synthesizer__noise": [True],
        },
    ],
}

if __name__ == "__main__":
    raise NotImplementedError("This is a configuration file. It is not meant to be run directly.")