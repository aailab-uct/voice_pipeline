"""Sample configuration file"""

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from custom_classifiers import KANClassifier

# experiment with seed for SMOTE
resampler_seeds = sorted(set(range(0, 105, 1)) | set(range(100, 1100, 20)))

config = {
    "experiment_name": "svd_basic_seed",
    "experiment_description": "basic run with not only artificial samples for comparison for seed experiments",
    # ========== GRID SEARCH ==========
    # list of dicts for GridSearchCV's param_grid - each dict is a separate branch
    "param_grid": [
        {
            # --- classifier branch: KNN ---
            "clf": [KNeighborsClassifier()],
            "clf__n_neighbors": [3],
            # --- augmentation/grid params shared ---
            "raw_aug__enabled": [False],  # True, False
            "feat_aug__enabled": [True],  # True, False
            "feat_aug__std_factor": [0.01],  # 0.01, 0.02, 0.05
            "resampler__random_state": [40],
            # "resampler__random_state": [0, 2, 6, 10, 20, 30, 40, 50, 60, 100, 500, 1000]
            # "resampler__random_state": resampler_seeds,
        },
        # {
        #     # --- classifier branch: KAN ---
        #     "clf": [KANClassifier(seed=42, device="cpu")],
        #     "clf__hidden": [8],
        #     "clf__grid": [5],
        #     "clf__k": [3],
        #     "clf__steps": [100],
        #     "clf__lr": [1e-3],
        #     "clf__seed": [40],
        #     # --- augmentation/grid params shared ---
        #     "raw_aug__enabled": [False],  # True, False
        #     "feat_aug__enabled": [True],  # True, False
        #     "feat_aug__std_factor": [0.02],  # 0.01, 0.02, 0.05
        #     # "resampler__random_state": [0, 2, 6, 10, 20, 30, 40, 50, 60, 100, 500, 1000, 10000]
        #     "resampler__random_state": resampler_seeds,
        # },
    ],
    # ========== FIXED PIPELINE SETTINGS ==========
    # IMPORTANT: single values only (no lists). Lists belong in param_grid.
    "pipeline": {
        # raw augmenter defaults (enabled is controlled by param_grid)
        "raw_aug__noise_std": 0.005,
        "raw_aug__random_state": 42,
        # MFCC extractor
        "mfcc__sr": 50000,
        "mfcc__n_mfcc": 20,
        # other feature extractors
        "timedomain__enabled": True,
        "timedomain__sr": 50000,
        "zcr__enabled": True,
        "zcr__sr": 50000,
        "spectral__enabled": True,
        "spectral__sr": 50000,
        # feature augmenter defaults (enabled + std_factor are controlled by param_grid)
        "feat_aug__random_state": 40,
        # resampler
        "resampler__enabled": True,
        # "resampler__random_state": 42,
        "resampler__artificial_only": False,
        "resampler__sampling_strategy": {0: 1000, 1: 1000},  # set number of samples in each class, "auto"
        # selector setting
        "selector__k": 20,
    },
    # ========== CV / scoring ==========
    "cv": StratifiedGroupKFold(n_splits=5),
    "scoring": "balanced_accuracy",
    # ========== DATASET BUILDING ==========
    "excluded_pathologies": {
        "Sängerstimme",
        "Gesangsstimme",
        "Sigmatismus",
        "Balbuties",
        "Dysodie",
        "Poltersyndrom",
        "Non-fluency-Syndrom",
    },
    "include_age": False,
    "include_sex": True,
    "allowed_vowels": {"a"},
    "allowed_intonations": {"n"},
    "selected_datasets": {"SVD"},  # "VOICED"
    "selected_fs": 50000,
}

if __name__ == "__main__":
    raise NotImplementedError("This is a configuration file. It is not meant to be run directly.")
