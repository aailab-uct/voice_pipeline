from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from kan_sklearn import KANClassifier

config = {
    "experiment_name": "default",
    "experiment_description": "KNN vs KAN + augmentation grid",

    # ========== GRID SEARCH ==========
    # list of dicts = two branches (KNN / KAN) with their own params
    "param_grid": [
        {
            # --- classifier branch: KNN ---
            "clf": [KNeighborsClassifier()],
            "clf__n_neighbors": [3],

            # --- augmentation/grid params shared ---
            "raw_aug__enabled": [True],     # True, False
            "feat_aug__enabled": [True],    # True, False
            "feat_aug__std_factor": [0.01], # 0.01, 0.02, 0.05
        },
        {
            # --- classifier branch: KAN ---
            "clf": [KANClassifier(seed=42, device="cpu")],
            "clf__hidden": [8],
            "clf__grid": [5],
            "clf__k": [3],
            "clf__steps": [100],
            "clf__lr": [1e-3],

            # --- augmentation/grid params shared ---
            "raw_aug__enabled": [True],      # True, False
            "feat_aug__enabled": [True],     # True, False
            "feat_aug__std_factor": [0.02],  # 0.01, 0.02, 0.05
        },
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
        "feat_aug__random_state": 42,

        # resampler
        "resampler__enabled": True,
        "resampler__random_state": 42,
        "resampler__artificial_only": True,
        "resampler__sampling_strategy": {0: 1000, 1: 1000},     # set number of samples in each class, "auto"

    },

    # ========== CV / scoring ==========
    "cv": StratifiedGroupKFold(n_splits=5),
    "scoring": "balanced_accuracy",

    # ========== DATASET BUILDING ==========
    "excluded_pathologies": {"Sängerstimme", "Gesangsstimme"},
    "include_age": True,
    "include_sex": True,
    "allowed_vowels": {"a"},
    "allowed_intonations": {"n"},
    "selected_datasets": {"SVD", "VOICED"},
    "selected_fs": 50000,
}
