from sklearn.model_selection import StratifiedGroupKFold

config = {
    "experiment_name": "default",
    "experiment_description": "Default experiment configuration",
    # model selection parameters
    "param_grid": {
        "raw_aug__enabled": [False],
        "feat_aug__enabled": [True, False],
        "feat_aug__std_factor": [0.01, 0.02, 0.05],
        "clf__n_neighbors": [3, 5, 7],
    },
    "pipeline": {
        # TODO
    },
    "cv": StratifiedGroupKFold(n_splits=5),
    "scoring": "balanced_accuracy",
    # dataset building parameters
    "excluded_pathologies": {"Sängerstimme", "Gesangsstimme"},
    "include_age": True,
    "include_sex": True,
    "allowed_vowels": {"a"},
    "allowed_intonations": {"n"},
    "selected_datasets": {"SVD"},
    "selected_fs": 50000,
}
