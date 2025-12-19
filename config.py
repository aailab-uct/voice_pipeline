from sklearn.model_selection import StratifiedGroupKFold

config = {
    "experiment_name": "default",
    "experiment_description": "Default experiment configuration",
    # model selection parameters
    "param_grid": {
        "raw_aug__enabled": [True],
        "feat_aug__enabled": [True],
    },
    "pipeline": {
        # TODO
    },
    "cv": StratifiedGroupKFold(n_splits=10),
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
