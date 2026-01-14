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
        # TODO - in progress
        # raw_aug
        "raw_aug__enabled": [True],
        "raw_aug__noise_std": [0.001, 0.002, 0.003],
        "raw_aug__random_state": [None, 0, 1],

        # mfcc
        "feat_aug__enabled": [True],
        "feat_aug__sr": [50000],
        "feat_aug__n_mfcc": [20],

        #timedomain
        "timedomain__enabled": [True],
        "timedomain__sr": [50000],

        # zrc
        "zcr__enabled": [True],
        "zcr__sr": [50000],

        # spectral
        "spectral__enabled": [True],
        "spectral__sr": [50000],

        # feat_aug
        "features__enabled": [True],
        "features__std_factor": [0.01, 0.02, 0.05],
        "features__random_state": [None, 0, 1],

        # resampler
        "resampler__enabled": [True],
        "resampler__random_state": [None, 0, 1],
        "resampler__artifical_only": [True],

    },
    "cv": StratifiedGroupKFold(n_splits=5),
    "scoring": "balanced_accuracy",
    # dataset building parameters
    "excluded_pathologies": {"Sängerstimme", "Gesangsstimme"},
    "include_age": True,
    "include_sex": True,
    "allowed_vowels": {"a"},
    "allowed_intonations": {"n"},
    "selected_datasets": {"SVD", "VOICED"},
    "selected_fs": 50000,
}
