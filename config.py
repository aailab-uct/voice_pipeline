from sklearn.model_selection import StratifiedGroupKFold

config = {
    "experiment_name": "default",
    "experiment_description": "Default experiment configuration",
    # model selection parameters
    "param_grid": {
        # Here you should define the parameters to be tuned
        # raw_aug
        "raw_aug__noise_std": [0.001, 0.002],

        # timedomain
        "timedomain__use_diff_pitch": [True, False],


        "feat_aug__enabled": [True, False],
        "feat_aug__std_factor": [0.01, 0.02],
        "clf__n_neighbors": [3, 5],
        "raw_aug__noise_std": [0.001, 0.002],

    },
    "pipeline": {
        # Here should you define the default pipeline parameters, which should not be tuned
        "raw_aug__enabled": False,
        "timedomain__enabled": True,
        "mfcc__enabled": True,
        "zcr__enabled": True,


        #timedomain
        "timedomain__enabled": True,

        # zrc
        "zcr__enabled": True,

        # spectral
        "spectral__enabled": True,

        # feat_aug
        "feat_aug__enabled": True,
        # resampler
        "resampler__enabled": True,
        "resampler__artificial_only": True,

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
    "drop_nan_f0": True,
}
