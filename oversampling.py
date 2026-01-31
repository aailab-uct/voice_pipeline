"""Module for oversampling transformers."""

import numpy as np
from imblearn.base import BaseSampler
from imblearn.over_sampling._smote.base import BaseSMOTE
from sklearn.utils._param_validation import Interval, StrOptions
from numbers import Real, Integral
from src.nn import VAEModel


def _compute_number_per_class(
    classes: np.ndarray, sampling_strategy: list | int | float | dict | str = "auto"
) -> dict[int, int]:
    """
    Compute desired number of samples per class based on sampling_strategy.

    :param classes: Array of class labels
    :type classes: np.ndarray
    :param sampling_strategy: Sampling strategy to determine desired counts
        - "auto": resample all classes to the maximum class count
        - dict|list: specific counts or multipliers per class
        - int: specific count for all classes
        - float: multiply max class count by float for all classes
    :type sampling_strategy: list | int | float | dict | str
    :return: Desired number of samples per class as a dict {class_label: count}
    :rtype: dict[int, int]
    """
    unique, counts = np.unique(classes, return_counts=True)  # get class distribution
    old_counts = dict(zip(unique, counts))  # store as dict

    if sampling_strategy == "auto":  # balance all classes to the max count
        max_count = max(counts)
        new_counts = {cls: max_count for cls in classes}
    elif isinstance(sampling_strategy, dict) or isinstance(sampling_strategy, list):  # specific counts per class
        values = sampling_strategy.values() if isinstance(sampling_strategy, dict) else sampling_strategy
        if all(isinstance(v, int) for v in values):  # specific counts
            new_counts = {cls: sampling_strategy[cls] for cls in classes}
        elif all(isinstance(v, float) for v in values):  # specific multipliers for each class
            new_counts = {cls: int(old_counts[cls] * sampling_strategy[cls]) for cls in classes}
        else:
            raise ValueError("sampling_strategy values must be all int or all float")
    elif isinstance(sampling_strategy, int):  # specific count for all classes
        new_counts = {cls: sampling_strategy for cls in classes}
    elif isinstance(sampling_strategy, float):  #  multiply max class count by float for all classes
        new_counts = {cls: int(max(counts) * sampling_strategy) for cls in classes}
    else:
        raise ValueError("String sampling_strategy not supported")

    return new_counts


class TunableBaseSampler(BaseSampler):
    """Base class for tunable samplers with enable/disable functionality."""

    _sampling_type = "over-sampling"

    def __init__(
        self,
        enabled: bool,
        additional_group_classes: list | int | None,
        random_state: None | int,
        **kwargs,
    ) -> None:
        """
        :param enabled: Whether to enable this sampler
        :type enabled: bool
        :param additional_group_classes: Columns in X to use as additional group identifiers
        :type additional_group_classes: list | int | None
        :param random_state: Random seed for reproducibility
        :type random_state: None | int
        """
        del kwargs  # unused
        super().__init__()
        self.random_state = random_state
        self.enabled = enabled
        self.rng = np.random.default_rng(random_state)

        if additional_group_classes is None:
            self.additional_group_classes = []
        elif isinstance(additional_group_classes, list):
            self.additional_group_classes = additional_group_classes
        elif isinstance(additional_group_classes, int):
            self.additional_group_classes = [additional_group_classes]
        else:
            raise ValueError("additional_group_classes must be None or dict")

    def _fit_resample(self, X: np.ndarray, y: np.ndarray, **params) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
        if not self.enabled:
            return X, y

        classes = y
        # incorporate additional group classes into class definition (to ensure stratification)
        for col in self.additional_group_classes:
            X_col = X[:, col]
            unique_values = np.unique(X_col)  # get unique values in the column
            class_offset = np.max(classes) + 1  # used to offset class labels
            map_unique_to_class = {
                val: idx * class_offset for idx, val in enumerate(unique_values)
            }  # mapping from unique value to new class label
            # this ensures that even float classes are handled as separate classes
            mapped_classes = np.array(
                [map_unique_to_class[val] for val in X_col]
            )  # map original values to new class labels
            classes = classes + mapped_classes  # combine original classes with new mapped classes
        X_new, y_new = self._resample(X, y, classes, **params)
        return X_new, y_new

    def _resample(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray, **params) -> tuple[np.ndarray, np.ndarray]:
        """To be implemented in subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")


class RandomCopySampler(TunableBaseSampler):
    """Custom oversampler that randomly duplicates samples to match desired output.

    All samples from the resampled class have aproximately equal representation in the output - count of each sample
    differs by at most 1.
    """

    # This is needed for sklearn compatibility - defines valid parameters and their types
    _parameter_constraints: dict = {
        "enabled": ["boolean"],
        "sampling_strategy": [
            "array-like",
            Interval(Real, None, None, closed="neither"),  # type: ignore[type-var]
            StrOptions({"auto"}),  # type: ignore[type-var]
        ],
        "additional_group_classes": [
            "array-like",
            Interval(Integral, 0, None, closed="left"),  # type: ignore[type-var]
            None,
        ],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        enabled: bool = True,
        sampling_strategy: list | int | float | dict | str = "auto",
        additional_group_classes: list | int | None = None,
        random_state: None | int = None,
        **kwargs,
    ) -> None:
        """
        Custom oversampler that randomly duplicates samples to match desired output.

        All samples from the resampled class have aproximately equal representation in the output - count of each sample
        differs by at most 1.

        :param enabled: Whether to enable this sampler
        :type enabled: bool
        :param sampling_strategy: What sampling strategy to use

            - "auto": resample all classes to the maximum class count
            - dict|list: specific counts or multipliers per class
            - int: specific count for all classes
            - float: multiply max class count by float for all classes
        :type sampling_strategy: list | int | float | dict | str
        :param additional_group_classes: Columns in X to use as additional group identifiers
        :type additional_group_classes: list | int | None
        :param random_state: Random seed for reproducibility
        :type random_state: None | int
        :raises: ValueError
        """
        del kwargs  # unused
        super().__init__(
            enabled=enabled,
            additional_group_classes=additional_group_classes,
            random_state=random_state,
        )
        self.sampling_strategy = sampling_strategy  # this will be processed in _resample

    def _resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray,
        **params,
    ) -> tuple[np.ndarray, np.ndarray]:
        del params  # unused

        new_counts = _compute_number_per_class(classes)
        X_new = []
        y_new = []

        # Oversample each class to the desired count
        for cls, target in new_counts.items():
            cls_mask = classes == cls
            X_cls = X[cls_mask]
            y_cls = y[cls_mask]
            current_number = X_cls.shape[0]

            multiples = target // current_number
            remainder = target % current_number
            for _ in range(multiples):  # add full copies
                X_new.append(X_cls)
                y_new.append(y_cls)
            samples = self.rng.choice(current_number, size=remainder, replace=False)  # add remaining samples
            X_new.append(X_cls[samples])
            y_new.append(y_cls[samples])
        X_res = np.vstack(X_new)
        y_res = np.hstack(y_new)
        # Shuffle the resampled dataset
        perm = self.rng.permutation(X_res.shape[0])
        return X_res[perm], y_res[perm]


class SMOTESampler(TunableBaseSampler, BaseSMOTE):  # type: ignore[misc]
    """Custom SMOTE oversampler that can also generate only artificial samples without copying existing ones."""

    # Note that the code is mostly inherited from imblearn's BaseSMOTE class
    # Complexity arises from integrating with imblearn's architecture and adding custom functionality which is not
    # straightforward.
    _sampling_type = "over-sampling"

    def __init__(
        self,
        enabled: bool = True,
        sampling_strategy: list | int | float | dict | str = "auto",
        additional_group_classes: list | int | None = None,
        random_state: None | int = None,
        artificial_only: bool = False,
        **kwargs,
    ):
        super().__init__(
            enabled=enabled,
            additional_group_classes=additional_group_classes,
            random_state=random_state,
            k_neighbors=kwargs.get("k_neighbors", 5),
        )

        self.sampling_strategy = sampling_strategy
        self.artificial_only = artificial_only

    def _fit_resample(self, X: np.ndarray, y: np.ndarray, **params):  # type: ignore
        self._validate_estimator()
        return super()._fit_resample(X, y, **params)

    def _resample(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray, **params) -> tuple[np.ndarray, np.ndarray]:
        del params  # not used

        new_counts = _compute_number_per_class(classes, self.sampling_strategy)  # desired counts per class

        # copy inputs to output if not artificial only
        X_new = [] if self.artificial_only else [X.copy()]
        y_new = [] if self.artificial_only else [y.copy()]

        # Oversample each class to the desired count
        for cls, target in new_counts.items():
            n_samples = target - (0 if self.artificial_only else np.sum(classes == cls))
            if n_samples <= 0:
                continue
            cls_mask = classes == cls
            X_cls = X[cls_mask]
            y_cls = y[cls_mask]

            self.nn_k_.fit(X_cls)  # fit nearest neighbors and retrieve neighbors so we can generate samples
            nns = self.nn_k_.kneighbors(X_cls, return_distance=False)[:, 1:]  # type: ignore[attr-defined]

            X_syn, y_syn = self._make_samples(
                X_cls,
                y.dtype,
                y_cls[0],  # target label used to label synthetic samples
                X_cls,
                nns,
                n_samples,
                1.0,
            )  # generate synthetic samples
            X_new.append(X_syn)
            y_new.append(y_syn)
        X_res = np.vstack(X_new)
        y_res = np.hstack(y_new)
        # Shuffle the resampled dataset
        perm = self.rng.permutation(X_res.shape[0])
        return X_res[perm], y_res[perm]


class VAESampler(TunableBaseSampler):
    """Resampler using Variational Autoencoder (VAE) to generate synthetic samples."""

    def __init__(
        self,
        enabled: bool = True,
        additional_group_classes: list | int | None = None,
        random_state: None | int = None,
        artificial_only: bool = False,
        architecture: dict | None = None,
        epochs: int = 50,
        **kwargs,
    ) -> None:
        """
        :param enabled: Whether to enable this sampler
        :type enabled: bool
        :param additional_group_classes: Columns in X to use as additional group identifiers
        :type additional_group_classes: list | int | None
        :param random_state: Random seed for reproducibility
        :type random_state: None | int
        """
        del kwargs  # unused
        super().__init__(
            enabled=enabled,
            additional_group_classes=additional_group_classes,
            random_state=random_state,
        )

        self.vae_model = VAEModel(
            latent_dim=architecture.get("latent_dim", 10),
            hidden_layers=architecture.get("hidden_layers", [64, 32]),
            random_state=random_state,
            epochs=epochs,
        )
        self.artificial_only = artificial_only

    def _resample(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray, **params) -> tuple[np.ndarray, np.ndarray]:
        del params  # unused
        self.vae_model.train(X, y, classes)

        new_counts = _compute_number_per_class(classes)

        # copy inputs to output if not artificial only
        X_new = [] if self.artificial_only else [X.copy()]
        y_new = [] if self.artificial_only else [y.copy()]
        # Oversample each class to the desired count
        for cls, target in new_counts.items():
            n_samples = target - (0 if self.artificial_only else np.sum(classes == cls))
            if n_samples <= 0:
                continue
            X_syn, y_syn = self.vae_model.generate_samples(n_samples=n_samples, cls=cls)

            X_new.append(X_syn)
            y_new.append(y_syn)
        X_res = np.vstack(X_new)
        y_res = np.hstack(y_new)
        # Shuffle the resampled dataset
        perm = self.rng.permutation(X_res.shape[0])
        return X_res[perm], y_res[perm]
