"""Module for oversampling transformers."""

import numpy as np
from imblearn.base import BaseSampler
from imblearn.over_sampling._smote.base import BaseSMOTE
from sklearn.utils._param_validation import Interval, StrOptions
from numbers import Real, Integral


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
    elif isinstance(sampling_strategy, dict) or isinstance(
        sampling_strategy, list
    ):  # specific counts per class
        values = (
            sampling_strategy.values()
            if isinstance(sampling_strategy, dict)
            else sampling_strategy
        )
        if all(type(v) == int for v in values):  # specific counts
            new_counts = {cls: sampling_strategy[cls] for cls in classes}
        elif all(
            type(v) == float for v in values
        ):  # specific multipliers for each class
            new_counts = {
                cls: int(old_counts[cls] * sampling_strategy[cls]) for cls in classes
            }
        else:
            raise ValueError("sampling_strategy values must be all int or all float")
    elif isinstance(sampling_strategy, int):  # specific count for all classes
        new_counts = {cls: sampling_strategy for cls in classes}
    elif isinstance(
        sampling_strategy, float
    ):  #  multiply max class count by float for all classes
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
    ) -> None:
        """
        :param enabled: Whether to enable this sampler
        :type enabled: bool
        :param additional_group_classes: Columns in X to use as additional group identifiers
        :type additional_group_classes: list | int | None
        :param random_state: Random seed for reproducibility
        :type random_state: None | int
        """
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

    def _fit_resample(  # type: ignore
        self, X: np.ndarray, y: np.ndarray, **params
    ) -> tuple[np.ndarray, np.ndarray]:
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
            }  # mapping from unique value to new class label - this ensures that even float classes are handled as separate classes
            mapped_classes = np.array(
                [map_unique_to_class[val] for val in X_col]
            )  # map original values to new class labels
            classes = (
                classes + mapped_classes
            )  # combine original classes with new mapped classes

        X_new, y_new = self._resample(X, y, classes, **params)
        return X_new, y_new

    def _resample(
        self, X: np.ndarray, y: np.ndarray, classes: np.ndarray, **params
    ) -> tuple[np.ndarray, np.ndarray]:
        """To be implemented in subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")


class RandomCopySampler(TunableBaseSampler):
    """Custom oversampler that randomly duplicates samples to match desired output.
    All samples from the resampled class have aproximately equal representation in the output - count of each sample differs by at most 1.
    """

    # This is needed for sklearn compatibility - defines valid parameters and their types
    _parameter_constraints: dict = {
        "enabled": ["boolean"],
        "sampling_strategy": [
            "array-like",
            Interval(Real, None, None, closed="neither"), # type: ignore[type-var]
            StrOptions({"auto"}), # type: ignore[type-var]
        ],
        "additional_group_classes": [
            "array-like",
            Interval(Integral, 0, None, closed="left"), # type: ignore[type-var]
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
    ) -> None:
        """
        TODO: complete docstring

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
        super().__init__(
            enabled=enabled,
            additional_group_classes=additional_group_classes,
            random_state=random_state,
        )
        self.sampling_strategy = (
            sampling_strategy  # this will be processed in _resample
        )

    def _resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray,
        **params,
    ) -> tuple[np.ndarray, np.ndarray]:
        del params  # unused

        new_counts = _compute_number_per_class(classes)
        new_X = []
        new_y = []

        # Oversample each class to the desired count
        for cls, target in new_counts.items():
            cls_mask = classes == cls
            X_cls = X[cls_mask]
            y_cls = y[cls_mask]
            current_number = X_cls.shape[0]

            multiples = target // current_number
            remainder = target % current_number
            for _ in range(multiples):  # add full copies
                new_X.append(X_cls)
                new_y.append(y_cls)
            samples = self.rng.choice(  # add remaining samples
                current_number, size=remainder, replace=False
            )
            new_X.append(X_cls[samples])
            new_y.append(y_cls[samples])
        X_res = np.vstack(new_X)
        y_res = np.hstack(new_y)
        # Shuffle the resampled dataset
        perm = self.rng.permutation(X_res.shape[0])
        return X_res[perm], y_res[perm]
