"""Module for oversampling transformers."""

import numpy as np
from imblearn.base import BaseSampler
from imblearn.over_sampling._smote.base import BaseSMOTE
from sklearn.utils._param_validation import Interval, StrOptions
from numbers import Real, Integral
from src.nn import VAEModel
from numpy.typing import NDArray
from imblearn.over_sampling import BorderlineSMOTE, ADASYN, KMeansSMOTE
from imblearn.combine import SMOTEENN


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

    def _resample(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating] | NDArray[np.bool_],
        classes: NDArray[np.integer],
        **params,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        """Custom SMOTE oversampler that can also generate only artificial samples without copying existing ones.

        Note that the undersampling is performed only if artificial_only is True, i.e. no real samples are droped unless
        this is specified.

        :param enabled: Whether to enable this sampler
        :param sampling_strategy: What sampling strategy to use
            - "auto": resample all classes to the maximum class count
            - dict|list: specific counts or multipliers per class,
            - int: specific count for all classes,
            - float: multiply max class count by float for all classes
        :param additional_group_classes: Columns in X to use as additional group identifiers
        :param random_state: Random seed for reproducibility
        :param artificial_only: If True, only generate synthetic samples without copying existing ones
        """
        super().__init__(
            enabled=enabled,
            additional_group_classes=additional_group_classes,
            random_state=random_state,
            k_neighbors=kwargs.get("k_neighbors", 5),
        )

        self.sampling_strategy = sampling_strategy
        self.artificial_only = artificial_only

    def _fit_resample(self, X: np.ndarray, y: np.ndarray, **params):  # type: ignore
        self._validate_estimator()  # must be called before _resample to initialize nearest neighbors estimator
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

    # This is needed for sklearn compatibility - defines valid parameters and their types
    _parameter_constraints: dict = {
        "enabled": ["boolean"],
        "additional_group_classes": [
            "array-like",
            Interval(Integral, 0, None, closed="left"),  # type: ignore[type-var]
            None,
        ],
        "random_state": ["random_state"],
        "sampling_strategy": [
            "array-like",
            Interval(Real, None, None, closed="neither"),  # type: ignore[type-var]
            StrOptions({"auto"}),  # type: ignore[type-var]
        ],
        "artificial_only": ["boolean"],
        "architecture": [dict, None],
        "epochs": [Interval(Integral, 1, None, closed="left")],  # type: ignore[type-var]
    }

    def __init__(
        self,
        enabled: bool = True,
        additional_group_classes: list | int | None = None,
        random_state: None | int = None,
        sampling_strategy: list | int | float | dict | str = "auto",
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
        self.epochs = epochs
        self.architecture = architecture
        self.artificial_only = artificial_only
        self.sampling_strategy = sampling_strategy

    @property
    def architecture(self) -> dict:
        """Get the architecture of the VAE model."""
        return self._architecture

    @architecture.setter
    def architecture(self, value: dict | None) -> None:
        """Set the architecture of the VAE model."""
        if value is None:
            value = {"latent_dim": 10, "hidden_layers": [64, 32], "batch_size": 32}
        self._architecture = value

        # Reinitialize the VAE model with the new architecture
        self.vae_model = VAEModel(
            latent_dim=self._architecture.get("latent_dim", 10),
            hidden_layers=self._architecture.get("hidden_layers", [64, 32]),
            batch_size=self._architecture.get("batch_size", 32),
            random_state=self.random_state,
            epochs=self.epochs,
        )

    @property
    def epochs(self) -> int:
        """Get the number of training epochs for the VAE model."""
        return self._epochs

    @epochs.setter
    def epochs(self, value: int) -> None:
        """Set the number of training epochs for the VAE model."""
        self._epochs = value
        # If the VAE model is already initialized, update its epochs
        if hasattr(self, "vae_model"):
            self.vae_model.epochs = value

    def _resample(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating] | NDArray[np.bool_],
        classes: NDArray[np.integer],
        **params,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample the dataset using a VAE model to generate synthetic samples.
        :param X: Input features
        :type X: NDArray[np.floating]
        :param y: Target labels
        :type y: NDArray[np.floating] | NDArray[np.bool_]
        :param classes: Class labels for each sample
        :type classes: NDArray[np.integer]
        :return: Resampled features and labels
        :rtype: tuple[np.ndarray, np.ndarray]"""
        del params  # unused

        self.vae_model.train(X, y, classes)

        new_counts = _compute_number_per_class(classes, self.sampling_strategy)  # desired counts per class

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


# ── 1) Borderline-SMOTE ──────────────────────────────────────────────────────

class BorderlineSMOTESampler(TunableBaseSampler):
    """
    Borderline-SMOTE: generates synthetic samples only from minority instances
    near the decision boundary.

    Extra parameters (beyond TunableBaseSampler):
        kind : str, default="borderline-1"
            "borderline-1" — oversample from borderline minority instances
            "borderline-2" — also considers majority neighbors for interpolation
        k_neighbors : int, default=5
            Number of nearest neighbors for SMOTE interpolation.
        m_neighbors : int, default=10
            Number of nearest neighbors to determine if a sample is borderline.
    """

    _parameter_constraints: dict = {
        "enabled": ["boolean"],
        "additional_group_classes": ["array-like", Interval(Integral, 0, None, closed="left"), None],
        "random_state": ["random_state"],
        "sampling_strategy": ["array-like", Interval(Real, None, None, closed="neither"), StrOptions({"auto"})],
        "artificial_only": ["boolean"],
        "kind": [StrOptions({"borderline-1", "borderline-2"})],
        "k_neighbors": [Interval(Integral, 1, None, closed="left")],
        "m_neighbors": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
            self,
            enabled: bool = True,
            additional_group_classes: list | int | None = None,
            random_state: int = 42,
            artificial_only: bool = False,
            sampling_strategy: str | float | dict = "auto",
            kind: str = "borderline-1",
            k_neighbors: int = 5,
            m_neighbors: int = 10,
    ):
        super().__init__(
            enabled=enabled,
            additional_group_classes=additional_group_classes,
            random_state=random_state,
        )
        self.artificial_only = artificial_only
        self.sampling_strategy = sampling_strategy
        self.kind = kind
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors

    def _resample(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray, **params):
        del params

        sampler = BorderlineSMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
            m_neighbors=self.m_neighbors,
            kind=self.kind,
        )
        X_res, y_res = sampler.fit_resample(X, y)

        if self.artificial_only:
            X_res, y_res = X_res[len(X):], y_res[len(y):]

        return X_res, y_res


# ── 2) ADASYN ────────────────────────────────────────────────────────────────

class ADASYNSampler(TunableBaseSampler):
    """
    ADASYN: Adaptive Synthetic Sampling. Generates more synthetic samples
    in regions where the minority class is harder to learn (more majority
    neighbors).

    Extra parameters (beyond TunableBaseSampler):
        n_neighbors : int, default=5
            Number of nearest neighbors used to construct synthetic samples.
    """

    _parameter_constraints: dict = {
        "enabled": ["boolean"],
        "additional_group_classes": ["array-like", Interval(Integral, 0, None, closed="left"), None],
        "random_state": ["random_state"],
        "sampling_strategy": ["array-like", Interval(Real, None, None, closed="neither"), StrOptions({"auto"})],
        "artificial_only": ["boolean"],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
            self,
            enabled: bool = True,
            additional_group_classes: list | int | None = None,
            random_state: int = 42,
            artificial_only: bool = False,
            sampling_strategy: str | float | dict = "auto",
            n_neighbors: int = 5,
    ):
        super().__init__(
            enabled=enabled,
            additional_group_classes=additional_group_classes,
            random_state=random_state,
        )
        self.artificial_only = artificial_only
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors

    def _resample(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray, **params):
        del params

        sampler = ADASYN(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
        )
        X_res, y_res = sampler.fit_resample(X, y)

        if self.artificial_only:
            X_res, y_res = X_res[len(X):], y_res[len(y):]

        return X_res, y_res


# ── 3) SMOTE-ENN (hybrid: oversample + clean) ────────────────────────────────

class SMOTEENNSampler(TunableBaseSampler):
    """
    SMOTE + Edited Nearest Neighbors: first oversamples the minority class
    with SMOTE, then removes noisy/ambiguous samples from both classes
    using ENN.

    Note: artificial_only is NOT supported here — ENN deletes samples
    from both original and synthetic data, so the boundary between them
    is lost after cleaning.

    Extra parameters (beyond TunableBaseSampler):
        k_neighbors_smote : int, default=5
            k for the SMOTE interpolation step.
        n_neighbors_enn : int, default=3
            k for the ENN cleaning step.
    """

    _parameter_constraints: dict = {
        "enabled": ["boolean"],
        "additional_group_classes": ["array-like", Interval(Integral, 0, None, closed="left"), None],
        "random_state": ["random_state"],
        "sampling_strategy": ["array-like", Interval(Real, None, None, closed="neither"), StrOptions({"auto"})],
        "artificial_only": ["boolean"],
        "k_neighbors_smote": [Interval(Integral, 1, None, closed="left")],
        "n_neighbors_enn": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
            self,
            enabled: bool = True,
            additional_group_classes: list | int | None = None,
            random_state: int = 42,
            artificial_only: bool = False,
            sampling_strategy: str | float | dict = "auto",
            k_neighbors_smote: int = 5,
            n_neighbors_enn: int = 3,
    ):
        super().__init__(
            enabled=enabled,
            additional_group_classes=additional_group_classes,
            random_state=random_state,
        )
        self.artificial_only = artificial_only
        self.sampling_strategy = sampling_strategy
        self.k_neighbors_smote = k_neighbors_smote
        self.n_neighbors_enn = n_neighbors_enn

    def _resample(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray, **params):
        del params

        from imblearn.over_sampling import SMOTE as _SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours

        sampler = SMOTEENN(
            smote=_SMOTE(
                k_neighbors=self.k_neighbors_smote,
                random_state=self.random_state,
            ),
            enn=EditedNearestNeighbours(
                n_neighbors=self.n_neighbors_enn,
            ),
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )
        X_res, y_res = sampler.fit_resample(X, y)

        if self.artificial_only:
            import warnings
            warnings.warn(
                "SMOTEENNSampler: artificial_only=True is ignored — "
                "ENN cleaning removes from both original and synthetic data.",
                UserWarning,
                stacklevel=2,
            )

        return X_res, y_res


# ── 4) KMeans-SMOTE ──────────────────────────────────────────────────────────

class KMeansSMOTESampler(TunableBaseSampler):
    """
    KMeans-SMOTE: clusters the data first, then applies SMOTE within
    clusters that have a high proportion of minority samples.
    Generates samples in sparse minority regions, avoiding noise.

    Extra parameters (beyond TunableBaseSampler):
        k_neighbors : int, default=5
            Number of nearest neighbors for SMOTE interpolation.
        kmeans_estimator : int or KMeans, default=None
            Number of clusters (int) or a fitted/unfitted KMeans instance.
            If None, imblearn uses a default.
        cluster_balance_threshold : float, default=0.1
            Clusters with a minority ratio below this are skipped.
            Lower = more selective (only very minority-dense clusters).
    """

    _parameter_constraints: dict = {
        "enabled": ["boolean"],
        "additional_group_classes": ["array-like", Interval(Integral, 0, None, closed="left"), None],
        "random_state": ["random_state"],
        "sampling_strategy": ["array-like", Interval(Real, None, None, closed="neither"), StrOptions({"auto"})],
        "artificial_only": ["boolean"],
        "k_neighbors": [Interval(Integral, 1, None, closed="left")],
        "kmeans_estimator": [Interval(Integral, 1, None, closed="left"), None],
        "cluster_balance_threshold": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
            self,
            enabled: bool = True,
            additional_group_classes: list | int | None = None,
            random_state: int = 42,
            artificial_only: bool = False,
            sampling_strategy: str | float | dict = "auto",
            k_neighbors: int = 5,
            kmeans_estimator=None,
            cluster_balance_threshold: float = 0.1,
    ):
        super().__init__(
            enabled=enabled,
            additional_group_classes=additional_group_classes,
            random_state=random_state,
        )
        self.artificial_only = artificial_only
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.kmeans_estimator = kmeans_estimator
        self.cluster_balance_threshold = cluster_balance_threshold

    def _resample(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray, **params):
        del params

        kwargs = dict(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
            cluster_balance_threshold=self.cluster_balance_threshold,
        )
        if self.kmeans_estimator is not None:
            kwargs["kmeans_estimator"] = self.kmeans_estimator

        sampler = KMeansSMOTE(**kwargs)
        X_res, y_res = sampler.fit_resample(X, y)

        if self.artificial_only:
            X_res, y_res = X_res[len(X):], y_res[len(y):]

        return X_res, y_res