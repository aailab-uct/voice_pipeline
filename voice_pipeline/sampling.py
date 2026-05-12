"""Oversampling methods for imbalanced voice datasets."""

from numbers import Integral, Real

import numpy as np
from imblearn.base import BaseSampler
from imblearn.over_sampling._smote.base import BaseSMOTE
from numpy.typing import NDArray
from sklearn.utils._param_validation import Interval, StrOptions

try:
    from voice_pipeline.nn import VAEModel
except ImportError:
    VAEModel = None


def _compute_number_per_class(
    classes: np.ndarray, sampling_strategy: list | int | float | dict | str = "auto"
) -> dict[int, int]:
    """Compute target sample counts for each class.

    Parameters
    ----------
    classes : numpy.ndarray
        Class label for each row.
    sampling_strategy : list, int, float, dict, or str, default="auto"
        Strategy used to determine target counts. ``"auto"`` balances all
        classes to the largest class. Integers set an absolute target count.
        Floats scale the largest class count. Dictionaries and lists can
        provide per-class targets or multipliers.

    Returns
    -------
    dict
        Mapping from class label to target count.

    Raises
    ------
    ValueError
        If ``sampling_strategy`` is unsupported or mixed-type.
    """
    unique, counts = np.unique(classes, return_counts=True)
    old_counts = dict(zip(unique, counts))

    if sampling_strategy == "auto":
        max_count = max(counts)
        new_counts = {cls: max_count for cls in unique}
    elif isinstance(sampling_strategy, (dict, list)):
        values = sampling_strategy.values() if isinstance(sampling_strategy, dict) else sampling_strategy
        if all(isinstance(v, int) for v in values):
            new_counts = {cls: sampling_strategy[cls] for cls in unique}
        elif all(isinstance(v, float) for v in values):
            new_counts = {cls: int(old_counts[cls] * sampling_strategy[cls]) for cls in unique}
        else:
            raise ValueError("sampling_strategy values must be all int or all float")
    elif isinstance(sampling_strategy, int):
        new_counts = {cls: sampling_strategy for cls in unique}
    elif isinstance(sampling_strategy, float):
        new_counts = {cls: int(max(counts) * sampling_strategy) for cls in unique}
    else:
        raise ValueError("String sampling_strategy not supported")

    return new_counts


class TunableBaseSampler(BaseSampler):
    """Base class for samplers that can be enabled or disabled."""

    _sampling_type = "over-sampling"

    def __init__(
        self,
        enabled: bool,
        additional_group_classes: list | int | None,
        random_state: None | int,
        **kwargs,
    ) -> None:
        """Initialize a tunable sampler.

        Parameters
        ----------
        enabled : bool
            Whether this sampler should modify the data.
        additional_group_classes : list, int, or None
            Columns in ``X`` used as additional grouping labels during
            resampling.
        random_state : int or None
            Random seed for reproducibility.
        **kwargs
            Ignored compatibility arguments.

        Raises
        ------
        ValueError
            If ``additional_group_classes`` has an unsupported type.
        """
        del kwargs
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
        for col in self.additional_group_classes:
            X_col = X[:, col]
            unique_values = np.unique(X_col)
            class_offset = np.max(classes) + 1
            map_unique_to_class = {val: idx * class_offset for idx, val in enumerate(unique_values)}
            mapped_classes = np.array([map_unique_to_class[val] for val in X_col])
            classes = classes + mapped_classes
        X_new, y_new = self._resample(X, y, classes, **params)
        return X_new, y_new

    def _resample(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating] | NDArray[np.bool_],
        classes: NDArray[np.integer],
        **params,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample data in subclasses.

        Parameters
        ----------
        X : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Target labels.
        classes : numpy.ndarray
            Class labels used for resampling.
        **params
            Optional sampler-specific parameters.

        Returns
        -------
        tuple of numpy.ndarray
            Resampled features and labels.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class RandomCopySampler(TunableBaseSampler):
    """Duplicate existing rows to reach target class counts.

    Samples from each resampled class are represented as evenly as possible in
    the output. Counts for individual source samples differ by at most one.
    """

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
        """Initialize a random-copy sampler.

        Parameters
        ----------
        enabled : bool, default=True
            Whether this sampler should modify the data.
        sampling_strategy : list, int, float, dict, or str, default="auto"
            Strategy used to choose target class counts.
        additional_group_classes : list, int, or None, default=None
            Columns in ``X`` used as additional grouping labels.
        random_state : int or None, default=None
            Random seed for reproducibility.
        **kwargs
            Ignored compatibility arguments.
        """
        del kwargs
        super().__init__(
            enabled=enabled,
            additional_group_classes=additional_group_classes,
            random_state=random_state,
        )
        self.sampling_strategy = sampling_strategy

    def _resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray,
        **params,
    ) -> tuple[np.ndarray, np.ndarray]:
        del params

        new_counts = _compute_number_per_class(classes)
        X_new = []
        y_new = []

        for cls, target in new_counts.items():
            cls_mask = classes == cls
            X_cls = X[cls_mask]
            y_cls = y[cls_mask]
            current_number = X_cls.shape[0]

            multiples = target // current_number
            remainder = target % current_number
            for _ in range(multiples):
                X_new.append(X_cls)
                y_new.append(y_cls)
            samples = self.rng.choice(current_number, size=remainder, replace=False)
            X_new.append(X_cls[samples])
            y_new.append(y_cls[samples])
        X_res = np.vstack(X_new)
        y_res = np.hstack(y_new)
        perm = self.rng.permutation(X_res.shape[0])
        return X_res[perm], y_res[perm]


class SMOTESampler(TunableBaseSampler, BaseSMOTE):  # type: ignore[misc]
    """SMOTE sampler with optional synthetic-only output."""

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
        """Initialize a SMOTE sampler.

        Parameters
        ----------
        enabled : bool, default=True
            Whether this sampler should modify the data.
        sampling_strategy : list, int, float, dict, or str, default="auto"
            Strategy used to choose target class counts.
        additional_group_classes : list, int, or None, default=None
            Columns in ``X`` used as additional grouping labels.
        random_state : int or None, default=None
            Random seed for reproducibility.
        artificial_only : bool, default=False
            If ``True``, return only generated samples. If ``False``, append
            generated samples to the original data.
        **kwargs
            Optional compatibility arguments, including ``k_neighbors``.
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
        self._validate_estimator()
        return super()._fit_resample(X, y, **params)

    def _resample(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray, **params) -> tuple[np.ndarray, np.ndarray]:
        del params

        new_counts = _compute_number_per_class(classes, self.sampling_strategy)

        X_new = [] if self.artificial_only else [X.copy()]
        y_new = [] if self.artificial_only else [y.copy()]

        for cls, target in new_counts.items():
            n_samples = target - (0 if self.artificial_only else np.sum(classes == cls))
            if n_samples <= 0:
                continue
            cls_mask = classes == cls
            X_cls = X[cls_mask]
            y_cls = y[cls_mask]

            self.nn_k_.fit(X_cls)
            nns = self.nn_k_.kneighbors(X_cls, return_distance=False)[:, 1:]  # type: ignore[attr-defined]

            X_syn, y_syn = self._make_samples(
                X_cls,
                y.dtype,
                y_cls[0],
                X_cls,
                nns,
                n_samples,
                1.0,
            )
            X_new.append(X_syn)
            y_new.append(y_syn)
        X_res = np.vstack(X_new)
        y_res = np.hstack(y_new)
        perm = self.rng.permutation(X_res.shape[0])
        return X_res[perm], y_res[perm]


class VAESampler(TunableBaseSampler):
    """Generate synthetic samples with a variational autoencoder."""

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
        """Initialize a VAE sampler.

        Parameters
        ----------
        enabled : bool, default=True
            Whether this sampler should modify the data.
        additional_group_classes : list, int, or None, default=None
            Columns in ``X`` used as additional grouping labels.
        random_state : int or None, default=None
            Random seed for reproducibility.
        sampling_strategy : list, int, float, dict, or str, default="auto"
            Strategy used to choose target class counts.
        artificial_only : bool, default=False
            If ``True``, return only generated samples. If ``False``, append
            generated samples to the original data.
        architecture : dict or None, default=None
            VAE architecture settings. ``None`` uses the default architecture.
        epochs : int, default=50
            Number of VAE training epochs.
        **kwargs
            Ignored compatibility arguments.
        """
        del kwargs
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
        """Return the VAE architecture settings.

        Returns
        -------
        dict
            Current VAE architecture dictionary.
        """
        return self._architecture

    @architecture.setter
    def architecture(self, value: dict | None) -> None:
        """Set the VAE architecture settings.

        Parameters
        ----------
        value : dict or None
            Architecture dictionary. ``None`` selects the default architecture.

        Raises
        ------
        ImportError
            If the VAE model implementation is unavailable.
        """
        if VAEModel is None:
            raise ImportError(
                "VAESampler needs voice_pipeline.nn.VAEModel, but that module is not present in this repository."
            )
        if value is None:
            value = {"latent_dim": 10, "hidden_layers": [64, 32], "batch_size": 32}
        self._architecture = value

        self.vae_model = VAEModel(
            latent_dim=self._architecture.get("latent_dim", 10),
            hidden_layers=self._architecture.get("hidden_layers", [64, 32]),
            batch_size=self._architecture.get("batch_size", 32),
            random_state=self.random_state,
            epochs=self.epochs,
        )

    @property
    def epochs(self) -> int:
        """Return the number of VAE training epochs.

        Returns
        -------
        int
            Number of training epochs.
        """
        return self._epochs

    @epochs.setter
    def epochs(self, value: int) -> None:
        """Set the number of VAE training epochs.

        Parameters
        ----------
        value : int
            Number of training epochs.
        """
        self._epochs = value
        if hasattr(self, "vae_model"):
            self.vae_model.epochs = value

    def _resample(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating] | NDArray[np.bool_],
        classes: NDArray[np.integer],
        **params,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic samples and return resampled data.

        Parameters
        ----------
        X : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Target labels.
        classes : numpy.ndarray
            Class labels used for resampling.
        **params
            Ignored compatibility arguments.

        Returns
        -------
        tuple of numpy.ndarray
            Resampled features and labels.
        """
        del params

        self.vae_model.train(X, y, classes)

        new_counts = _compute_number_per_class(classes, self.sampling_strategy)

        X_new = [] if self.artificial_only else [X.copy()]
        y_new = [] if self.artificial_only else [y.copy()]
        for cls, target in new_counts.items():
            n_samples = target - (0 if self.artificial_only else np.sum(classes == cls))
            if n_samples <= 0:
                continue
            X_syn, y_syn = self.vae_model.generate_samples(n_samples=n_samples, cls=cls)

            X_new.append(X_syn)
            y_new.append(y_syn)
        X_res = np.vstack(X_new)
        y_res = np.hstack(y_new)
        perm = self.rng.permutation(X_res.shape[0])
        return X_res[perm], y_res[perm]
