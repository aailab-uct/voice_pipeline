"""Custom classifiers with sklearn-compatible interfaces."""

import numpy as np
import torch
from kan import KAN
from sklearn.base import BaseEstimator, ClassifierMixin


class KANClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn wrapper for a Kolmogorov-Arnold Network classifier.

    The class can be used inside sklearn pipelines and ``GridSearchCV``. Its
    parameters can be tuned with the usual ``clf__`` prefix.
    """

    def __init__(self, hidden=8, grid=5, k=3, steps=200, lr=1e-3, seed=0, threshold=0.5, device="cpu"):
        """Initialize a KAN classifier.

        Parameters
        ----------
        hidden : int, default=8
            Number of neurons in the hidden layer.
        grid : int, default=5
            Number of spline grid points.
        k : int, default=3
            Spline degree.
        steps : int, default=200
            Number of optimization steps.
        lr : float, default=1e-3
            Learning rate.
        seed : int, default=0
            Random seed used by PyTorch and KAN.
        threshold : float, default=0.5
            Probability threshold used for binary class prediction.
        device : str, default="cpu"
            Device used by PyTorch, for example ``"cpu"`` or ``"cuda"``.
        """
        # Network hyperparameters
        self.hidden = hidden
        self.grid = grid
        self.k = k
        self.steps = steps
        self.lr = lr

        # Other settings
        self.seed = seed
        self.threshold = threshold
        self.device = device

    def fit(self, X, y):
        """Fit the KAN model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Binary target labels.

        Returns
        -------
        KANClassifier
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        torch.manual_seed(self.seed)

        self.model_ = KAN(
            width=[X.shape[1], self.hidden, 1],
            grid=self.grid,
            k=self.k,
            seed=self.seed,
        ).to(self.device)

        X_t = torch.from_numpy(X).to(self.device)
        y_t = torch.from_numpy(y).to(self.device)

        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        self.model_.train()
        for _ in range(self.steps):
            opt.zero_grad()
            logits = self.model_(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            opt.step()

        return self

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        numpy.ndarray of shape (n_samples, 2)
            Columns contain ``P(class=0)`` and ``P(class=1)``.
        """
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_t)
            p1 = torch.sigmoid(logits).cpu().numpy()

        p0 = 1.0 - p1
        return np.hstack([p0, p1])

    def predict(self, X):
        """Predict binary class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted labels, encoded as 0 or 1.
        """
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
