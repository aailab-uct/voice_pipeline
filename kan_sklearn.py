import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from kan import KAN


class KANClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for Kolmogorov–Arnold Network (KAN).

    Thanks to inheriting from BaseEstimator and ClassifierMixin:
    - it works inside sklearn Pipeline
    - it works with GridSearchCV
    - its parameters can be tuned via clf__* in param_grid
    """

    def __init__(self, hidden=8, grid=5, k=3, steps=200, lr=1e-3, seed=0, threshold=0.5, device="cpu"):
        # Network hyperparameters
        self.hidden = hidden  # number of neurons in the hidden layer
        self.grid = grid  # number of grid points for spline functions in KAN
        self.k = k  # spline degree (smoothness)
        self.steps = steps  # number of training iterations
        self.lr = lr  # learning rate

        # Other settings
        self.seed = seed  # random seed for reproducibility
        self.threshold = threshold  # decision threshold for binary classification
        self.device = device  # "cpu" or "cuda"

    def fit(self, X, y):
        """
        Train the KAN model.

        X: shape (n_samples, n_features)
        y: shape (n_samples,)
        """

        # Convert inputs to float32 numpy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)  # shape (N, 1) for BCE loss

        # Set PyTorch random seed
        torch.manual_seed(self.seed)

        # Create KAN model
        # Architecture: input -> hidden -> 1 output (logit)
        self.model_ = KAN(
            width=[X.shape[1], self.hidden, 1],
            grid=self.grid,
            k=self.k,
            seed=self.seed,
        ).to(self.device)

        # Convert data to torch tensors
        X_t = torch.from_numpy(X).to(self.device)
        y_t = torch.from_numpy(y).to(self.device)

        # Optimizer
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        # Loss function for binary classification (expects logits)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # Switch model to training mode
        self.model_.train()

        # Training loop
        for _ in range(self.steps):
            opt.zero_grad()  # reset gradients
            logits = self.model_(X_t)  # forward pass
            loss = loss_fn(logits, y_t)  # compute loss
            loss.backward()  # backpropagation
            opt.step()  # update weights

        return self

    def predict_proba(self, X):
        """
        Return class probabilities in sklearn format:
        [P(class=0), P(class=1)]
        """

        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X).to(self.device)

        # Switch to evaluation mode
        self.model_.eval()

        with torch.no_grad():
            logits = self.model_(X_t)
            # Convert logits to probabilities for class 1
            p1 = torch.sigmoid(logits).cpu().numpy()

        # Probability of class 0
        p0 = 1.0 - p1

        return np.hstack([p0, p1])

    def predict(self, X):
        """
        Return final class labels (0 or 1) using the given threshold.
        """

        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
