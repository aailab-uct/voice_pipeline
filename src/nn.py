import numpy as np
import torch
from torch import nn


class VAEModel:
    def __init__(
        self, latent_dim: int, hidden_layers: list[int], random_state: int | None = None, epochs: int = 50
    ) -> None:
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.random_state = random_state
        self.epochs = epochs
        self.decoder = None  # type: nn.Module
        self.encoder = None  # type: nn.Module
        # use cuda or rocm if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if random_state is not None:
            torch.manual_seed(random_state)
        self.class_to_y = {}

    def train(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> None:
        if self.decoder is None:
            self._initialize_model(x_dim=X.shape[1], n_classes=len(classes))
        else:
            self.decoder.apply(self._reset_weights)
            self.encoder.apply(self._reset_weights)
        self.class_to_y = {cls: np.unique(y[y == cls])[0] for cls in classes}
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-3)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        class_tensor = torch.tensor(classes, dtype=torch.long).to(self.device)
        class_tensor_onehot = nn.functional.one_hot(class_tensor, num_classes=len(classes)).float().to(self.device)
        for epoch in range(self.epochs):
            self.encoder.train()
            self.decoder.train()
            optimizer.zero_grad()
            h = self.encoder.forward(X_tensor)
            mu, logvar = torch.chunk(h, 2, dim=1)
            z = self._reparameterize(mu, logvar)
            z_cond = torch.cat([z, class_tensor_onehot], dim=1)
            X_recon = self.decoder.forward(z_cond)
            recon_loss = nn.functional.mse_loss(X_recon, X_tensor, reduction="sum")
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld_loss
            loss.backward()
            optimizer.step()

    def generate_samples(self, n_samples: int, cls: int) -> tuple[np.ndarray, np.ndarray]:
        if self.decoder is None:
            raise ValueError("Model has not been trained yet.")
        self.decoder.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            class_index = list(self.class_to_y.keys()).index(cls)
            class_tensor = torch.zeros(n_samples, len(self.class_to_y)).to(self.device)
            class_tensor[:, class_index] = 1.0
            generated = self.decoder.forward(torch.cat([z, class_tensor], dim=1))
        return generated.cpu().numpy(), np.full(n_samples, self.class_to_y[cls])

    def _initialize_model(self, x_dim: int, n_classes: int) -> None:
        encoder_layers = [nn.Linear(x_dim, self.hidden_layers[0]), nn.ReLU()]
        for i in range(len(self.hidden_layers) - 1):
            encoder_layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(self.hidden_layers[-1], self.latent_dim * 2))  # mean and logvar
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [nn.Linear(self.latent_dim + n_classes, self.hidden_layers[-1]), nn.ReLU()]
        for i in range(len(self.hidden_layers) - 1, 0, -1):
            decoder_layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i - 1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(self.hidden_layers[0], x_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def _reset_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
