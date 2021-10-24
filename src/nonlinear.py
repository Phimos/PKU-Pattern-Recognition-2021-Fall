from functools import partial
from typing import Any, Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch import Tensor
from torch.optim import Adam

from base import ClassifierEstimator


class MultilayerPerceptrons(ClassifierEstimator):
    """
    Multilayer Perceptrons
    """

    def __init__(self, in_features: int = 56, hidden_units: List[int] = [16],
                 lr: float = 0.001, max_epochs: int = 1000, verbose: int = 50) -> None:
        super().__init__()
        self.lr = lr
        self.max_epochs = max_epochs

        self.model = self._build_layers([in_features] + hidden_units)
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.verbose = verbose

        self.history = []

    def _build_layers(self, hidden_units: List[int]) -> nn.Module:
        model = nn.Sequential()
        for i in range(1, len(hidden_units)):
            model.add_module(
                f"hidden_layer_{i}",
                nn.Linear(hidden_units[i - 1], hidden_units[i]),
            )
            model.add_module(f"relu_{i}", nn.ReLU())
        model.add_module(
            "output_layer", nn.Linear(hidden_units[-1], 1)
        )
        return model

    def fit(self, X: Tensor, y: Tensor) -> None:
        y = y.float()
        for epoch in range(1, self.max_epochs+1):
            y_pred = self.model(X).flatten()
            loss = self.loss(y_pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            accuracy = self.score(X, y)
            self.history.append(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "accuracy": accuracy,
                }
            )
            if epoch % self.verbose == 0:
                print(
                    f"Epoch {epoch}/{self.max_epochs} - Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f}"
                )

    def predict(self, X: Tensor) -> Tensor:
        y_pred = F.relu(torch.sign(self.model(X)))
        return y_pred.flatten().long()


class nonlinearSVM(ClassifierEstimator):
    def __init__(self, n_samples: int = 10000, in_features: int = 56, C: float = 1.0, kernel: str = "rbf", **kwargs: Any) -> None:
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.kernel_func = self.select_kernel_func(kernel, **kwargs)

        self.w = torch.zeros(in_features, 1)
        self.b = torch.zeros(1)

    def select_kernel_func(self, kernel: str, **kwargs: Any):
        def linear(x: Tensor, y: Tensor):
            return torch.dot(x, y.T)

        def rbf(x: Tensor, y: Tensor, gamma: float):
            return torch.exp(-gamma * torch.cdist(x, y, p=2))

        def polynomial(x: Tensor, y: Tensor, degree: float):
            return (linear(x, y) + 1) ** degree

        if kernel == "linear":
            return linear
        elif kernel == "rbf":
            return partial(rbf, gamma=kwargs.get("gamma", 1.0))
        elif kernel == "poly":
            return partial(polynomial, degree=kwargs.get("degree", 2.0))
        else:
            raise ValueError("Unknown kernel function")

    def fit(self, X: Tensor, y: Tensor) -> None:
        self.X = X
        self.y = torch.sign(y.float() - 0.5)
        self.alpha = torch.zeros(X.shape[0], 1)

        xi = F.relu(
            1 - self.y * ((self.y.T * self.alpha.T * self.kernel_func(X, X)).sum(dim=-1) + self.b))
        pass

    def objective_function(self, xi: Tensor):
        return 0.5 * self.w.norm() ** 2 + self.C * xi.sum()

    def dual_objective_function(self, X: Tensor, y: Tensor, alpha: Tensor, kernel_func: Callable):
        return alpha.sum() - 0.5 * ((y @ y.T) * (alpha @ alpha.T) * kernel_func(X, X)).sum()

    def update_alpha(self, idx1, idx2, alpha, y, E, K):
        if y[idx1] != y[idx2]:
            U = torch.clamp(alpha[idx2] - alpha[idx1], min=0)
            V = torch.clamp(self.C - alpha[idx1] + alpha[idx2], max=self.C)
        else:
            U = torch.clamp(alpha[idx1] + alpha[idx2] - self.C, min=0)
            V = torch.clamp(alpha[idx1] + alpha[idx2], max=self.C)

        E = (alpha.T * y.T * K).sum(dim=-1) + self.b - y

        alpha_2_old = alpha[idx2]
        alpha[idx2] += y[idx2] * (E[idx1] - E[idx2]) / \
            (K[idx1, idx1] + K[idx2, idx2] - 2 * K[idx1, idx2])
        alpha[idx2] = torch.clamp(alpha[idx2], min=U, max=V)
        alpha[idx1] += y[idx1] * y[idx2] * (alpha_2_old - alpha[idx2])
        return alpha

    def select_index1(self):
        pass

    def select_index2(self):
        pass

    def forward(self, X: Tensor) -> Tensor:
        return (self.alpha.T * self.y.T * self.kernel_func(X, self.X)).sum(dim=-1) + self.b

    def predict(self, X: Tensor) -> Tensor:
        return torch.relu(torch.sign(self.forward(X)))


if __name__ == '__main__':
    # Load data
    data = loadmat('data.mat')['data']
    X, y = data[:, :-1], data[:, -1]
    y[y == 2] = 0
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    indices = torch.randperm(X.shape[0])
    train_size, test_size = int(0.8 * X.shape[0]), int(0.2 * X.shape[0])

    X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
    y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]

    model = MultilayerPerceptrons()
    model.fit(X_train, y_train)
    print("Multilayer Perceptrons Accuracy:", model.score(X_test, y_test))
