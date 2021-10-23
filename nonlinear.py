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
