import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from base import ClassifierEstimator


class FisherLinearDiscriminant(ClassifierEstimator):
    def __init__(self) -> None:
        self.w = None
        self.b = None

    def fit(self, X: Tensor, y: Tensor) -> None:
        X1 = X[y == 1]
        X2 = X[y == 0]
        m1 = X1.mean(dim=0, keepdim=True)
        m2 = X2.mean(dim=0, keepdim=True)
        S1 = (X1 - m1).T @ (X1 - m1)
        S2 = (X2 - m2).T @ (X2 - m2)
        Sw = S1 + S2
        self.w = torch.inverse(Sw) @ (m1 - m2).T
        self.b = -0.5 * (m1 + m2) @ self.w

    def predict(self, X: Tensor) -> Tensor:
        assert self.w is not None and self.b is not None
        y_pred = F.relu(torch.sign(X @ self.w + self.b))
        return y_pred.flatten().long()


class LogisticRegression(ClassifierEstimator):
    def __init__(self, in_features: int = 56, lr: float = 0.001,
                 max_epochs: int = 1000, verbose: int = 50) -> None:
        self.lr = lr
        self.max_epochs = max_epochs

        self.model = nn.Linear(in_features, 1)
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.verbose = verbose

        self.history = []

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
