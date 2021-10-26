from functools import partial
from typing import Any, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.svm import SVC
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
        self.eps = 1e-8
        self.C = C
        self.kernel = kernel
        self.kernel_func = self.select_kernel_func(kernel, **kwargs)

        self.w = torch.zeros(in_features, 1)
        self.b = torch.zeros(1)

    def select_kernel_func(self, kernel: str, **kwargs: Any):
        def linear(x: Tensor, y: Tensor):
            return x @ y.T

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
        self.y = torch.sign(y.float() - 0.5).reshape(-1, 1)
        self.alpha = torch.zeros(X.shape[0], 1)
        self.K = self.kernel_func(X, X)

        for i in range(100):
            self.fx = self.forward(self.X).reshape(-1, 1)
            self.E = self.fx - self.y
            idx1, idx2 = self.select_indices()
            self.alpha = self.update_alpha(
                idx1, idx2, self.alpha, self.y, self.K)
            self.b = self.update_b()
            if i % 10 == 0:
                print(self.objective_function(),
                      self.dual_objective_function())
                print(self.score(X, y))

    def objective_function(self):
        xi = F.relu(1 - self.y * self.fx)
        print(xi.shape)
        return 0.5 * ((self.y @ self.y.T) * (self.alpha @ self.alpha.T) * self.K).sum() + self.C * xi.sum()

    def dual_objective_function(self):
        return self.alpha.sum() - 0.5 * ((self.y @ self.y.T) * (self.alpha @ self.alpha.T) * self.K).sum()

    def update_alpha(self, idx1, idx2, alpha, y, K):
        if y[idx1] != y[idx2]:
            U = torch.clamp(alpha[idx2] - alpha[idx1], min=0)
            V = torch.clamp(self.C - alpha[idx1] + alpha[idx2], max=self.C)
        else:
            U = torch.clamp(alpha[idx1] + alpha[idx2] - self.C, min=0)
            V = torch.clamp(alpha[idx1] + alpha[idx2], max=self.C)
        U, V = U.item(), V.item()
        print("U", U, "V", V)
        E = self.E

        alpha_2_old = alpha[idx2].item()
        alpha[idx2] += y[idx2] * (E[idx1] - E[idx2]) / \
            (K[idx1, idx1] + K[idx2, idx2] - 2 * K[idx1, idx2])
        alpha[idx2] = torch.clamp(alpha[idx2], min=U, max=V)
        alpha[idx1] += y[idx1] * y[idx2] * (alpha_2_old - alpha[idx2])
        return alpha

    def update_b(self):
        cond = ((self.alpha > 0) & (self.alpha < self.C))
        if cond.any():
            return (self.y - self.fx)[cond].mean().item()
        else:
            return 0

    def select_index1(self):
        out = self.y.squeeze() * self.forward(self.X) - 1
        alpha = self.alpha.squeeze()
        cond1 = (out > 0) & (alpha == 0)
        cond2 = (out == 0) & (alpha > 0) & (alpha < self.C)
        cond3 = (out < 0) & (alpha == self.C)
        kkt = cond1 | cond2 | cond3
        return torch.nonzero(~kkt)[0].item()

    def select_index2(self, idx1):
        E = self.E
        return torch.argmax(torch.abs(E[idx1] - E)).item()

    def select_indices(self):
        idx1 = self.select_index1()
        idx2 = self.select_index2(idx1)
        return idx1, idx2

    def forward(self, X: Tensor) -> Tensor:
        return (self.alpha.T * self.y.T * self.kernel_func(X, self.X)).sum(dim=-1) + self.b

    def predict(self, X: Tensor) -> Tensor:
        return torch.relu(torch.sign(self.forward(X)))


class KernelFisher(ClassifierEstimator):
    def __init__(self, kernel: str = "rbf", t: float = 0.01, ** kwargs: Any) -> None:
        super().__init__()
        self.eps = 1e-8
        self.t = t
        self.kernel = kernel
        self.kernel_func = self.select_kernel_func(kernel, **kwargs)

        self.alpha = None
        self.b = None

    def select_kernel_func(self, kernel: str, **kwargs: Any):
        def linear(x: Tensor, y: Tensor):
            return x @ y.T

        def rbf(x: Tensor, y: Tensor, gamma: float):
            return torch.exp(-gamma * torch.cdist(x, y, p=2))

        def polynomial(x: Tensor, y: Tensor, degree: float):
            return (linear(x, y) + 1) ** degree

        if kernel == "linear":
            return linear
        elif kernel == "rbf":
            return partial(rbf, gamma=kwargs.get("gamma", 1.0))
        elif kernel == "poly":
            return partial(polynomial, degree=kwargs.get("degree", 3.0))
        else:
            raise ValueError("Unknown kernel function")

    def fit(self, X: Tensor, y: Tensor) -> None:
        self.X = X
        K = self.kernel_func(X, X)
        mask0, mask1 = y == 0, y == 1
        K0, K1 = K[mask0], K[mask1]
        m0 = K0.mean(dim=0)
        m1 = K1.mean(dim=0)
        tau = (m1 - m0).reshape(-1, 1)
        d0 = K0 - m0
        d1 = K1 - m1
        N0 = d0.T @ d0
        N1 = d1.T @ d1
        N = N0 + N1
        self.alpha = torch.pinverse(N + self.t * K) @ tau
        self.b = -(0.5 * (m0 + m1)) @ self.alpha

    def predict(self, X: Tensor) -> Tensor:
        assert self.alpha is not None and self.b is not None
        ret = (self.alpha * self.kernel_func(self.X, X)).sum(dim=0) + self.b
        return torch.relu(torch.sign(ret))


if __name__ == '__main__':
    torch.random.manual_seed(42)

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

    # Train MLP
    model = MultilayerPerceptrons()
    model.fit(X_train, y_train)
    print("Multilayer Perceptrons Accuracy:", model.score(X_test, y_test))

    # Plot loss & accuracy
    plt.plot([item['loss'] for item in model.history])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.savefig('nn_loss.png', dpi=300)
    plt.close()

    plt.plot([item['accuracy'] for item in model.history])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.savefig('nn_accuracy.png', dpi=300)
    plt.close()

    # Different kernels
    for kernel in ["linear", "rbf", "poly"]:
        model = SVC(kernel=kernel, gamma=1, degree=3)
        model.fit(X_train.numpy(), y_train.numpy())
        print("Kernel:", kernel)
        print("SVM Train Accuracy:", model.score(
            X_train.numpy(), y_train.numpy()))
        print("SVM Test Accuracy:", model.score(
            X_test.numpy(), y_test.numpy()))

        model = KernelFisher(kernel=kernel, gamma=1, degree=3)
        model.fit(X_train, y_train)
        print("Kernel:", kernel)
        print("Kernel Fisher Train Accuracy:", model.score(X_train, y_train))
        print("Kernel Fisher Test Accuracy:", model.score(X_test, y_test))

    # Different gamma
    for gamma in ['scale', 'auto', 1]:
        model = SVC(kernel='rbf', gamma=gamma)
        model.fit(X_train.numpy(), y_train.numpy())
        print("Gamma:", gamma, "SVC Accuracy:",
              model.score(X_test.numpy(), y_test.numpy()))

        model = KernelFisher(kernel="rbf", gamma=model._gamma)
        model.fit(X_train, y_train)
        print("Gamma:", gamma, "Kernel Fisher Accuracy:",
              model.score(X_test, y_test))
