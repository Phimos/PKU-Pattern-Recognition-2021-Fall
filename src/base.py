from typing import Any

from torch import Tensor


class ClassifierEstimator(object):
    def __init__(self) -> None:
        pass

    def fit(self, X: Tensor, y: Tensor, **kwargs: Any) -> None:
        raise NotImplementedError

    def predict(self, X: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def score(self, X: Tensor, y: Tensor):
        y_pred = self.predict(X)
        return ((y_pred == y).sum() / len(y)).item()
