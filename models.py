import numpy as np

from sklearn import linear_model, ensemble

from keras import Sequential, layers

from typing import Union


class Model:
    def __init__(self, name: str) -> None:
        self.name = name

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError


class DummyModel(Model):
    def __init__(self) -> None:
        super().__init__("DummyModel")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        return

    def predict(self, X: np.ndarray) -> Union[float, np.ndarray]:
        return np.ones(len(X)) if len(X) > 1 else 1.0


class LogisticRegression(Model):
    def __init__(self) -> None:
        super().__init__("LogisticRegression")

        self.model = linear_model.LogisticRegression(max_iter=1000)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> Union[float, np.ndarray]:
        pred = self.model.predict(X)

        return pred if len(pred) > 1 else pred.flatten()[0]


class RandomForest(Model):
    def __init__(self) -> None:
        super().__init__("RandomForest")

        self.model = ensemble.RandomForestClassifier(n_estimators=50, max_depth=10)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> Union[float, np.ndarray]:
        pred = self.model.predict(X)

        return pred if len(pred) > 1 else pred.flatten()[0]
