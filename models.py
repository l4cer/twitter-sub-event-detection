import numpy as np

from sklearn import linear_model, ensemble

from xgboost import XGBClassifier

from keras import Sequential, layers

from typing import Union


class Model:
    def __init__(self, name: str) -> None:
        self.name = name

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError


class FrequencyClassifier(Model):
    def __init__(self) -> None:
        super().__init__("FrequencyClassifier")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        values, counts = np.unique(y, return_counts=True)
        self.value = values[np.argmax(counts)]

    def predict(self, X: np.ndarray) -> Union[float, np.ndarray]:
        pred = self.value * np.ones(len(X))

        return pred if len(pred) > 1 else pred.flatten()[0]


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


class XGBoost(Model):
    def __init__(self) -> None:
        super().__init__("XGBoost")

        self.model = XGBClassifier()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> Union[float, np.ndarray]:
        pred = self.model.predict(X)

        return pred if len(pred) > 1 else pred.flatten()[0]


class Dense(Model):
    def __init__(self) -> None:
        super().__init__("Dense")

        self.model = Sequential([
            layers.Input(shape=(200,)),
            layers.Dense(50, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense( 1, activation="softmax")
        ])

        self.model.compile(
            loss="binary_crossentropy",
            metrics=["accuracy"],
            optimizer="adam"
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y, epochs=10, batch_size=1)

    def predict(self, X: np.ndarray) -> Union[float, np.ndarray]:
        pred = self.model.predict(X, verbose=0)

        return pred if len(pred) > 1 else pred.flatten()[0]
