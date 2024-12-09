import os

import logging

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.callbacks import ModelCheckpoint

from typing import Tuple


BLUE = "\33[34m"
GRAY = "\33[90m"
NORMAL = "\33[0m"


logging.basicConfig(level=logging.INFO,
    format=f"{BLUE}%(asctime)s{NORMAL} %(message)s{NORMAL}", datefmt="%H:%M:%S")


def get_train_dataset(folder: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    X_train = np.load(os.path.join(folder, "X_train.npy"))
    y_train = np.load(os.path.join(folder, "y_train.npy"))

    return X_train.astype(np.int32), y_train.astype(np.float32)


def get_eval_dataset(folder: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    X_eval = np.load(os.path.join(folder, "X_eval.npy"))
    T_eval = np.load(os.path.join(folder, "T_eval.npy"))

    return X_eval.astype(np.int32), T_eval.astype(str)


def get_model(dict_size: int = 191989, embedding_dim: int = 40) -> Sequential:
    model = Sequential()
    model.add(Embedding(input_dim=dict_size, output_dim=embedding_dim, input_length=50))
    model.add(LSTM(40))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    if os.path.exists("checkpoint.model.keras"):
        logging.info("Loading saved weights")
        model.load_weights("checkpoint.model.keras")

    return model


def train_model() -> None:
    X_train, y_train = get_train_dataset()
    logging.info(f"{X_train.shape=} and {y_train.shape=}\n")

    model = get_model()

    logging.info("Model summary:")
    model.summary()

    model_checkpoint_callback = ModelCheckpoint(
        filepath="checkpoint.model.keras",
        monitor="accuracy",
        mode="max",
        save_freq="epoch")

    mask = np.random.rand(len(y_train)) < 0.01

    model.fit(
        X_train[mask],
        y_train[mask],
        epochs=1,
        batch_size=64,
        callbacks=[model_checkpoint_callback]
    )


def evaluate_model() -> None:
    model = get_model()

    X_eval, T_eval = get_eval_dataset()
    y_eval = model.predict(X_eval)

    predictions = []

    THRESHOLD = 0.5

    for t in np.unique(T_eval):
        predictions.append(
            [t, 1.0 if np.mean(y_eval[T_eval == t]) > THRESHOLD else 0.0])

    df = pd.DataFrame(predictions, columns=["ID", "EventType"])
    df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    # train_model()
    evaluate_model()
