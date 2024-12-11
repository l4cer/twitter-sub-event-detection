import os

import console

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from models import *


def evaluate(model: Model) -> None:
    console.log(f"Model: {{ITALIC}}{{VIOLET}}{model.name}")

    df = pd.read_pickle("data/train.pkl")

    embeddings = np.zeros((len(df["Embedding"]), 200))
    for index, embedding in enumerate(df["Embedding"].values):
        embeddings[index, :] = embedding[:]

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, df["EventType"].values, test_size=0.3)

    console.log("Training...")
    model.train(X_train, y_train)

    score = accuracy_score(y_test, model.predict(X_test))

    if score < 0.7:
        console.log(f"Test acc: {{RED}}{score:.4f}")
    else:
        console.log(f"Test acc: {{YELLOW}}{score:.4f}")

    df = pd.read_pickle("data/eval.pkl")
    df["EventType"] = df["Embedding"].apply(lambda x: model.predict(x[None, :]))

    df["MatchID"] = df["ID"].apply(lambda x: int(x.split("_")[0]))
    df["PeriodID"] = df["ID"].apply(lambda x: int(x.split("_")[1]))
    df.sort_values(by=["MatchID", "PeriodID"], inplace=True)

    if not os.path.exists("predictions"):
        os.mkdir("predictions")

    filename = f"predictions/{model.name.lower()}.csv"
    df.to_csv(filename, columns=["ID", "EventType"], index=False)

    console.log(f"Predictions saved in {{ITALIC}}{{GRAY}}{filename}")


if __name__ == "__main__":
    model = FrequencyClassifier()
    evaluate(model)
