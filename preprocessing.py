import os

import console

import numpy as np
import pandas as pd

import nltk
import gensim.downloader as api

from typing import Union


console.log("Downloading stop words in multiple languages")
nltk.download("stopwords", quiet=True)

stopwords = set()
for language in ["english", "french", "spanish", "portuguese"]:
    stopwords.update(set(nltk.corpus.stopwords.words(language)))

console.log("Loading 200-dimensional GloVe embedding model")
embedding = api.load("glove-twitter-200")

console.skip_lines(num=1)


def preprocessing(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(data, str):
        df = pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        df = data

    # Remove retweets
    df = df[df["Tweet"].str.findall(r"RT @[\w]+:").map(len) == 0]

    # Remove mentions
    df = df[df["Tweet"].str.findall(r"@[\w]+").map(len) == 0]

    # Remove URLs
    df["Tweet"] = df["Tweet"].str.replace(r"http\S+", "", regex=True)

    # Remove punctuation and special characters (including numbers)
    df["Tweet"] = df["Tweet"].str.replace(r"[^a-zA-Z\s]+", " ", regex=True)

    # Put space in camel case sentences
    df["Tweet"] = df["Tweet"].str.replace(r"([A-Z][a-z])", r" \1", regex=True)

    # Remove multiple spaces and strip tweets
    df["Tweet"] = df["Tweet"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Lowercase all remaining tweets
    df["Tweet"] = df["Tweet"].str.lower().astype(str)

    # Remove duplicated tweets
    df.drop_duplicates(inplace=True)

    # Tokenize and remove stop words
    df["Tokens"] = df["Tweet"].apply(
        lambda text: [word for word in text.split() if word not in stopwords and word in embedding])

    df = df[df["Tokens"].map(len) > 0]

    # Embedding tokens
    df["Embedding"] = df["Tokens"].apply(
        lambda tokens: np.mean([embedding[token] for token in tokens], axis=0))

    df = df.drop(
        columns=["MatchID", "PeriodID", "Timestamp", "Tweet", "Tokens"])

    # Average of embeddings with the same ID
    df = df.groupby(["ID"]).mean().reset_index()

    return df


def preprocess_folder(folder: str, save_destination: str) -> None:
    merged_df = None

    total = len(os.listdir(folder))
    for index, filename in enumerate(os.listdir(folder)):
        console.log(f"{index+1}/{total} Preprocessing {{ITALIC}}{{GRAY}}{folder}/{filename}")
        df = preprocessing(os.path.join(folder, filename))

        merged_df = df if merged_df is None else pd.concat([merged_df, df])

    console.log(f"Saving the dataframe in {{ITALIC}}{{GRAY}}{save_destination}")
    merged_df.to_pickle(save_destination)


if __name__ == "__main__":
    preprocess_folder("train_tweets", "data/train.pkl")
    console.skip_lines(num=1)
    preprocess_folder("eval_tweets", "data/eval.pkl")
