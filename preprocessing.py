import os

import logging

import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from typing import Union


BLUE = "\33[34m"
GRAY = "\33[90m"
NORMAL = "\33[0m"


logging.basicConfig(level=logging.INFO,
    format=f"{BLUE}%(asctime)s{NORMAL} %(message)s{NORMAL}", datefmt="%H:%M:%S")


def preprocessing(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(data, str):
        df = pd.read_csv(data)
        # df = df[:45]

    if isinstance(data, pd.DataFrame):
        df = data

    # Remove retweets
    df = df[df["Tweet"].str.findall(r"RT @[\w]+:").map(len) == 0]

    # Remove mentions
    df = df[df["Tweet"].str.findall(r"@[\w]+").map(len) == 0]

    # Remove duplicated tweets
    df.drop_duplicates(inplace=True)

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

    return df


def preprocess_folder(folder: str, save_csv: bool = False) -> pd.DataFrame:
    merged_df = None

    for filename in os.listdir(folder):
        logging.info(f"Preprocessing {GRAY}{folder}/{filename}")
        df = preprocessing(os.path.join(folder, filename))

        merged_df = df if merged_df is None else pd.concat([merged_df, df])

    if save_csv:
        logging.info(f"Saving as {GRAY}{folder}.csv")
        merged_df.to_csv(f"{folder}.csv", index=False)

    print()

    return merged_df


def tokenization_and_pad(df_train: pd.DataFrame, df_eval: pd.DataFrame) -> None:
    tokenizer = Tokenizer()

    logging.info(f"Fit tokenizer in {GRAY}train dataset")
    tokenizer.fit_on_texts(df_train["Tweet"].values)

    logging.info(f"Tokenization and pad of {GRAY}train dataset")
    df_train["Tokens"] = pad_sequences(
        tokenizer.texts_to_sequences(df_train["Tweet"].values),
        maxlen=50,
        padding="post"
    ).tolist()

    logging.info(f"Tokenization and pad of {GRAY}eval dataset")
    df_eval["Tokens"] = pad_sequences(
        tokenizer.texts_to_sequences(df_eval["Tweet"].values),
        maxlen=50,
        padding="post"
    ).tolist()

    return


if __name__ == "__main__":
    df_train = preprocess_folder("train_tweets")
    df_eval = preprocess_folder("eval_tweets")

    tokenization_and_pad(df_train, df_eval)

    df_train.to_csv(
        "train_tweets.csv",
        columns=["ID", "Timestamp", "Tokens", "EventType"],
        index=False
    )

    df_eval.to_csv(
        "eval_tweets.csv",
        columns=["ID", "Timestamp", "Tokens"],
        index=False
    )
