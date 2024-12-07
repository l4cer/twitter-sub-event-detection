import os

import pandas as pd

from typing import Union


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

    # Remove punctuation and special characters
    df["Tweet"] = df["Tweet"].str.replace(r"[^\w\s]+", " ", regex=True)

    # Remove multiple spaces and strip tweets
    df["Tweet"] = df["Tweet"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Lowercase all remaining tweets
    df["Tweet"] = df["Tweet"].str.lower()

    # Tokenization
    df["Tokens"] = df["Tweet"].str.split()

    return df


def preprocess_folder(folder: str, save_csv: bool = False) -> pd.DataFrame:
    merged_df = None

    for filename in os.listdir(folder):
        df = preprocessing(os.path.join(folder, filename))

        merged_df = df if merged_df is None else pd.concat([merged_df, df])

    if save_csv:
        columns = ["ID", "MatchID", "PeriodID", "Timestamp", "Tokens"]
        merged_df[columns].to_csv(f"{folder}.csv")

    return merged_df


if __name__ == "__main__":
    df = preprocess_folder("eval_tweets", save_csv=True)
    print(df)
