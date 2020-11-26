import pandas as pd


def read_raw_file(path, file):
    return pd.read_csv(f"{path}/{file}", index_col=0)


def save_dataset(df, path, file):
    df.to_csv(f"{path}/{file}", index=False, encoding="utf-8")
