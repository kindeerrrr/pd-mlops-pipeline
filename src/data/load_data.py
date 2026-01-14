import pandas as pd
from src.data.validation import validate_data


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower()
    return df


if __name__ == "__main__":
    df = load_data("data/raw/default_credit.csv")
    df = clean_data(df)
    df = validate_data(df)
    df.to_csv("data/processed/clean_data.csv", index=False)