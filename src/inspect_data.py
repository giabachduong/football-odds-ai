
from pathlib import Path

import pandas as pd


def main():
    """Load and inspect the historical Premier League dataset."""
    data_path = Path("data/raw/epl_2526.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)

    # Clean accidental whitespace from column names.
    df.columns = df.columns.str.strip()

    print("\n=== Dataset shape ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n=== Column names ===")
    print(df.columns.tolist())

    print("\n=== First 5 matches ===")
    print(df.head())

    print("\n=== Missing values in important columns ===")
    important_columns = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
    ]

    available_columns = [
        col for col in important_columns if col in df.columns
    ]

    print(df[available_columns].isna().sum())

    print("\n=== Data types ===")
    print(df[available_columns].dtypes)


if __name__ == "__main__":
    main()
