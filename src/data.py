
from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "yellow_tripdata_2025-01.parquet"
)


def load_data(path=DATA_PATH):
    """Load the January 2025 Yellow Taxi dataset."""

    columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
    ]

    return pd.read_parquet(path, columns=columns)


def clean_data(df):
    """Filter invalid trips and create the duration target."""

    duration = (
        df["tpep_dropoff_datetime"]
        - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    january = (
        (df["tpep_pickup_datetime"] >= "2025-01-01")
        & (df["tpep_pickup_datetime"] < "2025-02-01")
    )

    valid_duration = (duration > 0) & (duration <= 360)

    df_clean = df.loc[january & valid_duration].copy()

    df_clean["duration"] = duration.loc[df_clean.index]

    return df_clean
