
import numpy as np
import pandas as pd


def prepare_features(df):
    """Create model features and target."""

    pickup_hour = df["tpep_pickup_datetime"].dt.hour

    X = df[
        ["passenger_count", "PULocationID", "DOLocationID"]
    ].copy()

    X["pickup_hour_sin"] = np.sin(
        2 * np.pi * pickup_hour / 24
    )

    X["pickup_hour_cos"] = np.cos(
        2 * np.pi * pickup_hour / 24
    )

    X["pickup_day_of_week"] = (
        df["tpep_pickup_datetime"].dt.day_of_week
    )

    y = df["duration"]

    return X, y
