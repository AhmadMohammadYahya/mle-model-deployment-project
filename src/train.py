
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import load_data, clean_data
from src.features import prepare_features


NUMERICAL_FEATURES = [
    "passenger_count",
    "pickup_hour_sin",
    "pickup_hour_cos",
]

CATEGORICAL_FEATURES = [
    "PULocationID",
    "DOLocationID",
    "pickup_day_of_week",
]


def train_model(X, y):
    """Train and evaluate the Random Forest model."""

    sample_size = min(200_000, len(X))

    X = X.sample(n=sample_size, random_state=42)
    y = y.loc[X.index]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True,
        )),
    ])

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=30,
            max_depth=18,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Train the model.
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)

    # Calculate the mean baseline.
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_val)

    baseline_rmse = np.sqrt(
        mean_squared_error(y_val, baseline_predictions)
    )

    # Log the experiment using MLflow.
    with mlflow.start_run() as run:

        mlflow.log_params({
            "model": "RandomForestRegressor",
            "n_estimators": 30,
            "max_depth": 18,
            "min_samples_leaf": 5,
            "random_state": 42,
            "sample_size": sample_size,
            "test_size": 0.2,
            "imputation_strategy": "median",
            "onehot_encoding": True,
        })

        mlflow.log_metrics({
            "rmse": float(rmse),
            "mae": float(mae),
            "baseline_rmse": float(baseline_rmse),
        })

        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_pipeline",
            input_example=X_val.head(5),
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        print("MLflow Run ID:", run.info.run_id)

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Random Forest RMSE: {rmse:.2f}")
    print(f"Random Forest MAE: {mae:.2f}")
    print(f"Mean baseline RMSE: {baseline_rmse:.2f}")

    return model


def main():

    # Configure local MLflow tracking.
   #mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment("NYC-Taxi-Trip-Duration")

    df = load_data()
    df_clean = clean_data(df)

    X, y = prepare_features(df_clean)

    train_model(X, y)

    print("\nTraining and MLflow logging completed.")


if __name__ == "__main__":
    main()
