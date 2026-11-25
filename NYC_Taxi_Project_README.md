# NYC Taxi Trip Duration Prediction

A local-first machine learning project that predicts the duration of New York City Yellow Taxi trips using the January 2025 TLC trip-record dataset. The project covers data preparation, model training, experiment tracking with MLflow, model registration and export, a FastAPI prediction service, automated API tests, and optional containerized deployment with Docker.

## Project objective

Train and locally deploy a `RandomForestRegressor` that predicts taxi trip duration in minutes.

The main technologies are:

- **Python 3.13**
- **pandas and NumPy** for data preparation and feature engineering
- **scikit-learn** for preprocessing, baseline modelling, and Random Forest training
- **MLflow** for experiment tracking and model registration
- **FastAPI and Pydantic** for the prediction API and request validation
- **pytest and HTTPX** for API tests
- **uv** for Python environment and dependency management
- **Docker** for containerizing the API

## Dataset

The project uses the January 2025 NYC Yellow Taxi trip records published by the New York City Taxi and Limousine Commission (TLC).

- Dataset: [January 2025 Yellow Taxi trip records](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet)
- Source and other TLC datasets: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Prediction target: `duration`, measured in minutes

The raw Parquet file contains **3,475,226 rows**. The cleaning step retains **3,471,950 trips**.

### Data cleaning

The target is calculated from the difference between the drop-off and pickup timestamps:

```python
duration = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60
```

The cleaning step:

1. Keeps trips with pickup timestamps in January 2025.
2. Removes trips with a duration less than or equal to zero.
3. Removes trips with a duration greater than 360 minutes.
4. Creates the `duration` target column in minutes.

The raw dataset is not modified. The current data-loading module reads only the columns needed for this task.

## Feature engineering

The model uses the following six features:

| Feature | Description |
|---|---|
| `passenger_count` | Number of passengers; missing values are retained and imputed by the preprocessing pipeline |
| `PULocationID` | Pickup taxi-zone identifier |
| `DOLocationID` | Drop-off taxi-zone identifier |
| `pickup_hour_sin` | Sine transformation of the pickup hour |
| `pickup_hour_cos` | Cosine transformation of the pickup hour |
| `pickup_day_of_week` | Day of the week, where Monday is 0 and Sunday is 6 |

Pickup hour is represented with sine and cosine transformations so that hours near midnight are treated as close to one another:

```python
pickup_hour_sin = np.sin(2 * np.pi * pickup_hour / 24)
pickup_hour_cos = np.cos(2 * np.pi * pickup_hour / 24)
```

The model is designed around information that can be available when a trip is requested. Actual trip distance, drop-off time, fare, and payment information are not used as model inputs.

## Model training and evaluation

The training workflow is implemented in `src/train.py` and uses a scikit-learn `Pipeline` so that preprocessing and model inference stay together.

### Validation setup

- Train/validation split: **80% / 20%**
- Random state: **42**
- Training sample: up to **200,000 rows**
- Actual run: **160,000 training rows** and **40,000 validation rows**

### Preprocessing

- Numeric features use median imputation with missing-value indicators.
- Categorical features use `OneHotEncoder(handle_unknown="ignore")`.
- A `ColumnTransformer` applies the appropriate preprocessing to each feature group.

### Models

A `DummyRegressor` using the mean strategy is used as a simple baseline. The main model is a `RandomForestRegressor` with the following settings:

| Parameter | Value |
|---|---:|
| `n_estimators` | 30 |
| `max_depth` | 18 |
| `min_samples_leaf` | 5 |
| `random_state` | 42 |
| `n_jobs` | -1 |

### Validation results

The recorded validation run produced:

| Model | RMSE | MAE |
|---|---:|---:|
| Mean baseline (`DummyRegressor`) | 11.51 min RMSE | — |
| Random Forest | **9.06 min** | **6.20 min** |

The Random Forest reduced RMSE by approximately **21.3%** compared with the mean baseline.

RMSE is the primary comparison metric. MAE is also reported to give an intuitive view of the average absolute prediction error. These are validation results from the sampled dataset, not a guarantee of performance on every future trip.

## MLflow experiment tracking and model registry

Training runs are tracked locally with MLflow. The project uses a SQLite tracking database:

```text
sqlite:///mlflow.db
```

The experiment is named:

```text
NYC-Taxi-Trip-Duration
```

The trained model is logged with its preprocessing pipeline and registered in the MLflow Model Registry as:

```text
NYC-Taxi-Predictor
```

The selected model version is assigned the `champion` alias. The export script, `scripts/export_champion_model.py`, downloads the model referenced by that alias and exports it to:

```text
models/random_forest_pipeline/
```

This separates model management from API serving: the API loads the exported model directory and does not need to contact the MLflow tracking server when it starts.

### Start the MLflow UI

Run from the project root:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Export the current champion model

After setting or updating the `champion` alias in MLflow, run:

```powershell
uv run python scripts/export_champion_model.py
```

The exported model directory is used by both the local API and the Docker image. Run the export script again after promoting a different champion model.

## FastAPI prediction service

The API is implemented in `app/main.py`. Request and response schemas are defined in `app/data_model.py`.

### Available endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Returns an API welcome/status message |
| `GET` | `/health` | Returns the health status |
| `POST` | `/predict` | Predicts the duration of one trip |

Interactive API documentation is available at `/docs`.

### Run locally

From the project root:

```powershell
uv sync
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Example request

Send a `POST` request to `/predict` with JSON:

```json
{
  "pickup_datetime": "2025-01-15T14:30:00",
  "PULocationID": 161,
  "DOLocationID": 236,
  "passenger_count": 2
}
```

Example response structure:

```json
{
  "predicted_duration": 12.52
}
```

The prediction is expressed in minutes. The exact value depends on the exported model version.

`passenger_count` may also be `null`. The preprocessing pipeline imputes the missing value. One tested request with `passenger_count: null` returned:

```json
{
  "predicted_duration": 15.29
}
```

### Request validation

Pydantic validates the incoming request. For example, pickup and drop-off location IDs must be positive integers. Invalid input is rejected with an HTTP `422` validation response.

## Automated tests

The API tests are in `tests/test_app.py`. They cover:

- `GET /` returns the expected message.
- `GET /health` returns the healthy status.
- `POST /predict` accepts a valid request and returns a prediction response. The model prediction is mocked in this unit test.
- `POST /predict` rejects an invalid location ID.

Run the tests from the project root:

```powershell
uv run python -m pytest
```

The latest recorded test run completed with **4 passed**.

## Docker deployment

Docker packages the FastAPI application together with its exported model. The image does not need the raw training dataset or the local MLflow tracking database to serve predictions.

### Build the image

Run from the project root:

```powershell
docker build -t nyc-taxi-api:1.0 .
```

### Start the container

```powershell
docker run -d --name nyc-taxi-api -p 8000:8000 nyc-taxi-api:1.0
```

The API is then available at:

- Health endpoint: [http://localhost:8000/health](http://localhost:8000/health)
- Interactive documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### Container management

View the running container:

```powershell
docker ps
```

View application logs:

```powershell
docker logs nyc-taxi-api
```

Stop and remove the container:

```powershell
docker stop nyc-taxi-api
docker rm nyc-taxi-api
```

To rebuild after changing the application or exported model, stop and remove the old container, rebuild the image, and start a new container.

### Docker implementation notes

The Dockerfile:

- Uses `python:3.13.17-slim`.
- Copies `uv` from the official `ghcr.io/astral-sh/uv:0.13.6` image.
- Installs locked production dependencies using `uv sync --frozen --no-dev --no-install-project`.
- Copies the FastAPI application and exported model into the image.
- Starts Uvicorn on `0.0.0.0:8000` inside the container.

The `.dockerignore` file excludes local virtual environments, raw data, MLflow tracking files, tests, notebooks, and local development files from the Docker build context.

## Repository structure

```text
mle-model-deployment-project/
├── app/
│   ├── __init__.py
│   ├── data_model.py
│   └── main.py
├── data/
│   └── raw/
│       └── yellow_tripdata_2025-01.parquet
├── models/
│   └── random_forest_pipeline/
├── scripts/
│   └── export_champion_model.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   └── test_app.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

The raw dataset, local MLflow database and run artifacts, virtual environment, and other generated files should remain local and should not be committed to Git. The exported model directory is required when building the Docker image; decide whether to commit it or provide a reproducible export step based on the repository's size and submission requirements.

## Project limitations and possible improvements

The current model is a practical baseline for a local deployment exercise. Possible improvements include:

1. **Hyperparameter tuning:** use `GridSearchCV` or Optuna to test whether a better Random Forest configuration improves validation performance.
2. **Model comparison:** compare the Random Forest with another suitable regression model using the same validation split and metrics.
3. **More robust evaluation:** evaluate on a time-based holdout period and check performance across different pickup zones and times of day.
4. **Feature improvements:** investigate additional features that are genuinely available when a trip is requested, without introducing target leakage.
5. **Monitoring:** record prediction activity and monitor changes in input data and model performance when ground-truth trip durations become available.
6. **Deployment hardening:** add structured logging, configuration management, container health checks, and a production deployment strategy.

## Assignment questions

### 1. What is the RMSE of the final model?

The final recorded Random Forest validation RMSE is **9.06 minutes**. The mean baseline RMSE is 11.51 minutes, so the Random Forest reduces RMSE by approximately 21.3%.

### 2. What would you do differently if you had more time?

I would perform systematic hyperparameter tuning and compare the Random Forest with other regression algorithms. I would also evaluate the model on a time-based holdout dataset to better understand how well it generalizes to later trips. Finally, I would investigate additional booking-time features and add monitoring for prediction inputs and model performance.

## Reproducibility notes

- Run commands from the repository root unless stated otherwise.
- Use the Python version specified by `.python-version` and the locked dependencies in `uv.lock`.
- The API expects the exported model at `models/random_forest_pipeline/`.
- If the MLflow champion alias changes, export the champion model again before rebuilding the Docker image.
- The model metrics documented above are from the recorded training run; retraining may produce different values if the data, sample, split, or dependencies change.
