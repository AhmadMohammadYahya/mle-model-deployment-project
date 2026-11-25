from pathlib import Path

from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI

from app.data_model import PredictionRequest, PredictionResponse


'''
# Connect to the local MLflow database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load the trained model
MODEL_URI = (
    "runs:/957f5b5bebb34115821871db28e16628/"
    "random_forest_pipeline"
)

model = mlflow.sklearn.load_model(MODEL_URI)
'''


# Load the trained model from the local models directory
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "random_forest_pipeline"
)

model = mlflow.sklearn.load_model(str(MODEL_PATH))

'''

# Connect to the local MLflow tracking database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load the currently approved production model
MODEL_URI = "models:/NYC-Taxi-Predictor@champion"

model = mlflow.sklearn.load_model(MODEL_URI)
'''




app = FastAPI(
    title="NYC Taxi Trip Duration Prediction API",
    description="Predicts taxi trip duration in minutes.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {"message": "NYC Taxi Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    pickup_datetime = request.pickup_datetime
    pickup_hour = pickup_datetime.hour

    # Prepare the same features used during model training
    input_data = pd.DataFrame([{
        "passenger_count": request.passenger_count,
        "PULocationID": request.PULocationID,
        "DOLocationID": request.DOLocationID,
        "pickup_hour_sin": np.sin(2 * np.pi * pickup_hour / 24),
        "pickup_hour_cos": np.cos(2 * np.pi * pickup_hour / 24),
        "pickup_day_of_week": pickup_datetime.weekday(),
    }])

    prediction = model.predict(input_data)[0]

    return PredictionResponse(
        predicted_duration=round(float(prediction), 2)
    )
