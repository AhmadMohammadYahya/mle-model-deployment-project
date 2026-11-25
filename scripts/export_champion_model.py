
from pathlib import Path
import shutil
import tempfile

import mlflow
from mlflow.tracking import MlflowClient


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "NYC-Taxi-Predictor"
MODEL_ALIAS = "champion"

MODELS_DIR = PROJECT_ROOT / "models"
DEPLOYMENT_MODEL_DIR = MODELS_DIR / "random_forest_pipeline"


def export_champion_model():
    """Export the currently approved MLflow model for deployment."""

    # Use the project's existing MLflow database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    client = MlflowClient()

    # Find the model assigned to the champion alias
    champion = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    print("Champion model found:")
    print("Model:", champion.name)
    print("Version:", champion.version)
    print("Run ID:", champion.run_id)

    # Resolve the registered model URI
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    # Download the selected model artifacts
    downloaded_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri
    )

    print("\nDownloaded model from MLflow.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare a temporary directory for the new model
    with tempfile.TemporaryDirectory(dir=MODELS_DIR) as temp_dir:

        staging_dir = Path(temp_dir) / "random_forest_pipeline"

        shutil.copytree(downloaded_path, staging_dir)

        # Keep the existing deployment model as a backup
        backup_dir = MODELS_DIR / "random_forest_pipeline_backup"

        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        if DEPLOYMENT_MODEL_DIR.exists():
            DEPLOYMENT_MODEL_DIR.rename(backup_dir)

        try:
            # Promote the downloaded model to the deployment directory
            staging_dir.rename(DEPLOYMENT_MODEL_DIR)

        except Exception:
            # Restore the previous model if promotion fails
            if backup_dir.exists():
                backup_dir.rename(DEPLOYMENT_MODEL_DIR)
            raise

        # Remove the backup after successful promotion
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

    print("\nModel exported successfully!")
    print("Deployment path:", DEPLOYMENT_MODEL_DIR)


if __name__ == "__main__":
    export_champion_model()
