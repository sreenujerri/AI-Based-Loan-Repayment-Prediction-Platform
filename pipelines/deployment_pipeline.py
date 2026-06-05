import json
import numpy as np
import pandas as pd

from zenml import pipeline, step
from pydantic import BaseModel
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.ingest_data import ingest_df
from steps.data_clean import clean_data
from steps.training_model import training_data
from steps.evaluation_model import evaluating_model
#from .utils import get_data_for_test


# -------------------------
# Docker Settings
# -------------------------
docker_settings = DockerSettings(required_integrations=[MLFLOW])


# -------------------------
# Data Loader
# -------------------------
@step(enable_cache=False)
def dynamic_importer() -> str:
    return get_data_for_test()


# -------------------------
# Deployment Config (FIXED)
# -------------------------
class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0.9


@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    return accuracy >= config.min_accuracy


# -------------------------
# MLflow Service Loader
# -------------------------
@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not services:
        raise RuntimeError("No MLflow deployment found")

    return services[0]


# -------------------------
# Predictor Step
# -------------------------
@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:

    service.start(timeout=10)

    payload = json.loads(data)

    payload.pop("columns", None)
    payload.pop("index", None)

    # LOAN FEATURES (must match training)
    columns = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
    ]

    df = pd.DataFrame(payload["data"], columns=columns)

    prediction = service.predict(df.values)

    return prediction


# -------------------------
# TRAIN + DEPLOY PIPELINE
# -------------------------
@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):

    # ✅ FIXED: pass path explicitly
    df = ingest_df(data_path="data/raw/LoanData.csv")

    x_train, x_test, y_train, y_test = clean_data(df)

    model = training_data(x_train, y_train)

    accuracy = evaluating_model(model, x_test, y_test)

    deploy = deployment_trigger(accuracy=accuracy)

    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deploy,
        workers=workers,
        timeout=timeout,
    )


# -------------------------
# INFERENCE PIPELINE
# -------------------------
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(
    pipeline_name: str,
    pipeline_step_name: str,
):

    batch_data = dynamic_importer()

    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=True,
    )

    predictor(service=service, data=batch_data)