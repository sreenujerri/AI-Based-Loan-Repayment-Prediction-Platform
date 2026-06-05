from typing import cast

import click
from rich import print

from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService


# -----------------------
# Constants
# -----------------------
DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

PIPELINE_NAME = "continuous_deployment_pipeline"
MODEL_NAME = "model"
STEP_NAME = "mlflow_model_deployer_step"


# -----------------------
# CLI
# -----------------------
@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Choose mode: deploy, predict, or deploy_and_predict",
)
@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy required to deploy the model",
)
def main(config: str, min_accuracy: float):

    mlflow_model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    deploy = config in [DEPLOY, DEPLOY_AND_PREDICT]
    predict = config in [PREDICT, DEPLOY_AND_PREDICT]

    # -----------------------
    # Deploy Pipeline
    # -----------------------
    if deploy:
        continuous_deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60,
        )

    # -----------------------
    # Inference Pipeline
    # -----------------------
    if predict:
        inference_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_step_name=STEP_NAME,
        )

    # -----------------------
    # MLflow Info
    # -----------------------
    print(
        f"\nYou can run:\n"
        f"[green]mlflow ui --backend-store-uri '{get_tracking_uri()}'[/green]\n"
        f"to inspect your experiment runs.\n"
    )

    # -----------------------
    # Get deployed service
    # -----------------------
    existing_services = mlflow_model_deployer.find_model_server(
        pipeline_name=PIPELINE_NAME,
        pipeline_step_name=STEP_NAME,
        model_name=MODEL_NAME,
    )

    if not existing_services:
        print(
            "No MLflow prediction server is currently running.\n"
            "Run with '--config deploy' first."
        )
        return

    service = cast(MLFlowDeploymentService, existing_services[0])

    # -----------------------
    # Service Status
    # -----------------------
    if service.is_running:
        print(
            f"The MLflow prediction server is running at:\n"
            f"[green]{service.prediction_url}[/green]\n\n"
            f"To stop it, run:\n"
            f"[yellow]zenml model-deployer models delete {service.uuid}[/yellow]"
        )

    elif service.is_failed:
        print(
            f"The MLflow prediction server failed.\n"
            f"State: {service.status.state.value}\n"
            f"Error: {service.status.last_error}"
        )


# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    main()