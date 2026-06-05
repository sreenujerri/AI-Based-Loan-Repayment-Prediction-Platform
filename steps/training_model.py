import logging
import mlflow
import mlflow.sklearn
import pandas as pd

from numpy import ndarray
from sklearn.base import ClassifierMixin
from zenml import step
from zenml.client import Client

from steps.config import ModelNameConfig
from model.model_dev import (
    HyperparameterTuner,
    RandomForestModel,
    AdaBoostModel,
    LogisticRegressionModel,
)

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def training_data(
    x_train: ndarray,
    x_test: ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> ClassifierMixin:

    try:
        model = None

        # -------------------------
        # Model Selection
        # -------------------------
        if config.model_name == "RandomForest":
            model = RandomForestModel()

        elif config.model_name == "AdaBoost":
            model = AdaBoostModel()

        elif config.model_name == "LogisticRegression":
            model = LogisticRegressionModel()

        else:
            raise ValueError("Model name not supported")

        # -------------------------
        # MLflow Logging: Model Name
        # -------------------------
        mlflow.log_param("model_name", config.model_name)

        # -------------------------
        # Hyperparameter Tuning
        # -------------------------
        tuner = HyperparameterTuner(
            model,
            x_train,
            y_train,
            x_test,
            y_test
        )

        best_params = {}

        if config.fine_tuning:
            best_params = tuner.optimize()
            mlflow.log_params(best_params)

            trained_model = model.train(
                x_train,
                y_train,
                **best_params
            )
        else:
            trained_model = model.train(
                x_train,
                y_train
            )

        # -------------------------
        # Optional: Log metrics (if you compute them)
        # -------------------------
        # Example:
        # accuracy = trained_model.score(x_test, y_test)
        # mlflow.log_metric("accuracy", accuracy)

        # -------------------------
        # Log Model to MLflow
        # -------------------------
        mlflow.sklearn.log_model(
            trained_model,
            artifact_path="model_customer"
        )

        logging.info("Model training completed successfully")

        return trained_model

    except Exception as e:
        logging.error(e)
        raise e