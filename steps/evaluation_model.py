import logging
import pandas as pd
import mlflow

from numpy import ndarray
from zenml import step
from sklearn.base import ClassifierMixin

from typing import Tuple
from typing_extensions import Annotated

from model.evaluation_dev import (
    AccuracyClass,
    F1ScoreClass,
    RecallClass,
    PrecisionClass
)

from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluating_model(
    model: ClassifierMixin,
    x_test: ndarray,
    y_test: pd.Series
) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "f1_score"],
    Annotated[float, "recall"],
    Annotated[float, "precision"]
]:

    try:

        prediction = model.predict(x_test)

        acc = AccuracyClass().calculate_score(
            y_test,
            prediction
        )

        f1 = F1ScoreClass().calculate_score(
            y_test,
            prediction
        )

        recall = RecallClass().calculate_score(
            y_test,
            prediction
        )

        precision = PrecisionClass().calculate_score(
            y_test,
            prediction
        )

        # -------------------------
        # Log Metrics
        # -------------------------
        mlflow.log_metric(
            "accuracy",
            acc
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        logging.info(
            "Evaluation completed successfully"
        )

        return (
            acc,
            f1,
            recall,
            precision
        )

    except Exception:

        logging.exception(
            "Error in evaluating model"
        )

        raise