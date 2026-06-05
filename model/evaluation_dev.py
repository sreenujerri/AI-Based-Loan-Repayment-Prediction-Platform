import numpy as np
from abc import ABC, abstractmethod

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score
)


# -----------------------
# Abstract Class
# -----------------------
class Evaluation(ABC):

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


# -----------------------
# Accuracy
# -----------------------
class AccuracyClass(Evaluation):

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)


# -----------------------
# F1 Score
# -----------------------
class F1ScoreClass(Evaluation):

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(y_true, y_pred, average="weighted")


# -----------------------
# Recall
# -----------------------
class RecallClass(Evaluation):

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return recall_score(y_true, y_pred, average="weighted")


# -----------------------
# Precision
# -----------------------
class PrecisionClass(Evaluation):

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return precision_score(y_true, y_pred, average="weighted")