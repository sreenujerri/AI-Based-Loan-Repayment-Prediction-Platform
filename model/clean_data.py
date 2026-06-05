from abc import ABC, abstractmethod
import logging
import pandas as pd

from typing import Any, Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ---------------- Strategy Base ----------------
class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, data: Any):
        pass


# ---------------- Preprocessing ----------------
class PreprocessStrategy(DataStrategy):

    def handle_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:

        try:
            logging.info("Starting preprocessing")

            data = data.copy()

            # Feature engineering
            data["Income"] = (
                data["ApplicantIncome"] +
                data["CoapplicantIncome"]
            )

            # Drop unnecessary columns
            data.drop(
                ["Loan_ID", "ApplicantIncome", "CoapplicantIncome"],
                axis=1,
                inplace=True
            )

            # Handle Dependents column
            if "Dependents" in data.columns:
                data["Dependents"] = (
                    data["Dependents"]
                    .replace("3+", 3)
                    .astype(float)
                )

            # Split features and target
            X = data.drop("Loan_Status", axis=1)
            y = data["Loan_Status"].map({"Y": 1, "N": 0})

            # Column selection
            cat_cols = X.select_dtypes(include=["object"]).columns
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns

            # Pipelines
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ])

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ("cat", cat_pipeline, cat_cols),
                ("num", num_pipeline, num_cols)
            ])

            logging.info("Preprocessing completed")

            return X, y, preprocessor

        except Exception:
            logging.exception("Error in preprocessing")
            raise


# ---------------- Train-Test Split ----------------
class DataDivideStrategy(DataStrategy):

    def handle_data(self, data):

        try:
            X, y, preprocessor = data

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            # Fit on train only
            X_train = preprocessor.fit_transform(X_train)

            # Transform test
            X_test = preprocessor.transform(X_test)

            logging.info("Data splitting completed")

            return X_train, X_test, y_train, y_test

        except Exception:
            logging.exception("Error in data splitting")
            raise


# ---------------- Context Class ----------------
class DataCleaning:

    def __init__(self, data, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self):

        try:
            return self.strategy.handle_data(self.data)

        except Exception:
            logging.exception("Error in DataCleaning")
            raise