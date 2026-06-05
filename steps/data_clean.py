import logging
import pandas as pd
import numpy as np

from zenml import step
from typing import Tuple
from typing_extensions import Annotated

from model.clean_data import (
    PreprocessStrategy,
    DataDivideStrategy,
    DataCleaning
)


@step
def clean_data(
    df: pd.DataFrame
) -> Tuple[
    Annotated[np.ndarray, "x_train"],
    Annotated[np.ndarray, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:

    try:
        # Preprocessing
        process = DataCleaning(
            data=df,
            strategy=PreprocessStrategy()
        )

        processed_data = process.handle_data()

        # Train-test split
        divide = DataCleaning(
            data=processed_data,
            strategy=DataDivideStrategy()
        )

        x_train, x_test, y_train, y_test = divide.handle_data()

        logging.info("Data cleaning completed successfully")

        return x_train, x_test, y_train, y_test

    except Exception:
        logging.exception("Error in clean_data step")
        raise