import logging
import pandas as pd
from zenml import step


class IngestData:
    """
    Data ingestion class that loads data from a CSV file.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)


@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Args:
        data_path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()

        logging.info("Data ingestion completed successfully.")
        return df

    except Exception as e:
        logging.exception("Error while ingesting data")
        raise