from zenml.client import Client

from pipelines.training_pipeline import loan_Risk_pipeline
from steps.config import ModelNameConfig

if __name__ == "__main__":

    client = Client()

    print(
        client.active_stack.experiment_tracker.get_tracking_uri()
    )

    loan_Risk_pipeline(
        data="C:/Users/sreen/OneDrive/Desktop/Zenml/data/raw/LoanData.csv",
        config=ModelNameConfig()
    )