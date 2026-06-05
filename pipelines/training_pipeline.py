from zenml.pipelines import pipeline

from steps.ingest_data import ingest_df
from steps.data_clean import clean_data
from steps.training_model import training_data
from steps.evaluation_model import evaluating_model
from steps.config import ModelNameConfig

from zenml import pipeline

@pipeline(enable_cache=False)
def loan_Risk_pipeline(data: str, config: ModelNameConfig):

    df = ingest_df(data)

    clean_outputs = clean_data(df)

    x_train = clean_outputs[0]
    x_test = clean_outputs[1]
    y_train = clean_outputs[2]
    y_test = clean_outputs[3]

    model = training_data(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        config=config,
    )

    evaluating_model(model, x_test, y_test)
