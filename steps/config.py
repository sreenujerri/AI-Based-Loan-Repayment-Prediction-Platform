from typing import Literal
from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    model_name: Literal["RandomForest", "AdaBoost", "Linear"] = "RandomForest"
    fine_tuning: bool = True