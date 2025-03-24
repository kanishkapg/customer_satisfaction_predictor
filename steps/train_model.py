import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    model_name: str = "LinearRegression",  # Pass model_name directly
) -> RegressorMixin:
    
    try:
        if model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {model_name} not supported")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
