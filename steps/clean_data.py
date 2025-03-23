import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataPreProcessStrategy, DataDivideStrategy, DataCleaning
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> None:
    try:
        process_startegy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_startegy)
        processed_data = data_cleaning.handle_data()
        
        divide_startegy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_startegy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed successfully")
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e
