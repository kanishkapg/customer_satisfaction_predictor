import logging
import pandas as pd
from zenml import step

class IngestData:
    # Ingesting data from the data_path

    def __init__(self, data_path: str):
        self.path = data_path
    
    def get_data(self):
        logging.info(f'Reading data from {self.path}')  # Use self.path here
        return pd.read_csv(self.path)  # Use self.path here
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting data from the data_path

    Args:
        data_path: path to the data

    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_df = IngestData(data_path)
        df = ingest_df.get_data()
        return df
    except Exception as e:
        logging.error(f'Failed to ingest data: {e}')
        raise e