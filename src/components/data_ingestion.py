import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

@dataclass
class Ingestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=Ingestionconfig()

    def start_data_ingestion(self):
        logging.info('Data ingestion methods Starts')
        try:
            df=pd.read_csv(os.path.join('notebooks/data','insurance.csv'))
            logging.info('Failed to load the pandas dataframe csv file')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('train_test_split')
            train_set,test_set=train_test_split(df,test_size=0.30)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data ingestion is Completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured the data Ingestion stage')
            raise CustomException(e,sys)
        


