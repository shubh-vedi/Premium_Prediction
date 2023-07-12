import numpy as np 
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

import sys
import os
from dataclasses import dataclass

@dataclass
class Model_Training_config:
    train_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = Model_Training_config()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('split independent and dependent variable inti train and test data')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info('Model Reportstarted')
            models = {

                'linear_Regression': LinearRegression(),
                'DecisionTree' : DecisionTreeRegressor(),
                'Random_forest' : RandomForestRegressor(),
            }

            logging.info('completed model fit')
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            logging.info('modelrepoetdxsd')
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            #logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.train_model_file_path,
                 obj=best_model
            )

        except Exception as e:
            logging.info('Exception occur in trainingmodel')
            raise CustomException(e,sys)
            
    

