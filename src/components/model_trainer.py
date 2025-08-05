import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import setup_logging, logging

from src.utils import save_object
from src.utils import evaluate_models

setup_logging()

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting training and test input data.')
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:,:-1],
                test_arr[:, -1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Classifier': KNeighborsRegressor(),
                'XGBClassifier': XGBRegressor(),
                'CatBoosting Classifier': CatBoostRegressor(),
                'AdaBoost Classifier': AdaBoostRegressor()
            }

            model_report: dict = evaluate_models(X_train = X_train, y_train = y_train,X_test = X_test, y_test = y_test, models = models)

            #To get best model score from the model_report dict.
            best_model_score = max(sorted(model_report.values()))

            #To get best model name from the model_report dict.
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = model_report[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found.')
            logging.info('Best found model on train and test sets.')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

        except Exception as e:
            raise CustomException(e, sys)
