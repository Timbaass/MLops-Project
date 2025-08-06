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
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoosting Regressor': CatBoostRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report: dict = evaluate_models(
                X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, params = params
            )

            #To get best model score from the model_report dict.
            best_model_score = max(sorted(model_report.values(), key = lambda x: x[0]))[0]

            #To get best model name from the model_report dict.
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(max(model_report.values(), key = lambda x : x[0]))
            ]
            #Get best model object.
            best_model = models[best_model_name]

            #Apply the hyperparameters to the model.
            best_params = model_report[best_model_name][1]
            best_model.set_params(**best_params)
            best_model.fit(X_train, y_train)

            if best_model_score < 0.6:
                raise CustomException('No best model found.')
            logging.info('Best found model on train and test sets.')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            y_pred = best_model.predict(X_test)

            r2_score_value = r2_score(y_test, y_pred)

            return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)
