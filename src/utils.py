import os 
import sys 

import numpy as np
import pandas as pd

from src.exception import CustomException
import dill 

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    model_report = {}
    for model_name, model in models.items():
        param = params.get(model_name, {})  # Get parameters, default to empty dict
        try:
            gs = GridSearchCV(model, param, cv=5, scoring='r2', n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)

            model_report[model_name] = [score, gs.best_params_]
        except Exception as e:
            raise CustomException(e, sys)
    return model_report