import sys
import os

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import setup_logging, logging

from dataclasses import dataclass

from src.utils import save_object

setup_logging()

@dataclass
class DataTransformationConfig:
    preproccesor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''This function returns a data transformer object to prepare the data for training.'''
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Categorical columns encoding completed.')
            logging.info('Numerical columns standard scaling completed.')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e: 
            raise CustomException(e, sys)
        
    def initiate_data_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test sets converted to a DataFrame from csv.")

            logging.info('Obtaining preprocessing object.')

            preprocessing_object = self.get_data_transformer_object()

            target_column_name = 'math_score'

            input_feature_train_df = train_df.drop(target_column_name, axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(target_column_name, axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Appliying preprocessing object on train and test sets.')

            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info('Saved preprocessing object.')

            save_object(
                file_path = self.data_transformation_config.preproccesor_obj_file_path,
                obj = preprocessing_object
            )

            return (
                train_arr,
                test_arr
            )
                 
        except Exception as e:
            raise CustomException(e, sys)