import os
import sys

import joblib
import pandas as pd

from src.exception import CustomException
from src.logger import logger


class PredictPipeline:

    def __init__(self):

        self.model_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

        self.preprocessor_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

    def predict(self, input_data):

        try:

            logger.info("Prediction started")

            # Load trained GridSearchCV model
            gridcv = joblib.load(
                self.model_path
            )

            logger.info(
                "Model loaded successfully"
            )

            # Load preprocessor
            preprocessor = joblib.load(
                self.preprocessor_path
            )

            logger.info(
                "Preprocessor loaded successfully"
            )

            # Convert input into DataFrame
            if isinstance(input_data, dict):

                input_data = pd.DataFrame(
                    [input_data]
                )

            # Remove CustomerID if provided
            if "CustomerID" in input_data.columns:

                input_data = input_data.drop(
                    columns=["CustomerID"]
                )

            # Apply same preprocessing
            transformed_data = (
                preprocessor.transform(
                    input_data
                )
            )

            # Prediction
            prediction = gridcv.predict(
                transformed_data
            )

            logger.info(
                "Prediction completed successfully"
            )

            return prediction[0]

        except Exception as e:

            logger.error(
                "Error occurred during prediction"
            )

            raise CustomException(
                e,
                sys
            )