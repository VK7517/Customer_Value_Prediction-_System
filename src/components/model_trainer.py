import os
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


class ModelTrainer:

    def __init__(self):

        self.model_path = os.path.join(
            "models",
            "final_model.pkl"
        )

        self.preprocessor_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

        self.metrics_path = os.path.join(
            "artifacts",
            "model_metrics.csv"
        )

    def train_models(
        self,
        customer_features,
        preprocessor
    ):

        try:

            logger.info("Starting model training")

            # -----------------------------
            # Separate X and y
            # -----------------------------

            X = customer_features.drop(
                columns=[
                    "CustomerID",
                    "future_revenue"
                ]
            )

            y = customer_features[
                "future_revenue"
            ]

            # -----------------------------
            # Train/Test Split
            # -----------------------------

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42
                )
            )

            logger.info(
                f"Training data shape: {X_train.shape}"
            )

            logger.info(
                f"Testing data shape: {X_test.shape}"
            )

            # -----------------------------
            # Preprocessing
            # -----------------------------

            X_train_transformed = (
                preprocessor.fit_transform(X_train)
            )

            X_test_transformed = (
                preprocessor.transform(X_test)
            )

            # -----------------------------
            # Models
            # -----------------------------

            models = {

                "Linear Regression":
                    LinearRegression(),

                "Ridge":
                    Ridge(alpha=1.0),

                "Decision Tree":
                    DecisionTreeRegressor(
                        random_state=42
                    ),

                "Random Forest":
                    RandomForestRegressor(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1
                    ),

                "Gradient Boosting":
                    GradientBoostingRegressor(
                        random_state=42
                    ),

                "XGBoost":
                    XGBRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=6,
                        random_state=42,
                        n_jobs=-1,
                        objective="reg:squarederror"
                    )
            }

            results = []

            best_model = None
            best_model_name = None
            best_r2 = -np.inf

            # -----------------------------
            # Train & Evaluate
            # -----------------------------

            for model_name, model in models.items():

                logger.info(
                    f"Training {model_name}"
                )

                model.fit(
                    X_train_transformed,
                    y_train
                )

                predictions = model.predict(
                    X_test_transformed
                )

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                )

                r2 = r2_score(
                    y_test,
                    predictions
                )

                results.append({
                    "Model": model_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                })

                logger.info(
                    f"{model_name} | "
                    f"MAE={mae:.4f} | "
                    f"RMSE={rmse:.4f} | "
                    f"R2={r2:.4f}"
                )

                # Higher R2 is better
                if r2 > best_r2:

                    best_r2 = r2
                    best_model = model
                    best_model_name = model_name

            # -----------------------------
            # Model Comparison
            # -----------------------------

            results_df = pd.DataFrame(
                results
            )

            results_df = results_df.sort_values(
                by="R2",
                ascending=False
            )

            os.makedirs(
                os.path.dirname(
                    self.metrics_path
                ),
                exist_ok=True
            )

            results_df.to_csv(
                self.metrics_path,
                index=False
            )

            logger.info(
                f"Best model: {best_model_name}"
            )

            logger.info(
                f"Best R2: {best_r2:.4f}"
            )

            # -----------------------------
            # Save Model
            # -----------------------------

            os.makedirs(
                os.path.dirname(
                    self.model_path
                ),
                exist_ok=True
            )

            save_object(
                self.model_path,
                best_model
            )

            # -----------------------------
            # Save Preprocessor
            # -----------------------------

            save_object(
                self.preprocessor_path,
                preprocessor
            )

            logger.info(
                "Model and preprocessor saved successfully"
            )

            return {
                "best_model": best_model,
                "best_model_name": best_model_name,
                "best_r2": best_r2,
                "results": results_df
            }

        except Exception as e:

            logger.error(
                "Error occurred during model training"
            )

            raise CustomException(
                e,
                sys
            )