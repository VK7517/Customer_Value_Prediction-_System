import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logger


class TrainPipeline:

    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):

        try:

            logger.info("========== Training Pipeline Started ==========")

            # ---------------------------------
            # 1. Data Ingestion
            # ---------------------------------

            logger.info("Starting Data Ingestion")

            data = self.data_ingestion.initiate_data_ingestion()

            logger.info("Data Ingestion completed")

            # ---------------------------------
            # 2. Data Transformation
            # ---------------------------------

            logger.info("Starting Data Transformation")

            customer_features = (
                self.data_transformation.create_customer_features(
                    data
                )
            )

            logger.info(
                "Customer feature engineering completed"
            )

            # ---------------------------------
            # 3. Create Preprocessor
            # ---------------------------------

            preprocessor = (
                self.data_transformation
                .get_data_transformer_object(
                    customer_features
                )
            )

            logger.info(
                "Preprocessor created successfully"
            )

            # ---------------------------------
            # 4. Model Training
            # ---------------------------------

            logger.info("Starting Model Training")

            model_result = (
                self.model_trainer.train_models(
                    customer_features,
                    preprocessor
                )
            )

            logger.info(
                "Model training completed successfully"
            )

            logger.info(
                f"Best Model: "
                f"{model_result['best_model_name']}"
            )

            logger.info(
                f"Best R2: "
                f"{model_result['best_r2']:.4f}"
            )

            logger.info("========== Training Pipeline Completed ==========")

            return model_result

        except Exception as e:

            logger.error(
                "Training pipeline failed"
            )

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":

    pipeline = TrainPipeline()

    result = pipeline.run_pipeline()

    print("\nTraining completed successfully!")
    print(
        f"Best Model: "
        f"{result['best_model_name']}"
    )

    print(
        f"Best R2: "
        f"{result['best_r2']:.4f}"
    )

    print("\nModel Comparison:")
    print(result["results"])