import os
import sys
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from src.exception import CustomException
from src.logger import logger

load_dotenv()

class DataIngestion:

    def __init__(self):

        self.csv_path = os.path.join(
            "data",
            "processed",
            "Cleaned_Online_Retail.csv"
        )

        self.mysql_url = URL.create(
            drivername="mysql+mysqlconnector",
            username=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            host=os.getenv("MYSQL_HOST"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database=os.getenv("MYSQL_DATABASE")
        )
    def load_data(self):

        try:

            logger.info("Data ingestion started")

            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(
                    f"Dataset not found at: {self.csv_path}"
                )
            
            df = pd.read_csv(self.csv_path)
            logger.info(
                f"Dataset loaded successfully. Shape: {df.shape}"
            )

            return df

        except Exception as e:
            logger.error("Error while loading CSV dataset")
            raise CustomException(e, sys)

    def save_to_mysql(self, df):

        try:

            logger.info("Starting data upload to MySQL")

            engine = create_engine(self.mysql_url)

            df.to_sql(
                name="transactions",
                con=engine,
                if_exists="replace",
                index=False,
                chunksize=5000
            )

            logger.info(
                "Dataset successfully stored in MySQL table: transactions"
            )

        except Exception as e:

            logger.error("Error while saving dataset to MySQL")

            raise CustomException(e, sys)

    def initiate_data_ingestion(self):

        try:

            df = self.load_data()

            self.save_to_mysql(df)

            logger.info("Data ingestion completed successfully")

            return df

        except Exception as e:

            logger.error("Data ingestion failed")

            raise CustomException(e, sys)


# if __name__ == "__main__":

#     ingestion = DataIngestion()

#     ingestion.initiate_data_ingestion()