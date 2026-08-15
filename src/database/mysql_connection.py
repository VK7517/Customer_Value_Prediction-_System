import os

import mysql.connector
from dotenv import load_dotenv
import sys 
from src.exception import CustomException
from src.logger import logger

load_dotenv()


def get_mysql_connection():

    try:
        logger.info("Attempting to connect to MySQL database")

        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE")
        )

        if connection.is_connected():
            logger.info("MySQL connection established successfully")

        return connection

    except Exception as e:
        logger.error("Error while connecting to MySQL")
        raise CustomException(e, sys)