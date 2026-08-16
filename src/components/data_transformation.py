import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger


class DataTransformation:

    def __init__(self):

        self.cutoff_date = pd.Timestamp("2011-06-01")

    def create_customer_features(self, data):

        try:

            logger.info("Starting customer feature engineering")

            data = data.copy()

            # Convert InvoiceDate
            data["InvoiceDate"] = pd.to_datetime(
                data["InvoiceDate"]
            )

            # -----------------------------
            # Historical / Future Split
            # -----------------------------

            historical_df = data[
                data["InvoiceDate"] < self.cutoff_date
            ].copy()

            future_df = data[
                data["InvoiceDate"] > self.cutoff_date
            ].copy()

            logger.info(
                f"Historical data shape: {historical_df.shape}"
            )

            logger.info(
                f"Future data shape: {future_df.shape}"
            )

            # -----------------------------
            # Recency
            # -----------------------------

            last_purchase = (
                historical_df
                .groupby("CustomerID")["InvoiceDate"]
                .max()
            )

            recency = (
                self.cutoff_date - last_purchase
            ).dt.days

            recency = recency.rename("recency_days")

            # -----------------------------
            # Frequency
            # -----------------------------

            frequency = (
                historical_df
                .groupby("CustomerID")["InvoiceNo"]
                .nunique()
            )

            frequency = frequency.rename(
                "purchase_frequency"
            )

            # -----------------------------
            # Historical Revenue
            # -----------------------------

            monetary = (
                historical_df
                .groupby("CustomerID")["TransactionRevenue"]
                .sum()
            )

            monetary = monetary.rename(
                "historical_revenue"
            )

            # -----------------------------
            # Total Quantity
            # -----------------------------

            total_quantity = (
                historical_df
                .groupby("CustomerID")["Quantity"]
                .sum()
            )

            total_quantity = total_quantity.rename(
                "total_quantity"
            )

            # -----------------------------
            # Unique Products
            # -----------------------------

            unique_products = (
                historical_df
                .groupby("CustomerID")["StockCode"]
                .nunique()
            )

            unique_products = unique_products.rename(
                "unique_products"
            )

            # -----------------------------
            # Customer Tenure
            # -----------------------------

            first_purchase = (
                historical_df
                .groupby("CustomerID")["InvoiceDate"]
                .min()
            )

            tenure = (
                self.cutoff_date - first_purchase
            ).dt.days

            tenure = tenure.rename(
                "customer_tenure_days"
            )

            # -----------------------------
            # Combine Features
            # -----------------------------

            customer_features = pd.concat(
                [
                    recency,
                    frequency,
                    monetary,
                    total_quantity,
                    unique_products,
                    tenure
                ],
                axis=1
            )

            # -----------------------------
            # Average Order Value
            # -----------------------------

            customer_features[
                "average_order_value"
            ] = (
                customer_features["historical_revenue"]
                /
                customer_features[
                    "purchase_frequency"
                ]
            )

            # -----------------------------
            # Average Quantity per Order
            # -----------------------------

            customer_features[
                "avg_quantity_per_order"
            ] = (
                customer_features["total_quantity"]
                /
                customer_features[
                    "purchase_frequency"
                ]
            )

            # -----------------------------
            # Orders per Month
            # -----------------------------

            customer_features[
                "orders_per_month"
            ] = (
                customer_features[
                    "purchase_frequency"
                ]
                /
                (
                    customer_features[
                        "customer_tenure_days"
                    ] / 30 + 1
                )
            )

            # -----------------------------
            # Revenue - Last 90 Days
            # -----------------------------

            recent_90d = historical_df[
                historical_df["InvoiceDate"]
                >= self.cutoff_date -
                pd.Timedelta(days=90)
            ]

            revenue_90d = (
                recent_90d
                .groupby("CustomerID")[
                    "TransactionRevenue"
                ]
                .sum()
                .rename("revenue_90d")
            )

            customer_features = customer_features.join(
                revenue_90d,
                how="left"
            )

            customer_features[
                "revenue_90d"
            ] = customer_features[
                "revenue_90d"
            ].fillna(0)

            # -----------------------------
            # Revenue - Last 30 Days
            # -----------------------------

            recent_30d = historical_df[
                historical_df["InvoiceDate"]
                >= self.cutoff_date -
                pd.Timedelta(days=30)
            ]

            revenue_30d = (
                recent_30d
                .groupby("CustomerID")[
                    "TransactionRevenue"
                ]
                .sum()
                .rename("revenue_30d")
            )

            customer_features = customer_features.join(
                revenue_30d,
                how="left"
            )

            customer_features[
                "revenue_30d"
            ] = customer_features[
                "revenue_30d"
            ].fillna(0)

            # -----------------------------
            # Previous 90 Days Revenue
            # -----------------------------

            previous_90d = historical_df[
                (
                    historical_df["InvoiceDate"]
                    >= self.cutoff_date -
                    pd.Timedelta(days=180)
                )
                &
                (
                    historical_df["InvoiceDate"]
                    < self.cutoff_date -
                    pd.Timedelta(days=90)
                )
            ]

            previous_revenue = (
                previous_90d
                .groupby("CustomerID")[
                    "TransactionRevenue"
                ]
                .sum()
                .rename(
                    "previous_90d_revenue"
                )
            )

            customer_features = customer_features.join(
                previous_revenue,
                how="left"
            )

            customer_features[
                "previous_90d_revenue"
            ] = customer_features[
                "previous_90d_revenue"
            ].fillna(0)

            # -----------------------------
            # Spending Trend
            # -----------------------------

            customer_features[
                "spending_trend"
            ] = (
                customer_features[
                    "revenue_90d"
                ]
                -
                customer_features[
                    "previous_90d_revenue"
                ]
            )

            # -----------------------------
            # Future Revenue - TARGET
            # -----------------------------

            future_revenue = (
                future_df
                .groupby("CustomerID")[
                    "TransactionRevenue"
                ]
                .sum()
                .rename("future_revenue")
            )

            customer_features = customer_features.join(
                future_revenue,
                how="left"
            )

            customer_features[
                "future_revenue"
            ] = customer_features[
                "future_revenue"
            ].fillna(0)

            # -----------------------------
            # Country
            # -----------------------------

            customer_country = (
                historical_df
                .groupby("CustomerID")["Country"]
                .agg(
                    lambda x:
                    x.mode().iloc[0]
                    if not x.mode().empty
                    else "Unknown"
                )
                .rename("country")
            )

            customer_features = customer_features.join(
                customer_country,
                how="left"
            )

            # -----------------------------
            # Reset Index
            # -----------------------------

            customer_features = (
                customer_features
                .reset_index()
            )

            logger.info(
                "Customer feature engineering completed"
            )

            logger.info(
                f"Final customer dataset shape: "
                f"{customer_features.shape}"
            )

            return customer_features

        except Exception as e:

            logger.error(
                "Error during customer feature engineering"
            )

            raise CustomException(e, sys)

    def get_data_transformer_object(
        self,
        data
    ):

        try:

            logger.info(
                "Creating preprocessing object"
            )

            X = data.drop(
                columns=[
                    "CustomerID",
                    "future_revenue"
                ]
            )

            numerical_columns = (
                X.select_dtypes(
                    include=[
                        "int64",
                        "float64"
                    ]
                ).columns.tolist()
            )

            categorical_columns = (
                X.select_dtypes(
                    include=[
                        "object",
                        "category"
                    ]
                ).columns.tolist()
            )

            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "encoder",
                        OneHotEncoder(
                            drop="first",
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        numerical_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        categorical_pipeline,
                        categorical_columns
                    )
                ]
            )

            logger.info(
                "Preprocessing object created successfully"
            )

            return preprocessor

        except Exception as e:

            logger.error(
                "Error while creating preprocessing object"
            )

            raise CustomException(e, sys)