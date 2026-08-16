import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------


st.set_page_config(
    page_title="Customer Lifetime Value Prediction",
    page_icon="📊",
    layout="wide"
)


# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("📊 Customer Lifetime Value Prediction")

st.write(
    "Enter customer purchasing information to estimate "
    "the customer's future revenue."
)

with st.sidebar:

    st.header("Project Information")

    st.write(
        """
        **Customer Lifetime Value Prediction**

        Machine Learning regression system
        for predicting future customer revenue.
        """
    )

    st.divider()

    st.write("**Technology Stack**")

    st.write(
        """
        • Python  
        • Pandas  
        • Scikit-learn  
        • XGBoost  
        • MySQL  
        • Streamlit
        """
    )

# -------------------------------------------------
# Input Form
# -------------------------------------------------

with st.form("customer_prediction_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        recency_days = st.number_input(
            "Recency (Days)",
            min_value=0,
            value=30
        )

        purchase_frequency = st.number_input(
            "Purchase Frequency",
            min_value=1,
            value=5
        )

        historical_revenue = st.number_input(
            "Historical Revenue",
            min_value=0.0,
            value=1000.0
        )

        total_quantity = st.number_input(
            "Total Quantity",
            min_value=0,
            value=200
        )

        unique_products = st.number_input(
            "Unique Products",
            min_value=1,
            value=20
        )

        customer_tenure_days = st.number_input(
            "Customer Tenure (Days)",
            min_value=0,
            value=180
        )

        average_order_value = st.number_input(
            "Average Order Value",
            min_value=0.0,
            value=200.0
        )

    with col2:

        avg_quantity_per_order = st.number_input(
            "Average Quantity per Order",
            min_value=0.0,
            value=40.0
        )

        orders_per_month = st.number_input(
            "Orders per Month",
            min_value=0.0,
            value=0.83
        )

        revenue_90d = st.number_input(
            "Revenue - Last 90 Days",
            min_value=0.0,
            value=500.0
        )

        revenue_30d = st.number_input(
            "Revenue - Last 30 Days",
            min_value=0.0,
            value=200.0
        )

        previous_90d_revenue = st.number_input(
            "Previous 90-Day Revenue",
            min_value=0.0,
            value=300.0
        )

        spending_trend = st.number_input(
            "Spending Trend",
            value=200.0
        )

        country = st.text_input(
            "Country",
            value="United Kingdom"
        )

    submit = st.form_submit_button(
        "Predict Future Revenue"
    )


# -------------------------------------------------
# Prediction
# -------------------------------------------------

if submit:

    input_data = {
        "recency_days": recency_days,
        "purchase_frequency": purchase_frequency,
        "historical_revenue": historical_revenue,
        "total_quantity": total_quantity,
        "unique_products": unique_products,
        "customer_tenure_days": customer_tenure_days,
        "average_order_value": average_order_value,
        "avg_quantity_per_order": avg_quantity_per_order,
        "orders_per_month": orders_per_month,
        "revenue_90d": revenue_90d,
        "revenue_30d": revenue_30d,
        "previous_90d_revenue": previous_90d_revenue,
        "spending_trend": spending_trend,
        "country": country
    }

    input_df = pd.DataFrame([input_data])

    try:

        pipeline = PredictPipeline()

        prediction = pipeline.predict(
            input_df
        )

        st.success(
            "Prediction completed successfully!"
        )

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Future Revenue",
                f"₹ {prediction:,.2f}"
            )

        with col2:
            st.metric(
                "Prediction Status",
                "Completed"
            )

    except Exception as e:

        st.error(
            "Unable to generate prediction."
        )

        st.exception(e)

st.divider()

st.subheader("About the Project")

st.write(
    """
    This application predicts a customer's future revenue
    using historical purchasing behavior.

    The model uses customer-level features such as:
    - Recency
    - Purchase Frequency
    - Historical Revenue
    - Average Order Value
    - Product Diversity
    - Recent Revenue
    - Spending Trend
    - Customer Tenure
    - Country
    """
)