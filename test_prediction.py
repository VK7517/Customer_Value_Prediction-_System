from src.pipeline.predict_pipeline import PredictPipeline


sample_customer = {
    "recency_days": 30,
    "purchase_frequency": 5,
    "historical_revenue": 1000,
    "total_quantity": 200,
    "unique_products": 20,
    "customer_tenure_days": 180,
    "average_order_value": 200,
    "avg_quantity_per_order": 40,
    "orders_per_month": 0.83,
    "revenue_90d": 500,
    "revenue_30d": 200,
    "previous_90d_revenue": 300,
    "spending_trend": 200,
    "country": "United Kingdom"
}


pipeline = PredictPipeline()

prediction = pipeline.predict(
    sample_customer
)

print(
    f"Predicted Future Revenue: {prediction:.2f}"
)