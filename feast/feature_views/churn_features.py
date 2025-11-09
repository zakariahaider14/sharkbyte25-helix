"""
Feast Feature Definitions for Telco Customer Churn Prediction Model

This module defines all features used by the Telco churn prediction model,
including customer demographics, service usage, and behavioral patterns.
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, BigQuerySource
from feast.types import Float32, Int32, String, Bool

# Define the entity (what we're making predictions about)
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="Customer entity for churn predictions"
)

# Define the BigQuery data source for customer demographics
customer_demographics_source = BigQuerySource(
    table="mlops_dev.customer_demographics",
    timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define the feature view for customer demographics
customer_demographics_view = FeatureView(
    name="customer_demographics",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="customer_id", dtype=String),
        Field(name="age", dtype=Int32),
        Field(name="gender", dtype=String),
        Field(name="senior_citizen", dtype=Bool),
        Field(name="partner", dtype=Bool),
        Field(name="dependents", dtype=Bool),
        Field(name="tenure_months", dtype=Int32),
        Field(name="contract_type", dtype=String),  # "Month-to-month", "One year", "Two year"
    ],
    source=customer_demographics_source,
    tags={"team": "mlops", "model": "churn-prediction", "type": "demographics"},
    description="Customer demographic and contract information"
)

# Define the BigQuery data source for service usage
service_usage_source = BigQuerySource(
    table="mlops_dev.customer_service_usage",
    timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define the feature view for service usage
service_usage_view = FeatureView(
    name="customer_service_usage",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="customer_id", dtype=String),
        Field(name="phone_service", dtype=Bool),
        Field(name="multiple_lines", dtype=Bool),
        Field(name="internet_service_type", dtype=String),  # "DSL", "Fiber optic", "No"
        Field(name="online_security", dtype=Bool),
        Field(name="online_backup", dtype=Bool),
        Field(name="device_protection", dtype=Bool),
        Field(name="tech_support", dtype=Bool),
        Field(name="streaming_tv", dtype=Bool),
        Field(name="streaming_movies", dtype=Bool),
        Field(name="paperless_billing", dtype=Bool),
    ],
    source=service_usage_source,
    tags={"team": "mlops", "model": "churn-prediction", "type": "service_usage"},
    description="Customer service subscriptions and usage patterns"
)

# Define the BigQuery data source for billing information
billing_source = BigQuerySource(
    table="mlops_dev.customer_billing",
    timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define the feature view for billing information
billing_view = FeatureView(
    name="customer_billing",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="customer_id", dtype=String),
        Field(name="monthly_charges", dtype=Float32),
        Field(name="total_charges", dtype=Float32),
        Field(name="payment_method", dtype=String),
        Field(name="paperless_billing", dtype=Bool),
        Field(name="avg_monthly_charges_6m", dtype=Float32),
        Field(name="charges_increase_rate", dtype=Float32),
    ],
    source=billing_source,
    tags={"team": "mlops", "model": "churn-prediction", "type": "billing"},
    description="Customer billing and payment information"
)

# Define the BigQuery data source for behavioral features
behavioral_source = BigQuerySource(
    table="mlops_dev.customer_behavior",
    timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define the feature view for behavioral features
behavioral_view = FeatureView(
    name="customer_behavior",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="customer_id", dtype=String),
        Field(name="support_tickets_count", dtype=Int32),
        Field(name="technical_tickets_count", dtype=Int32),
        Field(name="billing_complaints", dtype=Int32),
        Field(name="days_since_last_support_ticket", dtype=Int32),
        Field(name="avg_response_time_hours", dtype=Float32),
        Field(name="service_disruptions_count", dtype=Int32),
    ],
    source=behavioral_source,
    tags={"team": "mlops", "model": "churn-prediction", "type": "behavioral"},
    description="Customer support interaction and behavioral patterns"
)
