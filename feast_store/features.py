"""
Feast Feature Definitions
Defines feature views for COVID and Churn predictions
"""

from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# ============================================================================
# ENTITIES
# ============================================================================

# COVID entity (country-based)
covid_entity = Entity(
    name="country",
    join_keys=["country_name"],
    description="Country entity for COVID-19 data"
)

# Churn entity (customer-based)
churn_entity = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="Customer entity for churn prediction"
)

# ============================================================================
# DATA SOURCES
# ============================================================================

# COVID data source (from MySQL via parquet export)
covid_source = FileSource(
    path="data/covid_features.parquet",
    timestamp_field="timestamp",
)

# Churn data source (from MySQL via parquet export)
churn_source = FileSource(
    path="data/churn_features.parquet",
    timestamp_field="event_timestamp",
)

# ============================================================================
# FEATURE VIEWS
# ============================================================================

# COVID Feature View
covid_features = FeatureView(
    name="covid_features",
    entities=[covid_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="confirmed_cases", dtype=Int64),
        Field(name="deaths", dtype=Int64),
        Field(name="recovered", dtype=Int64),
        Field(name="active_cases", dtype=Int64),
        Field(name="death_rate", dtype=Float32),
        Field(name="recovery_rate", dtype=Float32),
        Field(name="risk_score", dtype=Float32),
    ],
    online=True,
    source=covid_source,
    tags={"team": "ml_ops", "model": "covid_prediction"},
)

# Churn Feature View
churn_features = FeatureView(
    name="churn_features",
    entities=[churn_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="SeniorCitizen", dtype=Int64),
        Field(name="Partner", dtype=Int64),
        Field(name="Dependents", dtype=Int64),
        Field(name="tenure", dtype=Int64),
        Field(name="PhoneService", dtype=Int64),
        Field(name="MultipleLines", dtype=Int64),
        Field(name="OnlineSecurity", dtype=Int64),
        Field(name="OnlineBackup", dtype=Int64),
        Field(name="DeviceProtection", dtype=Int64),
        Field(name="TechSupport", dtype=Int64),
        Field(name="StreamingTV", dtype=Int64),
        Field(name="StreamingMovies", dtype=Int64),
        Field(name="PaperlessBilling", dtype=Int64),
        Field(name="MonthlyCharges", dtype=Float32),
        Field(name="TotalCharges", dtype=Float32),
        Field(name="gender_encoded", dtype=Int64),
        Field(name="Contract_encoded", dtype=Int64),
        Field(name="PaymentMethod_encoded", dtype=Int64),
        Field(name="InternetService_encoded", dtype=Int64),
    ],
    online=True,
    source=churn_source,
    tags={"team": "ml_ops", "model": "churn_prediction"},
)
