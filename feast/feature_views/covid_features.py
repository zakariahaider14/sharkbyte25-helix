"""
Feast Feature Definitions for COVID-19 Prediction Model

This module defines all features used by the COVID-19 prediction model,
including their sources, transformations, and serving configurations.
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, BigQuerySource
from feast.types import Float32, Int32, String

# Define the entity (what we're making predictions about)
country = Entity(
    name="country",
    join_keys=["country_name"],
    description="Country entity for COVID-19 predictions"
)

# Define the BigQuery data source for COVID-19 features
covid_data_source = BigQuerySource(
    table="mlops_dev.covid_features",
    timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define the feature view for COVID-19 data
covid_feature_view = FeatureView(
    name="covid_features",
    entities=[country],
    ttl=timedelta(days=30),
    schema=[
        Field(name="country_name", dtype=String),
        Field(name="confirmed_cases", dtype=Int32),
        Field(name="deaths", dtype=Int32),
        Field(name="recovered", dtype=Int32),
        Field(name="active_cases", dtype=Int32),
        Field(name="case_fatality_rate", dtype=Float32),
        Field(name="recovery_rate", dtype=Float32),
        Field(name="population", dtype=Int32),
        Field(name="cases_per_million", dtype=Float32),
        Field(name="deaths_per_million", dtype=Float32),
        Field(name="testing_rate", dtype=Float32),
        Field(name="vaccination_rate", dtype=Float32),
        Field(name="new_cases_7day_avg", dtype=Int32),
        Field(name="new_deaths_7day_avg", dtype=Int32),
    ],
    source=covid_data_source,
    tags={"team": "mlops", "model": "covid-prediction"},
    description="COVID-19 epidemiological and demographic features for prediction"
)

# Define a feature view for time-series features
covid_timeseries_source = BigQuerySource(
    table="mlops_dev.covid_timeseries_features",
    timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

covid_timeseries_feature_view = FeatureView(
    name="covid_timeseries_features",
    entities=[country],
    ttl=timedelta(days=30),
    schema=[
        Field(name="country_name", dtype=String),
        Field(name="trend_direction", dtype=String),  # "up", "down", "stable"
        Field(name="trend_strength", dtype=Float32),
        Field(name="volatility", dtype=Float32),
        Field(name="doubling_time_days", dtype=Float32),
    ],
    source=covid_timeseries_source,
    tags={"team": "mlops", "model": "covid-prediction", "type": "timeseries"},
    description="Time-series derived features for COVID-19 trend analysis"
)
