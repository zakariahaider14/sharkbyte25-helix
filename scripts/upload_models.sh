#!/bin/bash

# HELIX Phase 3 - Upload Models and Data to GCS

set -e

echo "📤 Uploading Models and Data to GCS"
echo "===================================="
echo ""

# Load configuration
if [ ! -f .env.gcp ]; then
    echo "❌ Error: .env.gcp not found. Run phase3_setup.sh first."
    exit 1
fi

source .env.gcp

GREEN='\033[0;32m'
NC='\033[0m'

# Upload trained models
echo "📦 Uploading ML Models..."
echo "------------------------"

if [ -f "ml_models/saved_models/covid_model.joblib" ]; then
    echo "Uploading COVID model..."
    gsutil cp ml_models/saved_models/covid_model.joblib \
        gs://${ML_MODELS_BUCKET}/covid/v1.0/covid_model.joblib
    echo -e "${GREEN}✅ COVID model uploaded${NC}"
else
    echo "⚠️  COVID model not found at ml_models/saved_models/covid_model.joblib"
fi

if [ -f "ml_models/saved_models/churn_model.joblib" ]; then
    echo "Uploading Churn model..."
    gsutil cp ml_models/saved_models/churn_model.joblib \
        gs://${ML_MODELS_BUCKET}/churn/v1.0/churn_model.joblib
    echo -e "${GREEN}✅ Churn model uploaded${NC}"
else
    echo "⚠️  Churn model not found at ml_models/saved_models/churn_model.joblib"
fi

# Upload processed data
echo ""
echo "📊 Uploading Processed Data..."
echo "------------------------------"

if [ -d "data_pipeline/processed_data" ]; then
    echo "Uploading processed datasets..."
    gsutil -m cp -r data_pipeline/processed_data/*.csv \
        gs://${PROCESSED_DATA_BUCKET}/ 2>/dev/null || echo "No CSV files found or already uploaded"
    echo -e "${GREEN}✅ Data uploaded${NC}"
else
    echo "⚠️  Processed data directory not found"
fi

# Verify uploads
echo ""
echo "🔍 Verifying Uploads..."
echo "----------------------"

echo "Models bucket:"
gsutil ls gs://${ML_MODELS_BUCKET}/** || echo "No models found"

echo ""
echo "Data bucket:"
gsutil ls gs://${PROCESSED_DATA_BUCKET}/ || echo "No data found"

echo ""
echo -e "${GREEN}✅ Upload complete!${NC}"
echo ""
