#!/bin/bash

# HELIX Phase 3 - Load Data to BigQuery

set -e

echo "📊 Loading Data to BigQuery"
echo "==========================="
echo ""

# Load configuration
if [ ! -f .env.gcp ]; then
    echo "❌ Error: .env.gcp not found. Run phase3_setup.sh first."
    exit 1
fi

source .env.gcp

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Load COVID features
echo "🦠 Loading COVID Features..."
echo "---------------------------"

if gsutil ls gs://${PROCESSED_DATA_BUCKET}/covid_train.csv &> /dev/null; then
    echo "Loading COVID training data..."
    bq load \
        --source_format=CSV \
        --autodetect \
        --replace \
        ${BQ_DATASET}.covid_features \
        gs://${PROCESSED_DATA_BUCKET}/covid_train.csv
    
    echo -e "${GREEN}✅ COVID features loaded${NC}"
    
    # Show sample
    echo ""
    echo "Sample data:"
    bq query --use_legacy_sql=false \
        "SELECT * FROM \`${PROJECT_ID}.${BQ_DATASET}.covid_features\` LIMIT 5"
else
    echo -e "${YELLOW}⚠️  COVID data not found in GCS${NC}"
fi

# Load Churn features
echo ""
echo "📞 Loading Churn Features..."
echo "---------------------------"

if gsutil ls gs://${PROCESSED_DATA_BUCKET}/churn_train.csv &> /dev/null; then
    echo "Loading Churn training data..."
    bq load \
        --source_format=CSV \
        --autodetect \
        --replace \
        ${BQ_DATASET}.churn_features \
        gs://${PROCESSED_DATA_BUCKET}/churn_train.csv
    
    echo -e "${GREEN}✅ Churn features loaded${NC}"
    
    # Show sample
    echo ""
    echo "Sample data:"
    bq query --use_legacy_sql=false \
        "SELECT * FROM \`${PROJECT_ID}.${BQ_DATASET}.churn_features\` LIMIT 5"
else
    echo -e "${YELLOW}⚠️  Churn data not found in GCS${NC}"
fi

# Show dataset info
echo ""
echo "📋 Dataset Summary"
echo "-----------------"

echo "COVID features:"
bq query --use_legacy_sql=false \
    "SELECT COUNT(*) as total_records FROM \`${PROJECT_ID}.${BQ_DATASET}.covid_features\`"

echo ""
echo "Churn features:"
bq query --use_legacy_sql=false \
    "SELECT COUNT(*) as total_records FROM \`${PROJECT_ID}.${BQ_DATASET}.churn_features\`"

echo ""
echo -e "${GREEN}✅ BigQuery data load complete!${NC}"
echo ""
