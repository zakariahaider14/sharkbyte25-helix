#!/bin/bash

# HELIX Phase 3 - GCP Deployment Setup Script
# This script helps you set up and deploy HELIX to Google Cloud Platform

set -e  # Exit on error

echo "🚀 HELIX Phase 3 - GCP Deployment Setup"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Google Cloud SDK not found${NC}"
    echo ""
    echo "Please install it first:"
    echo "  brew install --cask google-cloud-sdk"
    echo ""
    echo "Or download from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo -e "${GREEN}✅ Google Cloud SDK found${NC}"
echo ""

# Get project ID
echo "📋 GCP Project Setup"
echo "-------------------"
read -p "Enter your GCP Project ID (or press Enter to create new): " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    read -p "Enter a name for your new project (e.g., helix-mlops): " PROJECT_NAME
    PROJECT_ID="${PROJECT_NAME}-$(date +%s)"
    echo ""
    echo "Creating project: $PROJECT_ID"
    gcloud projects create $PROJECT_ID --name="$PROJECT_NAME"
    echo -e "${GREEN}✅ Project created${NC}"
fi

# Set project
echo ""
echo "Setting active project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Check billing
echo ""
echo -e "${YELLOW}⚠️  Important: Billing must be enabled for this project${NC}"
echo "Visit: https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
read -p "Press Enter once billing is enabled..."

# Set region
echo ""
read -p "Enter your preferred region (default: us-central1): " REGION
REGION=${REGION:-us-central1}
gcloud config set compute/region $REGION

echo ""
echo "📦 Enabling Required APIs..."
echo "----------------------------"

APIS=(
    "compute.googleapis.com"
    "container.googleapis.com"
    "cloudbuild.googleapis.com"
    "artifactregistry.googleapis.com"
    "run.googleapis.com"
    "bigquery.googleapis.com"
    "storage.googleapis.com"
    "secretmanager.googleapis.com"
    "redis.googleapis.com"
    "cloudresourcemanager.googleapis.com"
    "iam.googleapis.com"
    "logging.googleapis.com"
    "monitoring.googleapis.com"
)

for api in "${APIS[@]}"; do
    echo "Enabling $api..."
    gcloud services enable $api --project=$PROJECT_ID
done

echo -e "${GREEN}✅ All APIs enabled${NC}"

# Create service account
echo ""
echo "👤 Creating Service Account..."
echo "------------------------------"

SA_NAME="helix-deployer"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create $SA_NAME \
    --display-name="HELIX Deployment Service Account" \
    --project=$PROJECT_ID || echo "Service account already exists"

# Grant roles
ROLES=(
    "roles/storage.admin"
    "roles/bigquery.admin"
    "roles/run.admin"
    "roles/artifactregistry.admin"
    "roles/secretmanager.admin"
    "roles/redis.admin"
    "roles/logging.admin"
    "roles/monitoring.admin"
)

for role in "${ROLES[@]}"; do
    echo "Granting $role..."
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="$role" \
        --quiet
done

echo -e "${GREEN}✅ Service account configured${NC}"

# Create GCS buckets
echo ""
echo "🪣 Creating Cloud Storage Buckets..."
echo "------------------------------------"

BUCKETS=(
    "${PROJECT_ID}-raw-data"
    "${PROJECT_ID}-processed-data"
    "${PROJECT_ID}-ml-models"
    "${PROJECT_ID}-feast-data"
)

for bucket in "${BUCKETS[@]}"; do
    echo "Creating bucket: gs://${bucket}"
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://${bucket}/ || echo "Bucket already exists"
done

echo -e "${GREEN}✅ Buckets created${NC}"

# Create Artifact Registry repository
echo ""
echo "📦 Creating Artifact Registry..."
echo "--------------------------------"

gcloud artifacts repositories create helix-images \
    --repository-format=docker \
    --location=$REGION \
    --description="HELIX Docker Images" \
    --project=$PROJECT_ID || echo "Repository already exists"

echo -e "${GREEN}✅ Artifact Registry ready${NC}"

# Create BigQuery dataset
echo ""
echo "📊 Creating BigQuery Dataset..."
echo "-------------------------------"

bq mk --dataset \
    --location=$REGION \
    --description="HELIX MLOps Feature Store" \
    ${PROJECT_ID}:helix_features || echo "Dataset already exists"

echo -e "${GREEN}✅ BigQuery dataset ready${NC}"

# Save configuration
echo ""
echo "💾 Saving Configuration..."
echo "-------------------------"

cat > .env.gcp <<EOF
# HELIX Phase 3 - GCP Configuration
# Generated: $(date)

PROJECT_ID=$PROJECT_ID
REGION=$REGION
SERVICE_ACCOUNT=$SA_EMAIL

# Buckets
RAW_DATA_BUCKET=${PROJECT_ID}-raw-data
PROCESSED_DATA_BUCKET=${PROJECT_ID}-processed-data
ML_MODELS_BUCKET=${PROJECT_ID}-ml-models
FEAST_DATA_BUCKET=${PROJECT_ID}-feast-data

# BigQuery
BQ_DATASET=helix_features

# Artifact Registry
ARTIFACT_REGISTRY=${REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images

# Service URLs (will be populated after deployment)
COVID_SERVICE_URL=
CHURN_SERVICE_URL=
FRONTEND_URL=
EOF

echo -e "${GREEN}✅ Configuration saved to .env.gcp${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 Phase 3 Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload models: ./scripts/upload_models.sh"
echo "2. Load data to BigQuery: ./scripts/load_bigquery.sh"
echo "3. Build Docker images: ./scripts/build_images.sh"
echo "4. Deploy to Cloud Run: ./scripts/deploy_services.sh"
echo ""
echo "Configuration saved in: .env.gcp"
echo ""
