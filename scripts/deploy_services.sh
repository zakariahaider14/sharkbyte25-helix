#!/bin/bash

# HELIX Phase 3 - Deploy Services to Cloud Run

set -e

echo "🚀 Deploying HELIX to Cloud Run"
echo "================================"
echo ""

# Load configuration
if [ ! -f .env.gcp ]; then
    echo "❌ Error: .env.gcp not found. Run phase3_setup.sh first."
    exit 1
fi

source .env.gcp

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get Gemini API key
echo "🔑 Secret Configuration"
echo "----------------------"
read -sp "Enter your Gemini API Key: " GEMINI_KEY
echo ""

# Store in Secret Manager
echo "Storing API key in Secret Manager..."
echo -n "$GEMINI_KEY" | gcloud secrets create gemini-api-key \
    --data-file=- \
    --replication-policy="automatic" \
    --project=$PROJECT_ID 2>/dev/null || \
    echo -n "$GEMINI_KEY" | gcloud secrets versions add gemini-api-key \
    --data-file=- \
    --project=$PROJECT_ID

echo -e "${GREEN}✅ Secret stored${NC}"
echo ""

# Configure Docker for Artifact Registry
echo "🐳 Configuring Docker..."
echo "-----------------------"
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

echo -e "${GREEN}✅ Docker configured${NC}"
echo ""

# Build and push COVID service
echo "📦 Building COVID Service..."
echo "---------------------------"

COVID_IMAGE="${ARTIFACT_REGISTRY}/covid-service:v1.0"

docker build -t $COVID_IMAGE -f Dockerfile.covid .
docker push $COVID_IMAGE

echo -e "${GREEN}✅ COVID service image pushed${NC}"
echo ""

# Build and push Churn service
echo "📦 Building Churn Service..."
echo "---------------------------"

CHURN_IMAGE="${ARTIFACT_REGISTRY}/churn-service:v1.0"

docker build -t $CHURN_IMAGE -f Dockerfile.churn .
docker push $CHURN_IMAGE

echo -e "${GREEN}✅ Churn service image pushed${NC}"
echo ""

# Deploy COVID service
echo "🚀 Deploying COVID Service to Cloud Run..."
echo "------------------------------------------"

gcloud run deploy covid-service \
    --image=$COVID_IMAGE \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --set-env-vars="GCS_BUCKET=${ML_MODELS_BUCKET},PROJECT_ID=${PROJECT_ID}" \
    --set-secrets="GEMINI_API_KEY=gemini-api-key:latest" \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=10 \
    --timeout=300 \
    --project=$PROJECT_ID

COVID_URL=$(gcloud run services describe covid-service \
    --region=$REGION \
    --project=$PROJECT_ID \
    --format='value(status.url)')

echo -e "${GREEN}✅ COVID service deployed${NC}"
echo -e "${BLUE}URL: $COVID_URL${NC}"
echo ""

# Deploy Churn service
echo "🚀 Deploying Churn Service to Cloud Run..."
echo "------------------------------------------"

gcloud run deploy churn-service \
    --image=$CHURN_IMAGE \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --set-env-vars="GCS_BUCKET=${ML_MODELS_BUCKET},PROJECT_ID=${PROJECT_ID}" \
    --set-secrets="GEMINI_API_KEY=gemini-api-key:latest" \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=10 \
    --timeout=300 \
    --project=$PROJECT_ID

CHURN_URL=$(gcloud run services describe churn-service \
    --region=$REGION \
    --project=$PROJECT_ID \
    --format='value(status.url)')

echo -e "${GREEN}✅ Churn service deployed${NC}"
echo -e "${BLUE}URL: $CHURN_URL${NC}"
echo ""

# Test services
echo "🧪 Testing Services..."
echo "---------------------"

echo "Testing COVID service health..."
curl -s "${COVID_URL}/health" | jq '.' || echo "Health check failed"

echo ""
echo "Testing Churn service health..."
curl -s "${CHURN_URL}/health" | jq '.' || echo "Health check failed"

echo ""

# Update .env.gcp with service URLs
sed -i.bak "s|COVID_SERVICE_URL=.*|COVID_SERVICE_URL=${COVID_URL}|" .env.gcp
sed -i.bak "s|CHURN_SERVICE_URL=.*|CHURN_SERVICE_URL=${CHURN_URL}|" .env.gcp
rm .env.gcp.bak

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Service URLs:"
echo "  COVID: $COVID_URL"
echo "  Churn: $CHURN_URL"
echo ""
echo "Test endpoints:"
echo "  curl ${COVID_URL}/health"
echo "  curl ${CHURN_URL}/health"
echo ""
echo "View logs:"
echo "  gcloud run services logs read covid-service --region=$REGION"
echo "  gcloud run services logs read churn-service --region=$REGION"
echo ""
echo "Monitor in console:"
echo "  https://console.cloud.google.com/run?project=$PROJECT_ID"
echo ""
