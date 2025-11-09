#!/bin/bash

# HELIX Frontend Deployment to Cloud Run

set -e

echo "🚀 Deploying HELIX Frontend to Cloud Run"
echo "=========================================="
echo ""

# Load configuration
if [ ! -f .env.gcp ]; then
    echo "❌ Error: .env.gcp not found. Run phase3_setup.sh first."
    exit 1
fi

source .env.gcp

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}📦 Building Frontend Image...${NC}"
echo "-------------------------------------"

# Build for AMD64 (Cloud Run requirement)
docker build --platform linux/amd64 \
  -t ${ARTIFACT_REGISTRY}/helix-frontend:v1.0 \
  -f Dockerfile.frontend .

echo -e "${GREEN}✅ Frontend image built${NC}"
echo ""

echo -e "${BLUE}📤 Pushing to Artifact Registry...${NC}"
echo "-----------------------------------"

docker push ${ARTIFACT_REGISTRY}/helix-frontend:v1.0

echo -e "${GREEN}✅ Image pushed${NC}"
echo ""

echo -e "${BLUE}🚀 Deploying to Cloud Run...${NC}"
echo "-----------------------------"

gcloud run deploy helix-frontend \
  --image=${ARTIFACT_REGISTRY}/helix-frontend:v1.0 \
  --platform=managed \
  --region=$REGION \
  --allow-unauthenticated \
  --set-env-vars="NODE_ENV=production,COVID_SERVICE_URL=${COVID_SERVICE_URL},CHURN_SERVICE_URL=${CHURN_SERVICE_URL}" \
  --set-secrets="GEMINI_API_KEY=gemini-api-key:latest" \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=5 \
  --timeout=300 \
  --project=$PROJECT_ID

# Get the frontend URL
FRONTEND_URL=$(gcloud run services describe helix-frontend \
  --region=$REGION \
  --project=$PROJECT_ID \
  --format='value(status.url)')

echo ""
echo -e "${GREEN}✅ Frontend deployed successfully!${NC}"
echo ""
echo "=========================================="
echo -e "${BLUE}🌐 Your HELIX Application is Live!${NC}"
echo "=========================================="
echo ""
echo "Frontend URL: $FRONTEND_URL"
echo ""
echo "Services:"
echo "  - Frontend: $FRONTEND_URL"
echo "  - COVID API: $COVID_SERVICE_URL"
echo "  - Churn API: $CHURN_SERVICE_URL"
echo ""
echo "Test it:"
echo "  Open: $FRONTEND_URL"
echo "  Or:   $FRONTEND_URL/agents"
echo ""

# Update .env.gcp
sed -i.bak "s|FRONTEND_URL=.*|FRONTEND_URL=${FRONTEND_URL}|" .env.gcp
rm .env.gcp.bak

echo "Configuration saved to .env.gcp"
echo ""
