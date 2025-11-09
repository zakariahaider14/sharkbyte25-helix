#!/bin/bash

# Import Existing GCP Resources into Pulumi
# This tells Pulumi about resources that already exist

set -e

PROJECT_ID="gen-lang-client-0520631489"
REGION="us-central1"

echo "🔄 Importing existing GCP resources into Pulumi..."
echo "=================================================="

cd "$(dirname "$0")"

# Ensure we're using the right stack
pulumi stack select prod 2>/dev/null || pulumi stack init prod

echo ""
echo "📦 Importing Storage Buckets..."

# Import buckets (if they exist)
pulumi import gcp:storage/bucket:Bucket raw-data-bucket ${PROJECT_ID}-raw-data 2>/dev/null || echo "  ⏭️  raw-data-bucket already imported or doesn't exist"
pulumi import gcp:storage/bucket:Bucket processed-data-bucket ${PROJECT_ID}-processed-data 2>/dev/null || echo "  ⏭️  processed-data-bucket already imported or doesn't exist"
pulumi import gcp:storage/bucket:Bucket ml-models-bucket ${PROJECT_ID}-ml-models 2>/dev/null || echo "  ⏭️  ml-models-bucket already imported or doesn't exist"
pulumi import gcp:storage/bucket:Bucket feast-data-bucket ${PROJECT_ID}-feast-data 2>/dev/null || echo "  ⏭️  feast-data-bucket already imported or doesn't exist"
pulumi import gcp:storage/bucket:Bucket mlflow-artifacts-bucket ${PROJECT_ID}-mlflow-artifacts 2>/dev/null || echo "  ⏭️  mlflow-artifacts-bucket already imported or doesn't exist"

echo ""
echo "📊 Importing BigQuery Dataset..."
pulumi import gcp:bigquery/dataset:Dataset helix-features ${PROJECT_ID}:helix_features 2>/dev/null || echo "  ⏭️  helix-features already imported or doesn't exist"

echo ""
echo "🐳 Importing Artifact Registry..."
pulumi import gcp:artifactregistry/repository:Repository helix-images projects/${PROJECT_ID}/locations/${REGION}/repositories/helix-images 2>/dev/null || echo "  ⏭️  helix-images already imported or doesn't exist"

echo ""
echo "🔐 Importing Service Account..."
pulumi import gcp:serviceaccount/account:Account helix-deployer projects/${PROJECT_ID}/serviceAccounts/helix-deployer@${PROJECT_ID}.iam.gserviceaccount.com 2>/dev/null || echo "  ⏭️  helix-deployer already imported or doesn't exist"

echo ""
echo "🔑 Importing Secrets..."
pulumi import gcp:secretmanager/secret:Secret gemini-api-key projects/${PROJECT_ID}/secrets/gemini-api-key 2>/dev/null || echo "  ⏭️  gemini-api-key already imported or doesn't exist"

echo ""
echo "☁️  Importing Cloud Run Services..."
pulumi import gcp:cloudrun/service:Service covid-service locations/${REGION}/namespaces/${PROJECT_ID}/services/covid-service 2>/dev/null || echo "  ⏭️  covid-service already imported or doesn't exist"
pulumi import gcp:cloudrun/service:Service churn-service locations/${REGION}/namespaces/${PROJECT_ID}/services/churn-service 2>/dev/null || echo "  ⏭️  churn-service already imported or doesn't exist"
pulumi import gcp:cloudrun/service:Service helix-frontend locations/${REGION}/namespaces/${PROJECT_ID}/services/helix-frontend 2>/dev/null || echo "  ⏭️  helix-frontend already imported or doesn't exist"

echo ""
echo "✅ Import complete!"
echo ""
echo "Next steps:"
echo "1. Run 'pulumi preview' to see what would change"
echo "2. Run 'pulumi up' to apply incremental updates"
