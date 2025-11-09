# HELIX : GCP Deployment Guide

**Deploying HELIX to Google Cloud Platform with Kubernetes, BigQuery, and Cloud Run**

This guide walks through deploying the HELIX MLOps platform to Google Cloud Platform (GCP) for production use.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [GCP Services Used](#gcp-services-used)
5. [Deployment Steps](#deployment-steps)
6. [Configuration](#configuration)
7. [Monitoring & Logging](#monitoring--logging)
8. [Cost Estimation](#cost-estimation)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This phase deploys HELIX to a production-ready GCP deployment with:

- **Scalability**: Auto-scaling with GKE (Google Kubernetes Engine)
- **Reliability**: Multi-zone deployment with 99.9% uptime SLA
- **Performance**: Cloud CDN, load balancing, and optimized data access
- **Security**: IAM, VPC, Secret Manager, and encrypted storage
- **Observability**: Cloud Logging, Monitoring, and Trace

### **Migration Path**

| Component | Phase 2 (Local) | Phase 3 (GCP) |
|-----------|-----------------|---------------|
| **Database** | MySQL (localhost) | Cloud SQL / BigQuery |
| **Feature Store** | Feast (SQLite + Parquet) | Feast (BigQuery + Redis) |
| **ML Services** | FastAPI (localhost:8000/8001) | Cloud Run / GKE |
| **Frontend** | Vite dev server (localhost:3000) | Cloud Run / Firebase Hosting |
| **Model Storage** | Local files | Google Cloud Storage |
| **Secrets** | .env file | Secret Manager |
| **Monitoring** | Console logs | Cloud Logging + Monitoring |

---

## 🏗️ Architecture

### **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INTERNET / USERS                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CLOUD LOAD BALANCER                              │
│                   (Global HTTP(S) LB)                               │
└────────────────┬───────────────────────────────┬────────────────────┘
                 │                               │
                 ▼                               ▼
┌────────────────────────────┐  ┌────────────────────────────────────┐
│   FRONTEND (Cloud Run)     │  │    BACKEND (Cloud Run / GKE)       │
│   • React App              │  │    • Node.js + Express + tRPC      │
│   • Static Assets          │  │    • Gemini AI Agent               │
│   • CDN Cached             │  │    • Intent Classification         │
└────────────────────────────┘  └────────────┬───────────────────────┘
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        │                     │                     │
                        ▼                     ▼                     ▼
        ┌───────────────────────┐  ┌──────────────────┐  ┌─────────────────┐
        │  COVID ML Service     │  │  Churn Service   │  │  Secret Manager │
        │  (Cloud Run)          │  │  (Cloud Run)     │  │  • API Keys     │
        │  • FastAPI            │  │  • FastAPI       │  │  • DB Creds     │
        │  • XGBoost Model      │  │  • XGBoost Model │  └─────────────────┘
        └───────────┬───────────┘  └────────┬─────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
        ┌─────────────────────────────────────────────────────────┐
        │              FEAST FEATURE STORE                        │
        │  ┌──────────────────────┐    ┌──────────────────────┐  │
        │  │  BigQuery            │    │  Redis (Memorystore) │  │
        │  │  (Offline Store)     │    │  (Online Store)      │  │
        │  │  • Historical Data   │    │  • Real-time Cache   │  │
        │  └──────────────────────┘    └──────────────────────┘  │
        └─────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────────────┐
        │           GOOGLE CLOUD STORAGE (GCS)                    │
        │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
        │  │  Raw Data    │  │  Processed   │  │  ML Models   │  │
        │  │  Bucket      │  │  Data Bucket │  │  Bucket      │  │
        │  └──────────────┘  └──────────────┘  └──────────────┘  │
        └─────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────────────┐
        │         MONITORING & LOGGING                            │
        │  • Cloud Logging  • Cloud Monitoring  • Cloud Trace     │
        └─────────────────────────────────────────────────────────┘
```

### **Kubernetes Architecture (GKE Option)**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GKE CLUSTER (Multi-Zone)                         │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    Ingress Controller                         │ │
│  │              (NGINX / GKE Ingress)                            │ │
│  └────────────────────────┬──────────────────────────────────────┘ │
│                           │                                         │
│         ┌─────────────────┼─────────────────┐                      │
│         ▼                 ▼                 ▼                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │  Frontend   │  │  Backend    │  │  Backend    │                │
│  │  Pod        │  │  Pod 1      │  │  Pod 2      │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│                           │                                         │
│         ┌─────────────────┼─────────────────┐                      │
│         ▼                 ▼                 ▼                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │  COVID      │  │  COVID      │  │  COVID      │                │
│  │  Service    │  │  Service    │  │  Service    │                │
│  │  Pod 1      │  │  Pod 2      │  │  Pod 3      │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │  Churn      │  │  Churn      │  │  Churn      │                │
│  │  Service    │  │  Service    │  │  Service    │                │
│  │  Pod 1      │  │  Pod 2      │  │  Pod 3      │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │           Horizontal Pod Autoscaler (HPA)                     │ │
│  │  • Scale 3-20 pods based on CPU/Memory                        │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Prerequisites

### **1. GCP Account Setup**

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Create new project
export PROJECT_ID="helix-mlops-prod"
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID

# Enable billing (required)
# Visit: https://console.cloud.google.com/billing
```

### **2. Enable Required APIs**

```bash
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com \
  redis.googleapis.com \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com
```

### **3. Install Tools**

```bash
# Kubernetes CLI
gcloud components install kubectl

# Docker (for building images)
# macOS: brew install docker
# Linux: apt-get install docker.io

# Terraform (optional, for IaC)
brew install terraform

# Helm (for K8s package management)
brew install helm
```

### **4. Service Account Setup**

```bash
# Create service account
gcloud iam service-accounts create helix-deployer \
  --display-name="HELIX Deployment Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:helix-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/container.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:helix-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:helix-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.admin"

# Download key
gcloud iam service-accounts keys create ~/helix-key.json \
  --iam-account=helix-deployer@${PROJECT_ID}.iam.gserviceaccount.com
```

---

## ☁️ GCP Services Used

| Service | Purpose | Cost Estimate |
|---------|---------|---------------|
| **Cloud Run** | Serverless container hosting | ~$50-200/month |
| **GKE** | Kubernetes cluster (alternative) | ~$200-500/month |
| **BigQuery** | Data warehouse & Feast offline store | ~$20-100/month |
| **Cloud Storage** | Object storage for models & data | ~$10-50/month |
| **Memorystore (Redis)** | Feast online store | ~$50-150/month |
| **Secret Manager** | Secure credential storage | ~$1-5/month |
| **Cloud Load Balancing** | Global load balancer | ~$20-50/month |
| **Cloud Logging** | Centralized logging | ~$10-30/month |
| **Cloud Monitoring** | Metrics & alerting | ~$10-30/month |
| **Artifact Registry** | Docker image storage | ~$5-20/month |
| **Cloud Build** | CI/CD pipeline | ~$10-30/month |
| **Total Estimated** | | **~$386-1,165/month** |

---

## 🚀 Deployment Steps

### **Step 1: Setup GCS Buckets**

```bash
export PROJECT_ID="helix-mlops-prod"
export REGION="us-central1"

# Create buckets
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://${PROJECT_ID}-raw-data
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://${PROJECT_ID}-processed-data
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://${PROJECT_ID}-ml-models
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://${PROJECT_ID}-feast-data

# Set lifecycle policies (optional - delete old data after 90 days)
cat > lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 90}
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://${PROJECT_ID}-raw-data
```

### **Step 2: Upload Data & Models**

```bash
# Upload trained models
gsutil cp ml_models/saved_models/covid_model.joblib \
  gs://${PROJECT_ID}-ml-models/covid/v1.0/

gsutil cp ml_models/saved_models/churn_model.joblib \
  gs://${PROJECT_ID}-ml-models/churn/v1.0/

# Upload processed data
gsutil cp data_pipeline/processed_data/*.csv \
  gs://${PROJECT_ID}-processed-data/
```

### **Step 3: Setup BigQuery**

```bash
# Create dataset
bq mk --dataset \
  --location=$REGION \
  --description="HELIX MLOps Feature Store" \
  ${PROJECT_ID}:helix_features

# Load COVID data
bq load \
  --source_format=CSV \
  --autodetect \
  helix_features.covid_features \
  gs://${PROJECT_ID}-processed-data/covid_train.csv

# Load Churn data
bq load \
  --source_format=CSV \
  --autodetect \
  helix_features.churn_features \
  gs://${PROJECT_ID}-processed-data/churn_train.csv

# Verify
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) as total FROM `helix_features.covid_features`'
```

### **Step 4: Setup Memorystore (Redis)**

```bash
# Create Redis instance for Feast online store
gcloud redis instances create helix-feast-online \
  --size=1 \
  --region=$REGION \
  --redis-version=redis_6_x \
  --tier=basic

# Get connection info
gcloud redis instances describe helix-feast-online \
  --region=$REGION \
  --format="get(host,port)"
```

### **Step 5: Configure Feast for GCP**

Create `feast_store/feature_store_gcp.yaml`:

```yaml
project: helix_mlops
registry: gs://helix-mlops-prod-feast-data/registry.db
provider: gcp
online_store:
  type: redis
  connection_string: "redis-host:6379"  # Replace with actual Redis host
offline_store:
  type: bigquery
  project_id: helix-mlops-prod
  dataset: helix_features
```

Apply Feast configuration:

```bash
cd feast_store
feast -c feature_store_gcp.yaml apply

# Materialize features to Redis
feast -c feature_store_gcp.yaml materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

### **Step 6: Store Secrets**

```bash
# Store Gemini API key
echo -n "YOUR_GEMINI_API_KEY" | \
  gcloud secrets create gemini-api-key \
  --data-file=- \
  --replication-policy="automatic"

# Store database URL (if using Cloud SQL)
echo -n "mysql://user:pass@host/db" | \
  gcloud secrets create database-url \
  --data-file=- \
  --replication-policy="automatic"

# Grant access to service account
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:helix-deployer@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### **Step 7: Build & Push Docker Images**

```bash
# Configure Docker for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Create Artifact Registry repository
gcloud artifacts repositories create helix-images \
  --repository-format=docker \
  --location=$REGION \
  --description="HELIX Docker Images"

# Build and push COVID service
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/covid-service:v1.0 \
  -f services/Dockerfile.covid .

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/covid-service:v1.0

# Build and push Churn service
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/churn-service:v1.0 \
  -f services/Dockerfile.churn .

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/churn-service:v1.0

# Build and push Frontend
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/frontend:v1.0 \
  -f Dockerfile .

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/frontend:v1.0
```

### **Step 8: Deploy to Cloud Run**

#### **Option A: Cloud Run (Serverless)**

```bash
# Deploy COVID service
gcloud run deploy covid-service \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/covid-service:v1.0 \
  --platform=managed \
  --region=$REGION \
  --allow-unauthenticated \
  --set-env-vars="GCS_BUCKET=${PROJECT_ID}-ml-models" \
  --set-secrets="GEMINI_API_KEY=gemini-api-key:latest" \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=10

# Deploy Churn service
gcloud run deploy churn-service \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/churn-service:v1.0 \
  --platform=managed \
  --region=$REGION \
  --allow-unauthenticated \
  --set-env-vars="GCS_BUCKET=${PROJECT_ID}-ml-models" \
  --set-secrets="GEMINI_API_KEY=gemini-api-key:latest" \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=10

# Deploy Frontend
gcloud run deploy helix-frontend \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/frontend:v1.0 \
  --platform=managed \
  --region=$REGION \
  --allow-unauthenticated \
  --set-env-vars="COVID_SERVICE_URL=https://covid-service-xxx.run.app,CHURN_SERVICE_URL=https://churn-service-xxx.run.app" \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=5

# Get service URLs
gcloud run services list --platform=managed
```

#### **Option B: GKE (Kubernetes)**

```bash
# Create GKE cluster
gcloud container clusters create helix-cluster \
  --region=$REGION \
  --num-nodes=3 \
  --machine-type=n1-standard-2 \
  --enable-autoscaling \
  --min-nodes=3 \
  --max-nodes=10 \
  --enable-autorepair \
  --enable-autoupgrade \
  --enable-stackdriver-kubernetes

# Get credentials
gcloud container clusters get-credentials helix-cluster --region=$REGION

# Create namespace
kubectl create namespace helix

# Deploy services
kubectl apply -f k8s/covid-deployment.yaml -n helix
kubectl apply -f k8s/churn-deployment.yaml -n helix
kubectl apply -f k8s/frontend-deployment.yaml -n helix
kubectl apply -f k8s/ingress.yaml -n helix

# Check status
kubectl get pods -n helix
kubectl get services -n helix
kubectl get ingress -n helix
```

### **Step 9: Setup Load Balancer & Domain**

```bash
# Reserve static IP
gcloud compute addresses create helix-ip \
  --global

# Get IP address
gcloud compute addresses describe helix-ip \
  --global \
  --format="get(address)"

# Configure DNS (example with Cloud DNS)
gcloud dns managed-zones create helix-zone \
  --dns-name="helix.example.com." \
  --description="HELIX MLOps Platform"

gcloud dns record-sets transaction start --zone=helix-zone

gcloud dns record-sets transaction add <STATIC_IP> \
  --name="helix.example.com." \
  --ttl=300 \
  --type=A \
  --zone=helix-zone

gcloud dns record-sets transaction execute --zone=helix-zone
```

### **Step 10: Setup Monitoring**

```bash
# Create Cloud Monitoring dashboard
gcloud monitoring dashboards create --config-from-file=monitoring/dashboard.json

# Create alerting policy
gcloud alpha monitoring policies create \
  --notification-channels=<CHANNEL_ID> \
  --display-name="HELIX High Error Rate" \
  --condition-display-name="Error rate > 5%" \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s
```

---

## ⚙️ Configuration

### **Environment Variables**

Update services to read from Secret Manager:

```python
# services/covid_service_real.py
from google.cloud import secretmanager

def get_secret(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

GEMINI_API_KEY = get_secret("gemini-api-key")
```

### **Feast Configuration**

Update `feast_store/features.py` for BigQuery:

```python
from feast import FeatureView, Field
from feast.types import Float64, Int64, String
from feast.data_source import BigQuerySource
from datetime import timedelta

# COVID Features
covid_source = BigQuerySource(
    table="helix-mlops-prod.helix_features.covid_features",
    timestamp_field="event_timestamp",
)

covid_fv = FeatureView(
    name="covid_features",
    entities=["country_name"],
    ttl=timedelta(days=1),
    schema=[
        Field(name="confirmed_cases", dtype=Int64),
        Field(name="deaths", dtype=Int64),
        Field(name="recovered", dtype=Int64),
        Field(name="active_cases", dtype=Int64),
        Field(name="death_rate", dtype=Float64),
        Field(name="recovery_rate", dtype=Float64),
        Field(name="risk_score", dtype=Float64),
    ],
    source=covid_source,
)
```

---

## 📊 Monitoring & Logging

### **Cloud Logging**

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=covid-service" \
  --limit=50 \
  --format=json

# Stream logs
gcloud logging tail "resource.type=cloud_run_revision" \
  --format="value(textPayload)"
```

### **Cloud Monitoring**

Access dashboards at: https://console.cloud.google.com/monitoring

**Key Metrics to Monitor:**
- Request latency (p50, p95, p99)
- Error rate (4xx, 5xx)
- CPU & memory utilization
- Request count
- Model inference time
- Feast feature retrieval time

### **Cloud Trace**

Enable distributed tracing:

```python
from google.cloud import trace_v1

tracer = trace_v1.TraceServiceClient()
```

---

## 💰 Cost Estimation

### **Monthly Cost Breakdown (Medium Traffic)**

| Service | Usage | Cost |
|---------|-------|------|
| Cloud Run (3 services) | 1M requests, 2GB RAM | $100 |
| GKE (alternative) | 3 nodes, n1-standard-2 | $300 |
| BigQuery | 100GB storage, 1TB queries | $50 |
| Cloud Storage | 500GB storage, 1TB egress | $30 |
| Memorystore Redis | 1GB basic tier | $50 |
| Load Balancer | 1TB traffic | $30 |
| Cloud Logging | 50GB logs | $20 |
| Cloud Monitoring | Standard metrics | $15 |
| Artifact Registry | 10GB images | $10 |
| Secret Manager | 10 secrets | $2 |
| **Total (Cloud Run)** | | **~$307/month** |
| **Total (GKE)** | | **~$507/month** |

**Cost Optimization Tips:**
- Use Cloud Run for variable traffic (pay-per-request)
- Use GKE for consistent high traffic (cheaper at scale)
- Enable autoscaling to minimize idle resources
- Use committed use discounts (30-70% savings)
- Set up budget alerts

---

## 🔧 Troubleshooting

### **Cloud Run Issues**

```bash
# Check service logs
gcloud run services logs read covid-service --limit=100

# Describe service
gcloud run services describe covid-service --region=$REGION

# Test service locally
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/covid-service:v1.0
```

### **BigQuery Issues**

```bash
# Test query
bq query --use_legacy_sql=false \
  'SELECT * FROM `helix_features.covid_features` LIMIT 10'

# Check dataset permissions
bq show helix_features
```

### **Feast Issues**

```bash
# Test Feast connection
feast -c feature_store_gcp.yaml feature-views list

# Materialize features manually
feast -c feature_store_gcp.yaml materialize \
  2024-01-01T00:00:00 \
  2024-12-31T23:59:59
```

### **Networking Issues**

```bash
# Test service connectivity
curl -X POST https://covid-service-xxx.run.app/predict/covid \
  -H "Content-Type: application/json" \
  -d '{"country_name": "USA"}'

# Check VPC connector (if using)
gcloud compute networks vpc-access connectors describe helix-connector \
  --region=$REGION
```

---

## 🚀 CI/CD Pipeline

### **Cloud Build Configuration**

Create `cloudbuild.yaml`:

```yaml
steps:
  # Build COVID service
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${_REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/covid-service:$SHORT_SHA', '-f', 'services/Dockerfile.covid', '.']
  
  # Push COVID service
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${_REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/covid-service:$SHORT_SHA']
  
  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'covid-service'
      - '--image=${_REGION}-docker.pkg.dev/${PROJECT_ID}/helix-images/covid-service:$SHORT_SHA'
      - '--region=${_REGION}'
      - '--platform=managed'

substitutions:
  _REGION: us-central1

options:
  machineType: 'N1_HIGHCPU_8'
```

### **Setup Trigger**

```bash
# Connect GitHub repository
gcloud builds triggers create github \
  --repo-name=helix \
  --repo-owner=your-username \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml
```

---

## 📚 Additional Resources

- [GCP Documentation](https://cloud.google.com/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [Feast on GCP](https://docs.feast.dev/how-to-guides/feast-gcp-aws)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)

---

## 🎯 Next Steps

1. **Security Hardening**
   - Enable VPC Service Controls
   - Configure Cloud Armor (WAF)
   - Set up IAM conditions
   - Enable audit logging

2. **Performance Optimization**
   - Enable Cloud CDN
   - Configure caching strategies
   - Optimize BigQuery queries
   - Tune Redis cache

3. **Advanced Features**
   - A/B testing with Traffic Splitting
   - Model versioning with Vertex AI
   - Data drift detection
   - Automated retraining pipeline

4. **Disaster Recovery**
   - Multi-region deployment
   - Backup strategies
   - Failover procedures
   - RTO/RPO planning

---

**Phase 3 Complete!** 🎉 HELIX is now production-ready on GCP with enterprise-grade scalability, reliability, and observability.
