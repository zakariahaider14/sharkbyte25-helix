/**
 * HELIX MLOps Infrastructure with Pulumi
 * 
 * This Pulumi program provisions:
 * - Cloud Storage buckets for data and models
 * - BigQuery dataset for features
 * - Artifact Registry for Docker images
 * - Cloud Run services (COVID, Churn, Frontend, MLflow, Prometheus)
 * - Secret Manager for API keys
 * - IAM permissions
 */

import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";
import * as docker from "@pulumi/docker";

// Get configuration
const config = new pulumi.Config();
const gcpConfig = new pulumi.Config("gcp");
const projectId = gcpConfig.require("project");
const region = gcpConfig.get("region") || "us-central1";

// Stack reference for environment-specific configs
const stack = pulumi.getStack();

// ============================================================================
// 1. CLOUD STORAGE BUCKETS
// ============================================================================

const rawDataBucket = new gcp.storage.Bucket("raw-data-bucket", {
    name: `${projectId}-raw-data`,
    location: region.toUpperCase().split("-")[0], // US, EU, ASIA
    uniformBucketLevelAccess: true,
    versioning: {
        enabled: true,
    },
    lifecycleRules: [{
        action: { type: "Delete" },
        condition: { age: 90 }, // Delete after 90 days
    }],
});

const processedDataBucket = new gcp.storage.Bucket("processed-data-bucket", {
    name: `${projectId}-processed-data`,
    location: region.toUpperCase().split("-")[0],
    uniformBucketLevelAccess: true,
    versioning: {
        enabled: true,
    },
});

const mlModelsBucket = new gcp.storage.Bucket("ml-models-bucket", {
    name: `${projectId}-ml-models`,
    location: region.toUpperCase().split("-")[0],
    uniformBucketLevelAccess: true,
    versioning: {
        enabled: true,
    },
});

const feastDataBucket = new gcp.storage.Bucket("feast-data-bucket", {
    name: `${projectId}-feast-data`,
    location: region.toUpperCase().split("-")[0],
    uniformBucketLevelAccess: true,
});

const mlflowArtifactsBucket = new gcp.storage.Bucket("mlflow-artifacts-bucket", {
    name: `${projectId}-mlflow-artifacts`,
    location: region.toUpperCase().split("-")[0],
    uniformBucketLevelAccess: true,
    versioning: {
        enabled: true,
    },
});

// ============================================================================
// 2. BIGQUERY DATASET
// ============================================================================

const helixDataset = new gcp.bigquery.Dataset("helix-features", {
    datasetId: "helix_features",
    location: region.toUpperCase().split("-")[0],
    description: "HELIX feature store for COVID and Churn predictions",
});

// COVID features table
const covidFeaturesTable = new gcp.bigquery.Table("covid-features-table", {
    datasetId: helixDataset.datasetId,
    tableId: "covid_features",
    deletionProtection: false,
    schema: JSON.stringify([
        { name: "country_name", type: "STRING", mode: "REQUIRED" },
        { name: "date", type: "TIMESTAMP", mode: "REQUIRED" },
        { name: "confirmed_cases", type: "FLOAT64", mode: "NULLABLE" },
        { name: "deaths", type: "FLOAT64", mode: "NULLABLE" },
        { name: "recovered", type: "FLOAT64", mode: "NULLABLE" },
        { name: "active_cases", type: "FLOAT64", mode: "NULLABLE" },
        { name: "death_rate", type: "FLOAT64", mode: "NULLABLE" },
        { name: "recovery_rate", type: "FLOAT64", mode: "NULLABLE" },
        { name: "risk_score", type: "FLOAT64", mode: "NULLABLE" },
    ]),
});

// Churn features table
const churnFeaturesTable = new gcp.bigquery.Table("churn-features-table", {
    datasetId: helixDataset.datasetId,
    tableId: "churn_features",
    deletionProtection: false,
    schema: JSON.stringify([
        { name: "customer_id", type: "STRING", mode: "REQUIRED" },
        { name: "event_timestamp", type: "TIMESTAMP", mode: "REQUIRED" },
        { name: "SeniorCitizen", type: "INT64", mode: "NULLABLE" },
        { name: "Partner", type: "INT64", mode: "NULLABLE" },
        { name: "Dependents", type: "INT64", mode: "NULLABLE" },
        { name: "tenure", type: "FLOAT64", mode: "NULLABLE" },
        { name: "MonthlyCharges", type: "FLOAT64", mode: "NULLABLE" },
        { name: "TotalCharges", type: "FLOAT64", mode: "NULLABLE" },
    ]),
});

// ============================================================================
// 3. ARTIFACT REGISTRY
// ============================================================================

const artifactRegistry = new gcp.artifactregistry.Repository("helix-images", {
    repositoryId: "helix-images",
    location: region,
    format: "DOCKER",
    description: "Docker images for HELIX MLOps services",
});

// ============================================================================
// 4. SECRET MANAGER
// ============================================================================

const geminiApiKeySecret = new gcp.secretmanager.Secret("gemini-api-key", {
    secretId: "gemini-api-key",
    replication: {
        auto: {},
    },
});

// ============================================================================
// 5. SERVICE ACCOUNTS & IAM
// ============================================================================

const helixServiceAccount = new gcp.serviceaccount.Account("helix-deployer", {
    accountId: "helix-deployer",
    displayName: "HELIX Deployment Service Account",
});

// Grant permissions to service account
const storageAdminBinding = new gcp.projects.IAMMember("storage-admin", {
    project: projectId,
    role: "roles/storage.admin",
    member: pulumi.interpolate`serviceAccount:${helixServiceAccount.email}`,
});

const bigqueryAdminBinding = new gcp.projects.IAMMember("bigquery-admin", {
    project: projectId,
    role: "roles/bigquery.admin",
    member: pulumi.interpolate`serviceAccount:${helixServiceAccount.email}`,
});

const secretAccessorBinding = new gcp.projects.IAMMember("secret-accessor", {
    project: projectId,
    role: "roles/secretmanager.secretAccessor",
    member: pulumi.interpolate`serviceAccount:${helixServiceAccount.email}`,
});

// ============================================================================
// 6. CLOUD RUN SERVICES
// ============================================================================

// COVID Prediction Service
const covidService = new gcp.cloudrun.Service("covid-service", {
    name: "covid-service",
    location: region,
    template: {
        spec: {
            serviceAccountName: helixServiceAccount.email,
            containers: [{
                image: pulumi.interpolate`${region}-docker.pkg.dev/${projectId}/helix-images/covid-service:latest`,
                ports: [{ containerPort: 8080 }],
                resources: {
                    limits: {
                        memory: "2Gi",
                        cpu: "2",
                    },
                },
                envs: [
                    {
                        name: "GCS_BUCKET",
                        value: mlModelsBucket.name,
                    },
                    {
                        name: "PROJECT_ID",
                        value: projectId,
                    },
                    {
                        name: "PROMETHEUS_ENABLED",
                        value: "true",
                    },
                ],
            }],
        },
        metadata: {
            annotations: {
                "autoscaling.knative.dev/minScale": "0",
                "autoscaling.knative.dev/maxScale": "10",
            },
        },
    },
    traffics: [{
        percent: 100,
        latestRevision: true,
    }],
});

// Make COVID service publicly accessible
const covidServiceIamPolicy = new gcp.cloudrun.IamMember("covid-service-public", {
    service: covidService.name,
    location: region,
    role: "roles/run.invoker",
    member: "allUsers",
});

// Churn Prediction Service
const churnService = new gcp.cloudrun.Service("churn-service", {
    name: "churn-service",
    location: region,
    template: {
        spec: {
            serviceAccountName: helixServiceAccount.email,
            containers: [{
                image: pulumi.interpolate`${region}-docker.pkg.dev/${projectId}/helix-images/churn-service:latest`,
                ports: [{ containerPort: 8080 }],
                resources: {
                    limits: {
                        memory: "2Gi",
                        cpu: "2",
                    },
                },
                envs: [
                    {
                        name: "GCS_BUCKET",
                        value: mlModelsBucket.name,
                    },
                    {
                        name: "PROJECT_ID",
                        value: projectId,
                    },
                    {
                        name: "PROMETHEUS_ENABLED",
                        value: "true",
                    },
                ],
            }],
        },
        metadata: {
            annotations: {
                "autoscaling.knative.dev/minScale": "0",
                "autoscaling.knative.dev/maxScale": "10",
            },
        },
    },
    traffics: [{
        percent: 100,
        latestRevision: true,
    }],
});

const churnServiceIamPolicy = new gcp.cloudrun.IamMember("churn-service-public", {
    service: churnService.name,
    location: region,
    role: "roles/run.invoker",
    member: "allUsers",
});

// MLflow Tracking Server
const mlflowService = new gcp.cloudrun.Service("mlflow-service", {
    name: "mlflow-service",
    location: region,
    template: {
        spec: {
            serviceAccountName: helixServiceAccount.email,
            containers: [{
                image: "ghcr.io/mlflow/mlflow:latest",
                ports: [{ containerPort: 5000 }],
                resources: {
                    limits: {
                        memory: "1Gi",
                        cpu: "1",
                    },
                },
                args: [
                    "mlflow",
                    "server",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "5000",
                    "--backend-store-uri",
                    pulumi.interpolate`gs://${mlflowArtifactsBucket.name}/mlflow-backend`,
                    "--default-artifact-root",
                    pulumi.interpolate`gs://${mlflowArtifactsBucket.name}/mlflow-artifacts`,
                ],
            }],
        },
        metadata: {
            annotations: {
                "autoscaling.knative.dev/minScale": "1",
                "autoscaling.knative.dev/maxScale": "3",
            },
        },
    },
    traffics: [{
        percent: 100,
        latestRevision: true,
    }],
});

const mlflowServiceIamPolicy = new gcp.cloudrun.IamMember("mlflow-service-public", {
    service: mlflowService.name,
    location: region,
    role: "roles/run.invoker",
    member: "allUsers",
});

// Prometheus Service
const prometheusService = new gcp.cloudrun.Service("prometheus-service", {
    name: "prometheus-service",
    location: region,
    template: {
        spec: {
            serviceAccountName: helixServiceAccount.email,
            containers: [{
                image: "prom/prometheus:latest",
                ports: [{ containerPort: 9090 }],
                resources: {
                    limits: {
                        memory: "1Gi",
                        cpu: "1",
                    },
                },
            }],
        },
        metadata: {
            annotations: {
                "autoscaling.knative.dev/minScale": "1",
                "autoscaling.knative.dev/maxScale": "2",
            },
        },
    },
    traffics: [{
        percent: 100,
        latestRevision: true,
    }],
});

const prometheusServiceIamPolicy = new gcp.cloudrun.IamMember("prometheus-service-public", {
    service: prometheusService.name,
    location: region,
    role: "roles/run.invoker",
    member: "allUsers",
});

// Frontend Service
const frontendService = new gcp.cloudrun.Service("helix-frontend", {
    name: "helix-frontend",
    location: region,
    template: {
        spec: {
            serviceAccountName: helixServiceAccount.email,
            containers: [{
                image: pulumi.interpolate`${region}-docker.pkg.dev/${projectId}/helix-images/helix-frontend:latest`,
                ports: [{ containerPort: 8080 }],
                resources: {
                    limits: {
                        memory: "1Gi",
                        cpu: "1",
                    },
                },
                envs: [
                    {
                        name: "NODE_ENV",
                        value: "production",
                    },
                    {
                        name: "COVID_SERVICE_URL",
                        value: covidService.statuses[0].url,
                    },
                    {
                        name: "CHURN_SERVICE_URL",
                        value: churnService.statuses[0].url,
                    },
                    {
                        name: "MLFLOW_TRACKING_URI",
                        value: mlflowService.statuses[0].url,
                    },
                    {
                        name: "BUILT_IN_FORGE_API_URL",
                        value: "https://generativelanguage.googleapis.com/v1beta/openai",
                    },
                ],
            }],
        },
        metadata: {
            annotations: {
                "autoscaling.knative.dev/minScale": "0",
                "autoscaling.knative.dev/maxScale": "5",
            },
        },
    },
    traffics: [{
        percent: 100,
        latestRevision: true,
    }],
});

const frontendServiceIamPolicy = new gcp.cloudrun.IamMember("frontend-service-public", {
    service: frontendService.name,
    location: region,
    role: "roles/run.invoker",
    member: "allUsers",
});

// ============================================================================
// 7. EXPORTS
// ============================================================================

export const rawDataBucketName = rawDataBucket.name;
export const processedDataBucketName = processedDataBucket.name;
export const mlModelsBucketName = mlModelsBucket.name;
export const feastDataBucketName = feastDataBucket.name;
export const mlflowArtifactsBucketName = mlflowArtifactsBucket.name;

export const bigQueryDataset = helixDataset.datasetId;
export const artifactRegistryUrl = pulumi.interpolate`${region}-docker.pkg.dev/${projectId}/helix-images`;

export const covidServiceUrl = covidService.statuses[0].url;
export const churnServiceUrl = churnService.statuses[0].url;
export const mlflowServiceUrl = mlflowService.statuses[0].url;
export const prometheusServiceUrl = prometheusService.statuses[0].url;
export const frontendServiceUrl = frontendService.statuses[0].url;

export const serviceAccountEmail = helixServiceAccount.email;
