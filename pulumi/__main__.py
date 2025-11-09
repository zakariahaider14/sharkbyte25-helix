"""
MLOps Infrastructure as Code using Pulumi for Google Cloud Platform.

This module defines all cloud resources required for the dual-model MLOps system:
- GCS buckets for data, features, and model artifacts
- GKE cluster for model serving and monitoring
- BigQuery dataset for Feast offline store
- Cloud Memorystore (Redis) for Feast online store
- Artifact Registry for Docker images
- VPC network and security configurations
"""

import pulumi
import pulumi_gcp as gcp
import pulumi_kubernetes as k8s
import json

# Load configuration
config = pulumi.Config()
gcp_project = config.require('gcp:project')
gcp_region = config.require('gcp:region')
environment = config.get('mlops:environment') or 'dev'
cluster_name = config.get('mlops:cluster_name') or 'mlops-gke'
cluster_version = config.get('mlops:cluster_version') or '1.27'
node_count = config.get_int('mlops:node_count') or 3
machine_type = config.get('mlops:machine_type') or 'n1-standard-2'
gcs_bucket_prefix = config.get('mlops:gcs_bucket_prefix') or 'mlops'
bigquery_dataset = config.get('mlops:bigquery_dataset') or 'mlops'
redis_tier = config.get('mlops:redis_tier') or 'basic'
redis_size_gb = config.get_int('mlops:redis_size_gb') or 2
artifact_registry_repo = config.get('mlops:artifact_registry_repo') or 'mlops-docker'

# Set GCP provider configuration
gcp_provider = gcp.Provider('gcp', project=gcp_project, region=gcp_region)

# ============================================================================
# 1. GCS BUCKETS FOR DATA, FEATURES, AND MODEL ARTIFACTS
# ============================================================================

def create_gcs_bucket(name, description):
    """Create a GCS bucket with versioning and lifecycle policies."""
    return gcp.storage.Bucket(
        name,
        project=gcp_project,
        location=gcp_region,
        versioning=gcp.storage.BucketVersioningArgs(enabled=True),
        uniform_bucket_level_access=gcp.storage.BucketUniformBucketLevelAccessArgs(enabled=True),
        opts=pulumi.ResourceOptions(provider=gcp_provider)
    )

# Create GCS buckets
raw_data_bucket = create_gcs_bucket(
    f'{gcs_bucket_prefix}-raw-data-{environment}',
    'Raw datasets from Kaggle'
)

feature_data_bucket = create_gcs_bucket(
    f'{gcs_bucket_prefix}-feature-data-{environment}',
    'Processed features for model training'
)

model_artifacts_bucket = create_gcs_bucket(
    f'{gcs_bucket_prefix}-model-artifacts-{environment}',
    'Trained model artifacts and checkpoints'
)

metadata_store_bucket = create_gcs_bucket(
    f'{gcs_bucket_prefix}-metadata-store-{environment}',
    'Metadata, configs, and experiment tracking'
)

# ============================================================================
# 2. VPC NETWORK AND SECURITY
# ============================================================================

# Create VPC network
vpc_network = gcp.compute.Network(
    'mlops-vpc',
    project=gcp_project,
    auto_create_subnetworks=False,
    opts=pulumi.ResourceOptions(provider=gcp_provider)
)

# Create subnet for GKE cluster
gke_subnet = gcp.compute.Subnetwork(
    'mlops-gke-subnet',
    project=gcp_project,
    network=vpc_network.id,
    region=gcp_region,
    ip_cidr_range='10.0.0.0/20',
    secondary_ip_range=[
        gcp.compute.SubnetworkSecondaryIpRangeArgs(
            range_name='pods',
            ip_cidr_range='10.4.0.0/14',
        ),
        gcp.compute.SubnetworkSecondaryIpRangeArgs(
            range_name='services',
            ip_cidr_range='10.0.16.0/20',
        ),
    ],
    opts=pulumi.ResourceOptions(provider=gcp_provider)
)

# Create firewall rule to allow internal communication
internal_firewall = gcp.compute.Firewall(
    'mlops-internal-firewall',
    project=gcp_project,
    network=vpc_network.name,
    allows=[
        gcp.compute.FirewallAllowArgs(
            protocol='tcp',
            ports=['0-65535'],
        ),
        gcp.compute.FirewallAllowArgs(
            protocol='udp',
            ports=['0-65535'],
        ),
    ],
    source_ranges=['10.0.0.0/8'],
    opts=pulumi.ResourceOptions(provider=gcp_provider)
)

# ============================================================================
# 3. GKE CLUSTER FOR MODEL SERVING AND MONITORING
# ============================================================================

gke_cluster = gcp.container.Cluster(
    cluster_name,
    project=gcp_project,
    location=gcp_region,
    initial_node_count=node_count,
    min_master_version=cluster_version,
    node_version=cluster_version,
    network=vpc_network.name,
    subnetwork=gke_subnet.name,
    enable_stackdriver_kubernetes=True,
    enable_ip_allocation_policy=True,
    cluster_secondary_range_name='pods',
    services_secondary_range_name='services',
    addons_config=gcp.container.ClusterAddonsConfigArgs(
        http_load_balancing=gcp.container.ClusterAddonsConfigHttpLoadBalancingArgs(disabled=False),
        horizontal_pod_autoscaling=gcp.container.ClusterAddonsConfigHorizontalPodAutoscalingArgs(disabled=False),
        network_policy_config=gcp.container.ClusterAddonsConfigNetworkPolicyConfigArgs(disabled=False),
    ),
    node_pool=gcp.container.ClusterNodePoolArgs(
        name='default',
        initial_node_count=node_count,
        node_config=gcp.container.ClusterNodePoolNodeConfigArgs(
            machine_type=machine_type,
            disk_size_gb=100,
            oauth_scopes=[
                'https://www.googleapis.com/auth/cloud-platform',
            ],
        ),
        autoscaling=gcp.container.ClusterNodePoolAutoscalingArgs(
            min_node_count=1,
            max_node_count=10,
        ),
    ),
    opts=pulumi.ResourceOptions(provider=gcp_provider)
)

# ============================================================================
# 4. ARTIFACT REGISTRY FOR DOCKER IMAGES
# ============================================================================

artifact_registry = gcp.artifactregistry.Repository(
    artifact_registry_repo,
    project=gcp_project,
    location=gcp_region,
    repository_id=artifact_registry_repo,
    format='DOCKER',
    description='Docker images for MLOps model serving microservices',
    opts=pulumi.ResourceOptions(provider=gcp_provider)
)

# ============================================================================
# 5. BIGQUERY DATASET FOR FEAST OFFLINE STORE
# ============================================================================

bigquery_dataset_resource = gcp.bigquery.Dataset(
    'mlops-feast-offline',
    project=gcp_project,
    dataset_id=bigquery_dataset,
    location=gcp_region,
    description='Feast offline feature store for ML model training',
    default_table_expiration_ms=None,
    opts=pulumi.ResourceOptions(provider=gcp_provider)
)

# ============================================================================
# 6. CLOUD MEMORYSTORE (REDIS) FOR FEAST ONLINE STORE
# ============================================================================

redis_instance = gcp.redis.Instance(
    'mlops-feast-online',
    project=gcp_project,
    region=gcp_region,
    tier=redis_tier,
    memory_size_gb=redis_size_gb,
    redis_version='7.0',
    display_name='Feast Online Feature Store',
    authorized_network=vpc_network.id,
    connect_mode='DIRECT_PEERING',
    opts=pulumi.ResourceOptions(provider=gcp_provider)
)

# ============================================================================
# 7. KUBERNETES PROVIDER FOR CLUSTER MANAGEMENT
# ============================================================================

k8s_provider = k8s.Provider(
    'k8s-provider',
    kubeconfig=pulumi.Output.concat(
        '{"apiVersion": "v1", "clusters": [{"cluster": {"certificate-authority-data": "',
        gke_cluster.master_auth.cluster_ca_certificate,
        '", "server": "https://',
        gke_cluster.endpoint,
        '"}, "name": "',
        gke_cluster.name,
        '"}], "contexts": [{"context": {"cluster": "',
        gke_cluster.name,
        '", "user": "',
        gke_cluster.name,
        '"}, "name": "',
        gke_cluster.name,
        '"}], "current-context": "',
        gke_cluster.name,
        '", "kind": "Config", "preferences": {}, "users": [{"name": "',
        gke_cluster.name,
        '", "user": {"auth-provider": {"config": {"cmd-args": "config config-helper --representation=json", "cmd-path": "gcloud", "expiry-key": "{.credential.token_expiry}", "token-key": "{.credential.access_token}"}, "name": "gcp"}}}]}'
    ),
    opts=pulumi.ResourceOptions(provider=gcp_provider)
)

# ============================================================================
# 8. NAMESPACES FOR MICROSERVICES AND MONITORING
# ============================================================================

mlops_namespace = k8s.core.v1.Namespace(
    'mlops-namespace',
    metadata={'name': 'mlops'},
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)

monitoring_namespace = k8s.core.v1.Namespace(
    'monitoring-namespace',
    metadata={'name': 'monitoring'},
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)

# ============================================================================
# 9. EXPORT OUTPUTS
# ============================================================================

pulumi.export('gcp_project', gcp_project)
pulumi.export('gcp_region', gcp_region)
pulumi.export('environment', environment)

# GCS Buckets
pulumi.export('raw_data_bucket', raw_data_bucket.name)
pulumi.export('feature_data_bucket', feature_data_bucket.name)
pulumi.export('model_artifacts_bucket', model_artifacts_bucket.name)
pulumi.export('metadata_store_bucket', metadata_store_bucket.name)

# GKE Cluster
pulumi.export('gke_cluster_name', gke_cluster.name)
pulumi.export('gke_cluster_endpoint', gke_cluster.endpoint)
pulumi.export('gke_cluster_ca_certificate', gke_cluster.master_auth.cluster_ca_certificate)

# Artifact Registry
pulumi.export('artifact_registry_repository', artifact_registry.repository_id)
pulumi.export('artifact_registry_location', artifact_registry.location)

# BigQuery
pulumi.export('bigquery_dataset', bigquery_dataset_resource.dataset_id)

# Redis
pulumi.export('redis_host', redis_instance.host)
pulumi.export('redis_port', redis_instance.port)

# VPC Network
pulumi.export('vpc_network_name', vpc_network.name)
pulumi.export('gke_subnet_name', gke_subnet.name)
