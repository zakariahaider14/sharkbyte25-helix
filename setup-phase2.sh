#!/bin/bash

# Phase 2 Setup Script
# Automates the setup of real ML pipeline

set -e  # Exit on error

echo "=========================================="
echo "PHASE 2 SETUP - REAL ML PIPELINE"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check prerequisites
echo "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi
print_success "Python 3 found"

# Check pip
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 not found. Please install pip"
    exit 1
fi
print_success "pip3 found"

# Check MySQL
if ! command -v mysql &> /dev/null; then
    print_error "MySQL not found. Please install MySQL"
    exit 1
fi
print_success "MySQL found"

# Check Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    print_error "Kaggle credentials not found at ~/.kaggle/kaggle.json"
    echo "Please:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Create New API Token"
    echo "3. Move kaggle.json to ~/.kaggle/"
    exit 1
fi
print_success "Kaggle credentials found"

echo ""
echo "=========================================="
echo "STEP 1: Installing Dependencies"
echo "=========================================="

# Install data pipeline dependencies
print_info "Installing data pipeline dependencies..."
pip3 install -q -r data_pipeline/requirements.txt
print_success "Data pipeline dependencies installed"

# Install ML model dependencies
print_info "Installing ML model dependencies..."
pip3 install -q -r ml_models/requirements.txt
print_success "ML model dependencies installed"

# Install real service dependencies
print_info "Installing real service dependencies..."
pip3 install -q -r services/requirements_real.txt
print_success "Real service dependencies installed"

echo ""
echo "=========================================="
echo "STEP 2: Downloading Datasets"
echo "=========================================="

cd data_pipeline
python3 download_datasets.py

# Get the download paths
CHURN_PATH=$(find ~/.cache/kagglehub/datasets/blastchar/telco-customer-churn -type d -name "versions" 2>/dev/null | head -1)
if [ -z "$CHURN_PATH" ]; then
    CHURN_PATH=$(find ~/.cache/kagglehub/datasets/blastchar -type d 2>/dev/null | head -1)
fi

COVID_PATH=$(find ~/.cache/kagglehub/datasets/imdevskp/corona-virus-report -type d -name "versions" 2>/dev/null | head -1)
if [ -z "$COVID_PATH" ]; then
    COVID_PATH=$(find ~/.cache/kagglehub/datasets/imdevskp -type d 2>/dev/null | head -1)
fi

if [ -z "$CHURN_PATH" ] || [ -z "$COVID_PATH" ]; then
    print_error "Could not find downloaded datasets"
    print_info "Please run preprocessing manually with the correct paths"
    exit 1
fi

echo ""
echo "=========================================="
echo "STEP 3: Preprocessing Data"
echo "=========================================="

print_info "Preprocessing Churn data..."
python3 preprocess_churn.py "$CHURN_PATH"
print_success "Churn data preprocessed"

print_info "Preprocessing COVID data..."
python3 preprocess_covid.py "$COVID_PATH"
print_success "COVID data preprocessed"

echo ""
echo "=========================================="
echo "STEP 4: Loading Data into MySQL"
echo "=========================================="

print_info "Loading data into MySQL..."
python3 load_to_mysql.py
print_success "Data loaded into MySQL"

echo ""
echo "=========================================="
echo "STEP 5: Training ML Models"
echo "=========================================="

cd ../ml_models

print_info "Training COVID model..."
python3 train_covid_model.py
print_success "COVID model trained"

print_info "Training Churn model..."
python3 train_churn_model.py
print_success "Churn model trained"

echo ""
echo "=========================================="
echo "STEP 6: Setting up Feast Feature Store"
echo "=========================================="

cd ../feast_store

print_info "Preparing Feast data..."
python3 prepare_feast_data.py
print_success "Feast data prepared"

print_info "Applying Feast configuration..."
feast apply
print_success "Feast configuration applied"

print_info "Materializing features..."
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
print_success "Features materialized"

cd ..

echo ""
echo "=========================================="
echo "PHASE 2 SETUP COMPLETE!"
echo "=========================================="
echo ""
print_success "All components successfully set up!"
echo ""
echo "Next steps:"
echo "1. Start COVID service: python3 services/covid_service_real.py"
echo "2. Start Churn service: python3 services/churn_service_real.py"
echo "3. Start web app: pnpm dev"
echo "4. Open http://localhost:3000/agent"
echo ""
echo "Or use the automated startup script:"
echo "  ./start-real-services.sh"
echo ""
print_info "See PHASE2_SETUP.md for detailed documentation"
