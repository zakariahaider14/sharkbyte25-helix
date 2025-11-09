#!/bin/bash

# HELIX Pulumi Setup Script
# This script sets up the Pulumi environment and makes deployment easier

set -e

echo "🚀 HELIX Pulumi Setup"
echo "===================="

# Add Pulumi to PATH
export PATH=$PATH:$HOME/.pulumi/bin

# Set passphrase (you can change this)
export PULUMI_CONFIG_PASSPHRASE="helix-mlops-2024"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📦 Checking Pulumi installation...${NC}"
if ! command -v pulumi &> /dev/null; then
    echo "❌ Pulumi not found. Installing..."
    curl -fsSL https://get.pulumi.com | sh
    export PATH=$PATH:$HOME/.pulumi/bin
fi

echo -e "${GREEN}✅ Pulumi $(pulumi version) installed${NC}"

echo -e "${BLUE}🔐 Logging in to Pulumi (local backend)...${NC}"
pulumi login --local

echo -e "${BLUE}📋 Checking stack...${NC}"
if ! pulumi stack ls | grep -q "sharkbyte"; then
    echo "Creating new stack 'sharkbyte'..."
    pulumi stack init sharkbyte
else
    echo "Stack 'sharkbyte' already exists"
    pulumi stack select sharkbyte
fi

echo -e "${BLUE}⚙️  Configuring GCP settings...${NC}"
pulumi config set gcp:project gen-lang-client-0520631489
pulumi config set gcp:region us-central1

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "📝 Next steps:"
echo "  1. Preview infrastructure: ./preview.sh"
echo "  2. Deploy infrastructure: ./deploy.sh"
echo "  3. Destroy infrastructure: ./destroy.sh"
echo ""
echo "💡 Tip: Set PULUMI_CONFIG_PASSPHRASE in your shell:"
echo "   export PULUMI_CONFIG_PASSPHRASE='helix-mlops-2024'"
