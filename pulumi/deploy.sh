#!/bin/bash

# Deploy HELIX infrastructure with Pulumi

export PATH=$PATH:$HOME/.pulumi/bin
export PULUMI_CONFIG_PASSPHRASE="helix-mlops-2024"

echo "🚀 Deploying HELIX infrastructure..."
pulumi up --yes

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Service URLs:"
pulumi stack output
