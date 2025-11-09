#!/bin/bash

# Destroy HELIX infrastructure

export PATH=$PATH:$HOME/.pulumi/bin
export PULUMI_CONFIG_PASSPHRASE="helix-mlops-2024"

echo "⚠️  WARNING: This will destroy all infrastructure!"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

echo "🗑️  Destroying infrastructure..."
pulumi destroy --yes

echo ""
echo "✅ Infrastructure destroyed"
