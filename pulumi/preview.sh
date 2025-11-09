#!/bin/bash

# Preview Pulumi infrastructure changes

export PATH=$PATH:$HOME/.pulumi/bin
export PULUMI_CONFIG_PASSPHRASE="helix-mlops-2024"

echo "🔍 Previewing infrastructure changes..."
pulumi preview
