#!/bin/bash

# Test CI/CD Pipeline Configuration
# This script verifies your CI/CD setup is ready

set -e

echo "🔄 Testing CI/CD Pipeline Configuration"
echo "========================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0

# 1. Check GitHub workflow file exists
echo -e "\n${BLUE}1. Checking GitHub Actions workflow...${NC}"
if [ -f ".github/workflows/deploy.yml" ]; then
    echo -e "${GREEN}✅ Workflow file exists${NC}"
else
    echo -e "${RED}❌ Workflow file not found${NC}"
    ERRORS=$((ERRORS + 1))
fi

# 2. Check if git repository is initialized
echo -e "\n${BLUE}2. Checking Git repository...${NC}"
if [ -d ".git" ]; then
    echo -e "${GREEN}✅ Git repository initialized${NC}"
    
    # Check remote
    if git remote -v | grep -q "origin"; then
        REMOTE_URL=$(git remote get-url origin)
        echo -e "${GREEN}✅ Remote configured: ${REMOTE_URL}${NC}"
    else
        echo -e "${YELLOW}⚠️  No remote configured${NC}"
        echo "   Add remote with: git remote add origin <your-repo-url>"
    fi
else
    echo -e "${RED}❌ Not a git repository${NC}"
    ERRORS=$((ERRORS + 1))
fi

# 3. Check required files for CI/CD
echo -e "\n${BLUE}3. Checking required files...${NC}"

FILES=(
    "Dockerfile.covid"
    "Dockerfile.churn"
    "Dockerfile.frontend"
    "services/requirements_covid.txt"
    "services/requirements_churn.txt"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ $file missing${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

# 4. Check GitHub secrets (can't verify directly, just inform)
echo -e "\n${BLUE}4. Required GitHub Secrets:${NC}"
echo -e "${YELLOW}⚠️  You need to add these secrets in GitHub:${NC}"
echo "   - GCP_PROJECT_ID"
echo "   - GCP_SERVICE_ACCOUNT_KEY"
echo "   - PULUMI_ACCESS_TOKEN (optional)"
echo "   - GEMINI_API_KEY"
echo ""
echo "   Add them at: https://github.com/<your-repo>/settings/secrets/actions"

# 5. Check GCP authentication
echo -e "\n${BLUE}5. Checking GCP authentication...${NC}"
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    echo -e "${GREEN}✅ GCP authenticated as: ${ACCOUNT}${NC}"
else
    echo -e "${RED}❌ Not authenticated with GCP${NC}"
    echo "   Run: gcloud auth login"
    ERRORS=$((ERRORS + 1))
fi

# 6. Check GCP project
echo -e "\n${BLUE}6. Checking GCP project...${NC}"
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -n "$PROJECT_ID" ]; then
    echo -e "${GREEN}✅ GCP project: ${PROJECT_ID}${NC}"
else
    echo -e "${RED}❌ No GCP project set${NC}"
    echo "   Run: gcloud config set project YOUR_PROJECT_ID"
    ERRORS=$((ERRORS + 1))
fi

# 7. Check Docker
echo -e "\n${BLUE}7. Checking Docker...${NC}"
if command -v docker &> /dev/null; then
    if docker ps &> /dev/null; then
        echo -e "${GREEN}✅ Docker is running${NC}"
    else
        echo -e "${YELLOW}⚠️  Docker installed but not running${NC}"
        echo "   Start Docker Desktop"
    fi
else
    echo -e "${RED}❌ Docker not installed${NC}"
    ERRORS=$((ERRORS + 1))
fi

# 8. Test workflow syntax (if act is installed)
echo -e "\n${BLUE}8. Testing workflow syntax...${NC}"
if command -v act &> /dev/null; then
    if act -l &> /dev/null; then
        echo -e "${GREEN}✅ Workflow syntax is valid${NC}"
    else
        echo -e "${RED}❌ Workflow syntax error${NC}"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${YELLOW}⚠️  'act' not installed (optional)${NC}"
    echo "   Install with: brew install act"
    echo "   This allows local testing of GitHub Actions"
fi

# 9. Simulate CI/CD steps locally
echo -e "\n${BLUE}9. Simulating CI/CD steps locally...${NC}"

# Test: Build Docker image
echo -e "\n${BLUE}   Testing Docker build (COVID service)...${NC}"
if docker build -f Dockerfile.covid -t test-covid:local . &> /tmp/docker_build.log; then
    echo -e "${GREEN}   ✅ Docker build successful${NC}"
    docker rmi test-covid:local &> /dev/null || true
else
    echo -e "${RED}   ❌ Docker build failed${NC}"
    echo "   Check logs: /tmp/docker_build.log"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo -e "\n${BLUE}========================================${NC}"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ CI/CD configuration looks good!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Add GitHub secrets (see step 4 above)"
    echo "2. Push to GitHub: git push origin main"
    echo "3. Check Actions tab: https://github.com/<your-repo>/actions"
else
    echo -e "${RED}❌ Found $ERRORS issue(s)${NC}"
    echo "Please fix the issues above before deploying"
fi
echo -e "${BLUE}========================================${NC}"
