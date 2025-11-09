#!/bin/bash

# HELIX Service Testing Script

set -e

echo "🧪 HELIX Service Testing"
echo "========================"
echo ""

# Load service URLs
if [ -f .env.gcp ]; then
    source .env.gcp
else
    COVID_SERVICE_URL="https://covid-service-397484632647.us-central1.run.app"
    CHURN_SERVICE_URL="https://churn-service-397484632647.us-central1.run.app"
fi

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test COVID Service
echo -e "${BLUE}🦠 Testing COVID Service${NC}"
echo "URL: $COVID_SERVICE_URL"
echo ""

echo "1. Health Check..."
curl -s ${COVID_SERVICE_URL}/health | jq '.'
echo ""

echo "2. Simple Prediction (USA)..."
curl -s -X POST ${COVID_SERVICE_URL}/predict/covid \
  -H "Content-Type: application/json" \
  -d '{
    "country_name": "USA",
    "confirmed_cases": 100000,
    "deaths": 2000,
    "recovered": 95000
  }' | jq '.'
echo ""

echo "3. Detailed Prediction (India)..."
curl -s -X POST ${COVID_SERVICE_URL}/predict/covid \
  -H "Content-Type: application/json" \
  -d '{
    "country_name": "India",
    "confirmed_cases": 500000,
    "deaths": 10000,
    "recovered": 450000,
    "population": 1400000000,
    "vaccination_rate": 0.65,
    "testing_rate": 1000.0
  }' | jq '.'
echo ""

echo "4. Minimal Data Prediction (Brazil)..."
curl -s -X POST ${COVID_SERVICE_URL}/predict/covid \
  -H "Content-Type: application/json" \
  -d '{
    "country_name": "Brazil",
    "confirmed_cases": 50000
  }' | jq '.'
echo ""

echo -e "${GREEN}✅ COVID Service Tests Complete${NC}"
echo ""
echo "=================================================="
echo ""

# Test Churn Service
echo -e "${BLUE}📞 Testing Churn Service${NC}"
echo "URL: $CHURN_SERVICE_URL"
echo ""

echo "1. Health Check..."
curl -s ${CHURN_SERVICE_URL}/health | jq '.'
echo ""

echo "2. Simple Prediction (CUST_001)..."
curl -s -X POST ${CHURN_SERVICE_URL}/predict/churn \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "tenure_months": 24,
    "monthly_charges": 85.5
  }' | jq '.'
echo ""

echo "3. Detailed Prediction (CUST_123)..."
curl -s -X POST ${CHURN_SERVICE_URL}/predict/churn \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_123",
    "age": 45,
    "tenure_months": 36,
    "monthly_charges": 95.0,
    "total_charges": 3420.0,
    "contract_type": "Two year",
    "internet_service_type": "Fiber optic",
    "tech_support": true,
    "online_security": true,
    "support_tickets_count": 1
  }' | jq '.'
echo ""

echo "4. High Risk Customer (CUST_999)..."
curl -s -X POST ${CHURN_SERVICE_URL}/predict/churn \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_999",
    "tenure_months": 2,
    "monthly_charges": 120.0,
    "total_charges": 240.0,
    "contract_type": "Month-to-month",
    "support_tickets_count": 5
  }' | jq '.'
echo ""

echo -e "${GREEN}✅ Churn Service Tests Complete${NC}"
echo ""
echo "=================================================="
echo ""

# Performance Test
echo -e "${YELLOW}⚡ Performance Test (10 requests)${NC}"
echo ""

echo "COVID Service Response Times:"
for i in {1..10}; do
    START=$(date +%s%N)
    curl -s -X POST ${COVID_SERVICE_URL}/predict/covid \
      -H "Content-Type: application/json" \
      -d '{"country_name": "Test", "confirmed_cases": 1000}' > /dev/null
    END=$(date +%s%N)
    DURATION=$((($END - $START) / 1000000))
    echo "  Request $i: ${DURATION}ms"
done
echo ""

echo "Churn Service Response Times:"
for i in {1..10}; do
    START=$(date +%s%N)
    curl -s -X POST ${CHURN_SERVICE_URL}/predict/churn \
      -H "Content-Type: application/json" \
      -d '{"customer_id": "TEST", "tenure_months": 12}' > /dev/null
    END=$(date +%s%N)
    DURATION=$((($END - $START) / 1000000))
    echo "  Request $i: ${DURATION}ms"
done
echo ""

echo -e "${GREEN}✅ All Tests Complete!${NC}"
echo ""
echo "Summary:"
echo "  COVID Service: $COVID_SERVICE_URL"
echo "  Churn Service: $CHURN_SERVICE_URL"
echo ""
