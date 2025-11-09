#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   MLOps Gemini Agent - Local Development Startup      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ Error: .env file not found${NC}"
    echo -e "${YELLOW}Please create .env file from .env.example:${NC}"
    echo -e "   cp .env.example .env"
    echo -e "   # Then edit .env with your configuration"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}⚠️  node_modules not found. Running pnpm install...${NC}"
    pnpm install
fi

# Check if Python dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Python dependencies not found. Installing...${NC}"
    pip3 install -r services/requirements.txt
fi

echo -e "${GREEN}✓ Prerequisites checked${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $COVID_PID $CHURN_PID $WEB_PID 2>/dev/null
    wait $COVID_PID $CHURN_PID $WEB_PID 2>/dev/null
    echo -e "${GREEN}✓ All services stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start COVID service
echo -e "${BLUE}🦠 Starting COVID-19 Prediction Service (port 8000)...${NC}"
python3 services/mock_covid_service.py > logs/covid.log 2>&1 &
COVID_PID=$!
sleep 2

# Check if COVID service started successfully
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${RED}❌ Failed to start COVID service${NC}"
    cat logs/covid.log
    kill $COVID_PID 2>/dev/null
    exit 1
fi
echo -e "${GREEN}✓ COVID service running (PID: $COVID_PID)${NC}"

# Start Churn service
echo -e "${BLUE}📞 Starting Churn Prediction Service (port 8001)...${NC}"
python3 services/mock_churn_service.py > logs/churn.log 2>&1 &
CHURN_PID=$!
sleep 2

# Check if Churn service started successfully
if ! curl -s http://localhost:8001/health > /dev/null; then
    echo -e "${RED}❌ Failed to start Churn service${NC}"
    cat logs/churn.log
    kill $COVID_PID $CHURN_PID 2>/dev/null
    exit 1
fi
echo -e "${GREEN}✓ Churn service running (PID: $CHURN_PID)${NC}"

# Start Web application
echo -e "${BLUE}🌐 Starting Web Application (port 3000)...${NC}"
pnpm dev > logs/web.log 2>&1 &
WEB_PID=$!
sleep 3

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              All Services Started Successfully!         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Service Status:${NC}"
echo -e "   COVID Service:  ${GREEN}http://localhost:8000${NC} (PID: $COVID_PID)"
echo -e "   Churn Service:  ${GREEN}http://localhost:8001${NC} (PID: $CHURN_PID)"
echo -e "   Web App:        ${GREEN}http://localhost:3000${NC} (PID: $WEB_PID)"
echo -e "   Agent UI:       ${GREEN}http://localhost:3000/agent${NC}"
echo ""
echo -e "${YELLOW}📝 Logs:${NC}"
echo -e "   COVID: tail -f logs/covid.log"
echo -e "   Churn: tail -f logs/churn.log"
echo -e "   Web:   tail -f logs/web.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for all processes
wait $COVID_PID $CHURN_PID $WEB_PID
