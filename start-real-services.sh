#!/bin/bash

# Start Real ML Services
# Starts COVID service, Churn service, and web application

echo "=========================================="
echo "STARTING REAL ML SERVICES"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check if models exist
if [ ! -f "ml_models/saved_models/covid_model.joblib" ]; then
    echo "❌ COVID model not found. Please run setup-phase2.sh first"
    exit 1
fi

if [ ! -f "ml_models/saved_models/churn_model.joblib" ]; then
    echo "❌ Churn model not found. Please run setup-phase2.sh first"
    exit 1
fi

# Start COVID service
echo "🦠 Starting COVID Prediction Service..."
python3 services/covid_service_real.py > logs/covid_service.log 2>&1 &
COVID_PID=$!
sleep 2

# Check if COVID service started
if ! kill -0 $COVID_PID 2>/dev/null; then
    echo "❌ COVID service failed to start. Check logs/covid_service.log"
    exit 1
fi

# Health check
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ COVID service running on http://localhost:8000"
else
    echo "⚠️  COVID service started but health check failed"
fi

# Start Churn service
echo "📞 Starting Churn Prediction Service..."
python3 services/churn_service_real.py > logs/churn_service.log 2>&1 &
CHURN_PID=$!
sleep 2

# Check if Churn service started
if ! kill -0 $CHURN_PID 2>/dev/null; then
    echo "❌ Churn service failed to start. Check logs/churn_service.log"
    kill $COVID_PID 2>/dev/null
    exit 1
fi

# Health check
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Churn service running on http://localhost:8001"
else
    echo "⚠️  Churn service started but health check failed"
fi

# Start web application
echo "🌐 Starting Web Application..."
pnpm dev > logs/webapp.log 2>&1 &
WEBAPP_PID=$!
sleep 3

# Check if web app started
if ! kill -0 $WEBAPP_PID 2>/dev/null; then
    echo "❌ Web app failed to start. Check logs/webapp.log"
    kill $COVID_PID $CHURN_PID 2>/dev/null
    exit 1
fi

echo "✅ Web application running on http://localhost:3000"

echo ""
echo "=========================================="
echo "ALL SERVICES RUNNING"
echo "=========================================="
echo ""
echo "📊 Service Status:"
echo "  COVID Service:  http://localhost:8000"
echo "  Churn Service:  http://localhost:8001"
echo "  Web App:        http://localhost:3000"
echo "  Agent UI:       http://localhost:3000/agent"
echo ""
echo "📝 Logs:"
echo "  COVID:  tail -f logs/covid_service.log"
echo "  Churn:  tail -f logs/churn_service.log"
echo "  Web:    tail -f logs/webapp.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for all background jobs
wait
