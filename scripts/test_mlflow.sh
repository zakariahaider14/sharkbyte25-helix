#!/bin/bash

# Test MLflow Integration
# This script verifies MLflow is working correctly

set -e

echo "🧪 Testing MLflow Integration"
echo "=============================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Check MLflow installation
echo -e "\n${BLUE}1. Checking MLflow installation...${NC}"
if command -v mlflow &> /dev/null; then
    echo -e "${GREEN}✅ MLflow installed: $(mlflow --version)${NC}"
else
    echo -e "${RED}❌ MLflow not installed${NC}"
    echo "Install with: pip install -r requirements_mlflow.txt"
    exit 1
fi

# 2. Start local MLflow server
echo -e "\n${BLUE}2. Starting local MLflow server...${NC}"
mlflow server --host 127.0.0.1 --port 5000 &
MLFLOW_PID=$!
echo "MLflow server PID: $MLFLOW_PID"
sleep 3

# 3. Test MLflow server
echo -e "\n${BLUE}3. Testing MLflow server...${NC}"
if curl -s http://127.0.0.1:5000/health > /dev/null; then
    echo -e "${GREEN}✅ MLflow server is running${NC}"
else
    echo -e "${RED}❌ MLflow server not responding${NC}"
    kill $MLFLOW_PID 2>/dev/null || true
    exit 1
fi

# 4. Run a simple MLflow experiment
echo -e "\n${BLUE}4. Running test experiment...${NC}"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

cat > /tmp/test_mlflow_experiment.py << 'EOF'
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set experiment
mlflow.set_experiment("test-experiment")

# Generate sample data
X, y = make_classification(n_samples=100, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run(run_name="test-run"):
    # Log parameters
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("max_depth", 3)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"✅ Test experiment completed! Accuracy: {accuracy:.4f}")
    print(f"✅ Run ID: {mlflow.active_run().info.run_id}")
EOF

python /tmp/test_mlflow_experiment.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Test experiment successful!${NC}"
else
    echo -e "${RED}❌ Test experiment failed${NC}"
    kill $MLFLOW_PID 2>/dev/null || true
    exit 1
fi

# 5. List experiments
echo -e "\n${BLUE}5. Listing MLflow experiments...${NC}"
mlflow experiments list

# 6. Open MLflow UI
echo -e "\n${BLUE}6. MLflow UI Access:${NC}"
echo -e "   ${GREEN}http://127.0.0.1:5000${NC}"
echo ""
echo "Press Enter to stop MLflow server and exit..."
read

# Cleanup
kill $MLFLOW_PID 2>/dev/null || true
rm /tmp/test_mlflow_experiment.py 2>/dev/null || true

echo -e "\n${GREEN}✅ MLflow test complete!${NC}"
