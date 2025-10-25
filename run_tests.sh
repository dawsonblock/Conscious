#!/bin/bash
# FDQC v4.0 - Test Runner Script
# Quick bash script to run all tests with proper environment setup

set -e  # Exit on error

echo "=========================================="
echo "  FDQC v4.0 Test Runner"
echo "=========================================="
echo ""

# Setup Python path
export PYTHONPATH="/home/ubuntu/.local/lib/python3.13/site-packages:$PYTHONPATH"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run test
run_test() {
    local test_name=$1
    local command=$2
    
    echo -e "${YELLOW}Testing: $test_name${NC}"
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        return 1
    fi
}

echo "1. Checking Dependencies..."
run_test "numpy" "python3 -c 'import numpy'"
run_test "scipy" "python3 -c 'import scipy'"
run_test "matplotlib" "python3 -c 'import matplotlib'"
echo ""

echo "2. Testing Imports..."
run_test "fdqc_ai" "python3 -c 'from fdqc_ai import FDQC_AI'"
run_test "core" "python3 -c 'from core.fdqc_core import FDQCCore'"
run_test "visualization" "python3 -c 'from utils.visualization import visualize_all'"
echo ""

echo "3. Running Demos..."
run_test "simple_example" "timeout 20 python3 examples/simple_example.py"
run_test "basic_demo" "timeout 20 python3 demos/full_demo.py basic"
echo ""

echo "4. Running Comprehensive Build Test..."
python3 build_test.py --quick

echo ""
echo "=========================================="
echo "  Test Suite Complete!"
echo "=========================================="
