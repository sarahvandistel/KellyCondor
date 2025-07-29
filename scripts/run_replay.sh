#!/bin/bash

# KellyCondor Replay Runner
# This script runs the historical replay engine for backtesting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}KellyCondor Replay Engine${NC}"
echo "================================"

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo -e "${RED}Error: Please run this script from the KellyCondor root directory${NC}"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not detected${NC}"
    echo "Please activate your virtual environment before running this script"
    echo "Example: source .venv/bin/activate"
    echo ""
fi

# Default parameters
ACCOUNT_SIZE=${ACCOUNT_SIZE:-100000}
DATA_FILE=${DATA_FILE:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"replay_results"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --account-size)
            ACCOUNT_SIZE="$2"
            shift 2
            ;;
        --data-file)
            DATA_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --account-size SIZE    Account size for replay (default: 100000)"
            echo "  --data-file FILE       Path to historical data file"
            echo "  --output-dir DIR       Output directory for results (default: replay_results)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  ACCOUNT_SIZE           Account size for replay"
            echo "  DATA_FILE              Path to historical data file"
            echo "  OUTPUT_DIR             Output directory for results"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Configuration:${NC}"
echo "  Account Size: $ACCOUNT_SIZE"
echo "  Data File: ${DATA_FILE:-'Not specified'}"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Run the replay engine
echo -e "${YELLOW}Starting replay engine...${NC}"

if [ -n "$DATA_FILE" ]; then
    echo "Using data file: $DATA_FILE"
    python -m kellycondor.replay --account-size "$ACCOUNT_SIZE" --data-file "$DATA_FILE" --output-dir "$OUTPUT_DIR"
else
    echo "Running with sample data..."
    python -m kellycondor.replay --account-size "$ACCOUNT_SIZE" --output-dir "$OUTPUT_DIR"
fi

echo ""
echo -e "${GREEN}Replay completed!${NC}"
echo "Results saved in: $OUTPUT_DIR" 