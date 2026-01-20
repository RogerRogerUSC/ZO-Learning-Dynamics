#!/bin/bash

# Script to run all configs in dynamics_experiments directory
# Usage: ./run_all_dynamics_experiments.sh [--num-test-samples N]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIGS_DIR="$SCRIPT_DIR/zo_llm/configs/text_classification/dynamics_experiments"

# Parse optional arguments
NUM_TEST_SAMPLES=5
if [[ "$1" == "--num-test-samples" ]] && [[ -n "$2" ]]; then
    NUM_TEST_SAMPLES="$2"
fi

# Check if configs directory exists
if [ ! -d "$CONFIGS_DIR" ]; then
    echo "Error: Configs directory not found: $CONFIGS_DIR"
    exit 1
fi

# Find all YAML files in dynamics_experiments
CONFIG_FILES=("$CONFIGS_DIR"/*.yaml)

# Check if any config files were found
if [ ${#CONFIG_FILES[@]} -eq 0 ] || [ ! -f "${CONFIG_FILES[0]}" ]; then
    echo "Error: No YAML config files found in $CONFIGS_DIR"
    exit 1
fi

# Count total configs
TOTAL_CONFIGS=${#CONFIG_FILES[@]}
echo "Found $TOTAL_CONFIGS config files to run"
echo "Number of test samples: $NUM_TEST_SAMPLES"
echo "=========================================="
echo ""

# Track success and failures
SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_CONFIGS=()

# Run each config file
for i in "${!CONFIG_FILES[@]}"; do
    CONFIG_FILE="${CONFIG_FILES[$i]}"
    CONFIG_NAME=$(basename "$CONFIG_FILE")
    CONFIG_RELATIVE_PATH="text_classification/dynamics_experiments/$CONFIG_NAME"
    
    # Calculate progress
    PROGRESS=$((i + 1))
    
    echo "[$PROGRESS/$TOTAL_CONFIGS] Running: $CONFIG_NAME"
    echo "----------------------------------------"
    
    # Run the experiment
    if uv run "$SCRIPT_DIR/llm_dynamics_main.py" \
        --config-path "$CONFIG_RELATIVE_PATH" \
        --num-test-samples "$NUM_TEST_SAMPLES"; then
        echo "✓ Successfully completed: $CONFIG_NAME"
        ((SUCCESS_COUNT++))
    else
        echo "✗ Failed: $CONFIG_NAME"
        ((FAILURE_COUNT++))
        FAILED_CONFIGS+=("$CONFIG_NAME")
    fi
    
    echo ""
done

# Print summary
echo "=========================================="
echo "Summary:"
echo "  Total configs: $TOTAL_CONFIGS"
echo "  Successful: $SUCCESS_COUNT"
echo "  Failed: $FAILURE_COUNT"
echo ""

if [ $FAILURE_COUNT -gt 0 ]; then
    echo "Failed configs:"
    for failed_config in "${FAILED_CONFIGS[@]}"; do
        echo "  - $failed_config"
    done
    exit 1
else
    echo "All experiments completed successfully!"
    exit 0
fi
