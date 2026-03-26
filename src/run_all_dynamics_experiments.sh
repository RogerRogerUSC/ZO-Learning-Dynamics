#!/bin/bash

# Script to run all configs in dynamics_experiments directory
# Usage: ./run_all_dynamics_experiments.sh [--task TASK] [--num-test-samples N]
#   --task      Optional: run only configs under dynamics_experiments/TASK/ (e.g. sst2, sst5)
#   --num-test-samples  Number of test samples (default: 5)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIGS_ROOT="$SCRIPT_DIR/zo_llm/configs/text_classification"
DYNAMICS_DIR="$CONFIGS_ROOT/dynamics_experiments"

# Parse optional arguments
NUM_TEST_SAMPLES=5
TASK_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-test-samples)
            NUM_TEST_SAMPLES="$2"
            shift 2
            ;;
        --task)
            TASK_FILTER="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Check if configs directory exists
if [ ! -d "$DYNAMICS_DIR" ]; then
    echo "Error: Configs directory not found: $DYNAMICS_DIR"
    exit 1
fi

# Find YAML files recursively, optionally filtered by task
if [ -n "$TASK_FILTER" ]; then
    SEARCH_DIR="$DYNAMICS_DIR/$TASK_FILTER"
    if [ ! -d "$SEARCH_DIR" ]; then
        echo "Error: Task directory not found: $SEARCH_DIR"
        exit 1
    fi
    mapfile -t CONFIG_FILES < <(find "$SEARCH_DIR" -name "*.yaml" | sort)
else
    mapfile -t CONFIG_FILES < <(find "$DYNAMICS_DIR" -name "*.yaml" | sort)
fi

# Check if any config files were found
if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "Error: No YAML config files found"
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
    # Compute path relative to CONFIGS_ROOT so llm_dynamics_main.py can resolve it
    CONFIG_RELATIVE_PATH="${CONFIG_FILE#$CONFIGS_ROOT/}"

    # Calculate progress
    PROGRESS=$((i + 1))

    echo "[$PROGRESS/$TOTAL_CONFIGS] Running: $CONFIG_RELATIVE_PATH"
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
        FAILED_CONFIGS+=("$CONFIG_RELATIVE_PATH")
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
