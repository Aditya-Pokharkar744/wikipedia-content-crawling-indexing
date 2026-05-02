#!/bin/bash
# run_query.sh - Fixed version with proper bash handling

show_help() {
    echo "Usage: ./run_query.sh [OPTIONS] \"QUERY\""
    echo "Example: ./run_query.sh \"machine learning\" -k 5"
}

# Initialize variables
QUERY=""
K=5
INDEX_DIR="index"
THRESHOLD=0.3
CPU_MODE=""
MODEL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -k)
            K="$2"
            shift 2
            ;;
        --index)
            INDEX_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --cpu)
            CPU_MODE="--cpu"
            shift
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            QUERY="$1"
            shift
            break
            ;;
    esac
done

# Validate query
if [[ -z "$QUERY" ]]; then
    echo "Error: Query argument is required!"
    show_help
    exit 1
fi

# Build command
CMD=(
    python3 query_index.py
    "$QUERY"
    -k "$K"
    --index-dir "$INDEX_DIR"
    --threshold "$THRESHOLD"
    $CPU_MODE
    ${MODEL:+--model "$MODEL"}
)

# Execute command
echo "Searching for: '$QUERY'"
"${CMD[@]}"