#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <seed-File:seed.txt> <num-pages: 10000> <hops-away: 6> <output-dir>"
    exit 1
fi

SEED_FILE=$1
NUM_PAGES=$2
HOPS_AWAY=$3
OUTPUT_DIR=$4

# Run the crawler
python3 wikipedia_spider.py "$SEED_FILE" "$NUM_PAGES" "$HOPS_AWAY" "$OUTPUT_DIR"