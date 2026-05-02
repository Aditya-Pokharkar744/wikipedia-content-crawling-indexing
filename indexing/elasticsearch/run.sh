#!/bin/bash

function print_usage {
    echo "Usage:"
    echo "  $0 index <directory> [--force] [--index-name NAME]  # Index documents"
    echo "  $0 search <query> [--k NUMBER] [--index-name NAME]  # Search documents"
    echo ""
    echo "Options:"
    echo "  --force       Force reindexing of already indexed files"
    echo "  --k NUMBER    Number of search results to return (default: 10)"
    echo "  --index-name  Name of the Elasticsearch index (default: documents_index)"
}

if [ "$#" -lt 2 ]; then
    print_usage
    exit 1
fi

COMMAND="$1"
shift

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed."
    exit 1
fi

python3 document_indexer.py "$COMMAND" "$@"