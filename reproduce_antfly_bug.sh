#!/usr/bin/env bash
#
# Minimal shell script to reproduce the Antfly crash bug.
#
# This script replicates what VectorDBBench does using only curl and jq:
# 1. Creates a table via REST API
# 2. Creates a vector index
# 3. Inserts batches of embeddings until the server crashes
#
# Dependencies: curl, jq (for JSON processing)
#
# Usage:
#   ./reproduce_antfly_bug.sh [HOST] [PORT] [NUM_VECTORS] [BATCH_SIZE]
#
# Example:
#   ./reproduce_antfly_bug.sh localhost 8080 50000 100

set -euo pipefail

# Configuration
HOST="${1:-localhost}"
PORT="${2:-8080}"
NUM_VECTORS="${3:-50000}"
BATCH_SIZE="${4:-100}"
DIMENSION=1536
TABLE_NAME="vdb"
INDEX_NAME="embedding_idx"

BASE_URL="http://${HOST}:${PORT}"
API_URL="${BASE_URL}/api/v1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}$(date '+%Y-%m-%d %H:%M:%S')${NC} | ${GREEN}INFO${NC}: $*"
}

log_error() {
    echo -e "${BLUE}$(date '+%Y-%m-%d %H:%M:%S')${NC} | ${RED}ERROR${NC}: $*"
}

log_warning() {
    echo -e "${BLUE}$(date '+%Y-%m-%d %H:%M:%S')${NC} | ${YELLOW}WARNING${NC}: $*"
}

# Check dependencies
check_dependencies() {
    if ! command -v curl &> /dev/null; then
        log_error "curl is required but not installed. Install it with: sudo pacman -S curl"
        exit 1
    fi
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed. Install it with: sudo pacman -S jq"
        exit 1
    fi
}

# Generate a random float between 0 and 1
random_float() {
    awk -v seed="$RANDOM" 'BEGIN { srand(seed); printf "%.6f", rand() }'
}

# Generate a random embedding vector as a JSON array
generate_embedding() {
    local dim=$1
    local vec="["
    for ((i=0; i<dim; i++)); do
        if [ $i -gt 0 ]; then
            vec="${vec},"
        fi
        vec="${vec}$(random_float)"
    done
    vec="${vec}]"
    echo "$vec"
}

# Wait for Antfly server to be ready
wait_for_health() {
    log_info "Waiting for Antfly server to be ready..."
    local retries=30
    while [ $retries -gt 0 ]; do
        if curl -s -f "${BASE_URL}/health" > /dev/null 2>&1; then
            log_info "Antfly server is ready."
            return 0
        fi
        retries=$((retries - 1))
        sleep 1
    done
    log_error "Failed to connect to Antfly server. Is it running?"
    return 1
}

# Create table
create_table() {
    log_info "Creating table '${TABLE_NAME}'"
    local response
    local http_code

    http_code=$(curl -s -w "%{http_code}" -o /tmp/create_table_response.json \
        -X POST "${API_URL}/table/${TABLE_NAME}" \
        -H "Content-Type: application/json" \
        -d '{"schema":{"key":"id"}}' \
        --max-time 30)

    response=$(cat /tmp/create_table_response.json)

    if [ "$http_code" = "200" ]; then
        log_info "Table '${TABLE_NAME}' created successfully"
        return 0
    elif [ "$http_code" = "400" ] && echo "$response" | grep -qi "already exists"; then
        log_info "Table '${TABLE_NAME}' already exists"
        return 0
    else
        log_error "Failed to create table: HTTP $http_code - $response"
        return 1
    fi
}

# Create vector index
create_index() {
    log_info "Creating index '${INDEX_NAME}' on table '${TABLE_NAME}' (dimension: ${DIMENSION})"
    local response
    local http_code

    http_code=$(curl -s -w "%{http_code}" -o /tmp/create_index_response.json \
        -X POST "${API_URL}/table/${TABLE_NAME}/index/${INDEX_NAME}" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"${INDEX_NAME}\",\"type\":\"vector_v2\",\"dimension\":${DIMENSION},\"field\":\"embedding\"}" \
        --max-time 30)

    response=$(cat /tmp/create_index_response.json)

    if [ "$http_code" = "201" ]; then
        log_info "Index '${INDEX_NAME}' created successfully"
        return 0
    elif echo "$response" | grep -qi "already exists"; then
        log_info "Index '${INDEX_NAME}' already exists"
        return 0
    else
        log_warning "Index creation returned HTTP $http_code: ${response:0:200}"
        return 0  # Continue anyway, like VectorDBBench does
    fi
}

# Insert a batch of embeddings
insert_batch() {
    local start_id=$1
    local batch_size=$2
    local end_id=$((start_id + batch_size - 1))

    log_info "Inserting ${batch_size} embeddings into '${TABLE_NAME}' (IDs: ${start_id}-${end_id})"

    # Build JSON payload
    local json_payload='{"inserts":{'
    for ((i=0; i<batch_size; i++)); do
        local doc_id=$((start_id + i))
        local embedding
        embedding=$(generate_embedding "$DIMENSION")

        if [ $i -gt 0 ]; then
            json_payload="${json_payload},"
        fi
        json_payload="${json_payload}\"${doc_id}\":{\"id\":${doc_id},\"embedding\":${embedding}}"
    done
    json_payload="${json_payload}}}"

    # Save to temporary file to avoid command line length issues
    echo "$json_payload" > /tmp/batch_insert.json

    # Send the request
    local http_code
    http_code=$(curl -s -w "%{http_code}" -o /tmp/insert_response.json \
        -X POST "${API_URL}/table/${TABLE_NAME}/batch" \
        -H "Content-Type: application/json" \
        --data-binary @/tmp/batch_insert.json \
        --max-time 300)

    if [ "$http_code" = "200" ]; then
        log_info "Successfully inserted ${batch_size} embeddings"
        return 0
    else
        local response
        response=$(cat /tmp/insert_response.json 2>/dev/null || echo "")
        log_error "Failed to insert data into Antfly: HTTP $http_code - $response"
        return 1
    fi
}

# Main execution
main() {
    echo "================================================================================"
    log_info "Antfly Crash Bug Reproduction Script (Shell Version)"
    echo "================================================================================"
    log_info "Server: ${BASE_URL}"
    log_info "Total vectors: ${NUM_VECTORS}"
    log_info "Dimension: ${DIMENSION}"
    log_info "Batch size: ${BATCH_SIZE}"
    log_info "Table: ${TABLE_NAME}"
    log_info "Index: ${INDEX_NAME}"
    echo "================================================================================"

    # Check dependencies
    check_dependencies

    # Step 1: Wait for server
    wait_for_health || exit 1

    # Step 2: Create table
    create_table || exit 1

    # Step 3: Wait for Raft consensus
    log_info "Waiting 5 seconds for table shards to initialize (Raft consensus)..."
    sleep 5

    # Step 4: Create index
    create_index || exit 1

    # Step 5: Insert embeddings in batches
    echo "================================================================================"
    log_info "Starting batch inserts..."
    echo "================================================================================"

    local total_inserted=0
    local batch_count=0
    local num_batches=$(( (NUM_VECTORS + BATCH_SIZE - 1) / BATCH_SIZE ))

    for ((batch_num=0; batch_num<num_batches; batch_num++)); do
        # Determine batch size (last batch might be smaller)
        local current_batch_size=$BATCH_SIZE
        local remaining=$((NUM_VECTORS - total_inserted))
        if [ $remaining -lt $BATCH_SIZE ]; then
            current_batch_size=$remaining
        fi

        if [ $current_batch_size -le 0 ]; then
            break
        fi

        # Insert the batch
        if ! insert_batch "$total_inserted" "$current_batch_size"; then
            log_error "CRASH DETECTED: Failed after ${total_inserted} vectors (${batch_count} batches)"
            log_error "This is the bug we're trying to reproduce!"
            exit 1
        fi

        total_inserted=$((total_inserted + current_batch_size))
        batch_count=$((batch_count + 1))

        log_info "Progress: ${total_inserted}/${NUM_VECTORS} vectors inserted (${batch_count}/${num_batches} batches)"
    done

    echo "================================================================================"
    log_info "SUCCESS: All ${total_inserted} vectors inserted successfully!"
    log_info "The bug did not manifest this time. Try running again or with more vectors."
    echo "================================================================================"

    # Cleanup
    rm -f /tmp/create_table_response.json /tmp/create_index_response.json /tmp/batch_insert.json /tmp/insert_response.json
}

# Run main function
main
