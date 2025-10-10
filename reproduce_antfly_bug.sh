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
        log_error "curl is required but not installed"
        exit 1
    fi
    if ! command -v python3 &> /dev/null; then
        log_error "python3 is required but not installed"
        exit 1
    fi
}

# Get Antfly process memory usage
get_antfly_memory() {
    local antfly_pid=$(pgrep -x antfly)
    if [ -n "$antfly_pid" ]; then
        # Get RSS (resident memory) and VSZ (virtual memory) in KB
        ps -p "$antfly_pid" -o pid,rss,vsz,comm --no-headers 2>/dev/null
    fi
}

# Check and fix Transparent Huge Pages (THP) setting
check_and_fix_thp() {
    local thp_enabled="/sys/kernel/mm/transparent_hugepage/enabled"

    # Check if THP control file exists
    if [ ! -f "$thp_enabled" ]; then
        log_warning "Cannot check Transparent Huge Pages setting (file not found)"
        return 0
    fi

    # Read current setting
    local current_setting=$(cat "$thp_enabled")

    # Check if THP is set to 'always' (problematic)
    if echo "$current_setting" | grep -q '\[always\]'; then
        echo ""
        log_warning "=============================================================================="
        log_warning "TRANSPARENT HUGE PAGES (THP) ISSUE DETECTED!"
        log_warning "=============================================================================="
        log_warning "Your system has THP set to 'always', which causes Antfly to use 329x more"
        log_warning "memory than necessary, leading to Out-Of-Memory crashes."
        log_warning ""
        log_warning "Current setting: $current_setting"
        log_warning "Recommended:     always [madvise] never"
        log_warning ""
        log_warning "This script can fix this temporarily (until reboot) by setting THP to 'madvise'."
        log_warning "This requires sudo privileges."
        log_warning "=============================================================================="
        echo ""

        # Prompt user
        read -p "Change THP to 'madvise' now? (y/n): " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Attempting to change THP setting to 'madvise'..."

            if echo madvise | sudo tee "$thp_enabled" > /dev/null 2>&1; then
                local new_setting=$(cat "$thp_enabled")
                log_info "Successfully changed THP setting!"
                log_info "New setting: $new_setting"
                echo ""
                log_info "Note: This change is temporary and will be lost on reboot."
                log_info "To make it permanent, see: DEBUG_ARCH_ENVIRONMENT.md"
                echo ""
            else
                log_error "Failed to change THP setting. You may need to run with sudo privileges."
                log_error "Continuing anyway, but expect memory issues..."
                sleep 2
            fi
        else
            log_warning "Continuing with THP='always'. Expect high memory usage and possible OOM crashes!"
            sleep 2
        fi
    elif echo "$current_setting" | grep -q '\[madvise\]'; then
        log_info "Transparent Huge Pages: madvise (optimal) ✓"
    elif echo "$current_setting" | grep -q '\[never\]'; then
        log_info "Transparent Huge Pages: never (acceptable) ✓"
    else
        log_warning "Transparent Huge Pages: unknown setting - $current_setting"
    fi
}

# Generate batch payload using Python (much faster than bash loops)
generate_batch_payload() {
    local start_id=$1
    local batch_size=$2
    local dimension=$3

    python3 -c "
import json
import random

start_id = $start_id
batch_size = $batch_size
dimension = $dimension

inserts = {}
for i in range(batch_size):
    doc_id = start_id + i
    embedding = [random.random() for _ in range(dimension)]
    inserts[str(doc_id)] = {'id': doc_id, 'embedding': embedding}

print(json.dumps({'inserts': inserts}))
"
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

# Wait for table shard to elect Raft leader
wait_for_shard_leader() {
    log_info "Waiting a bit for table shard to elect Raft leader..."
    sleep 5
    log_info "Proceeding with index creation"
}

# Create table
create_table() {
    log_info "Creating table 'vdb'"
    local response
    local http_code

    http_code=$(curl -s -w "%{http_code}" -o /tmp/response.json \
        -X POST "http://localhost:8080/api/v1/table/vdb" \
        -H "Content-Type: application/json" \
        -d '{"schema":{"key":"id"}}')

    response=$(cat /tmp/response.json)

    if [ "$http_code" = "200" ]; then
        log_info "Table created successfully"
        return 0
    elif [ "$http_code" = "400" ] && echo "$response" | grep -qi "already exists"; then
        log_info "Table already exists"
        return 0
    else
        log_error "Failed to create table: HTTP $http_code - $response"
        return 1
    fi
}

# Create vector index
create_index() {
    log_info "Creating index 'embedding_idx' (dimension: 1536)"
    local response
    local http_code

    http_code=$(curl -s -w "%{http_code}" -o /tmp/response.json \
        -X POST "http://localhost:8080/api/v1/table/vdb/index/embedding_idx" \
        -H "Content-Type: application/json" \
        -d '{"name":"embedding_idx","type":"vector_v2","dimension":1536,"field":"embedding"}')

    response=$(cat /tmp/response.json)

    if [ "$http_code" = "201" ]; then
        log_info "Index created successfully"
        return 0
    elif echo "$response" | grep -qi "already exists"; then
        log_info "Index already exists"
        return 0
    else
        log_warning "Index creation returned HTTP $http_code: ${response:0:200}"
        return 0  # Continue anyway
    fi
}

# Insert a batch of embeddings
insert_batch() {
    local start_id=$1
    local batch_size=$2
    local end_id=$((start_id + batch_size - 1))

    log_info "Inserting ${batch_size} embeddings (IDs: ${start_id}-${end_id})"

    # Generate JSON payload using Python (fast)
    generate_batch_payload "$start_id" "$batch_size" "$DIMENSION" > /tmp/batch_insert.json

    # Send the request
    local start_time
    start_time=$(date +%s)
    local http_code
    http_code=$(curl -s -w "%{http_code}" -o /tmp/response.json \
        -X POST "http://localhost:8080/api/v1/table/vdb/batch" \
        -H "Content-Type: application/json" \
        --data-binary @/tmp/batch_insert.json)
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $duration -gt 5 ]; then
        log_warning "Insert took ${duration} seconds (unusually long!)"
    fi

    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        log_info "Successfully inserted ${batch_size} embeddings"
        return 0
    else
        local response
        response=$(cat /tmp/response.json 2>/dev/null || echo "")
        log_error "Failed to insert: HTTP $http_code - $response"
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

    # Check and fix THP setting (critical for Arch Linux)
    check_and_fix_thp

    # Step 1: Wait for server
    wait_for_health || exit 1

    # Step 2: Create table
    create_table || exit 1

    # Step 3: Wait for Raft leader election
    wait_for_shard_leader

    # Step 4: Create index
    create_index || exit 1

    # Step 4.5: Verify index is ready (wait for background initialization)
    log_info "Waiting 10 seconds for index to initialize..."
    sleep 10

    # Quick verification
    if curl -s "http://localhost:8080/api/v1/table/vdb/index/embedding_idx" | grep -q "embedding_idx"; then
        log_info "Index verified and ready"
    else
        log_warning "Could not verify index (may still be initializing)"
    fi

    # Step 5: Insert embeddings in batches
    echo "================================================================================"
    log_info "Starting batch inserts..."
    echo "================================================================================"

    local total_inserted=0
    local batch_count=0
    local num_batches=$(( (NUM_VECTORS + BATCH_SIZE - 1) / BATCH_SIZE ))

    # Get initial memory usage
    local mem_info=$(get_antfly_memory)
    if [ -n "$mem_info" ]; then
        local pid=$(echo "$mem_info" | awk '{print $1}')
        local rss_kb=$(echo "$mem_info" | awk '{print $2}')
        local vsz_kb=$(echo "$mem_info" | awk '{print $3}')
        local rss_mb=$((rss_kb / 1024))
        local vsz_mb=$((vsz_kb / 1024))
        log_info "Initial Antfly memory: RSS=${rss_mb}MB, VSZ=${vsz_mb}MB (PID: ${pid})"
    fi

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
            exit 1
        fi

        total_inserted=$((total_inserted + current_batch_size))
        batch_count=$((batch_count + 1))

        # Log progress with memory usage every 10 batches
        if [ $((batch_count % 10)) -eq 0 ]; then
            local mem_info=$(get_antfly_memory)
            if [ -n "$mem_info" ]; then
                local rss_kb=$(echo "$mem_info" | awk '{print $2}')
                local vsz_kb=$(echo "$mem_info" | awk '{print $3}')
                local rss_mb=$((rss_kb / 1024))
                local vsz_mb=$((vsz_kb / 1024))
                log_info "Progress: ${total_inserted}/${NUM_VECTORS} vectors inserted (${batch_count}/${num_batches} batches) | Antfly Memory: RSS=${rss_mb}MB, VSZ=${vsz_mb}MB"
            else
                log_info "Progress: ${total_inserted}/${NUM_VECTORS} vectors inserted (${batch_count}/${num_batches} batches)"
            fi
        else
            log_info "Progress: ${total_inserted}/${NUM_VECTORS} vectors inserted (${batch_count}/${num_batches} batches)"
        fi
    done

    echo "================================================================================"
    log_info "SUCCESS: All ${total_inserted} vectors inserted successfully!"
    log_info "The bug did not manifest this time. Try running again or with more vectors."
    echo "================================================================================"

    # Cleanup
    rm -f /tmp/response.json /tmp/batch_insert.json
}

# Run main function
main
