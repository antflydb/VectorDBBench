# Antfly Crash Bug Report

## Environment
- **OS**: Arch Linux (Omarchy variant, Hyprland)
- **Kernel**: Linux 6.16.10-arch1-1
- **RAM**: 16GB
- **Antfly Version**: 0.0.0-dev4
- **Test Command**: `uvx --python 3.11 --from . --with '.[all]' python run_antfly_test.py`
- **Dataset**: OpenAI-SMALL-50K (50,000 vectors, 1536 dimensions)

## Summary
Antfly server crashes during VectorDBBench performance testing, typically after successfully inserting several batches of embeddings. The crash manifests as "Connection refused" errors when the benchmark attempts to continue inserting data.

## Reproduction Steps

1. Start Antfly server:
   ```bash
   CUDA_VISIBLE_DEVICES=-1 ollama serve  # In background
   /path/to/antflycli swarm 2>&1 | tee antfly.log
   ```

2. Run the VectorDBBench test:
   ```bash
   uvx --python 3.11 --from . --with '.[all]' python run_antfly_test.py
   ```

3. Observe crash after variable number of successful inserts (sometimes after a few batches, sometimes after several minutes)

## Observed Behavior

### Successful Phase
The benchmark successfully:
- Creates table `vdb`
- Creates vector index `embedding_idx` (dimension 1536)
- Inserts multiple batches of 100 embeddings each via `/api/v1/table/vdb/batch`
- Inserts progress normally for some time (variable duration)

### Crash Sequence
```
2025-10-09 00:38:49,434 | INFO: Inserting 100 embeddings into vdb
2025-10-09 00:38:58,939 | INFO: Successfully inserted 100 embeddings
2025-10-09 00:39:10,616 | INFO: Inserting 100 embeddings into vdb
2025-10-09 00:39:13,538 | INFO: Successfully inserted 100 embeddings
2025-10-09 00:39:25,378 | INFO: Inserting 100 embeddings into vdb
2025-10-09 00:39:25,874 | ERROR: Failed to insert data into Antfly: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
2025-10-09 00:39:25,974 | ERROR: Failed to insert data into Antfly: HTTPConnectionPool(host='localhost', port=8080): Max retries exceeded with url: /api/v1/table/vdb/batch (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f4a6e65a3d0>: Failed to establish a new connection: [Errno 111] Connection refused'))
```

### Antfly Server Logs (No Obvious Errors)
The Antfly server logs show normal operation with no crash or error messages around the time of failure:

```
2025-10-09T00:38:17.380-0700    INFO    store.coreDB    Starting index manager leader factory
2025-10-09T00:38:26.168-0700    DEBUG   metadataServer  Checking shard assignments...
    {"currentShards": [{"shard_id":"41edbc35dfc2456b","shard_info":{"byte_range":["","/w=="],"shard_stats":{"storage":{"disk_size":1087699915}}}}]}
2025-10-09T00:38:36.522-0700    DEBUG   metadataServer  Checking shard assignments...
    {"currentShards": [{"shard_id":"41edbc35dfc2456b","shard_info":{"byte_range":["","/w=="],"shard_stats":{"storage":{"disk_size":1106856430}}}}]}
2025-10-09T00:38:47.528-0700    DEBUG   metadataServer  Checking shard assignments...
    {"currentShards": [{"shard_id":"41edbc35dfc2456b","shard_info":{"byte_range":["","/w=="],"shard_stats":{"storage":{"disk_size":1147248060}}}}]}
```

**Note**: Disk size growth shows data is being written successfully up until the crash.

## Previous Related Issues

Earlier in testing, we encountered Raft consensus errors when creating indexes immediately after table creation:

```
2025-10-09T00:13:00.755-0700    ERROR    metadataServer    metadata/metadata.go:1676
    failed retrying adding index to shard    {"index": "embedding_idx", "shardID": "41edbc35dfc2456b",
    "error": "adding index to shard: finding leader: finding leader for shard 41edbc35dfc2456b:
    no raft status available for shard"}
```

This was mitigated by adding a 5-second wait after table creation before creating the index.

## Root Cause: CONFIRMED OOM (Out of Memory) Issue ✓

**UPDATE 2025-10-09 19:24:** Confirmed via system logs that **Antfly is killed by the Linux OOM killer** due to a severe memory leak.

### Evidence from journalctl:
```
Oct 09 19:24:50 rocinante kernel: Out of memory: Killed process 306187 (antfly)
    total-vm:19855980kB, anon-rss:14131176kB, file-rss:1456kB
```

### Critical Memory Leak Statistics:
- **Crash point:** After inserting only 7,000 vectors (14% of 50K dataset)
- **Expected memory usage:** ~42 MB (7,000 vectors × 1536 dim × 4 bytes)
- **Actual memory consumed:** ~13.8 GB (13,800 MB)
- **Memory amplification:** **329x** (Antfly uses 329× more RAM than the actual data!)
- **Virtual memory at crash:** ~19.4 GB

### Platform-Specific Issue:
- **MacOS binary:** Works perfectly, completes all 50,000 vectors without leak
- **Arch Linux binary:** Crashes at ~7,000 vectors due to OOM killer
- **Same version:** 0.0.0-dev4 on both platforms

This is a **critical platform-specific memory leak** in the Arch Linux build.

## Variability

- **Success Duration**: Highly variable - sometimes fails after a few batches, sometimes runs for several minutes
- **Earlier Success**: During initial testing, the benchmark progressed much further (got through data loading and started search performance tests) before encountering similar crashes
- **Data Corruption**: In one instance, received `"decoding embedding hashID for 256: insufficient bytes to decode uint64 int value"` errors, suggesting possible data corruption under stress

## Reproduction Script

A simplified reproduction script has been created: `reproduce_antfly_bug.sh`

### Usage:
```bash
./reproduce_antfly_bug.sh [HOST] [PORT] [NUM_VECTORS] [BATCH_SIZE]
./reproduce_antfly_bug.sh localhost 8080 50000 100  # Default values
```

### Features:
- **Memory monitoring**: Tracks Antfly RSS/VSZ every 10 batches
- **Timing warnings**: Alerts when inserts take >5 seconds (memory pressure indicator)
- **Minimal dependencies**: Only requires curl and python3
- **Detailed logging**: Shows exactly when and how the crash occurs

The script successfully reproduces the OOM crash on Arch Linux consistently.

## Next Steps for Antfly Developers

1. **Profile the Linux binary** with pprof/heaptrack to identify the leak source
2. **Compare build configurations** between macOS and Linux binaries
3. **Check platform-specific code** in vector storage, indexing, and Raft consensus
4. **Test with valgrind/AddressSanitizer** on Linux to catch memory errors
5. **Review goroutine lifecycle** for unbounded growth on Linux

## Additional Files

- `reproduce_antfly_bug.sh` - Minimal reproduction script with memory monitoring
- `OOM_ANALYSIS.md` - Detailed analysis of the memory leak
- `REPRODUCE_BUG_README.md` - Usage instructions for reproduction scripts

## Client Implementation Notes

The VectorDBBench Antfly client (in `vectordb_bench/backend/clients/antfly/antfly.py`):
- Uses Antfly REST API for all operations (batch inserts, queries)
- Creates table with schema: `{"key": "id"}`
- Creates vector index before data insertion: `{"type": "vector_v2", "dimension": 1536, "field": "embedding"}`
- Inserts data via `/api/v1/table/vdb/batch` with batches of 100 embeddings
- Each document: `{"id": <int>, "embedding": [<1536 floats>]}`

## Test Configuration

From `run_antfly_test.py`:
```python
{
    "db": DB.Antfly,
    "db_config": {},
    "db_case_config": AntflyConfig(),
    "case_config": CaseConfig(
        load_timeout=100000,
        run_timeout=100000,
    ),
}
```

Dataset: OpenAI embeddings, 50K vectors, 1536 dimensions, COSINE metric
