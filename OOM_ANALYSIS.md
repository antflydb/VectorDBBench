# Antfly OOM (Out of Memory) Analysis

## Problem Summary

**The Arch Linux binary of Antfly has a severe memory leak** that causes the process to be killed by the Linux OOM (Out of Memory) killer after inserting approximately 7,000 vectors (70 batches of 100 embeddings each, dimension 1536).

## Evidence from System Logs

### OOM Killer Event (2025-10-09 19:24:50)

```
Oct 09 19:24:49 rocinante kernel: antfly invoked oom-killer: gfp_mask=0x140cca(GFP_HIGHUSER_MOVABLE|__GFP_COMP), order=0, oom_score_adj=200

Oct 09 19:24:50 rocinante kernel: Out of memory: Killed process 306187 (antfly) total-vm:19855980kB, anon-rss:14131176kB, file-rss:1456kB, shmem-rss:0kB, UID:1000 pgtables:34516kB oom_score_adj:200
```

### Memory Usage at Time of Crash

- **Virtual Memory (total-vm):** ~19.4 GB
- **Anonymous RSS (anon-rss):** ~13.8 GB  ← Actual RAM consumed
- **File-backed RSS (file-rss):** ~1.4 MB
- **Vectors inserted:** 7,000 out of 50,000 (only 14% complete)
- **Data size:** ~42 MB of vector data (7,000 × 1536 × 4 bytes per float)

### Critical Findings

**The memory leak is catastrophic:**
- Expected memory for 7,000 vectors: ~42 MB
- Actual memory consumed: ~13,800 MB (13.8 GB)
- **Memory amplification factor: ~329x**

This means Antfly is using **329 times more memory** than the actual data size.

## Symptoms

1. **Progressive slowdown**: Insert operations that initially complete in <1 second start taking 10-162 seconds
2. **System becomes unusable**: The excessive memory consumption causes system-wide performance degradation
3. **Silent crash**: Antfly provides no error logging before being killed
4. **Cascading failures**: btop (system monitor) and hyprctl also crashed due to memory pressure

## Platform Comparison

### MacOS Binary (Working)
- Successfully completes all 50,000 vector inserts
- No memory leak observed
- Stable performance throughout

### Arch Linux Binary (Broken)
- Crashes after ~7,000 vectors (~14% progress)
- Severe memory leak (329x amplification)
- OOM killer terminates the process
- Same Antfly version (0.0.0-dev4)

## Root Cause Hypothesis

The issue is **platform-specific**, suggesting:

1. **Memory allocator differences**: The Go runtime or C/C++ allocator behaves differently on Linux vs macOS
2. **Build configuration issue**: The Arch binary may be built with different flags or optimizations
3. **mmap/memory management bug**: Linux-specific memory management code path has a leak
4. **Goroutine leak**: Unbounded goroutine creation on Linux (but not macOS)

## Reproduction Details

### Test Configuration
- **Dataset:** Random 1536-dimensional vectors
- **Batch size:** 100 vectors per batch
- **Total vectors:** 50,000
- **System RAM:** 16 GB
- **OS:** Arch Linux (Omarchy), Kernel 6.16.10-arch1-1

### Script Used
`reproduce_antfly_bug.sh` - Enhanced version with memory monitoring

### Timeline
```
19:21:57 - Batch 66 (6,600 vectors): Normal performance
19:21:58 - Batch 67-69 (6,700-6,900 vectors): Still normal
19:22:04 - Batch 70 (7,000 vectors): Insert takes 5+ seconds (first warning sign)
19:24:51 - Batch 70 fails after 162 seconds
19:24:49 - OOM killer invoked
19:24:50 - Antfly process killed
```

## HTTP Error Details

The final insert returned:
```
HTTP 100 - {"batch":"successful"}
```

**HTTP 100 (Continue)** is incorrect for this context - this should be HTTP 200. This suggests Antfly was in a degraded state and returned an incomplete/malformed response before crashing.

## Monitoring Enhancements

The reproduction script now includes:

1. **Memory tracking every 10 batches:**
   - RSS (Resident Set Size): Actual RAM used
   - VSZ (Virtual Size): Total virtual memory

2. **Timing warnings:**
   - Alerts when inserts take >5 seconds
   - Helps identify memory pressure early

3. **System resource impact:**
   - Monitors Antfly process specifically
   - Can detect progressive memory growth

## Recommended Next Steps

### For Antfly Developers

1. **Profile the Arch binary** with pprof/heaptrack to identify leak source
2. **Compare build configurations** between macOS and Linux binaries
3. **Check for platform-specific code paths** in:
   - Memory allocation (especially vector storage)
   - Index building
   - Raft/distributed consensus layer
4. **Test with valgrind/AddressSanitizer** on Linux to catch memory errors
5. **Review goroutine lifecycle** for unbounded growth

### For Bug Reporters

1. **Run with smaller datasets** to establish memory growth pattern
2. **Test on different Linux distros** (Ubuntu, Fedora) to isolate if it's Arch-specific
3. **Monitor with `pprof`** if possible:
   ```bash
   # If Antfly exposes pprof endpoint
   curl http://localhost:6060/debug/pprof/heap > heap.prof
   go tool pprof heap.prof
   ```

4. **Collect core dump** (if enabled):
   ```bash
   coredumpctl list
   coredumpctl dump <PID> -o antfly.core
   ```

## Files

- `reproduce_antfly_bug.sh` - Enhanced reproduction script with memory monitoring
- `ANTFLY_BUG_REPORT.md` - Original bug report
- `OOM_ANALYSIS.md` - This analysis document

## Additional System Information

```bash
# Check OOM killer history
journalctl --since "today" | grep -i oom

# Check if core dumps are enabled
coredumpctl list

# Monitor Antfly memory in real-time
watch -n 1 'ps aux | grep antfly'

# Check system memory pressure
cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable"
```
