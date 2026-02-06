# Evaluator

Executes LLM generated functions in isolated sandboxes using fork-based process isolation.

1. **Parse** extracts code from XML tags, markdown fences, or raw text, then validates with AST.
2. **Integrate** replaces the priority function body in the evaluation script.
3. **Cache graphs** loads graphs from LMDB on first use, caches in evaluator memory.
4. **Distribute** submits each test input (with graph) to the ThreadPoolExecutor worker pool.
5. **Compile** workers compile the program. Eval script is cached, only priority is recompiled.
6. **Fork** child process inherits parent's memory (including cached graphs) via copy-on-write.
7. **Sandbox** child sets memory limits, runs evaluation, returns result via pipe.
8. **Collect** parent reads results, publishes scores to database queue.

## Process Hierarchy

```
Evaluator process (caches graphs in memory)
├── ThreadPoolExecutor
│   ├── Thread 1 shares memory with evaluator (including graph cache)
│   ├── Thread 2
│   └── Thread N
│
└── Each thread forks:
    └── Child process (inherits graphs via copy-on-write), killed after timeout
```

## Fork-based Sandboxing

Uses `os.fork()` instead of `subprocess.Popen()` for massive performance gains:

- **Fork is fast**: ~1ms vs ~50ms for subprocess
- **No graph reload**: Child inherits parent's cached graphs via copy-on-write
- **No serialization**: Function and input already in child's memory
- **Full isolation**: Child crash doesn't affect parent

```
Worker thread                          Forked child
      │                                       │
      ├─── fork() ──────────────────────────► │ inherits memory (graphs, compiled code)
      │                                       │ sets memory limit
      │                                       │ executes priority function
      │  ◄─── pipe (result) ──────────────────┤ writes result
      │                                       │ os._exit(0)
```

The child:

1. Inherits parent's memory space (copy-on-write)
2. Sets memory limit via resource.setrlimit
3. Runs evaluation with cached graph
4. Writes result to pipe
5. Exits via `os._exit()` or gets killed after timeout

## Compilation Caching

Threads share memory and the compilation cache persists between evaluations. The program is separated into:

- **Base** imports, solve, evaluate, cached after first compilation
- **Priority** LLM generated function, recompiled each time

## Graph Caching

Graphs are loaded lazily on first use using FastGraph, a C++ graph class with NetworkX-compatible API. Uses CSR (Compressed Sparse Row) format for minimal memory.

- All threads share the same graph cache (thread-safe with Lock)
- Forked children inherit the cache via copy-on-write
- **Graphs are never reloaded during evaluation**

LLM priority functions use the same API: `G.neighbors(node)`, `G.degree(node)`, `G.nodes`.

## Performance

Fork overhead is ~35-40ms per evaluation regardless of graph size. Compare to subprocess approach which would reload graphs from LMDB each time (2+ minutes for IDS n=10).

| Graph Size | Fork + Eval Time | Graph Reload Time (old) |
|------------|------------------|------------------------|
| 1K nodes   | ~40ms            | ~30ms                  |
| 65K nodes  | ~40ms            | ~3s                    |
| 1M nodes   | ~40ms            | ~2 min                 |
| 4M nodes   | ~40ms            | ~75 min                |

## Memory Requirements

Each evaluator process loads and caches graphs. All values measured with C++ FastGraph. Total Time includes graph loading plus one greedy MIS evaluation with a trivial priority function.

### Deletion Binary s=1

| n | Nodes | Edges | Memory | Total Time |
|---|-------|-------|--------|------------|
| 6 | 64 | 543 | < 1 MB | 5 ms |
| 7 | 128 | 1,471 | < 1 MB | 4 ms |
| 8 | 256 | 3,839 | < 1 MB | 47 ms |
| 9 | 512 | 9,727 | < 1 MB | 23 ms |
| 10 | 1,024 | 24,063 | < 1 MB | 13 ms |
| 11 | 2,048 | 58,367 | < 1 MB | 24 ms |

**Total for n=6 to n=11: < 5 MB per evaluator**

### Deletion Binary s=2

| n | Nodes | Edges | Memory | Total Time |
|---|-------|-------|--------|------------|
| 7 | 128 | 5,173 | < 1 MB | 5 ms |
| 8 | 256 | 17,183 | < 1 MB | 58 ms |
| 9 | 512 | 54,895 | < 1 MB | 20 ms |
| 10 | 1,024 | 169,162 | < 1 MB | 48 ms |
| 11 | 2,048 | 504,451 | < 1 MB | 137 ms |
| 12 | 4,096 | 1,460,525 | < 1 MB | 386 ms |

**Total for n=7 to n=12: < 10 MB per evaluator**

### IDS Quaternary s=1

Memory has two values: **Final** (after loading, in-use) and **Peak** (during LMDB parsing spike).

| n | Nodes | Edges | Final Memory | Peak Memory | Eval Time |
|---|-------|-------|--------------|-------------|-----------|
| 6 | 4,096 | 392,358 | 40 MB | 40 MB | 0.1 s |
| 7 | 16,384 | 2,206,374 | 90 MB | 350 MB | 0.7 s |
| 8 | 65,536 | 11,815,590 | 200 MB | 1.4 GB | 3.4 s |
| 9 | 262,144 | 60,992,166 | 620 MB | 5.0 GB | 19.7 s |
| 10 | 1,048,576 | 305,965,734 | 2.5 GB | 20 GB | 2.2 min |
| 11 | 4,194,304 | 1,500,162,726 | 12.2 GB | ~50 GB | 75 min |

**Cumulative for n=6 to n=9: ~1.2 GB final, ~6.5 GB peak during loading**
**Cumulative for n=6 to n=10: ~3.5 GB final, ~25 GB peak during loading**

**WARNING**: Peak memory during LMDB loading is much higher than final memory. When starting multiple evaluators simultaneously, the peak spikes can cause OOM. Solutions:
1. **Stagger evaluator startup** - don't start all at once
2. **Pre-load graphs in main process** before forking evaluators
3. **Start with fewer evaluators** and scale up after graphs are cached

## Configuration

```python
EvaluatorConfig(
    timeout=30,                    # Seconds before sandbox killed
    max_workers=2,                 # Parallel workers per evaluator
    sandbox_memory_limit_gb=1.0,   # Memory limit per sandbox
    sandbox_debug=False,           # True to log sandbox errors to stderr
    prefetch_count=15,             # RabbitMQ message buffer

    evaluation_script_path="...",  # Path to evaluation script with evaluate and priority
    initial_functions_dir="...",   # Seed functions to start evolution
    s_values=[1],                  # Problem parameters
    start_n=[6],                   # Range start
    end_n=[11],                    # Range end
    mode="last",                   # Score aggregation, last, average, weighted, relative_difference
)
```

## Debugging

Graphs are cached in memory. No files written to disk during evaluation (fork-based).

To see why sandbox evaluations fail (compile errors, runtime exceptions), enable `sandbox_debug` in your config:

```python
EvaluatorConfig(
    sandbox_debug=True,  # Default: False. Logs errors with full tracebacks to stderr.
)
```

This prints `SANDBOX COMPILE ERROR` and `SANDBOX ERROR` messages to stderr (visible in SLURM `.err` logs). Off by default to avoid excessive output during normal runs.

```bash
# Check memory usage (graphs cached in evaluator)
ps aux | grep evaluator

# Check evaluator logs for errors
journalctl -u disfun-evaluator -f

# Monitor child processes (short-lived forks)
watch -n 0.1 'ps --ppid <evaluator_pid>'
```
