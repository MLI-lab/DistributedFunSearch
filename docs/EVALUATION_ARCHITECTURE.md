# Evaluation Architecture

This document explains how DistributedFunSearch evaluates LLM-generated code, including process isolation, resource constraints, and import caching.

## Table of Contents

1. [Overview](#overview)
2. [Process Hierarchy](#process-hierarchy)
3. [Resource Constraints](#resource-constraints)
4. [Import Behavior & Caching](#import-behavior--caching)
5. [Data Flow](#data-flow)
6. [Performance Characteristics](#performance-characteristics)

---

## Overview

Each LLM-generated `priority` function is executed in an **isolated subprocess** to:
- Prevent crashes from affecting the main evaluator
- Enforce memory limits on untrusted code
- Allow timeout/kill of runaway code
- Isolate side effects

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Main Process                                │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                 Evaluator Process (1 of N)                      │ │
│  │                                                                  │ │
│  │  ┌────────────────────────────────────────────────────────────┐ │ │
│  │  │           ProcessPoolExecutor (max_workers=M)              │ │ │
│  │  │                                                            │ │ │
│  │  │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │ │ │
│  │  │   │  Worker 1    │   │  Worker 2    │   │  Worker M    │  │ │ │
│  │  │   │              │   │              │   │              │  │ │ │
│  │  │   │ Spawns:      │   │ Spawns:      │   │ Spawns:      │  │ │ │
│  │  │   │ container_   │   │ container_   │   │ container_   │  │ │ │
│  │  │   │ main.py      │   │ main.py      │   │ main.py      │  │ │ │
│  │  │   └──────────────┘   └──────────────┘   └──────────────┘  │ │ │
│  │  └────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Process Hierarchy

Think of each level as a separate terminal/Python session:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Level 1: EVALUATOR PROCESS (1 per evaluator)                           │
│ Lifetime: Hours (entire experiment)                                    │
│ Like: A terminal that stays open, running async event loop             │
│                                                                        │
│ Job: Fetch messages from RabbitMQ, publish results back                │
│ Imports: aio_pika, asyncio (light, ~100MB)                             │
│                                                                        │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │ Level 2: WORKER PROCESSES (max_workers per evaluator)        │     │
│   │ Lifetime: Hours (reused across many evaluations)             │     │
│   │ Like: Terminals that stay open, waiting for work             │     │
│   │                                                              │     │
│   │ Job: Spawn sandboxes, wait for results, read/write files     │     │
│   │ Imports: subprocess, cloudpickle (light, ~10MB)              │     │
│   │ NOTE: Workers do NOT import numpy/networkx!                  │     │
│   │                                                              │     │
│   │   ┌────────────────────────────────────────────────────┐     │     │
│   │   │ Level 3: SANDBOX PROCESSES (fresh per evaluation)  │     │     │
│   │   │ Lifetime: Seconds (dies after each eval)           │     │     │
│   │   │ Like: Opening a new terminal, running one command  │     │     │
│   │   │                                                    │     │     │
│   │   │ Job: Import libraries, run priority(), exit        │     │     │
│   │   │ Imports: numpy, networkx, graph-tool (HEAVY!)      │     │     │
│   │   │ Cost: ~500-1000ms just for imports, every time     │     │     │
│   │   └────────────────────────────────────────────────────┘     │     │
│   └──────────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────┘
```

### Level 1: Evaluator Process

Created by `attach_evaluators.py` or the main process. Each evaluator:
- Connects to RabbitMQ (`evaluator_queue`)
- Runs async event loop to fetch/publish messages
- Spawns `max_workers` worker processes via `ProcessPoolExecutor`
- **Lifetime: Entire experiment (hours)**

```python
# evaluator.py
self.executor = ProcessPoolExecutor(max_workers=max_workers)
```

### Level 2: ProcessPoolExecutor Workers

Persistent Python processes that act as "sandbox launchers":
- Created once when evaluator starts
- **Reused for many evaluations** (long lifetime)
- Each worker spawns sandbox subprocesses and waits for results
- **Do NOT import heavy libraries** (numpy, networkx, etc.)
- Only import light utilities: `subprocess`, `cloudpickle`, `pathlib`

```python
# evaluator.py - submits work to executor
tasks = {
    self.executor.submit(run_evaluation, self.sandbox, program, ...): input
    for input in self.inputs
}
```

**What workers actually do:**
1. Receive task from evaluator
2. Write function to `prog.pickle`
3. Spawn sandbox subprocess
4. Wait for sandbox to finish (or timeout)
5. Read result from `output.pickle`
6. Return result to evaluator
7. **Stay alive for next task**

### Level 3: Sandbox Subprocess (container_main.py)

A **fresh subprocess** (new terminal) for **each evaluation**:
- Spawned via `subprocess.Popen()`
- **Imports ALL heavy libraries fresh** (numpy, networkx, graph-tool)
- Runs `container_main.py` with the pickled function
- Has memory limits enforced via `resource.setrlimit()`
- **Dies after each evaluation** (or killed after timeout)
- **Next evaluation = new process = reimport everything**

```python
# sandbox.py - worker spawns this
process = subprocess.Popen(
    [python_path, container_main.py, prog.pickle, input.pickle, output.pickle],
    start_new_session=True  # New process group for clean kills
)
stdout, stderr = process.communicate(timeout=self.timeout_secs)
```

**Why fresh each time?** Isolation. If one LLM-generated function corrupts memory or crashes, it cannot affect the next evaluation.

### Timeline Visualization

```
Time ──────────────────────────────────────────────────────────────────────►

EVALUATOR (Level 1) - Lives for entire experiment
████████████████████████████████████████████████████████████████████████████
│ fetch msg │ fetch msg │ fetch msg │ fetch msg │        ...
│           │           │           │           │
▼           ▼           ▼           ▼           ▼

WORKER 1 (Level 2) - Lives for entire experiment, reused
████████████████████████████████████████████████████████████████████████████
│ task 1    │ task 3    │ task 5    │ task 7    │        ...
│  │        │  │        │  │        │  │        │
│  ▼        │  ▼        │  ▼        │  ▼        │
│ ┌───┐     │ ┌───┐     │ ┌───┐     │ ┌───┐     │
│ │S1 │     │ │S3 │     │ │S5 │     │ │S7 │     │   S = Sandbox
│ └───┘     │ └───┘     │ └───┘     │ └───┘     │   (fresh each time)
│  dies     │  dies     │  dies     │  dies     │

WORKER 2 (Level 2) - Lives for entire experiment, reused
████████████████████████████████████████████████████████████████████████████
│ task 2    │ task 4    │ task 6    │ task 8    │        ...
│  │        │  │        │  │        │  │        │
│  ▼        │  ▼        │  ▼        │  ▼        │
│ ┌───┐     │ ┌───┐     │ ┌─────┐   │ ┌───┐     │
│ │S2 │     │ │S4 │     │ │S6   │   │ │S8 │     │   S6 = timeout
│ └───┘     │ └───┘     │ │KILL!│   │ └───┘     │   (killed, worker OK)
│  dies     │  dies     │ └─────┘   │  dies     │

SANDBOXES (Level 3) - Each one is SHORT-LIVED, imports fresh
S1: ┌─import─┬─run─┐ dies
S2: ┌─import─┬─run─┐ dies
S3:          ┌─import─┬─run─┐ dies
S4:          ┌─import─┬─run─┐ dies
S5:                   ┌─import─┬─run─┐ dies
S6:                   ┌─import─────────┐ TIMEOUT! killed
S7:                            ┌─import─┬─run─┐ dies
S8:                            ┌─import─┬─run─┐ dies
```

**Key insight:** Workers (Level 2) stay alive and are reused. Sandboxes (Level 3) die after each evaluation. Heavy imports (numpy, networkx) happen in sandboxes, so they're paid fresh every time.

---

## Resource Constraints

### Memory Limit

Set in `container_main.py` using `resource.setrlimit()`:

```python
# container_main.py
MEMORY_LIMIT_GB = float(os.environ.get('SANDBOX_MEMORY_LIMIT_GB', '1'))
MEMORY_LIMIT_BYTES = int(MEMORY_LIMIT_GB * 1024 * 1024 * 1024)
resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
```

Configurable via `EvaluatorConfig.sandbox_memory_limit_gb` (default: 1.0 GB).

**What happens when exceeded:**
- Process receives `MemoryError` or is killed by OS
- `sandbox.run()` returns `(None, False, 0.0, ...)`
- Evaluation marked as failed, continues to next

### Timeout

Set per-evaluation in `sandbox.py`:

```python
# sandbox.py
stdout, stderr = process.communicate(timeout=self.timeout_secs)
```

Configurable via `EvaluatorConfig.timeout` (default: 30 seconds).

**What happens on timeout:**
- Entire process group is killed: `os.killpg(os.getpgid(process.pid), 9)`
- Evaluation returns failure

### Thread Limiting

Prevents libraries from spawning excessive threads:

```python
# container_main.py - set BEFORE imports
os.environ.setdefault("OMP_NUM_THREADS", "1")       # OpenMP (graph-tool, NumPy)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # OpenBLAS (NumPy default)
os.environ.setdefault("MKL_NUM_THREADS", "1")       # Intel MKL
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")   # NumExpr
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS Accelerate
```

Without this, each sandbox would spawn ~50 threads, causing massive contention.

### CPU Affinity

Not currently enforced. Processes share available CPUs via OS scheduler.

---

## Import Behavior & Caching

### What Gets Imported Where?

| Component | Imports | Cached? | Memory Impact |
|-----------|---------|---------|---------------|
| Evaluator process | aio_pika, asyncio, etc. | Yes (process lifetime) | ~100MB |
| Executor workers | None (just calls sandbox.run) | N/A | Minimal |
| Sandbox (container_main.py) | pickle, sys, resource | Yes (per-sandbox) | ~10MB |
| Evaluation script (loaded) | numpy, networkx, graph-tool | **NO** (fresh each time) | ~300-500MB |

### The Key Insight: Sandbox Imports Are NOT Cached

Each `container_main.py` subprocess is **fresh**. When your evaluation script does:

```python
import numpy as np
import networkx as nx
```

These imports happen **every single evaluation**. This is by design (isolation), but has performance implications:

| Library | Import Time | Memory |
|---------|-------------|--------|
| numpy | ~100-200ms | ~50MB |
| networkx | ~200-300ms | ~30MB |
| graph-tool | ~500-1000ms | ~200MB |
| **Total** | **~1-1.5s** | **~300MB** |

### Why Not Cache Imports?

Each sandbox subprocess:
1. Is a fresh Python process (no shared state)
2. Cannot share memory with parent (no copy-on-write for Python objects)
3. Dies after each evaluation

**This is the tradeoff for isolation.**

### Mitigation Strategies

1. **Keep evaluation fast**: If your `priority` function runs in 10ms, the 1s import overhead dominates. Consider batching or using lighter libraries.

2. **Use ProcessPoolExecutor workers differently**: The executor workers themselves DO cache imports. If we moved more logic there, we could cache. But this reduces isolation.

3. **Use Daytona warm pools** (see DAYTONA_EVALUATION.md): Pre-warmed containers with imports already loaded.

---

## Data Flow

### Per-Evaluation Flow

```
1. Evaluator receives message from RabbitMQ
   └── Message contains: LLM-generated code, island_id, version

2. Code is parsed and compiled
   └── sandbox.py: DummySandbox.compile_code(program)
   └── Extracts `priority` function from full program

3. Function is pickled to disk
   └── sandbox.py: cloudpickle.dump(namespace[function_to_run], f)
   └── Path: sandbox{id}/call{count}/prog.pickle

4. Input is pickled (cached by hash)
   └── sandbox.py: cloudpickle.dump(test_input, f)
   └── Path: sandbox{id}/inputs/{sha256_hash}.pickle
   └── Reused if same input seen before

5. Subprocess is spawned
   └── subprocess.Popen([python, container_main.py, prog.pickle, input.pickle, output.pickle])

6. container_main.py executes:
   a. Sets memory limit (resource.setrlimit)
   b. Loads function from prog.pickle
   c. Loads input from input.pickle
   d. Calls: result = func(input_data, GRAPH_DIR)
   e. Writes result + cpu_time to output.pickle

7. Parent reads output.pickle
   └── Returns (result, success, cpu_time, paths...)

8. Cleanup
   └── call{count}/ directory removed
   └── Input files kept (reused)
   └── stderr logs kept (for debugging)
```

### File Structure

```
sandbox/
└── sandbox{evaluator_id}/
    ├── inputs/
    │   ├── {hash1}.pickle    # Cached input (n=6, s=1)
    │   ├── {hash2}.pickle    # Cached input (n=7, s=1)
    │   └── ...
    ├── call0/                 # Temporary, cleaned up
    │   ├── prog.pickle        # Pickled priority function
    │   └── output.pickle      # Result from execution
    ├── stderr_0.log          # Error output (kept)
    └── stderr_1.log
```

---

## Performance Characteristics

### Timing Breakdown (typical)

| Phase | Time | Notes |
|-------|------|-------|
| Receive message | ~1ms | RabbitMQ fetch |
| Parse/compile code | ~1-5ms | AST parsing |
| Pickle function | ~1ms | cloudpickle.dump |
| Subprocess spawn | ~5-10ms | fork + exec |
| Import libraries | ~100-1000ms | **Biggest cost** |
| Load graph | ~10-100ms | With caching: ~0ms |
| Execute priority | ~1-100ms | Depends on function |
| Read result | ~1ms | cloudpickle.load |
| **Total** | **~120-1200ms** | Dominated by imports |

### With Graph Caching Enabled

When `cache_graphs=True` in EvaluatorConfig:
- Graphs loaded into memory on first use
- Subsequent evaluations skip graph loading
- Reduces per-eval time by ~50-100ms

But note: Graph caching happens in the **evaluation script** (loaded each time), not in the sandbox. So it only helps within a single subprocess lifetime.

### Parallelism

With `num_evaluators=5` and `max_workers=2`:
- 5 evaluators × 2 workers = 10 concurrent evaluations
- Each evaluation spawns 1 subprocess
- Total: up to 10 sandbox processes at once

Memory impact: 10 × 300MB (imports) = ~3GB peak

---

## Summary

| Aspect | Current Implementation |
|--------|----------------------|
| **Isolation** | Strong (separate subprocess per eval) |
| **Memory limit** | Yes (resource.setrlimit, default 1GB) |
| **Timeout** | Yes (subprocess.communicate timeout) |
| **Import caching** | No (fresh process each time) |
| **Thread limiting** | Yes (OMP_NUM_THREADS=1, etc.) |
| **Filesystem isolation** | No (can read/write any file) |
| **Network isolation** | No (unrestricted) |

For stronger isolation (filesystem, network), see [DAYTONA_EVALUATION.md](./DAYTONA_EVALUATION.md).
