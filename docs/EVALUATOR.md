# Evaluator

Executes LLM-generated functions in isolated sandboxes.

## Flow

1. **Parse LLM output** Extract code from XML tags (`<code>...</code>`), markdown fences, or raw text. Validate with AST.

2. **Integrate into template** Replace the `priority` function body in the evaluation script with the parsed code.

3. **Submit to ProcessPoolExecutor** For each test input, submit a task to the worker pool.

4. **Compile in worker** Each worker compiles the program. The eval script (imports, helper functions) is cached, only the new `priority` function is recompiled each time.

5. **Pickle and spawn sandbox** The compiled `evaluate` function is pickled to disk. A fresh subprocess runs it with memory/timeout limits.

6. **Collect results** Read output from pickle, publish scores to database queue. Cancel remaining tasks on first failure.

## Process Hierarchy

All three levels are separate processes with their own memory:

```
Evaluator process
├── ProcessPoolExecutor (not a process, just a Python object that manages workers)
│   ├── Worker 1 (separate process, own memory, persists)
│   ├── Worker 2 (separate process, own memory, persists)
│   └── Worker 3 (separate process, own memory, persists)
│
└── Each worker spawns:
    └── Sandbox (separate process, own memory, fresh each time, killed after)
```

## ProcessPoolExecutor and Compilation Caching

The evaluator creates a `ProcessPoolExecutor(max_workers)`. Workers are separate Python processes with their own memory, and they persist between evaluations.

Each worker runs `sandbox.run()` which calls `DummySandbox.compile_code()`. This separates the program into:
- **Base** (~180 lines): imports, `load_graph`, `solve`, `evaluate`, cached after first compilation
- **Priority** (~10 lines): LLM-generated function, recompiled each time

The cache is a class variable in the worker process. Since workers persist, subsequent evaluations skip recompiling the base and only compile the new priority function.

## Sandbox Subprocess

Each evaluation spawns a fresh subprocess (`container_main.py`) that:
1. Sets memory limit (default 1 GB)
2. Loads the pre-compiled function from `prog.pickle`
3. Loads input from `input.pickle`
4. Executes and writes result to `output.pickle`
5. Exits (killed after timeout if needed)

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeout` | 30 | Seconds before sandbox is killed |
| `max_workers` | 3 | Parallel workers per evaluator |
| `sandbox_memory_limit_gb` | 1.0 | RAM limit per sandbox (GiB) |
| `prefetch_count` | 15 | RabbitMQ message buffer |

## Debugging

```bash
# Check sandbox errors
cat sandbox/sandbox{pid}/stderr_*.log

# Monitor sandbox processes
ps aux | grep container_main
```
