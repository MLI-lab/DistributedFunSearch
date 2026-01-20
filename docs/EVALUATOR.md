# Evaluator

Executes LLM-generated functions in isolated sandboxes:

1. **Parse** - Extract code from XML tags, markdown fences, or raw text. Validate with AST.
2. **Integrate** - Replace the `priority` function body in the evaluation script.
3. **Distribute** - Submit each test input to the ProcessPoolExecutor worker pool.
4. **Compile** - Workers compile the program. Eval script is cached, only `priority` is recompiled.
5. **Sandbox** - Compiled function is pickled to disk, executed in a subprocess with memory/timeout limits.
6. **Collect** - Read results, publish scores to database queue. Cancel remaining tasks on first failure.

## Process Hierarchy

```
Evaluator process
├── ProcessPoolExecutor
│   ├── Worker 1 (persists between evaluations)
│   ├── Worker 2
│   └── Worker N
│
└── Each worker spawns:
    └── Sandbox subprocess (fresh each time, killed after)
```

## Compilation Caching

Workers are separate processes with their own memory that persist between evaluations. Each worker separates the program into:

- **Base** (~180 lines): imports, `load_graph`, `solve`, `evaluate`, cached after first compilation
- **Priority** (~10 lines): LLM-generated function, recompiled each time

## Sandbox Subprocess

Each evaluation spawns a fresh subprocess that:

1. Sets memory limit (default 1 GiB)
2. Loads pre-compiled function from `prog.pickle`
3. Loads input from `input.pickle`
4. Executes and writes result to `output.pickle`
5. Exits (killed after timeout if needed)

## Configuration

```python
EvaluatorConfig(
    # Execution
    timeout=30,                    # Seconds before sandbox killed
    max_workers=2,                 # Parallel workers per evaluator
    sandbox_memory_limit_gb=1.0,   # RAM limit per sandbox
    prefetch_count=15,             # RabbitMQ message buffer

    # Problem definition
    evaluation_script_path="...",  # Path to evaluation script with evaluate() and priority()
    initial_functions_dir="...",   # Seed functions to start evolution
    s_values=[1],                  # Problem parameters
    start_n=[6],                   # Range start
    end_n=[11],                    # Range end
    mode="last",                   # Score aggregation: last, average, weighted, relative_difference
)
```

## Debugging

```bash
# Check sandbox errors
cat sandbox/sandbox{pid}/stderr_*.log

# Monitor sandbox processes
ps aux | grep container_main
```
