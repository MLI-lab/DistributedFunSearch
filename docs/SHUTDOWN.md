# Shutdown Logic

## Process Hierarchy

```
Main Process (__main__.py)
├── ProgramsDatabase (async task in main process, not a child)
├── Evaluator 1 (child process via multiprocessing)
│   └── ProcessPoolExecutor workers
│       └── Sandbox subprocesses
├── Evaluator 2 (child process)
├── Sampler 1 (child process)
└── Sampler 2 (child process)
```

## Ctrl+C Shutdown

When you press Ctrl+C (or max iterations reached, which sends SIGTERM to main), main process calls `_shutdown()` in `__main__.py` which:

1. Cancels ProgramsDatabase's async task (it gets CancelledError)
2. Captures all descendant PIDs using `psutil.Process(pid).children(recursive=True)` (must do this before terminating, or they become orphans)
3. Calls `proc.terminate()` on each child process, sending SIGTERM
4. Waits up to 30s for children to exit
5. Force kills any children still alive (checks `p.is_alive()`) and any descendants still running (checks `d.is_running()`)
6. Closes RabbitMQ connections and exits

Main knows which processes to track because it stores references in `task_manager.evaluator_processes` and `task_manager.sampler_processes`.

Each child process receives SIGTERM and has 30s to shut down gracefully before being force killed. Its signal handler calls `shutdown()` in `startup.py` which:

1. Calls `instance.shutdown()` (Evaluator or Sampler)
2. Evaluator: closes RabbitMQ, kills ProcessPoolExecutor workers and sandbox subprocesses
3. Sampler: closes RabbitMQ, releases vLLM GPU memory
4. Exits via `sys.exit(0)`

## Dynamic Scaling Shutdown

When ResourceManager scales down, it terminates individual child processes (not the whole system). In `resource_manager.py`, `terminate_process()`:

1. Captures descendant PIDs using `psutil.Process(pid).children(recursive=True)`
2. Calls `proc.terminate()` on the selected child, sending SIGTERM
3. Waits for configurable timeout (default 10s for evaluators, 30s for samplers)
4. Force kills the child and any orphaned descendants if still alive

The child receives SIGTERM and shuts down the same way as in Ctrl+C (signal handler calls `shutdown()` in `startup.py`).

## Files

| Component | Shutdown logic location |
|-----------|------------------------|
| Main process | `src/disfun/__main__.py` (`_shutdown`, `_terminate_processes`) |
| Child process entry | `src/disfun/startup.py` (`shutdown` function, signal handlers) |
| Evaluator cleanup | `src/disfun/evaluator.py` (`shutdown`, `shutdown_subprocesses`) |
| Sampler cleanup | `src/disfun/sampler.py` (`shutdown`, `cleanup`) |
| RabbitMQ connection | `src/disfun/utils/rabbitmq.py` (`ConnectionManager.close`) |

## Timeouts

| Operation | Timeout |
|-----------|---------|
| Cancel consume task | 5s |
| Main waiting for children | 30s |
| ProcessPoolExecutor shutdown | 5s |
