# Daytona Sandbox Integration Plan

This document evaluates using [Daytona](https://www.daytona.io/) for sandboxed code execution in DistributedFunSearch, comparing it to the current subprocess-based approach.

## Table of Contents

1. [Current Architecture](#current-architecture)
2. [Why Consider Daytona?](#why-consider-daytona)
3. [Architecture Options](#architecture-options)
4. [Resource & Cost Analysis](#resource--cost-analysis)
5. [Implementation Plan](#implementation-plan)
6. [Benchmark Strategy](#benchmark-strategy)
7. [Decision Matrix](#decision-matrix)

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Evaluator Process                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ProcessPoolExecutor (max_workers=2)         │    │
│  │  ┌─────────────────┐    ┌─────────────────┐             │    │
│  │  │   Worker 1      │    │   Worker 2      │             │    │
│  │  │ (subprocess)    │    │ (subprocess)    │             │    │
│  │  │                 │    │                 │             │    │
│  │  │ - Imports cached│    │ - Imports cached│             │    │
│  │  │ - Loads pickle  │    │ - Loads pickle  │             │    │
│  │  │ - Runs priority │    │ - Runs priority │             │    │
│  │  └─────────────────┘    └─────────────────┘             │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

Protections:
  ✅ Memory limit (resource.setrlimit)
  ✅ Timeout (subprocess.communicate timeout)
  ✅ Process group kill on timeout
  ❌ Filesystem access (can read/write any file)
  ❌ Network access (unrestricted)
  ❌ System calls (unrestricted)
```

**Current performance:**
- Subprocess startup: ~1-5ms
- Import cost: Cached (paid once per worker lifetime)
- Total eval overhead: ~5-10ms

---

## Why Consider Daytona?

### Security Gaps in Current Approach

Generated code can currently:
```python
def priority(node, G_gt, node_to_vertex, vertex_to_node, n, s):
    # These are all possible with current sandboxing:
    import os
    os.system("rm -rf /workspace/*")           # Delete files
    open("/etc/passwd").read()                  # Read sensitive files
    import urllib.request                       # Network access
    urllib.request.urlopen("http://evil.com")
    return 0.0
```

### What Daytona Provides

| Feature | Current | Daytona |
|---------|---------|---------|
| Filesystem isolation | ❌ | ✅ |
| Network isolation | ❌ | ✅ |
| Memory limits | ✅ | ✅ |
| CPU limits | Partial | ✅ |
| Timeout | ✅ | ✅ |
| Startup time | ~1-5ms | ~90-200ms (warm) |

---

## Architecture Options

### Option A: Fresh Sandbox Per Evaluation (Simple, Slow)

```
For each generated priority function:
  1. Create new Daytona sandbox (~90-200ms)
  2. Install imports (numpy, graph-tool) (~1-5 seconds!)
  3. Run evaluation (~1-30 seconds)
  4. Delete sandbox

Total overhead: ~2-6 seconds per evaluation
```

**Verdict: Too slow.** Import cost kills throughput.

---

### Option B: Warm Pool with Reuse (Recommended)

```
┌──────────────────────────────────────────────────────────────────┐
│                    Daytona Sandbox Pool                          │
│                                                                  │
│   Startup: Create N sandboxes, pre-load imports                  │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐     ┌──────────┐        │
│  │Sandbox 1 │ │Sandbox 2 │ │Sandbox 3 │ ... │Sandbox N │        │
│  │          │ │          │ │          │     │          │        │
│  │ numpy ✓  │ │ numpy ✓  │ │ numpy ✓  │     │ numpy ✓  │        │
│  │ gt ✓     │ │ gt ✓     │ │ gt ✓     │     │ gt ✓     │        │
│  │ (idle)   │ │ (busy)   │ │ (idle)   │     │ (busy)   │        │
│  └──────────┘ └──────────┘ └──────────┘     └──────────┘        │
│       │                          │                               │
│       ▼                          ▼                               │
│  Grab idle    ───────────►   Execute    ───────────►   Return   │
│  sandbox                     priority()               to pool    │
└──────────────────────────────────────────────────────────────────┘

Per-evaluation overhead: ~50-100ms (API call + execution start)
```

**How it works:**

```python
from daytona_sdk import Daytona, CreateSandboxParams

class DaytonaSandboxPool:
    def __init__(self, pool_size: int = 100):
        self.daytona = Daytona(config)
        self.pool_size = pool_size
        self.sandboxes = []
        self.available = asyncio.Queue()

    async def initialize(self):
        """Create pool and pre-warm with imports."""
        print(f"Creating {self.pool_size} sandboxes...")

        # Create all sandboxes
        for i in range(self.pool_size):
            sandbox = self.daytona.create(CreateSandboxParams(
                language="python",
                auto_stop_interval=0,  # Never auto-stop
            ))

            # Pre-load expensive imports
            sandbox.process.code_run("""
import numpy as np
import graph_tool.all as gt
# ... other imports from specification
            """)

            self.sandboxes.append(sandbox)
            await self.available.put(sandbox)

        print(f"Pool ready with {self.pool_size} warm sandboxes")

    async def execute(self, code: str) -> Any:
        """Execute code in a pooled sandbox."""
        sandbox = await self.available.get()
        try:
            result = sandbox.process.code_run(code)
            return result
        finally:
            # Return to pool (don't delete!)
            await self.available.put(sandbox)

    async def cleanup(self):
        """Cleanup all sandboxes on shutdown."""
        for sandbox in self.sandboxes:
            try:
                self.daytona.remove(sandbox)
            except Exception:
                pass
```

---

### Option C: One Sandbox Per Evaluator (Middle Ground)

```
┌─────────────────────────────────────────────────────────────────┐
│                     40 Evaluator Processes                       │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐         │
│  │ Evaluator 1 │  │ Evaluator 2 │  ...  │ Evaluator 40│         │
│  │             │  │             │       │             │         │
│  │ ┌─────────┐ │  │ ┌─────────┐ │       │ ┌─────────┐ │         │
│  │ │Sandbox 1│ │  │ │Sandbox 2│ │       │ │Sandbox40│ │         │
│  │ │(warm)   │ │  │ │(warm)   │ │       │ │(warm)   │ │         │
│  │ └─────────┘ │  │ └─────────┘ │       │ └─────────┘ │         │
│  └─────────────┘  └─────────────┘       └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘

Each evaluator:
  - Creates 1 Daytona sandbox at startup
  - Keeps it alive (auto_stop_interval=0)
  - Reuses for all evaluations
  - Cleans up on shutdown
```

**Simpler than Option B**, maps 1:1 to current architecture.

```python
class DaytonaEvaluator:
    def __init__(self, local_id: int):
        self.local_id = local_id
        self.daytona = Daytona(config)
        self.sandbox = None

    async def initialize(self):
        """Create and warm up sandbox."""
        self.sandbox = self.daytona.create(CreateSandboxParams(
            language="python",
            auto_stop_interval=0,
        ))

        # Pre-load imports (one-time cost)
        self.sandbox.process.code_run(IMPORTS_CODE)

    def run_evaluation(self, program: str, test_input) -> tuple:
        """Run priority function in sandbox."""
        # Send pickled function and input
        code = f"""
import cloudpickle
func = cloudpickle.loads({repr(cloudpickle.dumps(func))})
input_data = cloudpickle.loads({repr(cloudpickle.dumps(test_input))})
result = func(input_data)
print(cloudpickle.dumps(result))
"""
        response = self.sandbox.process.code_run(code)
        return cloudpickle.loads(response.result)

    async def cleanup(self):
        if self.sandbox:
            self.daytona.remove(self.sandbox)
```

---

### Option D: Hybrid (Current + Daytona for Untrusted)

Keep current subprocess approach for speed, add Daytona as optional "secure mode":

```python
class EvaluatorConfig:
    # ... existing fields ...
    sandbox_mode: str = "subprocess"  # "subprocess" | "daytona"
    daytona_api_key: str = None
```

Use cases:
- **Development/trusted code**: `sandbox_mode="subprocess"` (fast)
- **Production/untrusted**: `sandbox_mode="daytona"` (secure)

---

## Resource & Cost Analysis

### Memory Requirements (100 sandboxes)

| Component | Per Sandbox | × 100 |
|-----------|-------------|-------|
| Base container | ~50 MB | 5 GB |
| Python + numpy | ~150 MB | 15 GB |
| graph-tool | ~300 MB | 30 GB |
| **Total** | **~500 MB** | **~50 GB** |

**Conclusion:** 100 sandboxes feasible with 64+ GB RAM.

### Daytona Pricing (if cloud-hosted)

From their docs:
- Default resources: 1 vCPU, 1GB RAM, 3GB disk per sandbox
- Can request up to: 4 vCPU, 8GB RAM, 10GB disk
- Warm pool: "Sandboxes launch in milliseconds"
- Stopped sandboxes: Disk-only costs
- Archived sandboxes: Object storage costs (cheapest)

**For startups:** Check https://www.daytona.io/startups for credits.

### Latency Comparison

| Operation | Current (subprocess) | Daytona (warm pool) |
|-----------|---------------------|---------------------|
| Get worker/sandbox | ~0ms | ~10-50ms |
| Send code/data | ~1ms (disk) | ~10-20ms (API) |
| Start execution | ~1-5ms | ~50-100ms |
| **Total overhead** | **~5-10ms** | **~70-170ms** |

**Impact on throughput:**

With 30-second average evaluation time:
- Current: 30.01s per eval → ~120 evals/hour/evaluator
- Daytona: 30.15s per eval → ~119 evals/hour/evaluator
- **~1% throughput reduction** (acceptable)

With 1-second average evaluation time:
- Current: 1.01s per eval → ~3,560 evals/hour/evaluator
- Daytona: 1.15s per eval → ~3,130 evals/hour/evaluator
- **~12% throughput reduction** (noticeable)

---

## Implementation Plan

### Phase 1: Benchmark (1-2 days)

Create benchmark script to measure actual overhead:

```
tests/
└── benchmark_daytona.py    # Compare subprocess vs Daytona
```

**Metrics to measure:**
1. Cold start time (fresh sandbox)
2. Warm start time (reused sandbox)
3. Import time (numpy, graph-tool)
4. Code execution latency
5. Memory usage per sandbox
6. Max concurrent sandboxes

### Phase 2: Implement DaytonaSandbox (2-3 days)

```
src/disfun/
├── sandbox.py              # Existing (keep)
├── sandbox_daytona.py      # New Daytona implementation
└── sandbox_factory.py      # Factory to select implementation
```

**Interface (same as ExternalProcessSandbox):**

```python
class DaytonaSandbox:
    def __init__(self, api_key: str, timeout_secs: int, memory_limit_gb: float):
        ...

    def run(self, program: str, function_to_run: str, test_input,
            timeout_seconds: int, count: int) -> tuple[Any, bool, float, ...]:
        """Same interface as ExternalProcessSandbox.run()"""
        ...

    def cleanup_all(self):
        """Remove sandbox from Daytona."""
        ...
```

### Phase 3: Pool Manager (1-2 days)

```python
# src/disfun/sandbox_pool.py

class SandboxPool:
    """Manages pool of warm Daytona sandboxes."""

    def __init__(self, pool_size: int, config: DaytonaConfig):
        self.pool_size = pool_size
        self.config = config
        self.available: asyncio.Queue[Sandbox] = asyncio.Queue()
        self.all_sandboxes: list[Sandbox] = []

    async def initialize(self):
        """Create and warm all sandboxes."""
        ...

    async def acquire(self) -> Sandbox:
        """Get an available sandbox (blocks if none available)."""
        return await self.available.get()

    async def release(self, sandbox: Sandbox):
        """Return sandbox to pool."""
        await self.available.put(sandbox)

    async def shutdown(self):
        """Clean up all sandboxes."""
        for sb in self.all_sandboxes:
            self.daytona.remove(sb)
```

### Phase 4: Config Integration (1 day)

```python
# config.py additions

@dataclasses.dataclass(frozen=True)
class SandboxConfig:
    """Configuration for code execution sandbox."""
    mode: str = "subprocess"  # "subprocess" | "daytona"

    # Subprocess mode settings (existing)
    memory_limit_gb: float = 1.0
    timeout: int = 30

    # Daytona mode settings (new)
    daytona_api_key: str = None
    daytona_server_url: str = "https://api.daytona.io"
    daytona_pool_size: int = None  # None = 1 per evaluator
    daytona_auto_stop_interval: int = 0  # 0 = never
```

### Phase 5: Cleanup & Error Handling (1 day)

**Graceful shutdown:**
```python
async def shutdown(self):
    # 1. Stop accepting new work
    self._shutdown_requested = True

    # 2. Wait for in-flight evaluations (with timeout)
    await asyncio.wait_for(self._drain_queue(), timeout=60)

    # 3. Clean up Daytona sandboxes
    if self.sandbox_pool:
        await self.sandbox_pool.shutdown()
```

**Error recovery:**
```python
async def execute_with_retry(self, sandbox, code, max_retries=2):
    for attempt in range(max_retries):
        try:
            return sandbox.process.code_run(code)
        except DaytonaError as e:
            if "sandbox stopped" in str(e):
                # Sandbox died, get a new one
                sandbox = await self.pool.acquire_fresh()
            else:
                raise
```

---

## Benchmark Strategy

### Test Script

```python
# tests/benchmark_daytona.py

import asyncio
import time
import statistics
from daytona_sdk import Daytona, CreateSandboxParams

# Test configuration
NUM_SANDBOXES = [1, 10, 50, 100]
NUM_ITERATIONS = 20
IMPORTS = """
import numpy as np
try:
    import graph_tool.all as gt
    HAS_GT = True
except ImportError:
    HAS_GT = False
"""

EVAL_CODE = """
import numpy as np
arr = np.random.rand(1000, 1000)
result = np.sum(arr @ arr.T)
print(result)
"""

async def benchmark_cold_start():
    """Measure time to create fresh sandbox."""
    daytona = Daytona(config)
    times = []

    for i in range(NUM_ITERATIONS):
        start = time.perf_counter()
        sandbox = daytona.create(CreateSandboxParams(language="python"))
        times.append(time.perf_counter() - start)
        daytona.remove(sandbox)

    print(f"Cold start: {statistics.mean(times)*1000:.0f}ms ± {statistics.stdev(times)*1000:.0f}ms")

async def benchmark_warm_execution():
    """Measure execution time with pre-warmed sandbox."""
    daytona = Daytona(config)
    sandbox = daytona.create(CreateSandboxParams(
        language="python",
        auto_stop_interval=0,
    ))

    # Warm up with imports
    sandbox.process.code_run(IMPORTS)

    times = []
    for i in range(NUM_ITERATIONS):
        start = time.perf_counter()
        sandbox.process.code_run(EVAL_CODE)
        times.append(time.perf_counter() - start)

    print(f"Warm execution: {statistics.mean(times)*1000:.0f}ms ± {statistics.stdev(times)*1000:.0f}ms")
    daytona.remove(sandbox)

async def benchmark_pool_scaling():
    """Test how many concurrent sandboxes we can run."""
    daytona = Daytona(config)

    for pool_size in NUM_SANDBOXES:
        start = time.perf_counter()
        sandboxes = []

        for i in range(pool_size):
            sb = daytona.create(CreateSandboxParams(
                language="python",
                auto_stop_interval=0,
            ))
            sandboxes.append(sb)

        create_time = time.perf_counter() - start

        # Run concurrent executions
        start = time.perf_counter()
        await asyncio.gather(*[
            asyncio.to_thread(sb.process.code_run, "x = 1 + 1")
            for sb in sandboxes
        ])
        exec_time = time.perf_counter() - start

        print(f"Pool size {pool_size}: create={create_time:.1f}s, exec={exec_time:.1f}s")

        # Cleanup
        for sb in sandboxes:
            daytona.remove(sb)

if __name__ == "__main__":
    asyncio.run(benchmark_cold_start())
    asyncio.run(benchmark_warm_execution())
    asyncio.run(benchmark_pool_scaling())
```

### Compare with Current Implementation

```python
async def benchmark_current_subprocess():
    """Measure current subprocess sandbox performance."""
    from disfun.sandbox import ExternalProcessSandbox

    sandbox = ExternalProcessSandbox(
        base_path="./benchmark_sandbox",
        timeout_secs=30,
        local_id=0,
    )

    # Simple program
    program = """
def evaluate(input_data):
    import numpy as np
    arr = np.random.rand(1000, 1000)
    return np.sum(arr @ arr.T), None
"""

    times = []
    for i in range(NUM_ITERATIONS):
        start = time.perf_counter()
        result, success, cpu_time, *_ = sandbox.run(
            program, "evaluate", (6, 1, 2), 30, i
        )
        times.append(time.perf_counter() - start)

    print(f"Subprocess: {statistics.mean(times)*1000:.0f}ms ± {statistics.stdev(times)*1000:.0f}ms")
```

---

## Decision Matrix

| Criteria | Weight | Subprocess | Daytona (warm pool) |
|----------|--------|------------|---------------------|
| Startup latency | 20% | ⭐⭐⭐⭐⭐ (1-5ms) | ⭐⭐⭐ (70-170ms) |
| Security/isolation | 25% | ⭐⭐ (process only) | ⭐⭐⭐⭐⭐ (container) |
| Filesystem protection | 15% | ⭐ (none) | ⭐⭐⭐⭐⭐ (isolated) |
| Network protection | 10% | ⭐ (none) | ⭐⭐⭐⭐⭐ (isolated) |
| Implementation effort | 10% | ⭐⭐⭐⭐⭐ (done) | ⭐⭐⭐ (2-3 days) |
| Operational complexity | 10% | ⭐⭐⭐⭐⭐ (simple) | ⭐⭐⭐ (API dependency) |
| Cost | 10% | ⭐⭐⭐⭐⭐ (free) | ⭐⭐⭐ (API costs?) |

### Recommendations

**For research/development (current use):**
→ Keep subprocess sandboxing. Security risk is low, performance is critical.

**For production/multi-tenant:**
→ Use Daytona with warm pool. Security isolation is worth the overhead.

**Hybrid approach:**
→ Add Daytona as optional mode. Use subprocess for trusted experiments, Daytona for untrusted code.

---

## Next Steps

1. [ ] Sign up for Daytona (check startup program)
2. [ ] Get API key
3. [ ] Run benchmark script
4. [ ] Decide based on actual numbers
5. [ ] Implement if overhead is acceptable

---

## References

- [Daytona GitHub](https://github.com/daytonaio/daytona)
- [Daytona Documentation](https://www.daytona.io/docs/)
- [Sandbox Management Guide](https://www.daytona.io/docs/en/sandbox-management/)
- [Python SDK Reference](https://www.daytona.io/docs/en/python-sdk/)
- [Daytona Startups Program](https://www.daytona.io/startups)
