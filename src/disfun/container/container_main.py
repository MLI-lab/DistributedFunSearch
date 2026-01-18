# Limit threading libraries BEFORE any imports that use them.
# This prevents OpenBLAS, MKL, and OpenMP from spawning threads per process.
# Without this, each container spawns ~50 threads, causing massive CPU contention.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # OpenMP (graph-tool, some NumPy)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # OpenBLAS (NumPy default)
os.environ.setdefault("MKL_NUM_THREADS", "1")       # Intel MKL (alternative NumPy)
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")   # NumExpr
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS Accelerate

import pickle
import sys
import traceback
import time
import resource

# Set memory limit to prevent runaway generated code from consuming all RAM.
# Default is 1GB per sandbox process, configurable via SANDBOX_MEMORY_LIMIT_GB env var.
MEMORY_LIMIT_GB = float(os.environ.get('SANDBOX_MEMORY_LIMIT_GB', '1'))
MEMORY_LIMIT_BYTES = int(MEMORY_LIMIT_GB * 1024 * 1024 * 1024)
try:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
except (ValueError, resource.error):
    pass  # System limit may be lower than requested, proceed without limit

# Get graph directory from environment variable (required, set by sandbox.py)
GRAPH_DIR = os.environ.get('GRAPH_DIR')
if GRAPH_DIR is None:
    print("Error: GRAPH_DIR environment variable not set", file=sys.stderr)
    sys.exit(1)

def main(prog_file: str, input_file: str, output_file: str):
    """Executes a deserialized function with input and writes output to file."""
    try:
        # Load the function from the prog_file
        with open(prog_file, "rb") as f:
            func = pickle.load(f)

        # Load the input data from the input_file
        with open(input_file, "rb") as input_f:
            input_data = pickle.load(input_f)

        # Inject `GRAPH_DIR` into the function call
        start_cpu_time = time.process_time()
        ret = func(input_data, GRAPH_DIR)  # Pass GRAPH_DIR
        end_cpu_time = time.process_time()

        execution_time = end_cpu_time - start_cpu_time

        # Serialize and write the output to output_file
        with open(output_file, "wb") as of:
            pickle.dump({"result": ret, "cpu_time": execution_time}, of)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)  # Exit with error code 1 to indicate failure

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Incorrect number of arguments. Expected 3 arguments.", file=sys.stderr)
        sys.exit(-1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
