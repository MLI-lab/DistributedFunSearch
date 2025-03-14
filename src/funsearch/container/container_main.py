import logging
import pickle
import sys
import traceback
import time
import os

# Dynamically locate `src/graphs`
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # Move up two levels
GRAPH_DIR = os.path.join(SRC_DIR, "graphs")

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
        ret = func(input_data, GRAPH_DIR)  # ← Pass GRAPH_DIR
        end_cpu_time = time.process_time()

        execution_time = end_cpu_time - start_cpu_time

        # Serialize and write the output to output_file
        with open(output_file, "wb") as of:
            pickle.dump({"result": ret, "cpu_time": execution_time}, of)

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)  # Exit with error code 1 to indicate failure

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Incorrect number of arguments. Expected 3 arguments.", file=sys.stderr)
        sys.exit(-1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
