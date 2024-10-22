import logging
import pickle
import sys
import traceback

# Redirect all output (stdout and stderr) to a log file
#log_file_path = "container_main.log"

# Open the log file in write mode and redirect stdout and stderr
#sys.stdout = open(log_file_path, 'w')
#sys.stderr = sys.stdout  # Redirect stderr to the same file as stdout

def main(prog_file: str, input_file: str, output_file: str):
    """Executes a deserialized function with input and writes output to file."""
    try:
        # Load the function from the prog_file
        with open(prog_file, "rb") as f:
            func = pickle.load(f)

        # Load the input data from the input_file
        with open(input_file, "rb") as input_f:
            input_data = pickle.load(input_f)

        # Execute the function with the input data
        ret = func(input_data)

        # Serialize and write the output to output_file
        with open(output_file, "wb") as of:
            pickle.dump(ret, of)

    except Exception as e:
        # Print the full error traceback to stderr (which is redirected to the log file)
        print(f"Error occurred: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)  # Exit with error code 1 to indicate failure

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Incorrect number of arguments. Expected 3 arguments.", file=sys.stderr)
        sys.exit(-1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
