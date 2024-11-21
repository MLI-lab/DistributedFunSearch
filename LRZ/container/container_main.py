"""This file will be used as an executable script by the ExternalProcessSandbox.
 Designed to run in an isolated environment and execute a function that has been serialized (saved) with pickle.
"""
import logging
import pickle
import sys



def main(prog_file: str, input_file: str, output_file: str):
  """The method takes executable function as a cloudpickle file, then executes it with input data, and writes the output data to another file."""

  with open(prog_file, "rb") as f:
    func = pickle.load(f)

    with open(input_file, "rb") as input_f:
      input_data = pickle.load(input_f)

      #The deserialized function is then called with the deserialized input data, and the result is stored in ret.
      ret = func(input_data)
      #Serialize and write to output file 
      with open(output_file, "wb") as of:
        pickle.dump(ret, of)

# Using if __name__ == '__main__': allows to design scripts that can be run standalone to perform a specific task or imported as modules by other scripts without executing the main part of the script immediately.    
if __name__ == '__main__':
  # When a Python script is executed from the command line, sys.argv is a list that contains the command-line arguments passed to the script.
  # The first item in this list, sys.argv[0], is always the script's filename itself. 
  if len(sys.argv) != 4:
    sys.exit(-1)
  main(sys.argv[1], sys.argv[2], sys.argv[3])