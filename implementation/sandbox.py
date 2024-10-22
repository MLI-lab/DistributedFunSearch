import logging
import ast
import os
import pathlib
import sys
from typing import Any
import time
import subprocess
import cloudpickle
import warnings

# Set up the main logger for sandbox operations
log_file_path = pathlib.Path(__file__).parent / "sandbox.log"
logger = logging.getLogger('sandbox_logger')
logger.setLevel(logging.INFO)  # Set the log level

# Create file handler for main sandbox log
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)


# Set up a separate logger for warnings within the sandbox
warning_logger_sandbox = logging.getLogger('warning_logger_sandbox')
warning_handler_sandbox = logging.FileHandler('sandbox_warnings.log')
warning_handler_sandbox.setLevel(logging.WARNING)
warning_logger_sandbox.addHandler(warning_handler_sandbox)

# Custom handler that redirects warnings to the sandbox warning logger
def custom_warning_handler_sandbox(message, category, filename, lineno, file=None, line=None):
    warning_logger_sandbox.warning(f'{category.__name__}: {message} in {filename}, line {lineno}')

# Redirect warnings to the sandbox warning logger
warnings.showwarning = custom_warning_handler_sandbox

# Optionally, make sure all warnings are caught and not ignored
warnings.simplefilter("always")  # Ensure all warnings are caught


CONTAINER_MAIN = (pathlib.Path(__file__).parent / "container" / "container_main.py").absolute()



class DummySandbox():
    """Base class for Sandboxes that execute the generated code.

    Functionality: provides a way to dynamically execute Python code contained in a string,
    along with calling a function defined in that code with specific input.
    """
    def __init__(self, **kwargs):
        pass

    def run(
            self,
            program: str,
            function_to_run: str,
            test_input,
            timeout_seconds: int,
            count,
    ) -> tuple[Any, bool]:
        """Returns `function_to_run(test_input)` and whether execution succeeded."""

        namespace = DummySandbox.compile_code(program)
        return namespace[function_to_run](test_input)

    @staticmethod
    def compile_code(program: str):
        namespace = {}
        parsed_code = ast.parse(program)
        compiled_code = compile(parsed_code, filename="<ast>", mode="exec")
        exec(compiled_code, namespace)
        return namespace


class ExternalProcessSandbox(DummySandbox):
    """Sandbox that executes the code in a separate Python process in the same host."""

    def __init__(self, base_path: pathlib.Path, timeout_secs: int = 30, python_path: str = "python", local_id=None):
        super(ExternalProcessSandbox, self).__init__()
        self.local_id = local_id
        self.output_path = pathlib.Path(base_path) / f"sandbox{self.local_id}"
        self.timeout_secs = timeout_secs
        self.python_path = python_path
        self.input_path = self.output_path / "inputs"
        for p in [self.output_path, self.input_path]:
            if not p.exists():
                p.mkdir(parents=True)

    def _exec(self, call_data_path: pathlib.Path, input_path: pathlib.Path, error_file_path: pathlib.Path):
        """Use subprocess to execute python in a container.
        - main.py executes the LLM generated method from prog.pickle using input.pickle as input.
        - main.py writes the output of the method into output.pickle.
        """

        prog_path = call_data_path / "prog.pickle"  # Path for serialized Python function
        output_file = call_data_path / "output.pickle"  # Path where output will be written

        # Ensure directories exist before running the command
        if not call_data_path.exists():
            call_data_path.mkdir(parents=True, exist_ok=True)
        if not input_path.parent.exists():
            input_path.parent.mkdir(parents=True, exist_ok=True)
        if not error_file_path.parent.exists():
            error_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Construct the command to run the Python script with arguments
        cmd = [self.python_path, str(CONTAINER_MAIN), str(prog_path), str(input_path), str(output_file)]
        logger.debug(f"Executing the command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            # Use subprocess to run the command
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
            # Poll the process until it's done or times out
            while time.time() - start_time < self.timeout_secs:
                retcode = process.poll()  # Check if the process has finished
                if retcode is not None:  # If process has completed
                    if retcode == 0:
                        logger.debug("Process completed successfully")
                        return True
                    else:
                        logger.error(f"Process failed with return code {retcode}")
                        return False
                time.sleep(0.1)  # Sleep briefly before checking again

            # If we reach here, the process timed out
            logger.error("Process terminated due to timeout")
            process.kill()  # Forcefully terminate the process
            return False

        except Exception as e:
            logger.error(f"Error while executing process: {e}", exc_info=True)
            return False

    def run(
        self,
        program: str,
        function_to_run: str,
        test_input,
        timeout_seconds: int,
        count: int, 
    ) -> tuple[Any, bool, pathlib.Path]:  # Return the folder as part of the tuple

        call_data_folder = (self.output_path / f"call{count}").absolute()

        # Ensure the directory exists before running
        if not call_data_folder.exists():
            call_data_folder.mkdir(parents=True, exist_ok=True)

        input_hash = hash(test_input)
        input_path = (self.input_path / f"{input_hash}.pickle").absolute()

        # Ensure the input directory exists before creating the file
        if not self.input_path.exists():
            self.input_path.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            with open(input_path, "wb") as f:
                cloudpickle.dump(test_input, f)

        error_file = None  # Define error_file with a default value
        try:
            namespace = DummySandbox.compile_code(program)

            prog_file = (call_data_folder / f"prog.pickle").absolute()
            with open(prog_file, "wb+") as f:
                cloudpickle.dump(namespace[function_to_run], f)

            error_file = self.output_path / f"stderr_{count}.log"
            logger.debug("Before retcode = self._exec(call_data_folder, input_path, error_file)")
            retcode = self._exec(call_data_folder, input_path, error_file)

            if not retcode:
                return None, False, call_data_folder, input_path, error_file

            output_file = call_data_folder / f"output.pickle"
            with open(output_file, "rb") as f:
                out = cloudpickle.load(f)
                return out, True, call_data_folder, input_path, error_file  # Return the call_data_folder as part of the result
        except Exception as e:
            logger.debug(f"Could not execute code: {e}", exc_info=True)
            return None, False, call_data_folder, input_path, error_file # Ensure the folder is returned even on failure

    