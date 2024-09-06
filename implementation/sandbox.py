import logging
import ast
import os
import pathlib
import sys
from typing import Any
import time
import subprocess
import cloudpickle


log_file_path = pathlib.Path(__file__).parent / "sandbox.log"
logging.basicConfig(
    filename=log_file_path, 
    filemode='w',  # Overwrite the file each time the script runs
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s] %(message)s"  
)

logger = logging.getLogger('logger')


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
        count: int
    ) -> tuple[Any, bool]:
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
                return None, False

            output_file = call_data_folder / f"output.pickle"
            with open(output_file, "rb") as f:
                out = cloudpickle.load(f)
                return out, True
        except Exception as e:
            logger.debug(f"Could not execute code: {e}", exc_info=True)
            return None, False
        finally:
            # Perform cleanup regardless of success or failure
            self.cleanup(call_data_folder, input_path, error_file)

    def cleanup(self, call_data_folder: pathlib.Path, input_path: pathlib.Path, error_file: pathlib.Path):
        try:
            # Files to be cleaned
            output_file = call_data_folder / "output.pickle"
            prog_file = call_data_folder / "prog.pickle"

            # Deleting program and output files
            for file_path in [prog_file, output_file]:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Deleted {file_path}")

            # Deleting error log file
            if error_file and error_file.exists():
                error_file.unlink()
                logger.debug(f"Deleted error log file {error_file}")

            # Deleting input file
            if input_path.exists():
                input_path.unlink()
                logger.debug(f"Deleted input file {input_path}")

            # Remove the directory for call data if it's empty
            if call_data_folder.exists() and not any(call_data_folder.iterdir()):
                call_data_folder.rmdir()
                logger.debug(f"Deleted call data directory {call_data_folder}")

            # Optionally, clean up parent directories if they are empty
            for dir_path in [self.input_path, self.output_path]:
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    logger.debug(f"Deleted directory {dir_path}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
