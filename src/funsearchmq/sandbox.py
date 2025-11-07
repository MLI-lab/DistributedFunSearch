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
import hashlib
import psutil  


# Define the main container path
CONTAINER_MAIN = (pathlib.Path(__file__).parent / "container" / "container_main.py").absolute()

def ensure_dir_exists(path: pathlib.Path) -> pathlib.Path:
    """Ensure the directory exists."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

def cleanup_orphaned_sandbox_processes(logger=None):
    """
    Kill any orphaned container_main.py processes that are running without a parent evaluator.
    This is a safety mechanism to prevent zombie processes from consuming resources.

    Returns the number of processes killed.
    """
    killed_count = 0
    try:
        for proc in psutil.process_iter(['pid', 'cmdline', 'cpu_percent', 'create_time']):
            try:
                cmdline = proc.info.get('cmdline')
                if not cmdline:
                    continue

                # Check if this is a container_main.py process
                if 'container_main.py' in ' '.join(cmdline):
                    # Check if it's been running for more than 5 minutes (likely orphaned)
                    uptime = time.time() - proc.info['create_time']
                    if uptime > 300:  # 5 minutes
                        if logger:
                            logger.warning(f"Killing orphaned sandbox process PID {proc.info['pid']} (uptime: {uptime:.0f}s)")
                        proc.kill()
                        killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        if logger:
            logger.error(f"Error during orphaned process cleanup: {e}")

    return killed_count

class DummySandbox:
    """Base class for Sandboxes that execute generated code contained in a string,
    and call a function defined in that code with specific input.
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
    """Sandbox that executes the code in a separate Python process on the same host."""
    def __init__(self, base_path: pathlib.Path, timeout_secs: int = 30, python_path: str = "python", local_id=None):
        super(ExternalProcessSandbox, self).__init__()
        self.local_id = local_id
        self.output_path = ensure_dir_exists(pathlib.Path(base_path) / f"sandbox{self.local_id}")
        self.timeout_secs = timeout_secs
        self.python_path = python_path
        self.input_path = ensure_dir_exists(self.output_path / "inputs")

    def _exec(self, call_data_path: pathlib.Path, input_path: pathlib.Path, error_file_path: pathlib.Path) -> bool:
        """
        Use subprocess.Popen() to execute the Python container with proper process group management.
        The container (CONTAINER_MAIN) will execute the LLM-generated method from prog.pickle using input.pickle as input,
        writing the output to output.pickle.

        Process group management ensures that if this process dies, all child processes are killed.
        """
        prog_path = call_data_path / "prog.pickle"   # Serialized Python function
        output_file = call_data_path / "output.pickle"  # Where output will be written

        # Ensure directories exist
        ensure_dir_exists(call_data_path)
        ensure_dir_exists(input_path.parent)
        ensure_dir_exists(error_file_path.parent)

        # Construct the command
        cmd = [
            self.python_path,
            str(CONTAINER_MAIN),
            str(prog_path),
            str(input_path),
            str(output_file)
        ]

        process = None
        try:
            # Use Popen with start_new_session=True to create a new process group
            # This allows us to kill the entire process tree if needed
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd(),
                start_new_session=True  # Creates new process group
            )

            # Wait for process with timeout
            stdout, stderr = process.communicate(timeout=self.timeout_secs)

            # Always write stderr output to error_file (includes debug info like graph file paths)
            if stderr:
                with open(error_file_path, "wb") as ef:
                    ef.write(stderr)

            return (process.returncode == 0)

        except subprocess.TimeoutExpired:
            # Kill the entire process group on timeout
            if process:
                try:
                    # Kill the process group (negative PID kills the entire group)
                    os.killpg(os.getpgid(process.pid), 9)
                except (ProcessLookupError, PermissionError):
                    pass  # Process already dead
                try:
                    process.kill()  # Fallback
                    process.wait(timeout=1)
                except Exception:
                    pass
            return False

        except Exception as e:
            # Clean up on any error
            if process and process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), 9)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    process.kill()
                    process.wait(timeout=1)
                except Exception:
                    pass
            return False

    def _hash_input(self, test_input) -> str:
        """
        Use SHA-256 to hash the serialized input.
        This is more stable than Python's built-in hash().
        """
        serialized = cloudpickle.dumps(test_input)
        hash_obj = hashlib.sha256(serialized)
        return hash_obj.hexdigest()

    def run(
        self,
        program: str,
        function_to_run: str,
        test_input,
        timeout_seconds: int,
        count: int, 
    ) -> tuple[Any, bool, pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
        """
        Executes the function in a sandboxed environment.
        Returns:
            - The function result,
            - A boolean indicating success,
            - CPU time measured in the sandbox,
            - Path to the output directory,
            - Path to the input file,
            - Path to the error file.
        """
        call_data_folder = ensure_dir_exists((self.output_path / f"call{count}").absolute())

        # Create an input filename using SHA-256 hash
        input_hash = self._hash_input(test_input)
        input_file = (self.input_path / f"{input_hash}.pickle").absolute()
        ensure_dir_exists(self.input_path)

        # Create the input file if it doesn't exist
        if not input_file.exists():
            with open(input_file, "wb") as f:
                cloudpickle.dump(test_input, f)

        error_file = self.output_path / f"stderr_{count}.log"
        try:
            namespace = DummySandbox.compile_code(program)
            prog_file = (call_data_folder / f"prog.pickle").absolute()
            with open(prog_file, "wb+") as f:
                cloudpickle.dump(namespace[function_to_run], f)

            retcode = self._exec(call_data_folder, input_file, error_file)
            if not retcode:
                return None, False, 0.0, self.output_path, input_file, error_file

            output_file = call_data_folder / f"output.pickle"
            with open(output_file, "rb") as f:
                result_data = cloudpickle.load(f)
                result = result_data.get("result", None)
                cpu_time = result_data.get("cpu_time", 0.0)
                return result, True, cpu_time, self.output_path, input_file, error_file
        except Exception as e:
            return None, False, 0.0, self.output_path, input_file, error_file
