import ast
import os
import pathlib
from typing import Any
import time
import subprocess
import cloudpickle
import hashlib
import psutil
import shutil


# Define the main container path
CONTAINER_MAIN = (pathlib.Path(__file__).parent / "container" / "container_main.py").absolute()


def ensure_dir_exists(path: pathlib.Path) -> pathlib.Path:
    """Ensure the directory exists."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_orphaned_sandbox_processes(logger=None, max_age_seconds=300):
    """Kill orphaned container_main.py processes.

    Args:
        logger: Optional logger for status messages
        max_age_seconds: Only kill processes older than this (default 300s).
                         Set to 0 or None to kill all sandbox processes regardless of age.

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
                    # Check age if max_age_seconds is set
                    if max_age_seconds:
                        uptime = time.time() - proc.info['create_time']
                        if uptime <= max_age_seconds:
                            continue
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


def _kill_process_tree(process):
    """Kill a process and its entire process group.

    Attempts to kill via process group first, then falls back to direct kill.
    """
    if process is None:
        return

    try:
        os.killpg(os.getpgid(process.pid), 9)
    except (ProcessLookupError, PermissionError):
        pass

    try:
        process.kill()
        process.wait(timeout=1)
    except Exception:
        pass


class ExternalProcessSandbox:
    """Sandbox that executes generated code in a separate Python process.

    Uses subprocess isolation with process groups for proper cleanup on timeout.
    Caches compiled base namespace (imports, helpers) and only recompiles
    the priority function for each evaluation.
    """

    # Cache for compiled base namespace (spec without priority), per process
    _cached_namespace = None
    _cached_base_hash = None

    def __init__(
        self,
        base_path: pathlib.Path,
        timeout_secs: int = 30,
        python_path: str = "python",
        local_id=None,
        graph_dir=None,
        memory_limit_gb: float = 1.0,
    ):
        self.local_id = local_id
        self.output_path = ensure_dir_exists(pathlib.Path(base_path) / f"sandbox{self.local_id}")
        self.timeout_secs = timeout_secs
        self.python_path = python_path
        self.graph_dir = graph_dir
        self.memory_limit_gb = memory_limit_gb
        self.input_path = ensure_dir_exists(self.output_path / "inputs")

    @staticmethod
    def compile_code(program: str):
        """Compile program with caching. Only recompiles priority function.

        Separates the program into base (imports, helpers) and priority function.
        The base is cached and reused across evaluations.
        """
        tree = ast.parse(program)

        # Separate priority from rest of program
        priority_node = None
        base_nodes = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'priority':
                priority_node = node
            else:
                base_nodes.append(node)

        # Check if base changed (first call or different spec)
        base_tree = ast.Module(body=base_nodes, type_ignores=[])
        base_hash = hash(ast.dump(base_tree))

        if ExternalProcessSandbox._cached_base_hash != base_hash:
            ExternalProcessSandbox._cached_namespace = {}
            exec(compile(base_tree, '<ast>', 'exec'), ExternalProcessSandbox._cached_namespace)
            ExternalProcessSandbox._cached_base_hash = base_hash

        # Always compile and inject new priority into cached namespace
        if priority_node:
            priority_tree = ast.Module(body=[priority_node], type_ignores=[])
            exec(compile(priority_tree, '<ast>', 'exec'), ExternalProcessSandbox._cached_namespace)

        return ExternalProcessSandbox._cached_namespace

    def _exec(self, call_data_path: pathlib.Path, input_path: pathlib.Path, error_file_path: pathlib.Path) -> bool:
        """Execute the Python container in a subprocess.

        The container (CONTAINER_MAIN) executes the LLM generated method from prog.pickle
        using input.pickle as input, writing the output to output.pickle.

        Process group management ensures that if this process dies, all child processes are killed.
        """
        prog_path = call_data_path / "prog.pickle"
        output_file = call_data_path / "output.pickle"

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
            # Prepare environment variables
            env = os.environ.copy()
            if self.graph_dir:
                env['GRAPH_DIR'] = str(self.graph_dir)
            env['SANDBOX_MEMORY_LIMIT_GB'] = str(self.memory_limit_gb)

            # Use Popen with start_new_session to create a new process group.
            # This allows killing the entire process tree if needed.
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd(),
                env=env,
                start_new_session=True
            )

            # Wait for process with timeout
            stdout, stderr = process.communicate(timeout=self.timeout_secs)

            # Write stderr output to error file (includes debug info like graph file paths)
            if stderr:
                with open(error_file_path, "wb") as ef:
                    ef.write(stderr)

            return process.returncode == 0

        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            return False

        except Exception:
            if process and process.poll() is None:
                _kill_process_tree(process)
            return False

    def _hash_input(self, test_input) -> str:
        """Hash the serialized input using SHA256.

        This is more stable than Python's built in hash().
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
        count: int
    ) -> tuple[Any, bool, float, pathlib.Path, pathlib.Path, pathlib.Path]:
        """Execute the function in a sandboxed environment.

        Returns:
            result: The function result, or None if execution failed.
            success: Boolean indicating whether execution succeeded.
            cpu_time: CPU time measured in the sandbox.
            output_path: Path to the output directory.
            input_file: Path to the input file.
            error_file: Path to the error file.
        """
        call_data_folder = ensure_dir_exists((self.output_path / f"call{count}").absolute())

        # Create an input filename using SHA256 hash
        input_hash = self._hash_input(test_input)
        input_file = (self.input_path / f"{input_hash}.pickle").absolute()
        ensure_dir_exists(self.input_path)

        # Create the input file if it does not exist
        if not input_file.exists():
            with open(input_file, "wb") as f:
                cloudpickle.dump(test_input, f)

        error_file = self.output_path / f"stderr_{count}.log"
        try:
            namespace = ExternalProcessSandbox.compile_code(program)
            prog_file = (call_data_folder / "prog.pickle").absolute()
            with open(prog_file, "wb+") as f:
                cloudpickle.dump(namespace[function_to_run], f)

            retcode = self._exec(call_data_folder, input_file, error_file)
            if not retcode:
                return None, False, 0.0, self.output_path, input_file, error_file

            output_file = call_data_folder / "output.pickle"
            with open(output_file, "rb") as f:
                result_data = cloudpickle.load(f)
                result = result_data.get("result", None)
                cpu_time = result_data.get("cpu_time", 0.0)
                return result, True, cpu_time, self.output_path, input_file, error_file
        except Exception:
            return None, False, 0.0, self.output_path, input_file, error_file

    def cleanup_call_directories(self, count: int):
        """Clean up call directory after evaluation to save disk space.

        Removes the call{count} directory including prog.pickle and output.pickle.
        Keeps stderr logs and input files (inputs are reused).
        """
        try:
            call_data_folder = self.output_path / f"call{count}"
            if call_data_folder.exists():
                shutil.rmtree(call_data_folder, ignore_errors=True)
        except Exception:
            # Do not fail evaluation if cleanup fails
            pass

    def cleanup_all(self):
        """Clean up the entire sandbox directory for this evaluator.

        Call this during evaluator shutdown.
        """
        try:
            if self.output_path.exists():
                shutil.rmtree(self.output_path, ignore_errors=True)
        except Exception:
            # Do not fail if cleanup fails
            pass
