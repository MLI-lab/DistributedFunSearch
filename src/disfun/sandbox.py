"""Sandbox for executing LLM generated code via os.fork().

Child inherits parent's memory (copy on write), so cached graphs are available
without reload. Child runs evaluation, writes result to pipe, exits. Parent
waits with timeout.

Faster than subprocess sandboxing because child already has graphs in memory
(no LMDB reload) and no pickle serialization of function or input needed.
"""

import ast
import os
import signal
import resource
import time
import cloudpickle
import threading



class ExternalProcessSandbox:
    """Sandbox using fork() for fast, isolated execution.

    Child process inherits parent's memory via copy on write, so graphs
    cached in the evaluator are available without reloading.
    """

    # Cache for compiled base namespace (spec without priority), per process
    _cached_namespace = None
    _cached_base_hash = None
    _compile_lock = threading.Lock()  # thread synchronization for cache

    def __init__(
        self,
        timeout_secs: int = 30,
        graph_dir: str = None,
        memory_limit_gb: float = 1.0,
    ):
        self.timeout_secs = timeout_secs
        self.graph_dir = graph_dir
        self.memory_limit_gb = memory_limit_gb

    @staticmethod
    def compile_code(program: str):
        """Compile program with caching. Only recompiles priority function.

        Separates the program into base (imports, helpers) and priority function.
        The base is cached and reused across evaluations.
        Thread safe, uses lock to prevent race conditions in cache access.
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

        # Thread safe cache check and update
        with ExternalProcessSandbox._compile_lock:
            if ExternalProcessSandbox._cached_base_hash != base_hash:
                ExternalProcessSandbox._cached_namespace = {}
                exec(compile(base_tree, '<ast>', 'exec'), ExternalProcessSandbox._cached_namespace)
                ExternalProcessSandbox._cached_base_hash = base_hash

                # Preload lazy modules so child inherits them (avoids mmap in sandbox).
                # numpy.random is lazy loaded.
                ns = ExternalProcessSandbox._cached_namespace
                if 'np' in ns or 'numpy' in ns:
                    np_module = ns.get('np') or ns.get('numpy')
                    if np_module is not None:
                        try:
                            _ = np_module.random.rand(1)  # Force numpy.random load
                        except Exception:
                            pass

                # Preload scipy to avoid "failed to map segment" errors in forked children.
                # scipy's .so files need mmap which can fail under memory limits.
                try:
                    import scipy
                    import scipy.stats
                    import scipy.special
                    import scipy.spatial
                    # Touch lazy loaded submodules to force their .so files to load.
                    _ = scipy.stats.norm.pdf(0)
                except Exception:
                    pass

            # Always compile and inject new priority into cached namespace
            if priority_node:
                priority_tree = ast.Module(body=[priority_node], type_ignores=[])
                exec(compile(priority_tree, '<ast>', 'exec'), ExternalProcessSandbox._cached_namespace)

            return ExternalProcessSandbox._cached_namespace.copy()  # Return copy to avoid mutation

    def run(
        self,
        program: str,
        function_to_run: str,
        test_input,
    ) -> tuple:
        """Execute function in forked sandbox.

        Child inherits parent's memory (including cached graphs) via copy on write.
        Sets memory limits in child, runs evaluation, returns result via pipe.

        Returns:
            (result, success, cpu_time)
        """
        try:
            # Compile in parent, child will inherit compiled namespace.
            namespace = ExternalProcessSandbox.compile_code(program)
            func = namespace[function_to_run]
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return f"Compile error: {type(e).__name__}: {e}\n{tb}", False, 0.0

        # Create pipe for result communication
        read_fd, write_fd = os.pipe()

        pid = os.fork()

        if pid == 0:
            # Child process.
            # Detach from parent's signal handling before anything else.
            # After fork, child inherits asyncio's signal wakeup fd (a pipe shared
            # with the parent). Any signal delivered to the child would write to this
            # pipe, tricking the parent's event loop into thinking it received the
            # signal (causing spurious shutdowns). Resetting the wakeup fd and
            # ignoring signals prevents this.
            try:
                signal.set_wakeup_fd(-1)
            except (ValueError, OSError):
                pass
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)

            os.close(read_fd)

            # Suppress stdout/stderr from LLM generated code (e.g. print statements)
            # to prevent unbounded growth of sbatch .out/.err files.
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)  # stdout
            os.dup2(devnull, 2)  # stderr
            os.close(devnull)

            try:
                # Set memory limit (virtual address space)
                # Note: use RLIMIT_AS, not RLIMIT_DATA. The latter breaks mmap
                # for shared libraries like numpy ("failed to map segment").
                mem_bytes = int(self.memory_limit_gb * 1024 * 1024 * 1024)
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                except (ValueError, resource.error):
                    pass

                # Run evaluation
                start_cpu = time.process_time()
                result = func(test_input, self.graph_dir)
                cpu_time = time.process_time() - start_cpu

                # Serialize and write result
                result_bytes = cloudpickle.dumps({
                    "result": result,
                    "cpu_time": cpu_time
                })
                os.write(write_fd, result_bytes)
                os.close(write_fd)
                os._exit(0)

            except Exception as e:
                # Write error + traceback to pipe so parent can capture it
                import traceback
                error_msg = f"Runtime error: {type(e).__name__}: {e}\n{''.join(traceback.format_tb(e.__traceback__))}"
                try:
                    os.write(write_fd, error_msg.encode('utf-8', errors='replace'))
                except Exception:
                    pass
                try:
                    os.close(write_fd)
                except Exception:
                    pass
                os._exit(1)

        else:
            # Parent process.
            os.close(write_fd)

            try:
                # Wait for child with timeout
                deadline = time.time() + self.timeout_secs
                child_done = False

                while time.time() < deadline:
                    pid_result, status = os.waitpid(pid, os.WNOHANG)
                    if pid_result != 0:
                        child_done = True
                        break
                    time.sleep(0.01)

                if not child_done:
                    # Timeout, kill child.
                    try:
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
                    except (ProcessLookupError, ChildProcessError):
                        pass
                    os.close(read_fd)
                    return f"Timeout: exceeded {self.timeout_secs}s", False, 0.0

                # Check exit status
                if not (os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0):
                    # Read error details written by child to pipe
                    error_detail = ""
                    try:
                        chunks = []
                        while True:
                            chunk = os.read(read_fd, 65536)
                            if not chunk:
                                break
                            chunks.append(chunk)
                        if chunks:
                            error_detail = b''.join(chunks).decode('utf-8', errors='replace')
                    except Exception:
                        pass
                    os.close(read_fd)
                    if error_detail:
                        return error_detail, False, 0.0
                    if os.WIFSIGNALED(status):
                        sig = os.WTERMSIG(status)
                        return f"Runtime error: killed by signal {sig} (oom or crash)", False, 0.0
                    exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
                    return f"Runtime error: child exited with code {exit_code}", False, 0.0

                # Read result from pipe
                result_chunks = []
                while True:
                    chunk = os.read(read_fd, 65536)
                    if not chunk:
                        break
                    result_chunks.append(chunk)
                os.close(read_fd)

                if not result_chunks:
                    return None, False, 0.0

                result_bytes = b''.join(result_chunks)
                result_data = cloudpickle.loads(result_bytes)
                return result_data["result"], True, result_data["cpu_time"]

            except Exception as e:
                # Clean up child if still running
                try:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                except (ProcessLookupError, ChildProcessError):
                    pass
                try:
                    os.close(read_fd)
                except OSError:
                    pass
                return f"Parent error: {type(e).__name__}: {e}", False, 0.0

