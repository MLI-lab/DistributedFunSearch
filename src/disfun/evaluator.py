# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License - Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing - software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND - either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Asynchronous RabbitMQ Evaluator.

Differences from the original DeepMind FunSearch version

* Evaluates samples using a sandboxed execution environment that runs each
  generated program on test inputs in isolated subprocesses.
* Uses multiprocessing with CPU parallelism (via `ProcessPoolExecutor`) to evaluate
  multiple inputs in parallel.
* Tracks and publishes per-sample CPU time - along with GPU time and token counts
  received from the sampler.
* Publishes results back to the database queue with functional scores - hashed outputs,
  and a flag indicating whether an optimal solution was found.
* Logs full outputs for prompts with structurally identical few-shot examples,
  allowing downstream deduplication and analysis.
* Parses LLM output by extracting only the first valid function body from generated code.
  If the LLM generates multiple functions or malformed code - only the first parseable
  function body is kept and the rest is silently discarded.

"""


import ast
from typing import Any
import copy
import logging
from disfun import code_manipulation
from disfun import sandbox
import json
import aio_pika
import sys
import asyncio
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.multiprocessing import Manager # starts its own process on a cpu core
import gc
import psutil
import time




logger = logging.getLogger('main_logger')


def extract_evaluation_result(test_output, problem_instance):
    """
    Extract score and hash from evaluation output for a single problem instance.

    Customize this function to change what gets extracted from each evaluation.

    Args:
        test_output: The output from evaluate(problem_instance)
        problem_instance: The tuple defining the problem instance (e.g., (n, s, q))

    Returns:
        tuple: (score_key, score_value, hash_value) where:
            - score_key: The key for storing this score (defaults to full problem instance tuple)
            - score_value: Numeric score (test_output[0])
            - hash_value: Hash for deduplication (test_output[1]) or None
    """
    score_key = problem_instance  # Use full tuple as key for consistency
    score_value = test_output[0]
    hash_value = test_output[1] if test_output[1] is not None else None
    return score_key, score_value, hash_value


class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the start and end line numbers of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_start_line: int | None = None
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:
    """Collects the start and end line numbers of the target function."""
    if node.name == self._target_function_name:
      # node.lineno is the line with 'def' - node.end_lineno is the last line
      self._function_start_line = node.lineno
      self._function_end_line = node.end_lineno
    # calling it will continue normal traversal of the AST
    self.generic_visit(node)

  @property
  def function_start_line(self) -> int:
    """Line number of the 'def' line of function `target_function_name`."""
    assert self._function_start_line is not None  # Check internal correctness.
    return self._function_start_line

  # Allows to access the functions end line number after the AST has been visited
  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


def _extract_three_tier(generated_output: str) -> tuple[str, str | None, str | None]:
  """Extract thinking - thought - and code from model output using XML tags.

  Supports formats:
  1. Extended EoH: <thinking>...</thinking><thought>...</thought><code>...</code>
  2. EoH: <thought>...</thought><code>...</code>
  3. FunSearch: raw code (fallback to AST parsing)

  Args:
      generated_output: Raw LLM output

  Returns:
      tuple: (code_body, thought, thinking)
          - code_body: Cleaned function body (ready to insert into template)
          - thought: One-sentence algorithm description (EoH) or None
          - thinking: Full chain-of-thought reasoning (extended_eoh) or None
  """
  if not generated_output:
    return '', None, None

  # Try XML tag extraction
  import re
  thinking_match = re.search(r'<thinking>(.*?)</thinking>', generated_output, re.DOTALL)
  thought_match = re.search(r'<thought>(.*?)</thought>', generated_output, re.DOTALL)
  code_match = re.search(r'<code>(.*?)</code>', generated_output, re.DOTALL)

  # Extract components
  thinking = thinking_match.group(1).strip() if thinking_match else None
  thought = thought_match.group(1).strip() if thought_match else None

  if code_match:
    # <code> tags found - extract and parse code with AST
    raw_code = code_match.group(1).strip()
    logger.debug(f"Extracted code from <code> tags ({len(raw_code)} chars)")
    cleaned_code = _trim_function_body_ast(raw_code)
    logger.debug(f"Cleaned code after AST parsing ({len(cleaned_code) if cleaned_code else 0} chars):\n{cleaned_code if cleaned_code else '(empty)'}")
    return cleaned_code, thought, thinking

  # Fallback: No <code> tags - but may have <thought>/<thinking> tags
  # Remove thought/thinking tags from output and treat the rest as code
  code_to_parse = generated_output
  if thought_match or thinking_match:
    logger.debug("Found <thought> or <thinking> tags without <code> tags. Extracting thought/thinking and parsing remainder as code.")
    # Remove thought and thinking tags from the generated output
    code_to_parse = re.sub(r'<thinking>.*?</thinking>\s*', '', code_to_parse, flags=re.DOTALL)
    code_to_parse = re.sub(r'<thought>.*?</thought>\s*', '', code_to_parse, flags=re.DOTALL)
  else:
    logger.debug("No XML tags found, treating entire output as code")

  cleaned_code = _trim_function_body_ast(code_to_parse)
  logger.debug(f"Cleaned code after AST parsing ({len(cleaned_code) if cleaned_code else 0} chars):\n{cleaned_code if cleaned_code else '(empty)'}")
  return cleaned_code, thought, thinking


def _trim_function_body_ast(generated_code: str) -> str:
  """Extracts the body of the generated function using AST parsing.

  Handles both:
  1. Full function with header: `def priority_v1(...):\n    return 0.0`
  2. Function body only (fallback): `return 0.0`

  Parses the code as-is - extracts body - preserves exact indentation from LLM.
  """
  if not generated_code:
    return ''

  # Strip markdown code fences first (LLMs often wrap in ```python ... ```)
  generated_code = generated_code.strip()
  if generated_code.startswith('```'):
    lines = generated_code.split('\n')
    # Remove opening fence (```python or just ```)
    if lines[0].startswith('```'):
      lines = lines[1:]
    # Remove closing fence
    if lines and lines[-1].strip() == '```':
      lines = lines[:-1]
    generated_code = '\n'.join(lines).strip()
    logger.debug(f"Stripped markdown code fences, remaining code ({len(generated_code)} chars)")

  # Check if code contains a priority function definition
  import re
  priority_match = re.search(r'^\s*def\s+(priority(?:_v\d+)?)\s*\(', generated_code, re.MULTILINE)

  if priority_match:
    # Has priority function - extract its body only
    logger.debug("Found priority function definition, extracting its body")
    code = generated_code
    function_name = priority_match.group(1)

    tree = None
    deletion_count = 0
    # We keep trying and deleting code from the end until the parser succeeds.
    while tree is None:
      try:
        tree = ast.parse(code)
      except SyntaxError as e:
        deletion_count += 1
        deleted_line = code.splitlines()[e.lineno - 1] if e.lineno <= len(code.splitlines()) else "(unknown)"
        logger.warning(f"AST SyntaxError at line {e.lineno}: {e.msg}. Deleting line: {deleted_line[:100]}")
        code = '\n'.join(code.splitlines()[:e.lineno - 1])
        if deletion_count > 20:
          logger.error("Too many AST deletions (>20), stopping. Code likely invalid.")
          return ''
    if not code:
      logger.warning("AST parsing resulted in empty code after deletions")
      return ''
    if deletion_count > 0:
      logger.info(f"AST parsing required {deletion_count} line deletions to succeed")

    # Extract module-level imports (to move inside function body if needed)
    import ast as ast_module
    imports = []
    for node in tree.body:
      if isinstance(node, (ast_module.Import, ast_module.ImportFrom)):
        # Get the import statement as string
        import_line = code.splitlines()[node.lineno - 1]
        imports.append(import_line)
        logger.debug(f"Detected module-level import: {import_line}")
      elif isinstance(node, ast_module.FunctionDef):
        # Stop at function definition
        break

    # Extract body using visitor
    visitor = _FunctionLineVisitor(function_name)
    visitor.visit(tree)
    total_lines = len(code.splitlines())
    logger.debug(f"AST: function '{function_name}' spans lines {visitor.function_start_line}-{visitor.function_end_line} (total {total_lines} lines)")

    # Extract function body (skip the def line)
    body_lines = code.splitlines()[visitor.function_start_line:visitor.function_end_line]

    # Prepend imports (indented) to function body if they were outside the function
    if imports:
      logger.debug(f"Moving {len(imports)} module-level import(s) inside function body")
      indented_imports = ['    ' + imp for imp in imports]
      body_lines = indented_imports + body_lines

    result = '\n'.join(body_lines) + '\n\n'

    if imports:
      logger.debug(f"Rewritten function body with imports moved inside ({len(body_lines)} total lines):\n{result}")
    else:
      logger.debug(f"AST: extracted {len(body_lines)} body lines")

    return result

  else:
    # No priority function - wrap code in fake function
    # Check for unindented def at column 0 (standalone function to exclude)
    standalone_func_match = re.search(r'^def\s+\w+\s*\(', generated_code, re.MULTILINE)

    # Determine what code to wrap
    if standalone_func_match and generated_code[:standalone_func_match.start()].strip():
      # Found standalone function with code before it - use only code before
      code_to_wrap = generated_code[:standalone_func_match.start()].rstrip()
      logger.debug(f"Found standalone function, wrapping only code before it ({len(code_to_wrap)} chars)")
    else:
      # No standalone function, or no code before it - wrap everything
      code_to_wrap = generated_code
      logger.debug("Wrapping all code in fake function")

    # Wrap in fake function as-is (don't modify indentation - take LLM output as-is)
    code = 'def fake_function_header():\n' + code_to_wrap
    tree = None
    deletion_count = 0
    while tree is None:
      try:
        tree = ast.parse(code)
      except SyntaxError as e:
        deletion_count += 1
        deleted_line = code.splitlines()[e.lineno - 1] if e.lineno <= len(code.splitlines()) else "(unknown)"
        logger.warning(f"AST SyntaxError at line {e.lineno}: {e.msg}. Deleting line: {deleted_line[:100]}")
        code = '\n'.join(code.splitlines()[:e.lineno - 1])
        if deletion_count > 20:
          logger.error("Too many AST deletions (>20), stopping. Code likely invalid.")
          return ''
    if not code:
      logger.warning("AST parsing resulted in empty code after deletions")
      return ''
    if deletion_count > 0:
      logger.info(f"AST parsing required {deletion_count} line deletions to succeed")

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'

def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    # function_to_evolve is set to priority
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str, str | None, str | None]:
  """Returns the compiled generated function - full runnable program - thought - and thinking.

  Purpose: integrate a generated code as string into a larger program template.

  Returns:
      tuple: (evolved_function, program_str, thought, thinking)
          - evolved_function: Function object with updated body, thought, and thinking
          - program_str: Complete runnable program as string
          - thought: One-sentence algorithm description or None
          - thinking: Full chain-of-thought reasoning or None
  """
  # Extract code body and thought/thinking (markdown fences stripped inside _trim_function_body_ast)
  body, thought, thinking = _extract_three_tier(generated_code)
  if version_generated is not None:

    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)
  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)

  evolved_function.body = body
  evolved_function.thinking = thinking
  evolved_function.thought = thought
  return evolved_function, str(program), thought, thinking



def run_evaluation(sandbox, program, function_to_run, input, timeout_seconds, call_count, call_count_lock):
    with call_count_lock: # the with statement ensures the lock is released once the block is exited regardless of whether an exception is raised
        count = call_count.value
        call_count.value += 1

    result, runs_ok, cpu_time, call_data_folder, input_path, error_file = sandbox.run(program, function_to_run, input, timeout_seconds, count)

    # Clean up sandbox call directory to save disk space
    sandbox.cleanup_call_directories(count)

    return result, runs_ok, cpu_time, call_data_folder, input_path, error_file



class Evaluator:
    """Evaluates generated functions in sandboxed subprocesses.

    Note: Debug information about which graph files are being loaded is written to stderr
    in the sandbox subprocesses. These logs can be found in sandbox stderr_*.log files
    (e.g. - sandbox/sandbox<PID>/stderr_N.log). The stderr files are automatically cleaned up
    after evaluation completes.
    """
    def __init__(self, connection, channel, evaluator_queue, database_queue, template, function_to_evolve, function_to_run, inputs, sandbox_base_path, timeout_seconds, local_id, target_signatures, max_workers=2, graph_dir=None, cache_graphs=False, cache_size_limit_gb=2.0, rabbitmq_config=None):
        self.template = template
        self.function_to_evolve = function_to_evolve
        self.function_to_run = function_to_run
        self.inputs = inputs
        self.timeout_seconds = timeout_seconds
        self.local_id = local_id
        self.graph_dir = graph_dir  # Store graph_dir for passing to sandbox
        self.cache_graphs = cache_graphs  # Enable graph caching in specifications
        self.cache_size_limit_gb = cache_size_limit_gb  # Size limit for cached graphs
        self._shutdown_requested = False  # Flag to stop reconnection on shutdown

        # Use shared connection manager for RabbitMQ
        from disfun import process_utils
        self._conn_manager = process_utils.RabbitMQConnectionManager(
            config=rabbitmq_config,
            component_name=f"Evaluator {local_id}",
            queue_names=["evaluator_queue", "database_queue"],
            logger=logger
        )
        # Initialize with passed-in connection (may be None for deferred connection)
        self._conn_manager.connection = connection
        self._conn_manager.channel = channel
        if evaluator_queue:
            self._conn_manager.queues["evaluator_queue"] = evaluator_queue
        if database_queue:
            self._conn_manager.queues["database_queue"] = database_queue

        self.manager = Manager()
        self.call_count = self.manager.Value('i', 0)
        self.call_count_lock = self.manager.Lock()
        self.sandbox = sandbox.ExternalProcessSandbox(
            base_path=sandbox_base_path, timeout_secs=timeout_seconds, python_path=sys.executable, local_id=self.local_id, graph_dir=graph_dir)
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.cumulative_cpu_time = 0.0  # Track total CPU time
        self.cpu_time_lock = self.manager.Lock()  # Lock to protect updates to cumulative CPU time
        self.target_signatures = target_signatures # Example {(6,1): 10, (7,1): 16, (8,1): 30, (9,1): 52, (10,1): 94, (11,1): 172}

    def request_shutdown(self):
        """Signal that shutdown is requested - stops reconnection attempts."""
        self._shutdown_requested = True

    # Properties for backward compatibility - delegate to connection manager
    @property
    def connection(self):
        return self._conn_manager.connection

    @connection.setter
    def connection(self, value):
        self._conn_manager.connection = value

    @property
    def channel(self):
        return self._conn_manager.channel

    @channel.setter
    def channel(self, value):
        self._conn_manager.channel = value

    @property
    def evaluator_queue(self):
        return self._conn_manager.get_queue("evaluator_queue")

    @evaluator_queue.setter
    def evaluator_queue(self, value):
        self._conn_manager.queues["evaluator_queue"] = value

    @property
    def database_queue(self):
        return self._conn_manager.get_queue("database_queue")

    @database_queue.setter
    def database_queue(self, value):
        self._conn_manager.queues["database_queue"] = value

    async def _ensure_connection(self):
        """Delegate to shared connection manager."""
        return await self._conn_manager.ensure_connection()

    async def _close_connection(self):
        """Delegate to shared connection manager."""
        await self._conn_manager.close()

    async def async_cleanup(self):
        """Async cleanup - close RabbitMQ connections (matches Sampler interface)."""
        await self._conn_manager.close()

    async def shutdown(self):
        logger.info(f"Evaluator {self.local_id}: Initiating shutdown process.")
        try:
            if self.executor:
                logger.info(f"Evaluator {self.local_id}: Shutting down executor.")
                try:
                    # Shutdown executor with timeout to avoid hanging
                    await asyncio.wait_for(
                        asyncio.to_thread(self.executor.shutdown, wait=True),
                        timeout=5
                    )
                except TimeoutError:
                    logger.warning(f"Evaluator {self.local_id}: Executor shutdown timed out after 5s, forcing termination...")
                    # Force shutdown if timeout
                    self.executor.shutdown(wait=False)
                self.executor = None  # Set to None to avoid future attempts to use it
            else:
                logger.info(f"Evaluator {self.local_id}: Executor already shut down or not initialized.")

            # Ensure all child processes are terminated
            parent = psutil.Process()
            children = parent.children(recursive=True)

            if children:
                for child in children:
                    logger.info(f"Evaluator {self.local_id}: Terminating child process PID {child.pid}")
                    child.terminate()

                # Wait for processes to terminate with a timeout
                gone, still_alive = psutil.wait_procs(children, timeout=5)

                if still_alive:
                    for p in still_alive:
                        logger.warning(f"Evaluator {self.local_id}: Child process PID {p.pid} did not terminate. Forcing kill.")
                        p.kill()  # Forcefully kill any process that did not terminate
                else:
                    logger.info(f"Evaluator {self.local_id}: All child processes terminated successfully.")
            else:
                logger.info(f"Evaluator {self.local_id}: No running child processes to terminate.")

            # Final cleanup of any orphaned sandbox processes
            killed = sandbox.cleanup_orphaned_sandbox_processes(logger)
            if killed > 0:
                logger.info(f"Evaluator {self.local_id}: Cleaned up {killed} orphaned sandbox processes during shutdown")

            # Clean up sandbox directories
            if hasattr(self, '_sandbox') and self._sandbox:
                self._sandbox.cleanup_all()
                logger.info(f"Evaluator {self.local_id}: Cleaned up sandbox directories")

            # Run garbage collection to clean up resources
            gc.collect()

            logger.info(f"Evaluator {self.local_id}: Shutdown process complete.")
        except TimeoutError:
            logger.warning(f"Evaluator {self.local_id}: Timeout occurred during shutdown.")
        except Exception as e:
            logger.error(f"Evaluator {self.local_id}: Error during shutdown: {e}")


    async def consume_and_process(self):
        """Main consume loop with automatic connection recovery.

        Uses the same reconnection pattern as Sampler for consistency.
        Checks shutdown flag before any reconnection attempt.
        """
        reconnect_delay = 5.0
        max_reconnect_delay = 60.0

        while True:
            # Check shutdown flag at top of loop
            if self._shutdown_requested:
                logger.info(f"Evaluator {self.local_id}: Shutdown requested, exiting consume loop")
                break

            try:
                # Ensure connection is alive (reconnect if needed)
                if self._conn_manager.config is not None:
                    connected = await self._ensure_connection()
                    if not connected:
                        if self._shutdown_requested:
                            break
                        logger.error(f"Evaluator {self.local_id}: Connection failed, retrying in {reconnect_delay:.1f}s")
                        await asyncio.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                        continue

                # Reset delay on successful connection
                reconnect_delay = 5.0

                # Run consume loop
                await self._consume_loop()
                break

            except asyncio.CancelledError:
                logger.info(f"Evaluator {self.local_id}: Cancelled, exiting...")
                break

            except (aio_pika.exceptions.AMQPConnectionError,
                    aio_pika.exceptions.ChannelClosed,
                    aio_pika.exceptions.ChannelInvalidStateError,
                    ConnectionError, OSError) as e:
                if self._shutdown_requested:
                    logger.info(f"Evaluator {self.local_id}: Connection error during shutdown, exiting")
                    break

                logger.warning(f"Evaluator {self.local_id}: Connection error: {e}. Reconnecting in {reconnect_delay:.1f}s...")
                await self._close_connection()
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                continue

            except Exception as e:
                if self._shutdown_requested:
                    break

                logger.error(f"Evaluator {self.local_id}: Unexpected error: {e}. Reconnecting in {reconnect_delay:.1f}s...", exc_info=True)
                await self._close_connection()
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                continue

        # Cleanup after loop exits
        try:
            await asyncio.wait_for(self.shutdown(), timeout=30)
        except TimeoutError:
            logger.warning(f"Evaluator {self.local_id}: Shutdown timed out")
        except Exception as e:
            logger.error(f"Evaluator {self.local_id}: Error during shutdown: {e}")

    async def _consume_loop(self):
        """Inner consume loop - processes messages from queue."""
        async with self.channel:
            await self.channel.set_qos(prefetch_count=1)

            async with self.evaluator_queue.iterator() as stream:
                message_count = 0
                async for message in stream:
                    fetch_start_time = time.perf_counter()

                    async with message.process():
                        fetch_end_time = time.perf_counter()
                        fetch_duration = fetch_end_time - fetch_start_time
                        logger.debug(f"Time to fetch message from queue: {fetch_duration:.6f} seconds")

                        try:
                            await asyncio.wait_for(self.process_message(message), timeout=300)
                        except TimeoutError:
                            logger.warning("Processing message timed out.")
                        except Exception as e:
                            logger.error(f"Evaluator: Error while processing message: {e}")

                    # Periodically clean up orphaned sandbox processes
                    message_count += 1
                    if message_count % 10 == 0:
                        killed = sandbox.cleanup_orphaned_sandbox_processes(logger)
                        if killed > 0:
                            logger.info(f"Cleaned up {killed} orphaned sandbox processes")


    #async_time_execution
    #@async_track_memory
    async def process_message(self, message: aio_pika.IncomingMessage):
        call_folders_to_cleanup = []  # List to track created folders
        call_files_to_cleanup = []  # List to track created folders
        hash_value=None
        call_data_folder=None
        try:
            raw_data = message.body.decode()
            data = json.loads(raw_data)
            logger.debug(f"Data is {data}")
            logger.debug(f"Evaluator: Starts to analyze generated continuation of def priority: {data['sample']}")

            # Deserialize GPU time and lineage tracking
            gpu_time      = data.get("gpu_time", 0.0)
            input_tokens  = data.get("input_tokens", 0)
            output_tokens = data.get("output_tokens", 0)
            parent_ids    = data.get("parent_ids", [])  # Extract parent IDs for lineage tracking
            logger.debug(f"Received GPU time from Sampler: {gpu_time} seconds")
            logger.debug(f"Received input_tokens from Sampler: {input_tokens}")
            logger.debug(f"Received output_tokens from Sampler: {output_tokens}")
            logger.debug(f"Received parent_ids from Sampler: {parent_ids}")

            # Process the new function from the generated code
            new_function, program, thought, thinking = _sample_to_program(data["sample"], data.get("version_generated"), self.template, self.function_to_evolve)

            # Inject cache configuration into the program as global variables
            # These will be available to the specification code (load_graph - solve - etc.)
            cache_config_injection = f"""
# Graph caching configuration (injected by evaluator)
CACHE_GRAPHS = {self.cache_graphs}
CACHE_SIZE_LIMIT_GB = {self.cache_size_limit_gb}

"""
            program = cache_config_injection + program

            tasks = {}

            if new_function.body not in [None, '']:
                # Submit each test input as a task for multiprocessing
                logger.debug(f"Evaluator: Submitting {len(self.inputs)} evaluation tasks with inputs: {self.inputs}")
                logger.debug(f"Evaluator: Executor status: {self.executor}, _shutdown={getattr(self.executor, '_shutdown', 'N/A')}")
                tasks = {self.executor.submit(run_evaluation, self.sandbox, program, self.function_to_run, input, self.timeout_seconds, self.call_count, self.call_count_lock): input for input in self.inputs}
                logger.debug(f"Evaluator: Created {len(tasks)} tasks for execution")
            else:
                logger.info("New function body is None or empty. Skipping execution but publishing 'return'.")
                result = ("return", data['island_id'], {}, data['expected_version'], self.cumulative_cpu_time, gpu_time, input_tokens, output_tokens, False, parent_ids, thought, thinking)
                await self.publish_to_database(result, hash_value)  # Publish "return" result
                return  # Early return after publishing

            scores_per_test = {}
            # Waiting for results from all test inputs
            logger.debug(f"Evaluator: Starting to process {len(tasks)} futures")
            iteration_count = 0
            for future in as_completed(tasks):
                iteration_count += 1
                input = tasks[future]
                logger.debug(f"Evaluator: Processing future {iteration_count}/{len(tasks)} for input {input}")
                try:
                    test_output, runs_ok, cpu_time, call_data_folder, input_path, error_file= future.result(timeout=self.timeout_seconds)
                    logger.debug(f"Evaluator: Future result, runs_ok={runs_ok}, cpu_time={cpu_time}, test_output={test_output}")
                    call_folders_to_cleanup.append(call_data_folder)
                    call_files_to_cleanup.append(input_path)
                    call_files_to_cleanup.append(error_file)

                    # Accumulate CPU time
                    with self.cpu_time_lock:
                        self.cumulative_cpu_time += cpu_time

                    if runs_ok and test_output[0] is not None:
                        # Extract score - key - and hash using the extraction function
                        score_key, score_value, extracted_hash = extract_evaluation_result(test_output, input)
                        scores_per_test[score_key] = score_value
                        if extracted_hash is not None:
                            hash_value = extracted_hash
                        logger.info(f"Evaluator: Test passed for input {input}, score_key={score_key}, score={score_value}")
                    else:
                        # Read error details from error file if it exists
                        error_details = ""
                        if error_file and error_file.exists():
                            try:
                                with open(error_file) as f:
                                    error_content = f.read().strip()
                                    if error_content:
                                        # Only show first 500 chars of error
                                        error_details = f" - Error: {error_content[:500]}"
                            except Exception:
                                pass
                        logger.warning(f"Evaluator: Test failed for input {input}, runs_ok={runs_ok}, test_output={test_output}{error_details}")
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Task for input {input} timed out.")
                except concurrent.futures.CancelledError:
                    logger.warning(f"Task for input {input} was cancelled.")
                except Exception as e:
                    # Catch any other exceptions
                    logger.error(f"Error during task execution for input {input}: {e}")

            logger.debug(f"Evaluator: Completed processing all {iteration_count} futures, got {len(scores_per_test)} scores")



            if self.target_signatures:
                found_optimal_solution = all(
                    scores_per_test.get(dim, 0) >= self.target_signatures.get(dim, float("inf"))
                    for dim in self.target_signatures)
            else:
                found_optimal_solution = False

            # Prepare the result for publishing
            if len(scores_per_test) == len(self.inputs) and any(score != 0 for score in scores_per_test.values()):
                result = (new_function, data['island_id'], scores_per_test, data['expected_version'], self.cumulative_cpu_time, gpu_time, input_tokens, output_tokens, found_optimal_solution, parent_ids, thought, thinking)
                logger.debug(f"Scores are {scores_per_test}")
            else:
                result = ("return", data['island_id'], {}, data['expected_version'], self.cumulative_cpu_time, gpu_time, input_tokens, output_tokens, False, parent_ids, thought, thinking)

            # Publish the result
            await self.publish_to_database(result, hash_value)

            # Reset cumulative CPU time after publishing
            with self.cpu_time_lock:
                self.cumulative_cpu_time = 0.0

        except Exception as e:
            logger.error(f"Error in process_message: {e}")

        finally:
            # Cleanup sandbox folder after a delay
            # DISABLED - keeping sandbox for error analysis (99.99% failure rate investigation)
            # await asyncio.sleep(1)
            # if call_data_folder and call_data_folder.exists():
            #     shutil.rmtree(call_data_folder)
            pass


    async def publish_to_database(self, result, hash_value):
        try:

            function, island_id, scores_per_test, expected_version, cpu_time, gpu_time, input_tokens, output_tokens, found_optimal_solution, parent_ids, thought, thinking = result

            serialized_result = {
                "new_function": function.serialize() if hasattr(function, 'serialize') else str(function),
                "island_id": island_id,
                "scores_per_test": {str(key): value for key, value in scores_per_test.items()},
                "expected_version": expected_version,
                "hash_value": hash_value,
                "cpu_time": cpu_time,  # Include CPU time
                "gpu_time": gpu_time,   # Include GPU time
                "input_tokens":      input_tokens,
                "output_tokens":     output_tokens,
                "found_optimal_solution": found_optimal_solution,
                "parent_ids": parent_ids,  # Include parent IDs for lineage tracking
                "thought": thought,  # One-sentence algorithm description
                "thinking": thinking  # Full chain-of-thought reasoning
            }

            message_body = json.dumps(serialized_result)

            # Start timing before publishing
            publish_start_time = time.perf_counter()

            # Publishing the serialized result to the database queue
            await self.channel.default_exchange.publish(
                aio_pika.Message(body=message_body.encode()),
                routing_key='database_queue'
            )

            # End timing after publishing
            publish_end_time = time.perf_counter()
            publish_duration = publish_end_time - publish_start_time
            logger.debug(f"Time to publish message to queue: {publish_duration:.6f} seconds")

            logger.debug(f"Evaluator: Successfully published to database for island_id {island_id}.")

        except Exception as e:
            logger.error(f"Evaluator: Problem in publishing to database for island_id {island_id}: {e}")
            # Optionally re-raise the exception if the caller needs to handle it.
            raise

