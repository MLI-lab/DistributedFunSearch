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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Evaluator: executes LLM-generated functions in fork-based sandboxes.

Flow:
1. Parse LLM output from XML tags, markdown fences, or raw code
2. Integrate into evaluation template by replacing priority function body
3. Compile in worker thread with caching: base is cached, only priority recompiled
4. Fork child process (inherits parent memory via copy-on-write)
5. Child sets memory limits, runs evaluation, returns result via pipe
6. Publish scores to database queue

Graphs are loaded lazily from LMDB and cached. Forked children inherit the cache
via copy-on-write, so graphs are never reloaded. See docs/EVALUATOR.md for details.
"""


import ast
import copy
import logging
import os
from disfun.utils import code_manipulation
from disfun.utils.profiling import async_time_execution
from disfun import sandbox
import json
import aio_pika
import sys
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import gc
import psutil
import re

from disfun.utils.fast_graph import load_graph_from_lmdb


logger = logging.getLogger('main_logger')


# Graph cache shared across evaluator threads.
# Forked children inherit this cache via copy-on-write.
_graph_cache = {}
_graph_cache_lock = Lock()


def _build_graph_path(graph_dir, graph_type, s, n, q):
    """Build graph path: graph_dir/ids/quaternary/s1/graph_ids_s1_n6_q4.lmdb"""
    subdir = "deletion" if graph_type in ("deletion", "deletions", "d", "graph_d") else "ids"
    alphabet = "binary" if q == 2 else "quaternary" if q == 4 else f"q{q}"
    prefix = "graph_d" if subdir == "deletion" else "graph_ids"
    return os.path.join(graph_dir, subdir, alphabet, f"s{s}", f"{prefix}_s{s}_n{n}_q{q}.lmdb")


def get_cached_graph(n, s, q, graph_dir, graph_type="deletion"):
    """Get graph from cache or load and cache it.

    Thread-safe. Forked children inherit the cached graphs.
    Path structure: graph_dir/ids/quaternary/s1/graph_ids_s1_n6_q4.lmdb
    """
    key = (n, s, q)

    with _graph_cache_lock:
        if key in _graph_cache:
            return _graph_cache[key]

    graph_path = _build_graph_path(graph_dir, graph_type, s, n, q)
    if not os.path.exists(graph_path):
        logger.warning(f"Graph not found: {graph_path}")
        return None

    logger.info(f"Loading graph n={n}, s={s}, q={q} from {graph_path}")
    G = load_graph_from_lmdb(graph_path)

    with _graph_cache_lock:
        _graph_cache[key] = G
        logger.info(f"Cached graph n={n}, s={s}, q={q}, nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    return G


def extract_evaluation_result(test_output, problem_instance):
    """
    Extract score and hash from evaluation output for a single problem instance.
    Customize this function to change what gets extracted from each evaluation.

    Args:
        test_output: The output from evaluate(problem_instance)
        problem_instance: The tuple defining the problem instance (e.g., (n, s, q))

    Returns:
        tuple: (score_key, score_value, hash_value)
    """
    # Use (n, s) as score key, stripping q since it's constant per run
    score_key = (problem_instance[0], problem_instance[1])
    score_value = test_output[0]
    hash_value = test_output[1] if test_output[1] is not None else None
    return score_key, score_value, hash_value


def _find_function_lines(tree: ast.Module, function_name: str) -> tuple[int, int]:
  """Find start and end line numbers of a function in the AST."""
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == function_name:
      return node.lineno, node.end_lineno
  raise ValueError(f"Function '{function_name}' not found in AST")


def _parse_code(code: str, max_deletions: int = 20) -> tuple[ast.Module | None, str]:
  """Parse Python code, deleting invalid lines from the end until it compiles.

  Returns:
      (parsed_tree, remaining_code) where parsed_tree is None if parsing failed.
  """
  tree = None
  deletion_count = 0

  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      deletion_count += 1
      lines = code.splitlines()
      deleted_line = lines[e.lineno - 1] if e.lineno <= len(lines) else "(unknown)"
      logger.warning(f"AST SyntaxError at line {e.lineno}: {e.msg}. Deleting: {deleted_line[:100]}")
      code = '\n'.join(lines[:e.lineno - 1])
      if deletion_count > max_deletions:
        logger.error(f"Too many AST deletions (>{max_deletions}), code likely invalid.")
        return None, ''

  if not code:
    logger.warning("AST parsing resulted in empty code after deletions")
    return None, ''

  if deletion_count > 0:
    logger.info(f"AST parsing required {deletion_count} line deletions")

  return tree, code


def parse_llm_output(raw_output: str) -> tuple[str, str | None]:
  """
  Parse raw LLM output and extract a valid Python function body.

  Pipeline:
  1. Extract and strip <description> tags
  2. Extract code: <code> tags > last markdown fence > raw text
  3. Strip any nested markdown fences
  4. Detect structure: has def priority() -> extract body, else treat all as body
  5. Parse with AST (delete invalid lines until it compiles)

  Returns:
      (function_body, description)
  """
  if not raw_output:
    return '', None

  # ============================================================
  # STEP 1: Extract description and clean input
  # ============================================================
  description_match = re.search(r'<description>(.*?)</description>', raw_output, re.DOTALL)
  description = description_match.group(1).strip() if description_match else None

  # Remove description tags from text we'll search
  text = re.sub(r'<description>.*?</description>\s*', '', raw_output, flags=re.DOTALL)

  # ============================================================
  # STEP 2: Extract code (3-tier priority)
  # ============================================================

  # Tier 1: <code> tags (highest priority)
  # Try closed tags first, then unclosed (truncated output)
  code_match = re.search(r'<code>(.*?)</code>', text, re.DOTALL)
  if not code_match:
    # Fallback: unclosed <code> tag (output truncated before </code>)
    code_match = re.search(r'<code>(.*)', text, re.DOTALL)
    if code_match:
      logger.debug(f"Extracted code from unclosed <code> tag (truncated output)")
  if code_match:
    code = code_match.group(1)
    logger.debug(f"Extracted code from <code> tags ({len(code)} chars)")
  else:
    # Tier 2: Last markdown fence (most likely final answer)
    # Handles: ```python, ```py, ```Python, ```python3, or plain ```
    fence_matches = list(re.finditer(r'```(?:python|py|python3)?\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE))
    if fence_matches:
      code = fence_matches[-1].group(1)  # Take LAST match
      logger.debug(f"Extracted code from markdown fence {len(fence_matches)} of {len(fence_matches)} ({len(code)} chars)")
    else:
      # Tier 3: Use cleaned text directly
      code = text
      logger.debug(f"Using raw text as code ({len(code)} chars)")

  # Cleanup: strip any fences that might be nested inside (e.g., inside <code> tags)
  fence_in_code = re.search(r'```(?:python|py|python3)?\s*\n(.*?)```', code, re.DOTALL | re.IGNORECASE)
  if fence_in_code:
    code = fence_in_code.group(1)

  # Strip trailing whitespace and leading newlines, but PRESERVE leading indentation
  # (strip() would break indented body code by removing indent from first line only)
  code = code.lstrip('\n').rstrip()
  if not code:
    return '', description

  # ============================================================
  # STEP 3: Determine if this is body-only or full function
  # ============================================================

  # Look for def priority(...) anywhere
  priority_match = re.search(r'^\s*def\s+(priority(?:_v\d+)?)\s*\(', code, re.MULTILINE)

  if priority_match:
    # ---------------------------------------------------------
    # CASE A: Full function output, extract priority body
    # ---------------------------------------------------------
    function_name = priority_match.group(1)
    logger.debug(f"Found {function_name} definition, extracting body")

    tree, parsed_code = _parse_code(code)
    if tree is None:
      return '', description

    parsed_lines = parsed_code.splitlines()

    # Collect module level imports to move inside function body
    imports = []
    for node in tree.body:
      if isinstance(node, (ast.Import, ast.ImportFrom)):
        imports.append(parsed_lines[node.lineno - 1])
      elif isinstance(node, ast.FunctionDef):
        break

    # Collect helper functions (any function that is not priority)
    # Skip common test/utility function names
    skip_functions = {function_name, 'main', 'evaluate', 'test', 'unused'}
    helper_lines = []
    for node in tree.body:
      if isinstance(node, ast.FunctionDef) and node.name not in skip_functions:
        # Extract the full function source and add 4 spaces to each line
        func_start = node.lineno - 1
        func_end = node.end_lineno
        func_lines = parsed_lines[func_start:func_end]
        # Add 4 spaces to each line to make it a nested function
        indented_func = ['    ' + line for line in func_lines]
        helper_lines.extend(indented_func)
        helper_lines.append('')  # blank line after helper
        logger.debug(f"Including helper function '{node.name}' as nested function")

    # Extract priority body lines (everything after the def line)
    start_line, end_line = _find_function_lines(tree, function_name)
    body_lines = parsed_lines[start_line:end_line]

    # Prepend imports and helpers inside function body
    prefix_lines = []
    if imports:
      logger.debug(f"Moving {len(imports)} import(s) inside function body")
      prefix_lines.extend(['    ' + imp for imp in imports])
    if helper_lines:
      prefix_lines.extend(helper_lines)

    if prefix_lines:
      body_lines = prefix_lines + body_lines

    body = '\n'.join(body_lines) + '\n\n'

  else:
    # ---------------------------------------------------------
    # CASE B: Completion output, entire code is the body
    # ---------------------------------------------------------
    logger.debug("No priority function found, treating as body code")

    code_lines = code.splitlines()
    skip_functions = {'main', 'evaluate', 'test', 'unused', '_fake_', 'priority'}

    # Separate module level items (imports, functions at column 0) from body code (indented)
    import_lines = []
    helper_lines = []
    body_code_lines = []

    i = 0
    while i < len(code_lines):
      line = code_lines[i]

      # Skip empty lines at module level
      if not line.strip():
        i += 1
        continue

      # Import at column 0
      if re.match(r'^(import |from \w+ import )', line):
        import_lines.append('    ' + line)
        i += 1
        continue

      # Function at column 0
      func_match = re.match(r'^def\s+(\w+)\s*\(', line)
      if func_match:
        func_name = func_match.group(1)
        # Find end of function (next line at column 0 or end of code)
        func_start = i
        i += 1
        while i < len(code_lines) and (not code_lines[i].strip() or code_lines[i][0].isspace()):
          i += 1
        func_end = i

        if func_name not in skip_functions:
          func_lines = code_lines[func_start:func_end]
          indented_func = ['    ' + fl for fl in func_lines]
          helper_lines.extend(indented_func)
          helper_lines.append('')
          logger.debug(f"Including helper function '{func_name}' as nested function")
        continue

      # Indented code (body) or non indented body
      # Collect all remaining lines until we hit a module level item
      while i < len(code_lines):
        line = code_lines[i]
        if line.strip() and not line[0].isspace():
          # Check if it's a module level item
          if re.match(r'^(import |from \w+ import |def\s+\w+\s*\()', line):
            break
        body_code_lines.append(line)
        i += 1

    body_code = '\n'.join(body_code_lines)

    # Add indentation if code has none
    if body_code and body_code.strip() and not body_code.lstrip('\n')[0].isspace():
      body_code = '    ' + body_code.replace('\n', '\n    ')

    # Validate by wrapping in fake function and parsing
    wrapped = 'def _fake_():\n' + body_code
    tree, parsed_code = _parse_code(wrapped)
    if tree is None:
      return '', description

    # Extract validated body
    _, end_line = _find_function_lines(tree, '_fake_')
    body_lines = parsed_code.splitlines()[1:end_line]

    # Prepend imports and helpers to body
    prefix_lines = import_lines + helper_lines
    if prefix_lines:
      body_lines = prefix_lines + body_lines

    body = '\n'.join(body_lines) + '\n\n'

  logger.debug(f"Parsed LLM output: {len(body)} chars, description={'yes' if description else 'no'}")
  return body, description


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    # function_to_evolve is set to priority
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str, str | None]:
  """Integrates a generated code as string into a larger program template.

  Returns:
      tuple: (evolved_function, program_str (Complete runnable script as string), description)
  """
  # Parse LLM output: extract from XML tags, strip markdown, validate with AST
  body, description = parse_llm_output(generated_code)
  if version_generated is not None:

    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)
  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)

  evolved_function.body = body
  evolved_function.description = description
  return evolved_function, str(program), description


def run_evaluation(sandbox, program, function_to_run, input):
    return sandbox.run(program, function_to_run, input)


class Evaluator:
    """Evaluates generated functions in sandboxed subprocesses.

    Args:
        template: Program template with function to evolve
        inputs: List of test inputs for evaluation
        sandbox_base_path: Base directory for sandbox files
        local_id: Unique identifier for this evaluator (usually PID)
        evaluator_config: EvaluatorConfig with timeout, max_workers, etc.
        connection_manager: RabbitMQ ConnectionManager for queue access
        target_signatures: Dict mapping problem dimensions to target scores
    """
    def __init__(
        self,
        template,
        inputs,
        sandbox_base_path,
        local_id,
        evaluator_config,
        connection_manager,
        target_signatures=None,
        function_to_evolve='priority',
        function_to_run='evaluate',
    ):
        self.template = template
        self.inputs = inputs
        self.local_id = local_id
        self.target_signatures = target_signatures
        self.function_to_evolve = function_to_evolve
        self.function_to_run = function_to_run
        self.timeout_seconds = evaluator_config.timeout
        self.prefetch_count = evaluator_config.prefetch_count
        self._conn = connection_manager
        self.graph_dir = evaluator_config.graph_dir
        self.graph_type = getattr(evaluator_config, 'graph_type', 'deletion')

        # Sandbox and thread pool for parallel evaluation.
        # Graphs are cached here. Forked children inherit via copy-on-write.
        self.sandbox = sandbox.ExternalProcessSandbox(
            timeout_secs=evaluator_config.timeout,
            graph_dir=evaluator_config.graph_dir,
            memory_limit_gb=evaluator_config.sandbox_memory_limit_gb
        )
        self.executor = ThreadPoolExecutor(max_workers=evaluator_config.max_workers)
        self.cumulative_cpu_time = 0.0

    def _get_input_with_graph(self, input_tuple):
        """Augment input tuple with cached graph.

        Loads graph on first access, caches for reuse.
        Forked children inherit the cached graph via copy-on-write.
        """
        n, s, q = input_tuple[:3]
        G = get_cached_graph(n, s, q, self.graph_dir, self.graph_type)
        if G is not None:
            return (n, s, q, G)
        return input_tuple

    async def consume_and_process(self):
        """Main consume loop with automatic connection recovery."""

        while True:
            try:
                # Connect with retry (handles exponential backoff internally)
                if not await self._conn.connect_with_retry():
                    break  # Shutdown requested

                await self._consume_loop()
                break  # Normal exit

            except asyncio.CancelledError:
                logger.info(f"Evaluator {self.local_id}: Cancelled, exiting...")
                break

            except Exception as e:
                logger.warning(f"Evaluator {self.local_id}: {type(e).__name__}: {e}. Reconnecting...")
                await self._conn.close()

    async def _consume_loop(self):
        """Inner consume loop that processes messages from queue."""
        await self._conn.channel.set_qos(prefetch_count=self.prefetch_count)

        async with self._conn.get_queue("evaluator_queue").iterator() as stream:
            async for message in stream:
                async with message.process():
                    try:
                        await asyncio.wait_for(self.process_message(message), timeout=300)
                    except TimeoutError:
                        logger.warning("Processing message timed out.")
                    except Exception as e:
                        logger.error(f"Evaluator: Error while processing message: {e}")

    @async_time_execution("Evaluator")
    async def process_message(self, message: aio_pika.IncomingMessage):
        """
        Process a single message from evaluator_queue.
        """
        try:
            data = json.loads(message.body.decode())

            # Metadata from sampler
            gpu_time = data.get("gpu_time", 0.0)
            input_tokens = data.get("input_tokens", 0)
            output_tokens = data.get("output_tokens", 0)
            parent_ids = data.get("parent_ids", [])
            reflection_output = data.get("reflection_output")  # ReEvo reflection

            # Parse LLM output and integrate into evaluation template
            new_function, program, description = _sample_to_program(
                data["sample"], data.get("version_generated"), self.template, self.function_to_evolve
            )

            # Helper to build result tuple
            def make_result(func, scores, found_optimal=False):
                return (func, data['island_id'], scores, data['expected_version'],
                        self.cumulative_cpu_time, gpu_time, input_tokens, output_tokens,
                        found_optimal, parent_ids, description, reflection_output)

            # Early exit if parsing failed
            if not new_function.body:
                logger.warning(f"Evaluator: Parsing failed, empty body. Island {data['island_id']}")
                await self.publish_to_database(make_result("return", {}), None)
                return

            # Log parsed function body
            logger.info(f"Evaluator: Parsed function for island {data['island_id']}:\n{new_function.body}")

            # Submit evaluation tasks with cached graphs (forked children inherit cache)
            tasks = {
                self.executor.submit(run_evaluation, self.sandbox, program, self.function_to_run,
                                     self._get_input_with_graph(input)): input
                for input in self.inputs
            }

            # Collect results from completed futures
            scores_per_test = {}
            hash_value = None

            for future in as_completed(tasks):
                input = tasks[future]
                try:
                    test_output, runs_ok, cpu_time = future.result(timeout=self.timeout_seconds)
                    self.cumulative_cpu_time += cpu_time

                    if runs_ok and test_output[0] is not None:
                        score_key, score_value, extracted_hash = extract_evaluation_result(test_output, input)
                        scores_per_test[score_key] = score_value
                        if extracted_hash is not None:
                            hash_value = extracted_hash
                        logger.info(f"Evaluator: input {input} score={score_value}")
                    else:
                        logger.warning(f"Evaluator: input {input} failed")
                        break

                except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError, Exception) as e:
                    logger.warning(f"Evaluator: input {input} error: {type(e).__name__}")
                    break

            else:
                # Loop completed without break (all tests passed)
                if scores_per_test and any(scores_per_test.values()):
                    found_optimal = self.target_signatures and all(
                        scores_per_test.get(dim, 0) >= self.target_signatures.get(dim, float("inf"))
                        for dim in self.target_signatures
                    )
                    await self.publish_to_database(make_result(new_function, scores_per_test, found_optimal), hash_value)
                    self.cumulative_cpu_time = 0.0
                    return

            # Failure path: cancel pending tasks and publish return to database
            for f in tasks:
                f.cancel()
            await self.publish_to_database(make_result("return", {}), None)
            self.cumulative_cpu_time = 0.0

        except Exception as e:
            logger.error(f"Evaluator: process_message error: {e}")


    async def publish_to_database(self, result, hash_value):
        try:
            function, island_id, scores_per_test, expected_version, cpu_time, gpu_time, input_tokens, output_tokens, found_optimal_solution, parent_ids, description, reflection_output = result

            serialized_result = {
                "new_function": function.serialize() if hasattr(function, 'serialize') else str(function),
                "island_id": island_id,
                "scores_per_test": {str(key): value for key, value in scores_per_test.items()},
                "expected_version": expected_version,
                "hash_value": hash_value,
                "cpu_time": cpu_time,
                "gpu_time": gpu_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "found_optimal_solution": found_optimal_solution,
                "parent_ids": parent_ids,
                "description": description,
                "reflection_output": reflection_output,
            }

            await self._conn.channel.default_exchange.publish(
                aio_pika.Message(body=json.dumps(serialized_result).encode()),
                routing_key='database_queue'
            )

        except Exception as e:
            logger.error(f"Evaluator: Problem in publishing to database for island_id {island_id}: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown: close connection and cleanup subprocesses."""
        await self._conn.close()
        await self.shutdown_subprocesses()

    async def shutdown_subprocesses(self):
        """Cleanup ProcessPoolExecutor workers and sandbox subprocesses."""
        try:
            # Shutdown executor
            if self.executor:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True),
                        timeout=5
                    )
                except TimeoutError:
                    self.executor.shutdown(wait=False, cancel_futures=True)
                self.executor = None

            # Terminate all child processes
            children = psutil.Process().children(recursive=True)
            for child in children:
                child.terminate()
            _, still_alive = psutil.wait_procs(children, timeout=5)
            for p in still_alive:
                p.kill()

            # Cleanup orphaned sandbox processes and directories
            sandbox.cleanup_orphaned_sandbox_processes(logger)
            if self.sandbox:
                self.sandbox.cleanup_all()

            gc.collect()
            logger.info(f"Evaluator {self.local_id}: Subprocesses shutdown complete.")
        except Exception as e:
            logger.error(f"Evaluator {self.local_id}: Shutdown error: {e}")

