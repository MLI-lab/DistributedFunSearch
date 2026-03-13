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
3. Fork child process (inherits parent memory via copy-on-write)
4. Child compiles code with caching (base cached, only priority recompiled),
   sets memory limits, runs evaluation, returns result via pipe
5. Publish scores to database queue

Graphs are pre-loaded from LMDB during init and cached. Forked children inherit
the cache via copy-on-write, so graphs are never reloaded.
"""


import logging
import os
from disfun.utils.code_manipulation import sample_to_program
from disfun.utils.profiling import async_time_execution
from disfun import sandbox
import json
import aio_pika
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import gc
import psutil

from disfun.utils.fast_graph import load_graph_from_lmdb, load_nx_graph_from_lmdb


logger = logging.getLogger('main_logger')


# Graph cache shared across evaluator threads.
# Forked children inherit this cache via copy-on-write.
_graph_cache = {}
_graph_cache_lock = Lock()


def _build_graph_path(graph_dir, graph_type, s, n, q):
    """Build path like: graph_dir/ids/quaternary/s1/graph_ids_s1_n6_q4.lmdb"""
    is_deletion = graph_type in ("deletion", "deletions")
    subdir = "deletion" if is_deletion else "ids"
    alphabet = "binary" if q == 2 else "quaternary" if q == 4 else f"q{q}"
    prefix = "graph_d" if is_deletion else "graph_ids"
    return os.path.join(graph_dir, subdir, alphabet, f"s{s}", f"{prefix}_s{s}_n{n}_q{q}.lmdb")


def get_cached_graph(n, s, q, graph_dir, graph_type="deletion", use_nx=False):
    """Get graph from cache or load and cache it.

    Thread-safe. Forked children inherit the cached graphs.
    Returns None if graph_dir is not configured (no-graph mode).
    If use_nx is True, loads into a real nx.Graph instead of FastGraph.
    """
    if not graph_dir:
        return None

    key = (n, s, q, use_nx)

    with _graph_cache_lock:
        if key in _graph_cache:
            return _graph_cache[key]

    graph_path = _build_graph_path(graph_dir, graph_type, s, n, q)
    if not os.path.exists(graph_path):
        logger.warning(f"Graph not found: {graph_path}")
        return None

    loader = load_nx_graph_from_lmdb if use_nx else load_graph_from_lmdb
    logger.info(f"Loading graph n={n}, s={s}, q={q} from {graph_path} ({'nx.Graph' if use_nx else 'FastGraph'})")
    G = loader(graph_path)

    with _graph_cache_lock:
        _graph_cache[key] = G
        logger.info(f"Cached graph n={n}, s={s}, q={q}, nodes={len(G)}")

    return G


class Evaluator:
    """Evaluates generated functions in sandboxed subprocesses.

    Args:
        template: Program template with function to evolve
        inputs: List of test inputs for evaluation
        local_id: Unique identifier for this evaluator (usually PID)
        evaluator_config: EvaluatorConfig with timeout, max_workers, etc.
        connection_manager: RabbitMQ ConnectionManager for queue access
        target_signatures: Dict mapping problem dimensions to target scores
    """
    def __init__(
        self,
        template,
        inputs,
        local_id,
        evaluator_config,
        connection_manager,
        target_signatures=None,
        function_to_evolve='priority',
        function_to_run='evaluate',
        log_dir=None,
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
        self.use_nx = evaluator_config.evaluation_script_path.endswith('graph_nx.py')

        # Debug sample folders (category-first layout: debug_samples/{category}/eval{PID}_{counter}_island{id}/)
        self.debug_samples = getattr(evaluator_config, 'debug_samples', False)
        self._sample_counter = 0
        if self.debug_samples and log_dir:
            self.debug_dir = os.path.join(log_dir, "debug_samples")
            self._eval_prefix = f"eval{local_id}"
            os.makedirs(self.debug_dir, exist_ok=True)
            logger.info(f"Evaluator {local_id}: Debug samples enabled, writing to {self.debug_dir}")
        else:
            self.debug_dir = None

        # Sandbox and thread pool for parallel evaluation.
        # Graphs are cached here. Forked children inherit via copy-on-write.
        self.sandbox = sandbox.ExternalProcessSandbox(
            timeout_secs=evaluator_config.timeout,
            graph_dir=evaluator_config.graph_dir,
            memory_limit_gb=evaluator_config.sandbox_memory_limit_gb,
        )
        self.executor = ThreadPoolExecutor(max_workers=evaluator_config.max_workers)
        self.cumulative_cpu_time = 0.0

        # Pre-load graphs during init. Combined with staggered startup, this prevents
        # memory spikes from all evaluators loading simultaneously on first message.
        self._preload_graphs()

    def _preload_graphs(self):
        """Pre-load all graphs into cache during initialization."""
        if not self.graph_dir:
            logger.info(f"Evaluator {self.local_id}: No graph_dir configured, skipping graph preload (no-graph mode)")
            return
        unique_inputs = {(inp[0], inp[1], inp[2]) for inp in self.inputs if len(inp) >= 3}
        logger.info(f"Evaluator {self.local_id}: Pre-loading {len(unique_inputs)} graphs...")
        for n, s, q in sorted(unique_inputs):
            get_cached_graph(n, s, q, self.graph_dir, self.graph_type, self.use_nx)
        logger.info(f"Evaluator {self.local_id}: Pre-load complete.")

    def _get_input_with_graph(self, input_tuple):
        """Augment input tuple with cached graph.

        Graphs are preloaded during init, this just retrieves from cache.
        Forked children inherit the cached graph via copy-on-write.
        """
        n, s, q = input_tuple[:3]
        G = get_cached_graph(n, s, q, self.graph_dir, self.graph_type, self.use_nx)
        if G is not None:
            return (n, s, q, G)
        return input_tuple

    def _write_debug_sample(self, category, debug_data, debug_lines):
        """Write all debug files into a category subfolder (success/eval_failure/parse_failure)."""
        if not self.debug_dir:
            return
        island_id = debug_data.get("island_id", "x")
        name = f"{self._eval_prefix}_{self._sample_counter:06d}_island{island_id}"
        folder = os.path.join(self.debug_dir, category, name)
        try:
            os.makedirs(folder, exist_ok=True)
            prompt = debug_data.get("prompt", "")
            if prompt:
                with open(os.path.join(folder, "0_prompt.txt"), "w") as f:
                    f.write(prompt)
            with open(os.path.join(folder, "1_raw_llm_output.txt"), "w") as f:
                f.write(debug_data.get("raw_output", ""))
            thinking_trace = debug_data.get("thinking_trace")
            if thinking_trace:
                with open(os.path.join(folder, "1b_thinking_trace.txt"), "w") as f:
                    f.write(thinking_trace)
            with open(os.path.join(folder, "2_parsed_body.py"), "w") as f:
                f.write(debug_data.get("parsed_body", "Parse failed"))
            if debug_lines:
                with open(os.path.join(folder, "3_eval_results.txt"), "w") as f:
                    f.write("\n".join(debug_lines) + "\n")
        except Exception as e:
            logger.warning(f"Evaluator: debug_samples write error: {e}")

    async def consume_and_process(self):
        """Main consume loop with automatic connection recovery.

        Uses connect_with_retry() for exponential backoff (config: reconnect_delay, max_reconnect_delay).
        Gives up after max_reconnects consecutive failures to avoid infinite loops.
        """
        import os as _os
        pid = _os.getpid()
        max_reconnects = 10
        reconnect_count = 0

        while reconnect_count < max_reconnects:
            try:
                if not await self._conn.connect_with_retry():
                    logger.warning(f"Evaluator {self.local_id} (PID {pid}): Exit reason: shutdown requested during connect")
                    break  # Shutdown requested

                loop_start = asyncio.get_event_loop().time()
                await self._consume_loop()
                elapsed = asyncio.get_event_loop().time() - loop_start
                logger.info(f"Evaluator {self.local_id} (PID {pid}): consume_loop ended after {elapsed:.0f}s")
                # Only reset if it actually ran for a while (processed messages)
                if elapsed > 60:
                    reconnect_count = 0

            except asyncio.CancelledError:
                logger.warning(f"Evaluator {self.local_id} (PID {pid}): Exit reason: CancelledError (signal or shutdown)")
                break

            except Exception as e:
                logger.warning(f"Evaluator {self.local_id} (PID {pid}): {type(e).__name__}: {e}")
                await self._conn.close(shutdown=False)

            reconnect_count += 1
            logger.info(f"Evaluator {self.local_id} (PID {pid}): Reconnecting ({reconnect_count}/{max_reconnects})...")
        else:
            logger.error(f"Evaluator {self.local_id} (PID {pid}): Exit reason: {max_reconnects} consecutive reconnect failures")

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
            new_function, program, description, thinking_trace, failure_reason = sample_to_program(
                data["sample"], data.get("version_generated"), self.template, self.function_to_evolve
            )

            # Debug sample data (deferred write until outcome is known)
            self._sample_counter += 1
            debug_data = {
                "prompt": data.get("prompt", ""),
                "raw_output": data.get("sample", ""),
                "thinking_trace": thinking_trace,
                "parsed_body": new_function.body if new_function.body else f"Parse failed: {failure_reason or 'unknown'}",
                "island_id": data.get("island_id", "x"),
            }

            # Helper to build result tuple
            def make_result(func, scores, found_optimal=False):
                return (func, data['island_id'], scores, data['expected_version'],
                        self.cumulative_cpu_time, gpu_time, input_tokens, output_tokens,
                        found_optimal, parent_ids, description, reflection_output, thinking_trace)

            # Early exit if parsing failed
            if not new_function.body:
                logger.warning(f"Evaluator: Parsing failed ({failure_reason}), empty body. Island {data['island_id']}")
                self._write_debug_sample("parse_failure", debug_data, [f"Parse failed ({failure_reason or 'unknown'}) - never reached evaluation"])
                await self.publish_to_database(make_result("return", {}), None)
                return

            # Log parsed function body
            logger.info(f"Evaluator: Parsed function for island {data['island_id']}:\n{new_function.body}")

            # Submit evaluation tasks with cached graphs (forked children inherit cache)
            tasks = {
                self.executor.submit(self.sandbox.run, program, self.function_to_run,
                                     self._get_input_with_graph(input)): input
                for input in self.inputs
            }

            # Collect results from completed futures
            scores_per_test = {}
            hash_value = None
            debug_lines = []

            for future in as_completed(tasks):
                input = tasks[future]
                try:
                    test_output, runs_ok, cpu_time = future.result(timeout=self.timeout_seconds)
                    self.cumulative_cpu_time += cpu_time

                    if runs_ok and test_output[0] is not None:
                        score_key = (input[0], input[1])
                        score_value = test_output[0]
                        extracted_hash = test_output[1] if test_output[1] is not None else None
                        scores_per_test[score_key] = score_value
                        if extracted_hash is not None:
                            hash_value = extracted_hash
                        logger.info(f"Evaluator: input {input} score={score_value}")
                        debug_lines.append(f"{score_key}: score={score_value}")
                    else:
                        logger.warning(f"Evaluator: input {input} failed")
                        if isinstance(test_output, str):
                            reason = test_output
                        elif test_output is None:
                            reason = "sandbox returned None"
                        else:
                            reason = f"test_output={test_output}"
                        debug_lines.append(f"{input}: Failed — {reason}")
                        break

                except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError, Exception) as e:
                    logger.warning(f"Evaluator: input {input} error: {type(e).__name__}")
                    debug_lines.append(f"{input}: Error {type(e).__name__}: {e}")
                    break

            else:
                # Loop completed without break (all tests passed)
                if scores_per_test and any(scores_per_test.values()):
                    found_optimal = self.target_signatures and all(
                        scores_per_test.get(dim, 0) >= self.target_signatures.get(dim, float("inf"))
                        for dim in self.target_signatures
                    )
                    self._write_debug_sample("success", debug_data, debug_lines)
                    await self.publish_to_database(make_result(new_function, scores_per_test, found_optimal), hash_value)
                    self.cumulative_cpu_time = 0.0
                    return

            # Failure path: cancel pending tasks and publish return to database
            for f in tasks:
                f.cancel()
            self._write_debug_sample("eval_failure", debug_data, debug_lines)
            await self.publish_to_database(make_result("return", {}), None)
            self.cumulative_cpu_time = 0.0

        except Exception as e:
            logger.error(f"Evaluator: process_message error: {e}")


    async def publish_to_database(self, result, hash_value):
        try:
            function, island_id, scores_per_test, expected_version, cpu_time, gpu_time, input_tokens, output_tokens, found_optimal_solution, parent_ids, description, reflection_output, thinking_trace = result

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
                "thinking_trace": thinking_trace,
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
        """Cleanup ThreadPoolExecutor workers and sandbox subprocesses."""
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

            gc.collect()
            logger.info(f"Evaluator {self.local_id}: Subprocesses shutdown complete.")
        except Exception as e:
            logger.error(f"Evaluator {self.local_id}: Shutdown error: {e}")

