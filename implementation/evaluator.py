"""Class for evaluating programs proposed by the Sampler."""
import ast
from typing import Any
import copy
import logging
import code_manipulation
import sandbox
from pathlib import Path
import json
import aio_pika
import sys
import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.multiprocessing import Manager
import gc
from profiling import async_time_execution, async_track_memory


logger = logging.getLogger('main_logger')


class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      # node.end_lineo extracts the last name of the currently visited function when it matches the function name we initalized the class with 
      self._function_end_line = node.end_lineno
    # calling it will continue normal traversal of the AST 
    self.generic_visit(node)

  # Allows to access the functions end line number after the AST has been visited
  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''
  # wraps the code in a fake function header because generated_code is just the body of the function (completes def priority_vX):
  code = f'def fake_function_header():\n{generated_code}'
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
    # Loop continues until the parsing succeeds or there's no code left.  
    except SyntaxError as e:
      code = '\n'.join(code.splitlines()[:e.lineno - 1])
  if not code:
    return ''
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
) -> tuple[code_manipulation.Function, str]:
  """Returns the compiled generated function and the full runnable program.
     Purpose: integrate a generated code as string into a larger program template. 
  """
  body = _trim_function_body(generated_code) 
  if version_generated is not None:

    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)
  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)

  evolved_function.body = body
  return evolved_function, str(program)



def run_evaluation(sandbox, program, function_to_run, input, timeout_seconds, call_count, call_count_lock):
    with call_count_lock:
        count = call_count.value
        call_count.value += 1
    return sandbox.run(program, function_to_run, input, timeout_seconds, count)



class Evaluator:
    def __init__(self, connection, channel, evaluator_queue, database_queue, template, function_to_evolve, function_to_run, inputs, sandbox_base_path, timeout_seconds, local_id):
        self.connection = connection
        self.channel = channel
        self.evaluator_queue = evaluator_queue
        self.database_queue = database_queue
        self.template = template
        self.function_to_evolve = function_to_evolve
        self.function_to_run = function_to_run
        self.inputs = inputs
        self.timeout_seconds = timeout_seconds
        self.local_id = local_id
        self.manager = Manager()
        self.call_count = self.manager.Value('i', 0)
        self.call_count_lock = self.manager.Lock()
        self.sandbox = sandbox.ExternalProcessSandbox(
            base_path=sandbox_base_path, timeout_secs=timeout_seconds, python_path=sys.executable, local_id=self.local_id)
        self.executor = ProcessPoolExecutor(max_workers=10)


    async def consume_and_process(self):
        async with self.channel:
            await self.channel.set_qos(prefetch_count=1)
            async with self.evaluator_queue.iterator() as stream:
                try:
                    async for message in stream:
                        await self.process_message(message)
                except asyncio.CancelledError:
                    self.shutdown()  
                    raise  # Ensure the cancellation is propagated
                finally:
                    self.shutdown()
    @async_time_execution
    @async_track_memory
    async def process_message(self, message: aio_pika.IncomingMessage):
        async with message.process():
            raw_data = message.body.decode()
            data = json.loads(raw_data)
            logger.debug(f"Evaluator: Starts to analyse generated continuation of def priority: {data['sample']}")

            new_function, program = _sample_to_program(data["sample"], data.get("version_generated"), self.template, self.function_to_evolve)
            tasks = {}
            if new_function.body not in [None, '']:
                # Submit each test input as task for Multiprocessing
                tasks = {self.executor.submit(run_evaluation, self.sandbox, program, self.function_to_run, input, self.timeout_seconds, self.call_count, self.call_count_lock): input for input in self.inputs}
            else:
                logger.debug("New function body is None or empty. Skipping execution.")            
            scores_per_test = {}
            # Waiting for results from all test inputs
            for future in as_completed(tasks):
                input = tasks[future]
                test_output, runs_ok = future.result()
                logger.debug(f"Evaluator: test_output is {test_output} , runs_ok is {runs_ok} ")
                if runs_ok and test_output is not None: #and not _calls_ancestor(program, self.function_to_evolve)
                    scores_per_test[input] = test_output
                    logger.debug(f"Evaluator: scores_per_test {scores_per_test}")

            if scores_per_test:
                last_score = list(scores_per_test.values())[-1]
                if last_score != 0:
                    result = (new_function, data['island_id'], scores_per_test, data['expected_version'])
                    logger.debug(f"Scores are {scores_per_test}")
                else:
                    result = ("return", data['island_id'], {}, data['expected_version'])
            else: 
                result = ("return", data['island_id'], {}, data['expected_version'])
            try:
                await self.publish_to_database(result, message)
            except Exception as e:
                logger.error(f"Error in await self.publish_to_database(result) {e}")
                raise


    async def publish_to_database(self, result, message):
        function, island_id, scores_per_test, expected_version = result
        serialized_result = {
            "new_function": function.serialize() if hasattr(function, 'serialize') else str(function),
            "island_id": island_id,
            "scores_per_test": {str(key): value for key, value in scores_per_test.items()},
            "expected_version": expected_version
        }
        message_body = json.dumps(serialized_result)
        try:
            await self.channel.default_exchange.publish(
                aio_pika.Message(body=message_body.encode()), routing_key='database_queue'
            )
        except Exception as e:
            logger.error(f"Evaluator: Problem in publishing to database: {e}")

    def shutdown(self):
        self.executor.shutdown(wait=True)
        gc.collect()