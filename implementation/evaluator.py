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
import concurrent.futures  
from concurrent.futures import ProcessPoolExecutor, as_completed #try pathos for multiprocessing
from torch.multiprocessing import Manager
import gc
from profiling import async_time_execution, async_track_memory
import psutil
import shutil


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
    with call_count_lock: # the with statement ensures the lock is released once the block is exited regardless of whether an exception is raised 
        count = call_count.value
        call_count.value += 1

    result, runs_ok, call_data_folder, input_path, error_file = sandbox.run(program, function_to_run, input, timeout_seconds, count)
    return result, runs_ok, call_data_folder, input_path, error_file



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
        self.executor = ProcessPoolExecutor(max_workers=6)

    async def shutdown(self):
        logger.info(f"Evaluator {self.local_id}: Initiating shutdown process.")
        try: 
            if self.executor:
                logger.info(f"Evaluator {self.local_id}: Shutting down executor.")
                self.executor.shutdown(wait=False) # if evaluator spawns processes using the executor and then is cancelled those continue running, if wait=True executer could hand while waiting for the completion of the task if wait= False might fail to clean up properly 
                # Also if the subtasks are acessing any shared resouces eg call_count or the logger improper termination can cause issues if another process is waiting for them
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

            # Run garbage collection to clean up resources
            gc.collect()

            logger.info(f"Evaluator {self.local_id}: Shutdown process complete.")
        except asyncio.TimeoutError:
            logger.warning(f"Evaluator {self.local_id}: Timeout occurred during shutdown.")
        except Exception as e:
            logger.error(f"Evaluator {self.local_id}: Error during shutdown: {e}")


    async def consume_and_process(self):
        try:
            # Set channel QoS
            async with self.channel:
                await self.channel.set_qos(prefetch_count=1)

                # Start consuming messages
                async with self.evaluator_queue.iterator() as stream:
                    try:
                        async for message in stream:
                            async with message.process():
                                try:
                                    # Set a reasonable timeout for processing each message
                                    await asyncio.wait_for(self.process_message(message), timeout=300)  # Adjust the timeout as needed
                                except asyncio.TimeoutError:
                                    logger.warning("Processing message timed out.")
                                except Exception as e:
                                    logger.error(f"Error while processing message: {e}")
                    except asyncio.CancelledError:
                        logger.info("Consumer was cancelled.")
                        raise  # Propagate the cancellation upwards
                    except Exception as e:
                        logger.error(f"Error in message stream: {e}")
        except Exception as e:
            logger.error(f"Exception occurred in consume_and_process: {e}")
            raise  # Re-raise the exception to be handled by the caller if needed
        finally:
            try:
                # Call shutdown with a timeout
                await asyncio.wait_for(self.shutdown(), timeout=100)
            except asyncio.TimeoutError:
                logger.warning("Shutdown took too long and timed out.")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")


    #@async_time_execution
    #@async_track_memory
    async def process_message(self, message: aio_pika.IncomingMessage):
        call_folders_to_cleanup = []  # List to track created folders
        call_files_to_cleanup = []  # List to track created folders
        try:
            raw_data = message.body.decode()
            data = json.loads(raw_data)
            logger.debug(f"Evaluator: Starts to analyze generated continuation of def priority: {data['sample']}")

            # Process the new function from the generated code
            new_function, program = _sample_to_program(data["sample"], data.get("version_generated"), self.template, self.function_to_evolve)

            tasks = {}
            if new_function.body not in [None, '']:
                # Submit each test input as a task for multiprocessing
                tasks = {self.executor.submit(run_evaluation, self.sandbox, program, self.function_to_run, input, self.timeout_seconds, self.call_count, self.call_count_lock): input for input in self.inputs}
            else:
                logger.info("New function body is None or empty. Skipping execution but publishing 'return'.")
                result = ("return", data['island_id'], {}, data['expected_version'])
                await self.publish_to_database(result, message)  # Publish "return" result
                return  # Early return after publishing

            scores_per_test = {}
            # Waiting for results from all test inputs
            for future in as_completed(tasks):
                input = tasks[future]
                try:
                    test_output, runs_ok, call_data_folder, input_path, error_file= future.result(timeout=self.timeout_seconds)
                    logger.info(f"Evaluator: test_output is {test_output}, runs_ok is {runs_ok}")
                    call_folders_to_cleanup.append(call_data_folder)
                    call_files_to_cleanup.append(input_path)
                    call_files_to_cleanup.append(error_file)
                    if runs_ok and test_output is not None:
                        scores_per_test[input] = test_output
                        logger.debug(f"Evaluator: scores_per_test {scores_per_test}")
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Task for input {input} timed out.")
                except concurrent.futures.CancelledError:
                    logger.warning(f"Task for input {input} was cancelled.")
                except Exception as e:
                    # Catch any other exceptions
                    logger.error(f"Error during task execution for input {input}: {e}")

            # Prepare the result for publishing
            if len(scores_per_test) == len(self.inputs) and any(score != 0 for score in scores_per_test.values()):
                result = (new_function, data['island_id'], scores_per_test, data['expected_version'])
                logger.debug(f"Scores are {scores_per_test}")
            else:
                result = ("return", data['island_id'], {}, data['expected_version'])

            # Publish the result
            await self.publish_to_database(result, message)

        except Exception as e:
            logger.error(f"Error in process_message: {e}")
        
        finally: 
            # Cleanup call folders
            for folder in call_folders_to_cleanup:
                if folder.exists():
                    shutil.rmtree(folder)
            # Cleanup input files
            for input_file in call_files_to_cleanup:
                if input_file is not None and input_file.exists():
                    input_file.unlink()  # Remove the input file



    async def publish_to_database(self, result, message):
        try:
            # Extracting the result components
            function, island_id, scores_per_test, expected_version = result
        
            # Serializing the result
            serialized_result = {
                "new_function": function.serialize() if hasattr(function, 'serialize') else str(function),
                "island_id": island_id,
                "scores_per_test": {str(key): value for key, value in scores_per_test.items()},
                "expected_version": expected_version
            }
            message_body = json.dumps(serialized_result)
        
            # Publishing the serialized result to the database queue
            await self.channel.default_exchange.publish(
                aio_pika.Message(body=message_body.encode(), 
                #delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ), 
                routing_key='database_queue'
            )
            logger.info(f"Evaluator: Successfully published to database for island_id {island_id}.")
    
        except Exception as e:
            logger.error(f"Evaluator: Problem in publishing to database for island_id {island_id}: {e}")
            # Optionally re-raise the exception if the caller needs to handle it.
            raise
