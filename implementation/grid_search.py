import asyncio
import logging
import numpy as np
from yarl import URL
import json
import os
import dataclasses
import tempfile
import shutil
import pickle
import config as config_lib
import glob
from itertools import product  # Add this line
from funsearch import initialize_task_manager
import aio_pika

# Set the base directory and grid search directory
BASE_DIR = os.getcwd()
GRID_SEARCH_DIR = os.path.join(BASE_DIR, "GridSearch")
CHECKPOINT_DIR= os.path.join(BASE_DIR, "Checkpoints")
os.makedirs(GRID_SEARCH_DIR, exist_ok=True)  # Ensure the directory exists

# Setup Logger
logger = logging.getLogger('grid_search_logger')
logger.setLevel(logging.DEBUG)
log_file_path = os.path.join(GRID_SEARCH_DIR, 'grid_search.log')
file_handler = logging.FileHandler(log_file_path, mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Configuration file path
CONFIG_FILE_PATH = os.path.join(GRID_SEARCH_DIR, 'config_file.json')
RESULTS_FILE_PATH = os.path.join(GRID_SEARCH_DIR, 'results.csv')

class DataclassJSONEncoder(json.JSONEncoder):
    """Custom JSON Encoder for handling data classes."""
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)

def load_checkpoint(directory):
    try:
        # Adjust the path to specifically look within a certain directory
        checkpoint_files = glob.glob(os.path.join(directory, "checkpoint_*.pkl"))
        if not checkpoint_files:
            logger.warning("No checkpoint files found.")
            return None
        
        # Sort files by the timestamp in their filenames, assuming the format is "checkpoint_YYYY-MM-DD_HH-MM-SS.pkl"
        checkpoint_files.sort(key=lambda x: os.path.basename(x)[11:-4], reverse=True)  # sorts by the timestamp part of the filename
        latest_checkpoint_file = checkpoint_files[0]
        
        with open(latest_checkpoint_file, "rb") as f:
            data = pickle.load(f)
        
        logger.info(f"Checkpoint data loaded successfully from {latest_checkpoint_file}")
        return data
    except Exception as e:
        logger.error(f"Failed to load checkpoint data: {e}")
        return None


def evaluate_hyperparameters(checkpoint_data):
    if checkpoint_data is None:
        logger.error("No checkpoint data available for evaluation.")
        return float('nan')
    # Extract the values from the checkpoint data
    registered_programs = checkpoint_data.get('registered_programs', 0)
    best_scores_per_island = checkpoint_data.get('best_score_per_island', [])
    islands_state = checkpoint_data.get('islands_state', [])
    average_best_score = sum(best_scores_per_island) / len(best_scores_per_island) if best_scores_per_island else 0
    max_best_score = max(best_scores_per_island) if best_scores_per_island else 0
    total_islands_state_length = sum(len(island) for island in islands_state)
    overall_score = (registered_programs + average_best_score + total_islands_state_length + (2 * max_best_score)) / 5
    logger.info(f"Evaluation results - Registered Programs: {registered_programs}, Average Best Score: {average_best_score}, Max Best Score: {max_best_score}, Total Number of Clusters: {total_islands_state_length}, Overall Score: {overall_score}")
    return overall_score, (registered_programs, average_best_score, max_best_score, total_islands_state_length)

def save_results_to_file(param_config, evaluation_metric):
    with open(RESULTS_FILE_PATH, 'a') as file:
        # Only save the modified parameters (those in param_config)
        param_values = ', '.join(f"{k}={v}" for k, v in param_config.items())
        # Write config and metric data with appropriate labels
        file.write(f"config: {param_values}, metric: {evaluation_metric}\n")
        # Log the saved result
        logger.info(f"Results saved: config: {param_values}, metric: {evaluation_metric}")


def update_config(config, param_updates):
    for param, value in param_updates.items():
        if hasattr(config, param):
            setattr(config, param, value)
        else:
            raise ValueError(f"Parameter {param} does not exist in the Config class")
    return config

def load_last_grid_config():
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, 'r') as file:
                data = json.load(file)
            if data:
                return data[-1]  # Last evaluated config
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON. Starting with the first configuration.")
    return None

def save_last_grid_config(param_config):
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'r+') as file:
                data = json.load(file)
                data.append(param_config)
        else:
            data = [param_config]
        safe_file_write(CONFIG_FILE_PATH, data)
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from the existing file.")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")

def safe_file_write(path, data):
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(temp_fd, 'w') as tmp:
            json.dump(data, tmp, indent=4, cls=DataclassJSONEncoder)
        shutil.move(temp_path, path)
    except Exception as e:
        logger.error(f"Failed to write data safely: {e}")
        os.unlink(temp_path)

async def run_experiment_with_timeout(config, param_config, timeout_seconds):
    # Log the current state of config to verify its attributes before passing it on
    task_manager = initialize_task_manager(config)
    task = asyncio.create_task(task_manager.run())
    
    try:
        await asyncio.wait_for(task, timeout_seconds)
        logger.info("Experiment completed successfully.")
    except asyncio.TimeoutError:
        logger.error("Experiment timeout.")
        raise  # Re-raise the TimeoutError so it can be handled upstream
    except Exception as e:
        logger.error(f"Unexpected error during task execution: {e}")
        raise  # Re-raise the exception to propagate it
    finally:
        # Save the checkpoint data
        checkpoint_data = load_checkpoint(CHECKPOINT_DIR)
        if checkpoint_data:
            metric = evaluate_hyperparameters(checkpoint_data)
            save_results_to_file(param_config, metric)

        # Add a sleep delay before performing the cleanup
        logger.info("Sleeping for 2 minutes before cleaning up queues and consumers...")
        # Queue and consumer cleanup
        try:
            logger.info("Cleaning up consumers and queues...")
            
            # Add logic to clean up your RabbitMQ queues and consumers
            connection = await aio_pika.connect_robust(
                config.amqp_url,
                timeout=300,
                client_properties={"connection_attempts": 3, "retry_delay": 5},
                reconnect_interval=5
            )
            channel = await connection.channel()

            # Assuming you're using specific queues
            queues = ["evaluator_queue", "sampler_queue", "database_queue"]
            for queue_name in queues:
                try:
                    queue = await channel.declare_queue(queue_name, durable=False, auto_delete=True)
                    await queue.purge()  # Purge the queue to clear all messages
                    await queue.delete()  # Optionally delete the queue entirely if it's no longer needed
                    logger.info(f"Successfully cleaned up queue: {queue_name}")
                except Exception as e:
                    logger.error(f"Error while cleaning up queue {queue_name}: {e}")
            
            await channel.close()
            await connection.close()

        except Exception as e:
            logger.error(f"Error while cleaning up consumers and queues: {e}")


async def perform_grid_search(grid_dict, timeout_seconds):
    last_config = load_last_grid_config()
    param_names = list(grid_dict.keys())
    param_values = list(grid_dict.values())
    param_combinations = list(product(*param_values)) # Cartesian product of the the lists of values for each hyperparameter.
    start_index = param_combinations.index(tuple(last_config.values())) + 1 if last_config and tuple(last_config.values()) in param_combinations else 0 # if last conifg tuple not found start with first config
    for i in range(start_index, len(param_combinations)):    
        param_config = dict(zip(param_names, param_combinations[i])) # zipped pairs into a dictionary, e.g, {'learning_rate': 0.01, 'batch_size': 32}
        config = config_lib.Config()
        logger.info(f"Starting experiment with {param_config}")
        updated_config = update_config(config, param_config)
        save_last_grid_config(param_config)
        try: 
            await run_experiment_with_timeout(updated_config, param_config, timeout_seconds)
        except Exception as e: 
            logger.error(f"Error in await run experiment {e}")
            raise e
    logger.info("Grid search completed.")

if __name__ == "__main__":
    grid_params = {
        "temperature": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        "top_p": [0.6, 0.7, 0.8, 0.9, 1]
    }
    try:
        asyncio.run(perform_grid_search(grid_params, 3660))
    except Exception as e:
        logger.error(f"Grid search error: {e}")
        os._exit(1)
