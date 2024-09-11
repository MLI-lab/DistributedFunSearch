import asyncio
import logging
import numpy as np
from yarl import URL
from funsearch import TaskManager
from config import Config
import json
import os
import dataclasses
import tempfile
import shutil
import pickle
import config as config_lib
import glob

config = config_lib.Config()

# Set the current working directory
BASE_DIR = os.getcwd()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup Logger
logger = logging.getLogger('grid_search_logger')

# Set logging level (e.g., INFO, DEBUG, ERROR, etc.)
logger.setLevel(logging.INFO)

# File Handler for logging, appending logs to 'grid_search.log'
log_file_path = os.path.join(BASE_DIR, 'grid_search.log')
file_handler = logging.FileHandler(log_file_path, mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Configuration file path
CONFIG_FILE_PATH = os.path.join(BASE_DIR, 'config_file.json')
RESULTS_FILE_PATH = os.path.join(BASE_DIR, 'results.csv')

def load_experiment_setup():
    """Loads the experiment setup, including specification, inputs, and AMQP URL."""
    try:
        spec_file_path = os.path.join(BASE_DIR, 'implementation/specification.txt')
        with open(spec_file_path, 'r') as file:
            specification = file.read()
    except FileNotFoundError:
        logger.error("Specification file not found.")
        return None

    inputs = [(6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)]
    amqp_url = URL(f'amqp://{config.rabbitmq.username}:{config.rabbitmq.password}@{config.rabbitmq.host}:{config.rabbitmq.port}/').update_query(heartbeat=180000)
    return specification, inputs, amqp_url

def load_checkpoint():
    """Loads the most recent checkpoint data from a file."""
    try:
        # Search for all checkpoint files matching the pattern 'checkpoint_*.pkl'
        checkpoint_files = glob.glob(os.path.join(os.getcwd(), "checkpoint_*.pkl"))

        if not checkpoint_files:
            logger.warning("No checkpoint files found.")
            return None

        # Sort files by modification time (most recent first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)

        # The most recent checkpoint file
        latest_checkpoint_file = checkpoint_files[0]

        # Load the most recent checkpoint file
        with open(latest_checkpoint_file, "rb") as f:
            data = pickle.load(f)
        
        logger.info(f"Checkpoint data loaded successfully from {latest_checkpoint_file}")
        return data

    except Exception as e:
        logger.error(f"Failed to load checkpoint data: {e}")
        return None


def evaluate_hyperparameters(checkpoint_data):
    """
    Calculates an overall score from checkpoint data by averaging the registered programs,
    best scores per island, the maximum best score (with double weight), and the lengths of dictionaries in islands_state.

    Parameters:
    - checkpoint_data (dict): Checkpoint data containing necessary evaluation info.

    Returns:
    - float: Calculated overall score.
    - tuple: A tuple containing registered_programs, best_score_per_island, max_best_score, and islands_state_len.
    """
    if checkpoint_data is None:
        logger.error("No checkpoint data available for evaluation.")
        return float('nan')

    # Extract the values from the checkpoint data
    registered_programs = checkpoint_data.get('registered_programs', 0)
    best_scores_per_island = checkpoint_data.get('_best_score_per_island', [])
    islands_state = checkpoint_data.get('islands_state', [])

    # Calculate the average best score per island
    if best_scores_per_island:
        average_best_score = sum(best_scores_per_island) / len(best_scores_per_island)
        max_best_score = max(best_scores_per_island)  # Find the maximum score
    else:
        average_best_score = 0
        max_best_score = 0

    # Calculate the total number of entries across all dictionaries in islands_state
    total_islands_state_length = sum(len(island) for island in islands_state)

    # Calculate the weighted overall average
    # Registered programs and total_islands_state_length are equally weighted (1x)
    # Average best score is weighted 1x and max best score is weighted 2x
    overall_score = (
        registered_programs + average_best_score + total_islands_state_length + (2 * max_best_score)
    ) / 5

    logger.info(f"Evaluation results - Registered Programs: {registered_programs}, "
                f"Average Best Score: {average_best_score}, Max Best Score: {max_best_score}, "
                f"Total Islands State Length: {total_islands_state_length}, "
                f"Overall Score: {overall_score}")

    # Return the overall score and detailed components for logging or further analysis
    return overall_score, (registered_programs, average_best_score, max_best_score, total_islands_state_length)



def save_results_to_file(param_config, evaluation_metric):
    """Appends the results of an experiment to the results file in CSV format."""
    with open(RESULTS_FILE_PATH, 'a') as file:
        param_values = ','.join(f"{k}={v}" for k, v in param_config.items())
        file.write(f"{param_values},{evaluation_metric}\n")
        logger.info(f"Results saved: {param_values}, Metric={evaluation_metric}")

def update_config(config, param_updates):
    """Updates the configuration object based on param_updates dictionary."""
    for param, value in param_updates.items():
        if hasattr(config, param):
            setattr(config, param, value)
        else:
            raise ValueError(f"Parameter {param} does not exist in the Config class")
    return config

def load_last_grid_config():
    """Loads the last configuration from the configuration file to resume grid search."""
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
    """Saves the current grid configuration to the configuration file."""
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'r+') as file:
                data = json.load(file)
                data.append(param_config)
        else:
            data = [param_config]
        safe_file_write(CONFIG_FILE_PATH, data)
        logger.info(f"Saved current configuration: {param_config}")
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from the existing file.")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")

def safe_file_write(path, data):
    """Writes data safely to a file using atomic operations."""
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(temp_fd, 'w') as tmp:
            json.dump(data, tmp, indent=4, cls=DataclassJSONEncoder)
        shutil.move(temp_path, path)
    except Exception as e:
        logger.error(f"Failed to write data safely: {e}")
        os.unlink(temp_path)
        raise

async def run_experiment_with_timeout(specification, inputs, config, amqp_url, timeout_seconds):
    """Runs the experiment with a timeout."""
    try:
        logger.info(f"Starting experiment with config: {config}")
        task_manager = TaskManager(specification, inputs, config)
        task = asyncio.create_task(task_manager.run())
        await asyncio.wait_for(task, timeout_seconds)
        logger.info("Experiment completed successfully.")
    except asyncio.TimeoutError:
        logger.error("Experiment timeout.")
    except Exception as e:
        logger.error(f"Unexpected error during task execution: {e}")
    finally:
        checkpoint_data = load_checkpoint(os.path.join(BASE_DIR, 'checkpoint.pkl'))
        if checkpoint_data:
            metric = evaluate_hyperparameters(checkpoint_data)
            save_results_to_file(dataclasses.asdict(config), metric)

async def perform_grid_search(grid_dict, timeout_seconds):
    """Perform grid search over parameters defined in grid_dict."""
    setup = load_experiment_setup()
    if setup is None:
        logger.error("Failed to load experiment setup. Aborting grid search.")
        return

    specification, inputs, amqp_url = setup

    # Load the last configuration to resume
    last_config = load_last_grid_config()

    param_names = list(grid_dict.keys())
    param_values = list(grid_dict.values())

    from itertools import product
    param_combinations = list(product(*param_values))

    start_index = 0
    if last_config:
        last_values = [last_config[key] for key in param_names]
        if last_values in param_combinations:
            start_index = param_combinations.index(tuple(last_values)) + 1

    # Loop over each parameter combination starting from the saved position
    for i in range(start_index, len(param_combinations)):
        param_config = dict(zip(param_names, param_combinations[i]))

        logger.info(f"Running grid search with parameters: {param_config}")

        # Load default config and update with the current grid configuration
        config = Config()
        updated_config = update_config(config, param_config)

        # Save and run the experiment
        save_last_grid_config(param_config)
        await run_experiment_with_timeout(specification, inputs, updated_config, amqp_url, timeout_seconds)

    logger.info("Grid search completed.")

if __name__ == "__main__":
    grid_params = {
        "temperature": [0.9, 1.0, 1.1],
        "top_p": [0.6, 0.8, 0.9]
    }
    try:
        asyncio.run(perform_grid_search(grid_params, 1200))
    except Exception as e:
        logger.error(f"Grid search error: {e}")
        os._exit(1)
