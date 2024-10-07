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
from itertools import product
from funsearch import initialize_task_manager
import aio_pika

# Set the base directory and grid search directory
BASE_DIR = os.getcwd()
GRID_SEARCH_DIR = os.path.join(BASE_DIR, "GridSearch")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "Checkpoints")
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
LAST_CONFIG_FILE_PATH = os.path.join(GRID_SEARCH_DIR, 'last_config_file.json')

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
    total_programs = checkpoint_data.get('total_programs', 0)
    best_scores_per_island = checkpoint_data.get('best_score_per_island', [])
    islands_state = checkpoint_data.get('islands_state', [])
    
    average_best_score = sum(best_scores_per_island) / len(best_scores_per_island) if best_scores_per_island else 0
    max_best_score = max(best_scores_per_island) if best_scores_per_island else 0
    total_islands_state_length = sum(len(island) for island in islands_state)
    
    overall_score = (registered_programs + average_best_score + total_islands_state_length + (2 * max_best_score)) / 5
    
    logger.info(f"Evaluation results - Total_programs: {total_programs}, Registered Programs: {registered_programs}, "
                f"Average Best Score: {average_best_score}, Max Best Score: {max_best_score}, "
                f"Total Number of Clusters: {total_islands_state_length}, Overall Score: {overall_score}")
    
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
            current_value = getattr(config, param)
            # Ensure the type of the new value matches the existing value's type
            if isinstance(current_value, type(value)):
                setattr(config, param, value)
            else:
                # Attempt to convert to float if type mismatch occurs for temperature or top_p
                if param in ['temperature', 'top_p']:
                    try:
                        setattr(config, param, float(value))
                        logger.warning(f"Type mismatch for {param}. Converted value to float: {value}")
                    except ValueError:
                        raise ValueError(f"Failed to convert {param} to float. Received: {value}")
                else:
                    raise ValueError(f"Incorrect type for parameter {param}. Expected {type(current_value)}, got {type(value)}.")
        else:
            raise ValueError(f"Parameter {param} does not exist in the Config class")
    return config


def load_last_grid_config():
    """Load the last grid configuration from the JSON file."""
    if os.path.exists(LAST_CONFIG_FILE_PATH):
        try:
            with open(LAST_CONFIG_FILE_PATH, 'r') as file:
                raw_content = file.read().strip()  # Read raw content and strip any extra whitespace
                if not raw_content:
                    logger.info("Last config file is empty, starting with the first configuration.")
                    return None  # Return None if the file is empty

                logger.info(f"Raw content of last_config_file.json: {raw_content}")  # Debug log
                
                # Now try loading the JSON content
                data = json.loads(raw_content)

                if isinstance(data, dict):
                    logger.info(f"Loaded last config: {data}")
                    return data  # Return the config as it is already a dictionary
                else:
                    logger.info("Last config file is not in the expected format, starting with the first configuration.")
                    return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {e} - Content: {raw_content}")  # Log the raw content on error
            return None
    else:
        logger.info("Last config file does not exist, starting with the first configuration.")
        return None


def save_only_last_grid_config(param_config):
    try:
        # Save only the current param_config (overwriting the previous one)
        with open(LAST_CONFIG_FILE_PATH, 'w') as file:
            json.dump(param_config, file)  # Write the current configuration to file
        logger.info(f"Successfully saved the last configuration to {LAST_CONFIG_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error saving the last configuration: {e}")

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

async def evaluate_checkpoint_periodically(interval_seconds):
    while True:
        await asyncio.sleep(interval_seconds)
        checkpoint_data = load_checkpoint(CHECKPOINT_DIR)
        if checkpoint_data:
            overall_score, metrics = evaluate_hyperparameters(checkpoint_data)
            logger.info(f"Evaluation during periodic check: {metrics}")
        else:
            logger.warning("No valid checkpoint data for periodic evaluation.")

async def run_experiment(config, param_config):
    task_manager = initialize_task_manager(config)
    task = asyncio.create_task(task_manager.run())
    await task

import re

def read_retry_configs(file_path):
    """Read specific combinations from retry_configs.txt file."""
    retry_configs = []
    pattern = r"\{.*\}"  # Regex pattern to match dictionary-like structures
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Strip whitespace and check if the line matches the pattern
                line = line.strip()
                if re.match(pattern, line):
                    # Extract temperature and top_p values from the line
                    temp_match = re.search(r"'temperature':\s*(\d+(\.\d+)?)", line)
                    top_p_match = re.search(r"'top_p':\s*(\d+(\.\d+)?)", line)
                    
                    if temp_match and top_p_match:
                        temp_value = float(temp_match.group(1))
                        top_p_value = float(top_p_match.group(1))
                        
                        # Append the extracted values as a dictionary
                        retry_configs.append({"temperature": temp_value, "top_p": top_p_value})
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    return retry_configs


async def perform_grid_search(grid_dict=None):
    # Load the last config
    try:
        last_config = load_last_grid_config()
    except Exception as e: 
        print(f"Error in load_last_grid_config {e}")

    # Grid params are set to None, so no param_combinations from the grid
    grid_param_combinations = []

    # Read retry configs from the file
    retry_configs_file = "/franziska/implementation/GridSearch/retry_configs.txt"
    try:
        specific_combinations = read_retry_configs(retry_configs_file)
    except Exception as e: 
        print(f"Error in read_retry_configs {e}")
    # Combine grid and specific combinations
    try: 
        all_param_combinations = grid_param_combinations + specific_combinations
    except Exception as e: 
        print(f"Error in all_param_combinations {e}")
    # Remove duplicates if any
    unique_param_combinations = [dict(t) for t in {tuple(d.items()) for d in all_param_combinations}]

    # Determine start index based on last_config or start with the first configuration
    start_index = 0  # Default to start from the first configuration
    if last_config:
        for idx, config in enumerate(unique_param_combinations):
            if config == last_config:
                start_index = idx + 1  # Continue from the last processed config
                break
    else:
        logger.info("No previous config found, starting from the first configuration.")

    # Iterate through all the combinations, starting from the last processed config or the first one
    for i in range(start_index, len(unique_param_combinations)):
        param_config = unique_param_combinations[i]
        config = config_lib.Config()  
        logger.info(f"Starting experiment with {param_config}")
        
        try:
            # Update the configuration with the parameter settings
            updated_config = update_config(config, param_config)

            # Save the last configuration (for resuming if interrupted)
            save_last_grid_config(param_config)
            save_only_last_grid_config(param_config)

            # Run the experiment with the updated configuration
            await run_experiment(updated_config, param_config)

        except Exception as e:
            logger.error(f"Error during experiment execution: {e}")
            raise e

    logger.info("Grid search completed.")

# Updated main function
async def main():
    grid_params = None  # Set grid params to None as all grids have been performed
    
    # Start the grid search and the periodic evaluation in parallel
    await asyncio.gather(
        perform_grid_search(grid_params),
        evaluate_checkpoint_periodically(3610)  # Run evaluation every 310 seconds
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        os._exit(1)
