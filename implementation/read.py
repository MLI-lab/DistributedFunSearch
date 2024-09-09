import pickle
import pprint  
import time

# Load the checkpoint data
filepath = "/franziska/implementation/checkpoint.pkl"
with open(filepath, "rb") as f:
    checkpoint_data = pickle.load(f)

# Convert the '_last_reset_time' if it exists
if 'last_reset_time' in checkpoint_data:
    checkpoint_data['last_reset_time'] = time.ctime(checkpoint_data['last_reset_time'])

# List of keys you want to print
keys_to_print = [
    "registered_programs", 
    "total_programs", 
    "execution_failed",
    "best_score_per_island",
    "best_program_per_island", 
    "best_scores_per_test_per_island",
    "last_reset_time"
]

# Print only the specified keys
for key in keys_to_print:
    if key in checkpoint_data:
        print(f"{key}:")
        pprint.pprint(checkpoint_data[key])
        print("\n")  # Add a newline for better readability

# Optionally, print a confirmation message
print("Selected checkpoint data printed to the terminal.")
