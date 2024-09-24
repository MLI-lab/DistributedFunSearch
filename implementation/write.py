import pickle
import pprint  
import time

# Load the checkpoint data
filepath = "/franziska/implementation/Checkpoints/checkpoint_2024-09-16_15-01-06.pkl"
with open(filepath, "rb") as f:
    checkpoint_data = pickle.load(f)

# Convert the '_last_reset_time' if it exists
if 'last_reset_time' in checkpoint_data:
    checkpoint_data['last_reset_time'] = time.ctime(checkpoint_data['last_reset_time'])

# Write the entire checkpoint data to a text file
output_filepath = "/franziska/implementation/checkpoint_full.txt"
with open(output_filepath, "w") as output_file:
    pprint.pprint(checkpoint_data, stream=output_file)

print(f"Entire checkpoint data written to {output_filepath}")
