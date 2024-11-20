import os
import pickle

# Paths
checkpoint_folder = "/franziska/Funsearch/Analysis/Checkpoints_T0.1"
preprocessed_output = "preprocessed_programs.py"
error_log = "preprocessing_errors.log"

# Check if last position in a cluster signature is 172
def has_target_signature(cluster_signature, target=172):
    try:
        signature_tuple = tuple(map(int, cluster_signature.strip("()").split(",")))
        return signature_tuple[-1] == target
    except ValueError:
        return False

# Extract and preprocess programs
with open(preprocessed_output, "w") as f, open(error_log, "w") as log:
    f.write("# Preprocessed programs with last signature 172\n\n")
    for filename in os.listdir(checkpoint_folder):
        if filename.endswith(".pkl"):
            checkpoint_path = os.path.join(checkpoint_folder, filename)
            with open(checkpoint_path, "rb") as checkpoint_file:
                checkpoint_data = pickle.load(checkpoint_file)

            for island in checkpoint_data.get('islands_state', []):
                for cluster_signature, cluster_data in island.get('clusters', {}).items():
                    if has_target_signature(cluster_signature):
                        for program in cluster_data['programs']:
                            try:
                                # Write the program name, arguments, and body directly
                                f.write(f"def {program['name']}({program['args']}):\n")
                                f.write(program['body'] + "\n\n")  # Write the body as is
                            except Exception as e:
                                # Log the error
                                log.write(f"Error processing program: {e}\nProgram:\n{program}\n{'=' * 50}\n")

print(f"Preprocessed programs written to {preprocessed_output}")
print(f"Errors logged to {error_log}")
