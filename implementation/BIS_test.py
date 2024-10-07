import os
import pickle
import numpy as np
from similarity import compare_one_code_similarity_with_protection

def load_checkpoint(timestamp_index=None):
    """
    Load a checkpoint file. If timestamp_index is None, load the latest checkpoint.
    """
    # Get list of checkpoint files
    checkpoint_dir = "Checkpoints"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pkl')]
    checkpoint_files.sort()  # Ensure the files are sorted by timestamp

    if not checkpoint_files:
        print("No checkpoint files found.")
        return None

    if timestamp_index is None:
        # Load the latest checkpoint
        checkpoint_file = checkpoint_files[-1]
    else:
        if 0 <= timestamp_index < len(checkpoint_files):
            checkpoint_file = checkpoint_files[timestamp_index]
        else:
            print(f"Invalid timestamp index: {timestamp_index}")
            return None

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)
    print(f"Loaded checkpoint: {checkpoint_file}")
    return checkpoint_data

def compute_island_similarities(checkpoint_data, similarity_type='bag_of_nodes', protected_vars=[]):
    """
    Compute similarities between islands and print the top similarity scores.
    """
    islands_data = checkpoint_data.get('islands_state', [])
    num_islands = len(islands_data)
    if num_islands == 0:
        print("No island data found in the checkpoint.")
        return

    # Loop through each pair of islands
    for idx_a in range(num_islands):
        island_data_a = islands_data[idx_a]
        programs_a = []
        cluster_a_signatures = []

        # Collect all programs from island A
        for cluster_a_key, cluster_a in island_data_a.get('clusters', {}).items():
            print(cluster_a_key)
            programs = cluster_a.get('programs', [])
            programs_a.extend(programs)
            cluster_a_signatures.extend([cluster_a_key] * len(programs))

        if not programs_a:
            print(f"No programs found in Island {idx_a + 1}. Skipping.")
            continue

        for idx_b in range(idx_a + 1, num_islands):  # Avoid duplicate pairs
            island_data_b = islands_data[idx_b]
            programs_b = []
            cluster_b_signatures = []

            # Collect all programs from island B
            for cluster_b_key, cluster_b in island_data_b.get('clusters', {}).items():
                programs = cluster_b.get('programs', [])
                programs_b.extend(programs)
                cluster_b_signatures.extend([cluster_b_key] * len(programs))

            if not programs_b:
                print(f"No programs found in Island {idx_b + 1}. Skipping.")
                continue

            # Compute similarity between all programs in island A and B
            similarity_scores = []
            for i, prog_a in enumerate(programs_a):
                for j, prog_b in enumerate(programs_b):
                    try:
                        similarity = compare_one_code_similarity_with_protection(
                            prog_a, prog_b, similarity_type, protected_vars
                        )
                        print(similarity)
                        similarity_scores.append((similarity, idx_a + 1, idx_b + 1, cluster_a_signatures[i], cluster_b_signatures[j]))
                    except Exception as e:
                        print(f"Error comparing programs: {e}")
                        continue

            if not similarity_scores:
                print(f"No similarity scores computed between Island {idx_a + 1} and Island {idx_b + 1}.")
                continue

            # Sort the similarity scores to find the top 10 most similar program pairs
            similarity_scores.sort(reverse=True, key=lambda x: x[0])  # Sort by similarity score (highest first)
            top_10_similarities = similarity_scores[:10]  # Keep only the top 10 similarities

            # Print the top similarities
            print(f"\nTop similarities between Island {idx_a + 1} and Island {idx_b + 1}:")
            for sim in top_10_similarities:
                similarity, island_a_num, island_b_num, cluster_a_key, cluster_b_key = sim
                print(f"Similarity: {similarity:.4f}, Island {island_a_num} Cluster '{cluster_a_key}' vs Island {island_b_num} Cluster '{cluster_b_key}'")

def main():
    # Load the latest checkpoint
    checkpoint_data = load_checkpoint() # no time stap so will load the last one 
    print(checkpoint_data)

    if checkpoint_data is None:
        return

    # Define the similarity type and protected variables
    similarity_type = 'bag_of_nodes'  # Replace with your desired similarity type
    protected_vars = ['node', 'G', 'n', 's']  # Replace with your protected variables

    # Compute and print similarities between islands
    compute_island_similarities(checkpoint_data, similarity_type, protected_vars)

if __name__ == "__main__":
    main()
