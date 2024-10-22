import networkx as nx
import matplotlib.pyplot as plt
import itertools

# Your previously defined functions
def generate_graph(n, s):
    G = nx.Graph()
    sequences = [''.join(seq) for seq in itertools.product('01', repeat=n)]  # Generate all binary strings of length n
    # Adding nodes
    for seq in sequences:
        G.add_node(seq)
    # Adding edges
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if has_common_subsequence(sequences[i], sequences[j], n, s):
                G.add_edge(sequences[i], sequences[j])
    return G

def has_common_subsequence(seq1, seq2, n, s):
    threshold = n - s
    if threshold <= 0:
        return True  # Trivial case where subsequence length is 0 or negative
    # Initialize two rows for DP
    prev = [0] * (n + 1)
    current = [0] * (n + 1)
    # Fill the DP table row by row
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                current[j] = prev[j - 1] + 1
            else:
                current[j] = max(prev[j], current[j - 1])
            if current[j] >= threshold:
                return True
        prev, current = current, prev
    return False  # No LCS of adequate length was found

# Visualization function with smaller nodes and font size
def visualize_graph(G, title="Graph Visualization", filename="graph.pdf"):
    plt.figure(figsize=(10, 8))
    
    # Layout for the nodes
    pos = nx.spring_layout(G)  # Positions the nodes using a force-directed algorithm

    # Draw the graph with smaller node size and font size
    nx.draw(G, pos, with_labels=True, node_color="lightblue", 
            node_size=50,  # Smaller node size
            font_size=0,     # Smaller font size
            font_weight="bold", 
            edge_color="gray")
    
    # Set the title
    plt.title(title)
    
    # Save the graph as a PDF
    plt.savefig(filename, format="pdf")
    plt.close()  # Close the plot to prevent it from displaying immediately

    print(f"Graph saved as {filename}")

# Parameters for n and s
n = 11  # Length of binary sequences
s = 1  # Number of deletions

# Generate the graph based on n and s
G = generate_graph(n, s)

# Save the graph as a PDF
visualize_graph(G, title=f"{n}-bit, {s}-deletion correcting set", filename="n_bit_s_deletion_graph.pdf")
