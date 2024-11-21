import ast
import re
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher
import zss  # For tree edit distance
from code_manipulation import Function

# Function to remove comments and docstrings
def remove_docstrings_and_comments(source_code):
    source_code = re.sub(r'#.*', '', source_code)
    source_code = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', source_code)
    return source_code

# Normalizer class with selective variable replacement and default protected vars
class CodeNormalizerSelective(ast.NodeTransformer):
    def __init__(self, protected_vars=['node', 'G', 'n', 's']):
        self.var_count = 0
        self.func_count = 0
        self.class_count = 0
        self.var_names = {}
        self.func_names = {}
        self.class_names = {}
        self.protected_vars = protected_vars  # Default protected variable names

    def visit_Name(self, node):
        # Replace variable names if they are not in the protected list
        if node.id not in self.protected_vars:
            if node.id not in self.var_names:
                self.var_count += 1
                self.var_names[node.id] = f"var_{self.var_count}"
            node.id = self.var_names[node.id]
        return node

    def visit_FunctionDef(self, node):
        # Replace function names uniformly
        if node.name not in self.func_names:
            self.func_count += 1
            self.func_names[node.name] = f"func_{self.func_count}"
        node.name = self.func_names[node.name]
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        # Replace class names uniformly
        if node.name not in self.class_names:
            self.class_count += 1
            self.class_names[node.name] = f"class_{self.class_count}"
        node.name = self.class_names[node.name]
        self.generic_visit(node)
        return node

# Function to normalize the code by replacing variable, function, and class names
def normalize_code_selective(source_code, protected_vars, return_ast=False):
    clean_code = remove_docstrings_and_comments(source_code)
    tree = ast.parse(clean_code)
    normalizer = CodeNormalizerSelective(protected_vars)
    normalizer.visit(tree)
    return tree if return_ast else ast.unparse(tree)  # Return AST tree or string based on return_ast

# ------------------------- String Similarity -------------------------

def calculate_string_similarity_with_protection(code1, code2, protected_vars):
    normalized_code1 = normalize_code_selective(code1, protected_vars).strip()
    normalized_code2 = normalize_code_selective(code2, protected_vars).strip()
    
    # Strip extra spaces and newlines for accurate comparison
    normalized_code1 = " ".join(normalized_code1.split())
    normalized_code2 = " ".join(normalized_code2.split())
    
    # Use SequenceMatcher to compare normalized strings
    return SequenceMatcher(None, normalized_code1, normalized_code2).ratio()

# ------------------------- Bag of AST Nodes Similarity -------------------------

def ast_node_frequency(source_code, protected_vars):
    tree = normalize_code_selective(source_code, protected_vars, return_ast=True)
    node_types = [type(node).__name__ for node in ast.walk(tree)]  # Get the type of each node
    return Counter(node_types)

def cosine_similarity_dicts(dict1, dict2):
    keys = set(dict1.keys()).union(set(dict2.keys()))
    vec1 = np.array([dict1.get(key, 0) for key in keys])
    vec2 = np.array([dict2.get(key, 0) for key in keys])
    return cosine_similarity([vec1], [vec2])[0][0]

def compare_code_bag_of_nodes(code1, code2, protected_vars):
    freq1 = ast_node_frequency(code1, protected_vars)
    freq2 = ast_node_frequency(code2, protected_vars)
    return cosine_similarity_dicts(freq1, freq2)

# ------------------------- Tree Edit Distance -------------------------

def get_children(node):
    return node.children

def insert_cost(node):
    return 1

def remove_cost(node):
    return 1

def update_cost(node1, node2):
    return 0 if node1.label == node2.label else 1

def ast_to_zss_tree(node):
    return zss.Node(type(node).__name__, children=[ast_to_zss_tree(child) for child in ast.iter_child_nodes(node)])

def compute_tree_edit_distance(code1, code2, protected_vars):
    tree1 = ast_to_zss_tree(normalize_code_selective(code1, protected_vars, return_ast=True))
    tree2 = ast_to_zss_tree(normalize_code_selective(code2, protected_vars, return_ast=True))
    return zss.distance(tree1, tree2, get_children, insert_cost, remove_cost, update_cost)

# ------------------------- Combine All Similarity Measures -------------------------

def compare_code_similarity_with_protection(code1, code2, protected_vars=['node', 'G', 'n', 's']):
    string_similarity = calculate_string_similarity_with_protection(code1, code2, protected_vars)
    bag_of_nodes_similarity = compare_code_bag_of_nodes(code1, code2, protected_vars)
    tree_edit_distance = compute_tree_edit_distance(code1, code2, protected_vars)

    
def compare_one_code_similarity_with_protection(code1, code2, similarity_type, protected_vars=['node', 'G', 'n', 's']):
    """
    Compares two code snippets or Function representations using the specified similarity measure.

    Parameters:
    - code1: The first code snippet (string or dict representing a Function).
    - code2: The second code snippet (string or dict representing a Function).
    - similarity_type: The type of similarity measure to use. 
                       Options are "string", "bag_of_nodes", or "tree_edit_distance".
    - protected_vars: List of variable names that should be protected from renaming during comparison.

    Returns:
    - The similarity score based on the chosen similarity measure, or a dictionary of all similarity scores.
    """
    # If code1 or code2 is a dictionary, convert to a properly formatted function string
    if isinstance(code1, dict):
        function_obj1 = Function.from_dict(code1)  # Create a Function instance from the dictionary
        code1 = str(function_obj1)  # Use the __str__ method to format it as a string

    if isinstance(code2, dict):
        function_obj2 = Function.from_dict(code2)  # Create a Function instance from the dictionary
        code2 = str(function_obj2)  # Use the __str__ method to format it as a string

    # Calculate all similarity measures
    string_similarity = calculate_string_similarity_with_protection(code1, code2, protected_vars)
    bag_of_nodes_similarity = compare_code_bag_of_nodes(code1, code2, protected_vars)
    tree_edit_distance = compute_tree_edit_distance(code1, code2, protected_vars)

    # Return the selected similarity measure or all results in a dictionary
    if similarity_type == "string":
        return string_similarity
    elif similarity_type == "bag_of_nodes":
        return bag_of_nodes_similarity
    elif similarity_type == "tree_edit_distance":
        return tree_edit_distance
    elif similarity_type == "all":
        # Return a dictionary containing all similarity measures
        return {
            "String Similarity (with normalization)": string_similarity,
            "Bag of AST Nodes Similarity": bag_of_nodes_similarity,
            "Tree Edit Distance": tree_edit_distance
        }
    else:
        raise ValueError(f"Invalid similarity type: {similarity_type}. Choose 'string', 'bag_of_nodes', 'tree_edit_distance', or 'all'.")
