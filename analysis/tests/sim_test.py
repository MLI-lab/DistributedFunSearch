import textwrap
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from checkpoint.similarity import compare_code_similarity_with_protection

# Test cases to evaluate similarity methods and highlight weaknesses
def test_similarity_approaches():
    # Test 1: Identical functions with different variable names
    code1 = textwrap.dedent("""
    def add_numbers(a, b):
        return a + b
    """)

    code2 = textwrap.dedent("""
    def add_numbers(x, y):
        return x + y
    """)

    # Test 2: Same structure but different variable and function names
    code3 = textwrap.dedent("""
    def multiply_values(val1, val2):
        return val1 * val2
    """)

    code4 = textwrap.dedent("""
    def compute_product(x, y):
        return x * y
    """)

    # Test 3: Reordered operations
    code5 = textwrap.dedent("""
    def calculate(a, b):
        sum_val = a + b
        product_val = a * b
        return sum_val, product_val
    """)

    code6 = textwrap.dedent("""
    def calculate(a, b):
        product_val = a * b
        sum_val = a + b
        return sum_val, product_val
    """)

    # Test 4: Slight structural differences
    code7 = textwrap.dedent("""
    def calculate(a, b):
        sum_val = a + b
        return sum_val
    """)

    code8 = textwrap.dedent("""
    def calculate(a, b):
        if a > 0:
            sum_val = a + b
        return sum_val
    """)

    # Test 5: Comparing polynomials with different powers and coefficients
    code9 = textwrap.dedent("""
    def polynomial(x):
        return 3 * x**2 + 2 * x + 1
    """)

    code10 = textwrap.dedent("""
    def polynomial(x):
        return 5 * x**3 + 4 * x**2 + 2
    """)

    # Test 6: Different functionality but same structure
    code11 = textwrap.dedent("""
    def count_evens(numbers):
        count = 0
        for num in numbers:
            if num % 2 == 0:
                count += 1
        return count
    """)

    code12 = textwrap.dedent("""
    def sum_numbers(numbers):
        total = 0
        for num in numbers:
            total += num
        return total
    """)

    # Protected variables (test for priority_vX)
    protected_vars = ['node', 'G', 'n', 's']

    # Similarity Tests
    print("\nTest 1: Identical functions with different variable names")
    print_similarity_scores(code1, code2, protected_vars)

    print("\nTest 2: Same structure but different variable and function names")
    print_similarity_scores(code3, code4, protected_vars)

    print("\nTest 3: Reordered operations")
    print_similarity_scores(code5, code6, protected_vars)

    print("\nTest 4: Slight structural differences")
    print_similarity_scores(code7, code8, protected_vars)

    print("\nTest 5: Comparing polynomials with different powers and coefficients")
    print_similarity_scores(code9, code10, protected_vars)

    print("\nTest 6: Different functionality but same structure")
    print_similarity_scores(code11, code12, protected_vars)


# Utility function to print similarity scores
def print_similarity_scores(code1, code2, protected_vars):
    similarity_results = compare_code_similarity_with_protection(code1, code2, protected_vars)
    print(f"String Similarity (with normalization): {similarity_results['String Similarity (with normalization)']:.2f}")
    print(f"Bag of AST Nodes Similarity: {similarity_results['Bag of AST Nodes Similarity']:.2f}")
    print(f"Tree Edit Distance: {similarity_results['Tree Edit Distance']}")


# Run the tests
if __name__ == "__main__":
    test_similarity_approaches()
