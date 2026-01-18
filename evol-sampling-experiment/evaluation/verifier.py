# evaluation/verifier.py
"""
Simple binary verification: does model answer match ground truth?

Keep it simple:
1. Extract answer from model response (look for \\boxed{})
2. Normalize both answers
3. Compare (exact match, then sympy as fallback)
"""

import re
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} in model output.

    Handles nested braces properly.
    """
    if text is None:
        return None

    # Handle nested braces by matching balanced braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """Basic normalization of math answers."""
    if answer is None:
        return ""

    # Strip whitespace
    answer = answer.strip()

    # Remove dollar signs (sometimes used for math mode)
    answer = answer.replace("$", "")

    # Normalize fractions: \frac{a}{b} -> (a)/(b)
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', answer)

    # Remove \left and \right
    answer = answer.replace("\\left", "").replace("\\right", "")

    # Remove common LaTeX commands that don't affect the value
    answer = answer.replace("\\cdot", "*")
    answer = answer.replace("\\times", "*")
    answer = answer.replace("\\div", "/")

    # Normalize spacing
    answer = " ".join(answer.split())

    return answer


def try_numeric_comparison(pred: str, gt: str) -> Optional[bool]:
    """Try to compare answers numerically."""
    try:
        # Replace ^ with ** for exponentiation
        pred_expr = pred.replace("^", "**")
        gt_expr = gt.replace("^", "**")

        pred_float = float(eval(pred_expr))
        gt_float = float(eval(gt_expr))

        return abs(pred_float - gt_float) < 1e-6
    except:
        return None


def try_sympy_comparison(pred: str, gt: str) -> Optional[bool]:
    """Try to compare answers symbolically using sympy."""
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        # Try direct sympify first
        try:
            pred_sym = sympy.sympify(pred)
            gt_sym = sympy.sympify(gt)
            if sympy.simplify(pred_sym - gt_sym) == 0:
                return True
        except:
            pass

        # Try as expressions
        try:
            pred_sym = sympy.sympify(pred.replace("^", "**"))
            gt_sym = sympy.sympify(gt.replace("^", "**"))
            if sympy.simplify(pred_sym - gt_sym) == 0:
                return True
        except:
            pass

        return False
    except ImportError:
        return None
    except:
        return None


def verify_answer(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth.

    Simple approach:
    1. Try exact string match (after normalization)
    2. Try numeric comparison (handles "0.5" vs "1/2")
    3. Try sympy symbolic equality as fallback
    """
    if predicted is None:
        return False

    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Empty predictions are wrong
    if not pred_norm:
        return False

    # Exact match
    if pred_norm == gt_norm:
        return True

    # Try numeric comparison
    numeric_result = try_numeric_comparison(pred_norm, gt_norm)
    if numeric_result is True:
        return True

    # Try sympy symbolic equality
    sympy_result = try_sympy_comparison(pred_norm, gt_norm)
    if sympy_result is True:
        return True

    return False


def evaluate_solution(solution: str, ground_truth: str) -> bool:
    """Evaluate a single solution against ground truth."""
    predicted = extract_boxed_answer(solution)
    if predicted is None:
        return False
    return verify_answer(predicted, ground_truth)


def evaluate_solutions(solutions: list[str], ground_truth: str) -> list[bool]:
    """Evaluate a batch of solutions."""
    return [evaluate_solution(sol, ground_truth) for sol in solutions]


if __name__ == "__main__":
    # Quick tests
    print("Testing verifier...")

    # Test answer extraction
    test_cases = [
        ("The answer is \\boxed{42}", "42"),
        ("So we get \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("\\boxed{x^2 + 1}", "x^2 + 1"),
        ("No boxed answer here", None),
    ]

    print("\n--- Answer Extraction ---")
    for text, expected in test_cases:
        result = extract_boxed_answer(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{text[:40]}...' -> '{result}' (expected '{expected}')")

    # Test verification
    verify_cases = [
        ("42", "42", True),
        ("0.5", "1/2", True),
        ("\\frac{1}{2}", "0.5", True),
        ("2", "3", False),
        ("x^2", "x^2", True),
    ]

    print("\n--- Verification ---")
    for pred, gt, expected in verify_cases:
        result = verify_answer(pred, gt)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{pred}' vs '{gt}' -> {result} (expected {expected})")
