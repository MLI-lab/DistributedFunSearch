# Enc_Dec: Encoder/Decoder Evaluation for Deletion-Correcting Codes

This specification evaluates encoder/decoder validity for deletion-correcting codes parameterized by weight functions and moduli.

## Background

### Parameterization

**Encoder:**
- Uses r constraints, each with weight function f_j(i) and modulus m_j
- Parity positions P ⊆ {0, ..., n-1}
- Constraint: Σ x_i · f_j(i) ≡ σ_j (mod m_j)

**Decoder:**
- Matrix A ∈ Z^{r×n} with columns (f_0(i), ..., f_{r-1}(i))^T
- Deletion pattern v ∈ {0,1}^n where v_i = deleted bit value at position i
- Valid if all Av are distinct for different deletion patterns

### VT Codes

Varshamov-Tenengolts codes are a special case:
- r = 1 (single constraint)
- f(i) = i + 1 (1-indexed equivalent)
- m = n + 1

VT codes can correct **single deletions** (s=1) but fail for s≥2.

## Usage

### Quick Test

```bash
cd tests
python3 test_vt_codes.py
```

### Using the Evaluation Functions

```python
from evaluation.encoder_decoder import (
    vt_code,
    check_decoder_validity,
    check_decoder_validity_refined,
    check_encoder_validity
)

# Get VT code parameters for n=8
weight_funcs, moduli = vt_code(8)

# Check decoder validity for s=1 deletion
result = check_decoder_validity_refined(n=8, s=1, weight_funcs=weight_funcs, moduli=moduli)
print(f"Valid: {result['valid']}")
print(f"Dangerous collisions: {result['dangerous_collisions']}")
print(f"Benign collisions: {result['benign_collisions']}")

# Check for s=2 deletions (will fail)
result = check_decoder_validity_refined(n=8, s=2, weight_funcs=weight_funcs, moduli=moduli)
print(f"Valid: {result['valid']}")
print(f"Collision details: {result['dangerous_details'][:3]}")
```

### Custom Weight Functions

```python
# Define custom weight functions (example: f_0(i) = i+1, f_1(i) = (i+1)^2)
weight_funcs = [
    lambda i: i + 1,
    lambda i: (i + 1) ** 2
]
moduli = [n + 1, n * n + 1]

result = check_decoder_validity(n=10, s=2, weight_funcs=weight_funcs, moduli=moduli)
```

## Key Functions

### `check_decoder_validity(n, s, weight_funcs, moduli)`

Checks if all deletion patterns have distinct syndrome differences (Av).

Returns:
- `valid`: True if no collisions
- `unique_fraction`: Fraction of unique Av values
- `collisions`: List of (Av, patterns) pairs

### `check_decoder_validity_refined(n, s, weight_funcs, moduli)`

Refined check that separates:
- **Dangerous collisions**: Patterns with same Av where at least one deletes a 1-bit
- **Benign collisions**: Zero-only patterns (handled by shift correction)

### `check_encoder_validity(n, parity_positions, weight_funcs, moduli)`

Checks if parity bit assignments can achieve all required syndromes.

### `vt_code(n)`

Returns VT code parameters: f(i) = i + 1, m = n + 1

## Understanding Collisions

### Benign vs Dangerous

The paper notes that patterns deleting only zeros have Av=0 but are distinguishable via the shift correction term in the full syndrome difference Δ.

- **Benign**: All patterns in collision group delete only 0s → handled by shift
- **Dangerous**: At least one pattern deletes a 1-bit → true collision, breaks decoding

### Expected Results

| n   | s=1 | s=2 |
|-----|-----|-----|
| 4-12 | ✓   | ✗   |

VT codes pass for s=1 (0 dangerous collisions), fail for s≥2.

## File Structure

```
Enc_Dec/
├── evaluation/
│   └── encoder_decoder.py   # Core evaluation functions
├── tests/
│   └── test_vt_codes.py     # VT code verification
└── README.md
```

## Complexity

- Encoder check: O(2^|P|)
- Decoder check: O(C(n,s) · 2^s) = O(n^s · 2^s)
