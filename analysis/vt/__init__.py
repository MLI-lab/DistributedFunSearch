"""
VT Code Tools

This package contains tools for working with Varshamov-Tenengolts (VT) codes:
- generate.py: Generate VT codes for any n and syndrome value a
- compare.py: Compare any codebook against VT codes
- subset.py: Analyze subset relationships between codebooks

VT codes are single-deletion-correcting codes. VT_a(n) is defined as:
    VT_a(n) = {x in {0,1}^n : sum_{i=1}^{n} i * x_i ≡ a (mod n+1)}
"""
