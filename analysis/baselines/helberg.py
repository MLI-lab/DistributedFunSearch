"""
Helberg-Ferreira Multiple Insertion/Deletion Correcting Codes
=============================================================

Implementation based on:
"On Multiple Insertion/Deletion Correcting Codes"
Albertus S. J. Helberg and Hendrik C. Ferreira
IEEE Trans. Inform. Theory, Vol. 48, No. 1, January 2002

This module implements the generalized number-theoretic construction
for codes that can correct s random insertion and/or deletion errors.

Key Concepts:
-------------
1. Codeword Moment: For a binary sequence x = x_1 x_2 ... x_n,
   the moment is Σ v_i * x_i where v is the weight vector.

2. Weight Vector (v): A modified Fibonacci-like sequence where
   v_j = 1 + Σ_{k=1}^{s} v_{j-k}
   
3. Modulus (u): u = 1 + Σ_{j=0}^{s-1} v_{n-j}

4. Code Construction: A codeword x is valid if:
   Σ v_i * x_i ≡ a (mod u)
   for some fixed integer a.

For s=1, this reduces to the classic Levenshtein codes.
For s=2, the weight vector becomes: 1, 2, 4, 7, 12, 20, 33, ...

Usage:
------
Run directly for a cardinality table with n up to 20 and s in {2,3,4,5}:

    python helberg.py

For custom n and s values:

    from helberg import HelbergFerreiraCode, compute_cardinality_table

    code = HelbergFerreiraCode(n=12, s=3)
    print(code.analyze()['max_cardinality'])

    # Or compute a table
    compute_cardinality_table(max_n=25, s_values=[1, 2, 3])

Note: enumerates all 2^n sequences, so n above ~25 will be slow.
"""

import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Set
import time


class HelbergFerreiraCode:
    """
    Implementation of Helberg-Ferreira multiple insertion/deletion correcting codes.
    
    Attributes:
        n: Length of codewords
        s: Number of insertion/deletion errors to correct
        v: Weight vector
        u: Modulus
        partitions: Dictionary mapping residue a -> list of codewords
    """
    
    def __init__(self, n: int, s: int):
        """
        Initialize the code construction.
        
        Args:
            n: Length of codewords
            s: Number of insertion/deletion errors to correct
        """
        self.n = n
        self.s = s
        self.v = self._compute_weight_vector()
        self.u = self._compute_modulus()
        self.partitions = self._partition_sequences()
    
    def _compute_weight_vector(self) -> List[int]:
        """
        Compute the weight vector v using the recurrence:
        v_j = 1 + sum(v_{j-1}, v_{j-2}, ..., v_{j-s})
        v_i = 0 for i <= 0
        
        Returns:
            List [v_1, v_2, ..., v_n]
        """
        v = [0] * (self.n + 1)  # 1-indexed
        
        for j in range(1, self.n + 1):
            v[j] = 1
            for k in range(1, self.s + 1):
                if j - k >= 1:
                    v[j] += v[j - k]
        
        return v[1:]  # Return v_1 to v_n (0-indexed as list)
    
    def _compute_modulus(self) -> int:
        """
        Compute the modulus u = 1 + sum(v_{n-j} for j in 0 to s-1)
        
        Returns:
            Modulus u
        """
        u = 1
        for j in range(self.s):
            if self.n - 1 - j >= 0:
                u += self.v[self.n - 1 - j]
        return u
    
    def compute_moment(self, x: int) -> int:
        """
        Compute the codeword moment: sum(v_i * x_i)
        
        Args:
            x: Integer representation of binary codeword
        
        Returns:
            Codeword moment
        """
        moment = 0
        for i in range(self.n):
            bit = (x >> (self.n - 1 - i)) & 1
            moment += self.v[i] * bit
        return moment
    
    def compute_residue(self, x: int) -> int:
        """
        Compute the residue class: moment mod u
        
        Args:
            x: Integer representation of binary codeword
        
        Returns:
            Residue a = moment mod u
        """
        return self.compute_moment(x) % self.u
    
    def _partition_sequences(self) -> Dict[int, List[int]]:
        """
        Partition all 2^n binary sequences by their residue class.
        
        Returns:
            Dictionary mapping a -> list of codewords with residue a
        """
        partitions = defaultdict(list)
        
        for x in range(2**self.n):
            a = self.compute_residue(x)
            partitions[a].append(x)
        
        return dict(partitions)
    
    def get_codebook(self, a: int) -> List[int]:
        """
        Get the codebook for residue class a.
        
        Args:
            a: Residue class (0 <= a < u)
        
        Returns:
            List of codewords in the codebook
        """
        return self.partitions.get(a, [])
    
    def get_max_cardinality_codebook(self) -> Tuple[int, List[int]]:
        """
        Find the codebook with maximum cardinality.
        
        Returns:
            Tuple of (best residue a, list of codewords)
        """
        best_a = max(self.partitions.keys(), key=lambda a: len(self.partitions[a]))
        return best_a, self.partitions[best_a]
    
    def encode(self, message_bits: int, a: int = 0) -> Optional[int]:
        """
        Simple encoding: find a codeword in codebook a that matches
        the message (if small enough).
        
        For practical use, lookup tables or systematic encoding would be needed.
        
        Args:
            message_bits: Message as integer
            a: Residue class to use
        
        Returns:
            Codeword if message_bits < |C_a|, else None
        """
        codebook = self.get_codebook(a)
        if message_bits < len(codebook):
            return codebook[message_bits]
        return None
    
    @staticmethod
    def int_to_binary(x: int, n: int) -> str:
        """Convert integer to binary string of length n."""
        return format(x, f'0{n}b')
    
    @staticmethod
    def binary_to_int(binary_str: str) -> int:
        """Convert binary string to integer."""
        return int(binary_str, 2)
    
    def print_code_table(self):
        """Print the complete code table."""
        print(f"\nCode Table for n={self.n}, s={self.s}")
        print(f"Weight vector v = {self.v}")
        print(f"Modulus u = {self.u}")
        print("-" * 60)
        print(f"{'x':^{self.n+4}} | {'weight':^6} | {'moment':^8} | {'residue':^8}")
        print("-" * 60)
        
        for x in range(2**self.n):
            binary = self.int_to_binary(x, self.n)
            weight = bin(x).count('1')
            moment = self.compute_moment(x)
            residue = moment % self.u
            print(f"{binary:^{self.n+4}} | {weight:^6} | {moment:^8} | {residue:^8}")
    
    def analyze(self) -> Dict:
        """
        Analyze code properties.
        
        Returns:
            Dictionary with analysis results
        """
        best_a, codebook = self.get_max_cardinality_codebook()
        
        # Compute minimum Hamming distance
        min_dist = float('inf')
        if len(codebook) > 1:
            for i, cw1 in enumerate(codebook):
                for cw2 in codebook[i+1:]:
                    d = bin(cw1 ^ cw2).count('1')
                    min_dist = min(min_dist, d)
        
        # Weight distribution
        weights = defaultdict(int)
        for cw in codebook:
            w = bin(cw).count('1')
            weights[w] += 1
        
        card = len(codebook)
        rate = np.log2(card) / self.n if card > 1 else 0
        
        return {
            'n': self.n,
            's': self.s,
            'max_cardinality': card,
            'best_residue': best_a,
            'modulus': self.u,
            'weight_vector': self.v,
            'min_hamming_distance': min_dist if min_dist != float('inf') else None,
            'weight_distribution': dict(weights),
            'rate': rate,
            'codewords': [self.int_to_binary(cw, self.n) for cw in codebook]
        }


def compute_cardinality_table(max_n: int, s_values: List[int], 
                               verbose: bool = True) -> Dict[Tuple[int, int], Dict]:
    """
    Compute maximum cardinalities for all (n, s) combinations.
    
    Args:
        max_n: Maximum codeword length
        s_values: List of s values to compute
        verbose: Print progress
    
    Returns:
        Dictionary mapping (n, s) -> analysis results
    """
    results = {}
    
    for s in s_values:
        if verbose:
            print(f"\nComputing for s = {s}:")
        
        min_n = s + 2  # Minimum viable n
        
        for n in range(min_n, max_n + 1):
            start = time.time()
            
            code = HelbergFerreiraCode(n, s)
            analysis = code.analyze()
            analysis['time'] = time.time() - start
            
            results[(n, s)] = analysis
            
            if verbose:
                print(f"  n={n:2d}: |C|={analysis['max_cardinality']:6d}, "
                      f"u={analysis['modulus']:8d}, "
                      f"R={analysis['rate']:.4f}, "
                      f"d_min={analysis['min_hamming_distance']}, "
                      f"time={analysis['time']:.2f}s")
    
    return results


def print_cardinality_table(results: Dict, s_values: List[int], max_n: int):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("MAXIMUM CARDINALITIES FOR s-INSERTION/DELETION-CORRECTING CODES")
    print("=" * 80)
    
    header = f"{'n':^6}"
    for s in s_values:
        header += f" | {'s='+str(s):^12}"
    print(header)
    print("-" * len(header))
    
    for n in range(4, max_n + 1):
        row = f"{n:^6}"
        for s in s_values:
            if (n, s) in results:
                card = results[(n, s)]['max_cardinality']
                row += f" | {card:^12}"
            else:
                row += f" | {'-':^12}"
        print(row)


def verify_code_construction(n: int, s: int, verbose: bool = True) -> bool:
    """
    Verify that the construction produces valid s-correcting codes
    by checking the subword condition.
    
    The condition is: for any two codewords x, y:
    |δ(x, y)| < n - s
    where δ(x, y) is the largest common subword.
    
    Equivalently: after up to s deletions from each codeword,
    no common sequences should exist.
    
    Args:
        n: Codeword length
        s: Number of errors
        verbose: Print details
    
    Returns:
        True if verification passes
    """
    code = HelbergFerreiraCode(n, s)
    best_a, codebook = code.get_max_cardinality_codebook()
    
    if verbose:
        print(f"\nVerifying n={n}, s={s}: |C|={len(codebook)}, a={best_a}")
    
    if len(codebook) <= 1:
        return True
    
    def get_subwords(x: int, n: int, deletions: int) -> Set[Tuple[int, int]]:
        """Get all (subword, length) pairs after exactly 'deletions' deletions."""
        if deletions == 0:
            return {(x, n)}
        
        result = set()
        binary = format(x, f'0{n}b')
        
        for positions in combinations(range(n), deletions):
            subword = ''.join(binary[i] for i in range(n) if i not in positions)
            new_len = n - deletions
            if subword:
                result.add((int(subword, 2), new_len))
            else:
                result.add((0, 0))
        
        return result
    
    # Check all pairs
    for i, cw1 in enumerate(codebook):
        for cw2 in codebook[i+1:]:
            # For each pair, check if they could be confused
            # after d1 and d2 deletions where d1 + d2 <= 2s
            for d1 in range(s + 1):
                sw1 = get_subwords(cw1, n, d1)
                for d2 in range(s + 1):
                    if d1 + d2 > 2 * s:
                        continue
                    sw2 = get_subwords(cw2, n, d2)
                    
                    common = sw1 & sw2
                    if common:
                        if verbose:
                            print(f"  FAIL: {format(cw1, f'0{n}b')} and {format(cw2, f'0{n}b')} "
                                  f"share subword after {d1}+{d2} deletions")
                        return False
    
    if verbose:
        print("  PASS: All codeword pairs are distinguishable")
    return True


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("HELBERG-FERREIRA MULTIPLE INSERTION/DELETION CORRECTING CODES")
    print("=" * 70)
    
    # Example 1: Reproduce paper's example (n=4, s=2)
    print("\n[Example 1] Paper's example: n=4, s=2")
    print("-" * 40)
    code = HelbergFerreiraCode(4, 2)
    code.print_code_table()
    
    analysis = code.analyze()
    print(f"\nBest codebook (a={analysis['best_residue']}):")
    print(f"  Codewords: {analysis['codewords']}")
    print(f"  Cardinality: {analysis['max_cardinality']}")
    print(f"  Min Hamming distance: {analysis['min_hamming_distance']}")
    
    # Example 2: Larger code
    print("\n[Example 2] n=10, s=2")
    print("-" * 40)
    code = HelbergFerreiraCode(10, 2)
    analysis = code.analyze()
    print(f"Weight vector: {analysis['weight_vector']}")
    print(f"Modulus u: {analysis['modulus']}")
    print(f"Max cardinality: {analysis['max_cardinality']}")
    print(f"Best residue a: {analysis['best_residue']}")
    print(f"Code rate R: {analysis['rate']:.4f}")
    print(f"Min Hamming distance: {analysis['min_hamming_distance']}")
    print(f"Codewords: {analysis['codewords']}")
    
    # Example 3: Verify small codes
    print("\n[Example 3] Verification of small codes")
    print("-" * 40)
    for s in [2, 3]:
        for n in range(s + 2, min(s + 6, 12)):
            verify_code_construction(n, s, verbose=True)
    
    # Compute extended table
    print("\n[Example 4] Extended cardinality table (n ≤ 20)")
    print("-" * 40)
    results = compute_cardinality_table(max_n=20, s_values=[2, 3, 4, 5], verbose=True)
    print_cardinality_table(results, [2, 3, 4, 5], 20)