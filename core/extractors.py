"""
=============================================================================
RANDOMNESS EXTRACTORS  — core/extractors.py
=============================================================================

WHY WE NEED EXTRACTORS
───────────────────────
Even a real quantum circuit has imperfections:
  • State preparation error: |0⟩ isn't perfectly |0⟩ — residual |1⟩ amplitude
  • Gate infidelity: H isn't exactly (1/√2)[[1,1],[1,-1]]
  • Readout error: a measurement of |0⟩ occasionally reports 1

These create a bias  p = P(bit=1) = 0.5 + ε  where ε is small but nonzero.
An adversary who knows ε gains I = 1 - H(p) bits of information per bit,
where H(p) = -p log₂ p - (1-p) log₂(1-p) is the binary entropy.

An extractor transforms n biased bits into m ≤ n uniform bits such that
no algorithm can distinguish the output from truly uniform bits even given
full knowledge of the bias structure.

THE VON NEUMANN EXTRACTOR
──────────────────────────
Due to John Von Neumann (1951).  For a sequence of i.i.d. bits with bias p:

  1. Read pairs of bits (b₁, b₂).
  2. If b₁ = 0, b₂ = 1  → output 0   (probability p(1-p))
  3. If b₁ = 1, b₂ = 0  → output 1   (probability (1-p)p)
  4. If b₁ = b₂          → discard    (probability p² + (1-p)²)

Proof of unbiasedness:
  P(output 0) = p(1-p) / [p(1-p) + (1-p)p] = 1/2  ✓
  P(output 1) = (1-p)p / [p(1-p) + (1-p)p] = 1/2  ✓

Cost: expected n·H(p) output bits from n input bits.  For p = 0.51, H ≈ 0.9999
so we lose less than 0.01% of bits.  For a very biased coin (p=0.9), we'd lose
about 53% of bits — still always get unbiased output.

TOEPLITZ HASH EXTRACTOR  (stronger, for paranoid cryptographic use)
─────────────────────────────────────────────────────────────────────
For a stronger extractor that handles non-i.i.d. inputs (e.g., correlated
measurement errors across adjacent qubits on a chip), we use a Toeplitz
matrix multiplication over GF(2).

Given:
  - n input bits x ∈ {0,1}ⁿ  with min-entropy ≥ k
  - Random seed s ∈ {0,1}^{n+m-1}  (small, can be public)

The Toeplitz matrix T is (m × n) over GF(2), defined by seed s:
  T[i][j] = s[i+j]

Output:  y = T·x mod 2   (m-dimensional, over GF(2))

The Leftover Hash Lemma guarantees:
  ‖P_Y - Uniform‖₁ ≤ 2^{-(k-m)/2}

So if we set m = k - 2·security_param (e.g., k - 256), the output is
indistinguishable from uniform by any algorithm with 2^{256} time/memory.
"""

from __future__ import annotations

import hashlib
import os
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Von Neumann Extractor
# ─────────────────────────────────────────────────────────────────────────────

def von_neumann_extract(bits: list[int]) -> list[int]:
    """
    Apply the Von Neumann randomness extractor to a list of bits.

    Parameters
    ──────────
    bits : list[int]
        Raw bits (0 or 1) from the quantum circuit.  Can be biased.

    Returns
    ───────
    list[int]
        Shorter list of unbiased bits.  Length is random with expected
        value = len(bits) × H(p) / 2  where H(p) is the binary entropy
        of the input bias.

    Example
    ───────
    >>> raw  = [0, 1, 1, 1, 0, 0, 1, 0]  # biased input
    >>> out  = von_neumann_extract(raw)
    >>> out  # only pairs with different bits contribute
    [0, 1, 0]  # (0,1)→0, (1,1)→discard, (0,0)→discard, (1,0)→1... wait
    # Actually: (0,1)→0, (1,1)→discard, (0,0)→discard, (1,0)→1 = [0, 1]
    """
    output = []

    # Process bits in non-overlapping pairs
    # (using overlapping pairs would introduce correlation)
    for i in range(0, len(bits) - 1, 2):
        b1, b2 = bits[i], bits[i + 1]

        if b1 == 0 and b2 == 1:
            output.append(0)      # Case: (0,1) → output 0
        elif b1 == 1 and b2 == 0:
            output.append(1)      # Case: (1,0) → output 1
        # Case: b1 == b2 → discard (do nothing)

    return output


def von_neumann_efficiency(bits: list[int]) -> dict:
    """
    Measure how efficient the Von Neumann extractor was on this input.

    Returns a dict with:
      input_bits    : total raw bits consumed
      output_bits   : unbiased bits produced
      efficiency    : output_bits / input_bits
      estimated_p   : estimated bias of the raw bits (should be ≈ 0.5)
      entropy_rate  : H(p) = theoretical maximum efficiency
    """
    n      = len(bits)
    p_hat  = sum(bits) / n  if n > 0 else 0.5

    # Binary entropy H(p) — clamped to avoid log(0)
    eps = 1e-12
    p   = np.clip(p_hat, eps, 1 - eps)
    H   = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    output = von_neumann_extract(bits)

    return {
        "input_bits":   n,
        "output_bits":  len(output),
        "efficiency":   len(output) / n if n > 0 else 0,
        "estimated_p":  p_hat,
        "entropy_rate": H,
        "theoretical_efficiency": H / 2,   # VN produces H/2 bits per input bit
    }


# ─────────────────────────────────────────────────────────────────────────────
# Toeplitz Hash Extractor  (strong extractor for cryptographic applications)
# ─────────────────────────────────────────────────────────────────────────────

def toeplitz_extract(
    bits: list[int],
    output_length: Optional[int] = None,
    seed: Optional[bytes] = None,
) -> list[int]:
    """
    Apply a Toeplitz hash extractor over GF(2).

    This is a STRONG extractor: it works even if adjacent qubit measurements
    are correlated (e.g., due to crosstalk on real quantum hardware).

    Parameters
    ──────────
    bits : list[int]
        Raw input bits.  Min-entropy should be ≥ output_length + 256.

    output_length : int | None
        Desired output bits.  Defaults to len(bits) // 2.
        Must satisfy: output_length ≤ min_entropy - 256
        (the 256 is the security parameter — 2^-128 statistical distance).

    seed : bytes | None
        The seed s ∈ {0,1}^{n+m-1} as bytes.  If None, a fresh OS-random
        seed is used.  The seed can be public — security doesn't depend on
        its secrecy, only on its freshness (each invocation needs a new seed).

    Returns
    ───────
    list[int]
        output_length unbiased, strongly extracted bits.

    Mathematical guarantee (Leftover Hash Lemma)
    ─────────────────────────────────────────────
    If the input has min-entropy k, and output_length m = k - 256:
        Statistical distance from uniform ≤ 2^{-128}
    This holds regardless of the bias structure or correlations in the input.
    """
    n = len(bits)
    m = output_length or (n // 2)

    if m <= 0 or m > n:
        raise ValueError(f"output_length must be in [1, {n}]")

    # Generate seed: n + m - 1 bits define the full Toeplitz matrix
    seed_length_bytes = (n + m - 1 + 7) // 8
    if seed is None:
        seed = os.urandom(seed_length_bytes)

    # Expand seed to bit array using SHA-256 in counter mode
    # (deterministic so the same seed always gives the same matrix)
    seed_bits = _bytes_to_bits(seed)[:n + m - 1]
    while len(seed_bits) < n + m - 1:
        seed_bits.extend(_bytes_to_bits(
            hashlib.sha256(seed + len(seed_bits).to_bytes(4, 'big')).digest()
        ))
    seed_bits = seed_bits[:n + m - 1]

    # Build Toeplitz matrix T (m × n) over GF(2)
    # T[i][j] = seed_bits[i + j]  (this is the Toeplitz structure)
    # Using numpy for efficiency — over GF(2) multiplication is bitwise AND,
    # addition is XOR, and matrix-vector product is (A @ x) mod 2.
    T = np.zeros((m, n), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            T[i, j] = seed_bits[i + j]

    x = np.array(bits, dtype=np.uint8)

    # GF(2) matrix-vector product: y = T·x mod 2
    # This is the extraction: maps n (possibly correlated) bits to m uniform bits
    y = (T @ x) % 2

    return y.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _bytes_to_bits(data: bytes) -> list[int]:
    """Convert a bytes object to a list of bits (MSB first within each byte)."""
    bits = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def measure_bias(bits: list[int]) -> float:
    """
    Estimate the bias p = P(bit=1) of a bit string.
    Returns 0.5 for a perfectly unbiased sequence.
    """
    if not bits:
        return 0.5
    return sum(bits) / len(bits)


def min_entropy_estimate(bits: list[int], block_size: int = 8) -> float:
    """
    Estimate the min-entropy of a bit string using block frequency analysis.

    Min-entropy H_∞ = -log₂(max_probability) where max_probability is
    the probability of the most likely block pattern.

    For a truly uniform source on b-bit blocks: H_∞ = b bits.
    Compression below b means some patterns are more likely than others.
    """
    if len(bits) < block_size * 10:
        raise ValueError("Need at least 10× block_size bits for a meaningful estimate")

    n_blocks = len(bits) // block_size
    counts   = {}

    for i in range(n_blocks):
        block = tuple(bits[i * block_size : (i + 1) * block_size])
        counts[block] = counts.get(block, 0) + 1

    max_prob = max(counts.values()) / n_blocks
    h_inf    = -np.log2(max_prob) if max_prob > 0 else block_size

    return h_inf
