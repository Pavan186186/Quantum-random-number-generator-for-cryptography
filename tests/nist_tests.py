"""
=============================================================================
NIST SP 800-22 STATISTICAL TEST SUITE  — tests/nist_tests.py
=============================================================================

WHAT THESE TESTS DO
───────────────────
NIST Special Publication 800-22 defines 15 statistical tests for evaluating
random number generators used in cryptographic applications.  A sequence
passes each test with probability ≥ 1 - α (we use α = 0.01) if it is truly
random.  A good RNG should pass ALL 15 tests.

KEY MATHEMATICAL CONCEPTS
──────────────────────────
Every test:
  1. States a null hypothesis H₀: "the sequence is random"
  2. Computes a test statistic from the bit string
  3. Computes a p-value = P(statistic ≥ observed | H₀)
  4. Rejects H₀ (declares non-random) if p-value < α = 0.01

We implement the 5 most fundamental tests here.  The full NIST suite has
10 more (overlapping templates, Lempel-Ziv complexity, etc.) — see the
NIST publication for those.

TESTS IMPLEMENTED
─────────────────
1. Frequency (Monobit) Test
2. Frequency Within a Block Test
3. Runs Test
4. Longest Run of Ones in a Block Test
5. Serial Test (pair frequency)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple
from scipy import special


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """Result of a single NIST statistical test."""
    name:        str
    p_value:     float
    statistic:   float
    passed:      bool        # p_value > 0.01
    details:     dict        # test-specific intermediate values


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Frequency (Monobit) Test
# ─────────────────────────────────────────────────────────────────────────────

def frequency_test(bits: list[int]) -> TestResult:
    """
    NIST Test 1: Frequency (Monobit) Test.

    PURPOSE
    ───────
    Checks that the number of 0s and 1s are approximately equal.
    This is the most basic check: any RNG that fails this is badly broken.

    ALGORITHM
    ─────────
    1. Convert bits to {+1, -1}: sᵢ = 2bᵢ - 1
    2. Compute Sₙ = Σᵢ sᵢ  (sum of ±1 values)
    3. Compute test statistic: S_obs = |Sₙ| / √n
    4. p-value = erfc(S_obs / √2)  where erfc is the complementary error function

    DERIVATION
    ──────────
    Under H₀, E[sᵢ] = 0 and Var(sᵢ) = 1.  By CLT:
        Sₙ / √n → N(0, 1)  as n → ∞
    So S_obs follows a half-normal under H₀, and:
        p-value = P(|Z| ≥ S_obs) = erfc(S_obs/√2)
    where Z ~ N(0,1).

    PASS CRITERION
    ──────────────
    p-value ≥ 0.01  (equivalently, S_obs ≤ 2.576 for n large)
    """
    n   = len(bits)
    s   = [2 * b - 1 for b in bits]          # map 0→-1, 1→+1
    S_n = sum(s)                               # Σ sᵢ
    S_obs = abs(S_n) / math.sqrt(n)            # test statistic
    p_value = math.erfc(S_obs / math.sqrt(2)) # p-value

    return TestResult(
        name      = "Frequency (Monobit)",
        p_value   = p_value,
        statistic = S_obs,
        passed    = p_value >= 0.01,
        details   = {
            "n": n,
            "S_n": S_n,
            "proportion_ones": sum(bits) / n,
            "expected_proportion": 0.5,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Block Frequency Test
# ─────────────────────────────────────────────────────────────────────────────

def block_frequency_test(bits: list[int], M: int = 128) -> TestResult:
    """
    NIST Test 2: Frequency Within a Block Test.

    PURPOSE
    ───────
    Tests whether the proportion of 1s is approximately 1/2 within each
    non-overlapping block of M bits.  Catches cases where the frequency
    is correct globally but varies across the sequence.

    ALGORITHM
    ─────────
    1. Divide n bits into N = ⌊n/M⌋ blocks of M bits each.
    2. For each block j, compute πⱼ = (number of 1s in block j) / M
    3. Compute χ² = 4M · Σⱼ (πⱼ - 1/2)²
    4. p-value = igamc(N/2, χ²/2)  (incomplete gamma function)

    DERIVATION
    ──────────
    Under H₀, each πⱼ is the mean of M i.i.d. Bernoulli(1/2) variables.
    By CLT, √M(πⱼ - 1/2) → N(0, 1/4).
    So 4M(πⱼ - 1/2)² → χ²(1).
    Summing N independent such terms: χ² = 4M·Σ(πⱼ-1/2)² → χ²(N).
    p-value = P(χ²(N) ≥ χ²_obs) = igamc(N/2, χ²/2).

    PARAMETERS
    ──────────
    M: block size in bits.  NIST recommends M ≥ 20 and M > 0.01·n.
    """
    n = len(bits)
    N = n // M                    # number of complete blocks
    if N < 1:
        raise ValueError(f"Need at least {M} bits for block frequency test with M={M}")

    # Compute per-block proportions πⱼ
    proportions = []
    for j in range(N):
        block = bits[j * M : (j + 1) * M]
        pi_j  = sum(block) / M
        proportions.append(pi_j)

    # Chi-squared statistic
    chi2 = 4 * M * sum((p - 0.5) ** 2 for p in proportions)

    # p-value from upper incomplete gamma function
    p_value = special.gammaincc(N / 2, chi2 / 2)

    return TestResult(
        name      = "Block Frequency",
        p_value   = p_value,
        statistic = chi2,
        passed    = p_value >= 0.01,
        details   = {
            "n": n, "M": M, "N": N,
            "block_proportions_mean": float(np.mean(proportions)),
            "block_proportions_std":  float(np.std(proportions)),
            "chi2": chi2,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Runs Test
# ─────────────────────────────────────────────────────────────────────────────

def runs_test(bits: list[int]) -> TestResult:
    """
    NIST Test 3: Runs Test.

    PURPOSE
    ───────
    A "run" is a maximal sequence of identical consecutive bits (all 0s or
    all 1s).  This test checks whether the number of runs (oscillations
    between 0 and 1) is consistent with a random sequence.

    Too few runs → the sequence has long streaks → positive autocorrelation.
    Too many runs → the sequence alternates too regularly → negative autocorrelation.
    Either implies structure/memory, which is non-random.

    ALGORITHM
    ─────────
    1. Compute π = proportion of 1s.
       Pre-requisite: |π - 1/2| < 2/√n  else fail immediately (monobit fail).
    2. Count total runs V_n = Σᵢ |bᵢ₊₁ - bᵢ| + 1
       (a run boundary occurs wherever adjacent bits differ)
    3. Test statistic:
           Z = |V_n - 2n·π(1-π)| / (2√(2n)·π(1-π))
    4. p-value = erfc(Z / √2)

    DERIVATION
    ──────────
    Under H₀, the expected number of runs:
        E[V_n] = 2n·π(1-π)  ≈ n/2  for π ≈ 1/2

    Variance:
        Var(V_n) = 4n·π(1-π)·(1-2π(1-π)) / (n-1)
                 ≈ 2·π(1-π)·(1-2π(1-π))·n ... (leading term)

    The standardised statistic is asymptotically N(0,1) under H₀.
    The NIST formulation uses the simplified variance 2√(2n)·π(1-π).
    """
    n  = len(bits)
    pi = sum(bits) / n   # estimated proportion of 1s

    # Pre-test: if frequency test fails badly, runs test is undefined
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return TestResult(
            name      = "Runs",
            p_value   = 0.0,
            statistic = float('inf'),
            passed    = False,
            details   = {
                "reason": "Pre-condition failed: proportion outside acceptable range",
                "proportion_ones": pi,
                "threshold": 2 / math.sqrt(n),
            }
        )

    # Count run boundaries: positions where bᵢ ≠ bᵢ₊₁
    boundaries = sum(1 for i in range(n - 1) if bits[i] != bits[i + 1])
    V_n = boundaries + 1   # number of runs = number of transitions + 1

    # Test statistic
    expected_V = 2 * n * pi * (1 - pi)
    denom      = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    Z          = abs(V_n - expected_V) / denom
    p_value    = math.erfc(Z / math.sqrt(2))

    return TestResult(
        name      = "Runs",
        p_value   = p_value,
        statistic = Z,
        passed    = p_value >= 0.01,
        details   = {
            "n": n,
            "V_n": V_n,
            "expected_V": expected_V,
            "proportion_ones": pi,
            "boundaries": boundaries,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Longest Run of Ones in a Block
# ─────────────────────────────────────────────────────────────────────────────

def longest_run_test(bits: list[int]) -> TestResult:
    """
    NIST Test 4: Longest Run of Ones in a Block.

    PURPOSE
    ───────
    Detects that the longest run of 1s in each block matches the expected
    distribution for a random sequence.  A biased RNG might produce many
    short runs (if P(1) < 0.5) or very long runs (if P(1) > 0.5).

    ALGORITHM (for n ≥ 128 bits, M = 8, K = 3)
    ────────────────────────────────────────────
    1. Divide into blocks of M = 8 bits.
    2. Find the longest run of 1s in each block.
    3. Bin the longest run lengths into K+1 = 4 categories:
       v[0] = blocks where longest run ≤ 1
       v[1] = blocks where longest run = 2
       v[2] = blocks where longest run = 3
       v[3] = blocks where longest run ≥ 4
    4. Compute χ² = Σⱼ (v[j] - N·π[j])² / (N·π[j])
       where π[j] are precomputed theoretical probabilities.
    5. p-value = igamc(K/2, χ²/2)

    The theoretical probabilities π[j] for M=8 are precomputed from
    combinatorial analysis and tabulated in the NIST publication.
    """
    n = len(bits)

    # NIST parameter selection based on sequence length
    if n < 128:
        raise ValueError("Longest run test requires at least 128 bits")
    elif n < 6272:
        M, K = 8, 3
        # Theoretical probabilities for M=8 (NIST Table 2.4.4)
        pi = [0.2148, 0.3672, 0.2305, 0.1875]
        thresholds = [1, 2, 3, float('inf')]   # categories: ≤1, 2, 3, ≥4
    elif n < 750000:
        M, K = 128, 5
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        thresholds = [4, 5, 6, 7, 8, float('inf')]
    else:
        M, K = 10000, 6
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
        thresholds = [10, 11, 12, 13, 14, 15, float('inf')]

    N = n // M   # number of blocks

    def longest_run_of_ones(block):
        """Find the longest consecutive run of 1s in a block."""
        max_run = current_run = 0
        for b in block:
            if b == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    # Count longest runs per block
    longest_runs = []
    for j in range(N):
        block = bits[j * M : (j + 1) * M]
        longest_runs.append(longest_run_of_ones(block))

    # Bin into categories
    v = [0] * (K + 1)
    for lr in longest_runs:
        for cat, threshold in enumerate(thresholds):
            if cat < K and lr <= threshold:
                v[cat] += 1
                break
            elif cat == K:
                v[cat] += 1

    # Chi-squared statistic
    chi2    = sum((v[j] - N * pi[j])**2 / (N * pi[j]) for j in range(K + 1))
    p_value = special.gammaincc(K / 2, chi2 / 2)

    return TestResult(
        name      = "Longest Run of Ones",
        p_value   = p_value,
        statistic = chi2,
        passed    = p_value >= 0.01,
        details   = {
            "n": n, "M": M, "K": K, "N": N,
            "category_counts": v,
            "theoretical_probs": pi,
            "mean_longest_run": float(np.mean(longest_runs)),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Serial Test
# ─────────────────────────────────────────────────────────────────────────────

def serial_test(bits: list[int], m: int = 2) -> TestResult:
    """
    NIST Test 5: Serial Test.

    PURPOSE
    ───────
    Tests whether all m-bit patterns (overlapping) appear with equal frequency.
    For m=2, this checks whether 00, 01, 10, 11 each appear ~n/4 times.
    This detects short-range correlations between adjacent bits.

    ALGORITHM
    ─────────
    1. Augment the sequence: append the first m-1 bits to the end
       (making it circular — every bit is in m overlapping windows).
    2. For m and m-1 and m-2, count the frequency of each pattern.
    3. Compute ψ²_m = (2^m / n) · Σ (count(v) - n/2^m)²  for all v ∈ {0,1}^m
    4. Compute differences:
           ∇ψ²_m  = ψ²_m - ψ²_{m-1}
           ∇²ψ²_m = ψ²_m - 2·ψ²_{m-1} + ψ²_{m-2}
    5. p-values:
           p₁ = igamc(2^{m-2}, ∇ψ²_m / 2)
           p₂ = igamc(2^{m-3}, ∇²ψ²_m / 2)

    DERIVATION
    ──────────
    Under H₀, each of the 2^m patterns appears n/2^m times on average.
    The statistic ψ²_m is a rescaled chi-squared with 2^m degrees of freedom.
    The differences ∇ψ²_m and ∇²ψ²_m are approximately chi-squared with
    2^{m-2} and 2^{m-3} degrees of freedom respectively.

    We return both p-values; the test passes if both p-values ≥ 0.01.
    """
    n = len(bits)
    if n < 100:
        raise ValueError("Serial test requires at least 100 bits")

    def psi_sq(bits, m):
        """Compute ψ²_m for the given m."""
        if m == 0:
            return 0.0
        # Augment sequence (wrap around)
        augmented = bits + bits[:m - 1]
        # Count all m-bit patterns
        counts = {}
        for i in range(n):
            pattern = tuple(augmented[i : i + m])
            counts[pattern] = counts.get(pattern, 0) + 1
        # ψ²_m = (2^m / n) · Σ count(v)²  - n   (equivalent reformulation)
        psi2 = (2**m / n) * sum(c**2 for c in counts.values()) - n
        return psi2

    psi2_m   = psi_sq(bits, m)
    psi2_m1  = psi_sq(bits, m - 1)
    psi2_m2  = psi_sq(bits, m - 2)

    del1     = psi2_m - psi2_m1
    del2     = psi2_m - 2 * psi2_m1 + psi2_m2

    p_value1 = special.gammaincc(2**(m - 2), del1 / 2)
    p_value2 = special.gammaincc(2**(m - 3), del2 / 2) if m >= 3 else 1.0

    return TestResult(
        name      = f"Serial (m={m})",
        p_value   = min(p_value1, p_value2),   # both must pass
        statistic = max(del1, del2),
        passed    = (p_value1 >= 0.01) and (p_value2 >= 0.01),
        details   = {
            "n": n, "m": m,
            "psi2_m": psi2_m, "psi2_m1": psi2_m1,
            "delta1": del1, "delta2": del2,
            "p_value1": p_value1, "p_value2": p_value2,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full suite runner
# ─────────────────────────────────────────────────────────────────────────────

def run_nist_suite(bits: list[int], verbose: bool = True) -> list[TestResult]:
    """
    Run all implemented NIST tests on a bit sequence.

    Returns a list of TestResult objects.  Prints a formatted report if
    verbose=True.
    """
    tests = [
        lambda b: frequency_test(b),
        lambda b: block_frequency_test(b, M=min(128, len(b) // 10)),
        lambda b: runs_test(b),
        lambda b: longest_run_test(b),
        lambda b: serial_test(b, m=2),
        lambda b: serial_test(b, m=3),
    ]

    results = []
    for test_fn in tests:
        try:
            result = test_fn(bits)
            results.append(result)
        except Exception as e:
            logger_placeholder = f"Test failed with error: {e}"
            results.append(TestResult(
                name=str(test_fn), p_value=0.0, statistic=0.0,
                passed=False, details={"error": str(e)}
            ))

    if verbose:
        print("\n" + "═" * 62)
        print("  NIST SP 800-22 Statistical Test Results")
        print("═" * 62)
        print(f"  Input: {len(bits)} bits")
        print(f"  Significance level α = 0.01\n")
        print(f"  {'Test':<30} {'p-value':>10} {'Result':>8}")
        print("  " + "─" * 58)
        for r in results:
            symbol = "✓ PASS" if r.passed else "✗ FAIL"
            print(f"  {r.name:<30} {r.p_value:>10.6f} {symbol:>8}")
        n_pass = sum(1 for r in results if r.passed)
        print("  " + "─" * 58)
        print(f"  Passed: {n_pass}/{len(results)} tests")
        print("═" * 62 + "\n")

    return results
