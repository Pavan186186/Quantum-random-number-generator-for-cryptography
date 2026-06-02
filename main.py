"""
=============================================================================
QUANTUM RANDOM NUMBER GENERATOR — main.py
=============================================================================

Run this file to execute the full QRNG demo:
  1. Statevector demo (shows the math live)
  2. Generate bits and run all NIST tests
  3. Benchmark quantum vs classical throughput
  4. Compare randomness quality across sources
  5. Generate sample cryptographic keys

INSTALLATION
────────────
  pip install pennylane numpy scipy

USAGE
─────
  python main.py                    # full demo
  python main.py --quick            # fast mode (fewer bits)
  python main.py --bits 100000      # test with 100k bits
"""

from __future__ import annotations

import sys
import argparse
import time
import numpy as np

# Add project root to path
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from core.qrng       import QRNG, QRNGConfig
from core.extractors import (
    von_neumann_extract,
    von_neumann_efficiency,
    min_entropy_estimate,
    measure_bias,
    toeplitz_extract,
)
from tests.nist_tests import run_nist_suite
from crypto.keygen   import (
    generate_aes_key,
    generate_ec_private_key_seed,
    compare_quantum_vs_classical,
    benchmark_throughput,
)


def banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║       QUANTUM RANDOM NUMBER GENERATOR — PennyLane + Python       ║
║   True randomness from the Born rule, not computational hardness  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def demo_statevector():
    """Show the quantum state math live."""
    print("─" * 62)
    print("  STEP 1: Statevector Demo — The Math Made Visible")
    print("─" * 62)
    print("""
  Starting state: |0⟩ = [1, 0]ᵀ
  Apply Hadamard: H = (1/√2)[[1, 1], [1, -1]]
  H|0⟩ = (1/√2)[1, 1]ᵀ = (|0⟩ + |1⟩) / √2
""")
    cfg = QRNGConfig(n_qubits=1, n_shots=1000, apply_extractor=False)
    rng = QRNG(cfg)
    rng.state_vector_demo()


def demo_bit_generation(n_bits: int):
    """Generate bits and show raw statistics."""
    print("─" * 62)
    print(f"  STEP 2: Generating {n_bits} Quantum Random Bits")
    print("─" * 62)

    cfg  = QRNGConfig(n_qubits=8, n_shots=n_bits // 8 + 1, apply_extractor=False)
    rng  = QRNG(cfg)

    t0       = time.perf_counter()
    raw_bits = rng.generate_bits(n_bits)
    elapsed  = time.perf_counter() - t0

    bias      = measure_bias(raw_bits)
    h_inf_est = min_entropy_estimate(raw_bits)

    print(f"\n  Generated {len(raw_bits)} raw bits in {elapsed:.3f}s")
    print(f"  Proportion of 1s (should be ≈ 0.5): {bias:.6f}")
    print(f"  Estimated min-entropy (should be ≈ 8): {h_inf_est:.3f} bits/block")
    print(f"\n  First 64 bits: {''.join(str(b) for b in raw_bits[:64])}")

    # Show Von Neumann extractor efficiency
    print("\n  Running Von Neumann extractor...")
    eff = von_neumann_efficiency(raw_bits)
    extracted = von_neumann_extract(raw_bits)
    print(f"  Input bias (p̂):        {eff['estimated_p']:.6f}")
    print(f"  Theoretical H(p):      {eff['entropy_rate']:.6f} bits/bit")
    print(f"  Extractor efficiency:  {eff['efficiency']:.4f} (bits out / bits in)")
    print(f"  Bits after extraction: {len(extracted)}")

    return raw_bits, extracted, rng


def demo_nist_tests(bits: list[int], label: str):
    """Run and display NIST test suite."""
    print("─" * 62)
    print(f"  STEP 3: NIST SP 800-22 Tests — {label}")
    print("─" * 62)
    results = run_nist_suite(bits, verbose=True)
    n_pass  = sum(1 for r in results if r.passed)

    if n_pass == len(results):
        print(f"  ✓ All {n_pass} tests passed — output is statistically indistinguishable")
        print(f"    from a truly random source at α = 0.01 significance level.\n")
    else:
        print(f"  ⚠ {len(results) - n_pass} tests failed. This may indicate:")
        print(f"    - Device bias (try increasing n_bits or enabling extractor)")
        print(f"    - Insufficient sample size (NIST recommends ≥ 10^6 bits)\n")

    return results


def demo_crypto_keys(rng: QRNG):
    """Generate and display sample cryptographic keys."""
    print("─" * 62)
    print("  STEP 4: Cryptographic Key Generation")
    print("─" * 62)

    aes_key = generate_aes_key(rng, key_size=256)
    ec_seed = generate_ec_private_key_seed(rng)

    print(f"\n  AES-256 key (32 bytes, hex):")
    print(f"  {aes_key.hex()}")
    print(f"\n  ECDH/X25519 private key seed (32 bytes, hex):")
    print(f"  {ec_seed.hex()}")
    print(f"\n  Sample random integers:")
    for _ in range(5):
        print(f"    rng.generate_int(0, 999) = {rng.generate_int(0, 999)}")
    print(f"\n  Sample random floats:")
    for _ in range(5):
        f = rng.generate_float()
        print(f"    rng.generate_float()     = {f:.12f}")


def demo_toeplitz_extractor(raw_bits: list[int]):
    """Demonstrate the strong Toeplitz extractor."""
    print("─" * 62)
    print("  STEP 5: Toeplitz Strong Extractor Demo")
    print("─" * 62)
    print("""
  The Von Neumann extractor handles i.i.d. biased bits.
  The Toeplitz extractor handles CORRELATED bits (e.g., from crosstalk
  on real quantum hardware where adjacent qubits affect each other).

  Leftover Hash Lemma guarantee:
    If input has min-entropy k ≥ m + 256,
    output m bits are 2^{-128} close to uniform.
""")
    n      = len(raw_bits)
    m      = n // 2  # output half as many bits as input
    strong = toeplitz_extract(raw_bits, output_length=m)

    bias_in  = measure_bias(raw_bits)
    bias_out = measure_bias(strong)

    print(f"  Input:  {n} bits,  bias = {bias_in:.6f}")
    print(f"  Output: {len(strong)} bits, bias = {bias_out:.6f}")
    print(f"  Output bias should be closer to 0.5: {abs(bias_out - 0.5) < abs(bias_in - 0.5)}\n")


def main():
    parser = argparse.ArgumentParser(description="Quantum Random Number Generator Demo")
    parser.add_argument("--bits",  type=int, default=20000,
                        help="Number of bits to generate (default: 20000)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 5000 bits, skip comparison")
    args = parser.parse_args()

    n_bits = 5000 if args.quick else args.bits

    banner()

    # ── Stage 1: Show the quantum math ────────────────────────────────────────
    demo_statevector()

    # ── Stage 2: Generate bits ─────────────────────────────────────────────────
    raw_bits, extracted, rng = demo_bit_generation(n_bits)

    # ── Stage 3: NIST tests on raw bits ───────────────────────────────────────
    if len(raw_bits) >= 100:
        demo_nist_tests(raw_bits, label="Raw Quantum Bits")

    # ── Stage 3b: NIST tests on extracted bits ─────────────────────────────────
    if len(extracted) >= 100:
        demo_nist_tests(extracted, label="After Von Neumann Extraction")

    # ── Stage 4: Toeplitz extractor ────────────────────────────────────────────
    demo_toeplitz_extractor(raw_bits)

    # ── Stage 5: Crypto keys ───────────────────────────────────────────────────
    demo_crypto_keys(rng)

    # ── Stage 6: Benchmark ────────────────────────────────────────────────────
    print("─" * 62)
    print("  STEP 6: Throughput Benchmark")
    print("─" * 62)
    benchmark_throughput(rng, n_bits=min(n_bits, 20000))

    # ── Stage 7: Full comparison (skip in quick mode) ─────────────────────────
    if not args.quick:
        print("─" * 62)
        print("  STEP 7: Quantum vs Classical Comparison")
        print("─" * 62)
        compare_quantum_vs_classical(n_bits=min(n_bits, 10000))

    print("\n  Demo complete.  Quantum randomness is real. ✓\n")


if __name__ == "__main__":
    main()
