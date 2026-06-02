"""
=============================================================================
CRYPTOGRAPHIC KEY GENERATOR  — crypto/keygen.py
=============================================================================

This module wraps the QRNG to generate production-ready cryptographic
material: AES keys, RSA seeds, and ECDH private keys.

It also provides a comparison between quantum-generated and classical
pseudo-random keys, demonstrating the difference in predictability.
"""

from __future__ import annotations

import os
import hmac
import hashlib
import secrets
import struct
import time
from typing import Literal

# We import from sibling modules using relative paths
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core.qrng import QRNG, QRNGConfig


# ─────────────────────────────────────────────────────────────────────────────
# AES Key Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_aes_key(
    rng: QRNG,
    key_size: Literal[128, 192, 256] = 256,
) -> bytes:
    """
    Generate a quantum-random AES key of the specified size.

    AES-256 requires 32 bytes (256 bits) of key material.
    AES-128 requires 16 bytes.

    The key is generated directly from quantum measurements and passed through
    HKDF (HMAC-based Key Derivation Function) for domain separation:
        key = HKDF-SHA256(quantum_bytes, salt=OS-random, info=b"AES-key")

    HKDF doesn't add entropy — it just formats the key material according
    to RFC 5869, making it safe to use in any cryptographic context.
    """
    if key_size not in (128, 192, 256):
        raise ValueError("key_size must be 128, 192, or 256")

    n_bytes = key_size // 8
    # Step 1: Get raw quantum bytes (already entropy-rich)
    quantum_bytes = rng.generate_bytes(n_bytes)

    # Step 2: HKDF-SHA256 expand/extract for proper formatting
    # HKDF-Extract: PRK = HMAC-SHA256(salt, quantum_bytes)
    salt = os.urandom(32)    # fresh OS-random salt — doesn't need to be secret
    prk  = hmac.new(salt, quantum_bytes, hashlib.sha256).digest()

    # HKDF-Expand: key = T(1) = HMAC-SHA256(PRK, "" || 0x01 || info)
    info = b"QRNG-AES-key"
    t1   = hmac.new(prk, info + b"\x01", hashlib.sha256).digest()

    return t1[:n_bytes]


# ─────────────────────────────────────────────────────────────────────────────
# ECDH Private Key Seed
# ─────────────────────────────────────────────────────────────────────────────

def generate_ec_private_key_seed(rng: QRNG) -> bytes:
    """
    Generate 32 bytes of quantum-random material suitable as an
    elliptic curve private key seed for P-256 (secp256r1) or Curve25519.

    The seed must be:
      - Uniformly random in [1, n-1] where n is the curve order
      - n for P-256  = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
      - n for X25519 = 2^252 + 27742317777372353535851937790883648493

    Since our random 256-bit number is astronomically unlikely to be ≥ n
    (the curve orders are close to 2^256), we just generate 32 bytes and
    let the cryptographic library do the clamping/reduction.

    For X25519 specifically, RFC 7748 specifies bit clamping:
        secret[0]  &= 248   # clear bottom 3 bits
        secret[31] &= 127   # clear top bit
        secret[31] |= 64    # set second-highest bit
    """
    raw_seed = rng.generate_bytes(32)

    # Apply X25519 clamping (safe to do even if you'll use a different curve,
    # as it only clears/sets specific bits without reducing entropy significantly)
    seed = bytearray(raw_seed)
    seed[ 0] &= 248
    seed[31] &= 127
    seed[31] |= 64

    return bytes(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison: Quantum vs Classical
# ─────────────────────────────────────────────────────────────────────────────

def compare_quantum_vs_classical(n_bits: int = 10000) -> dict:
    """
    Generate the same number of bits from three sources and compare their
    statistical properties.  This is the core demonstration of the project.

    Sources compared:
    1. Quantum (via PennyLane simulator, no extractor)
    2. Quantum (with Von Neumann extractor)
    3. Classical CSPRNG (Python's secrets module, which uses OS entropy)
    4. Classical PRNG (Python's random.Random with fixed seed — PREDICTABLE)

    The quantum sources should be statistically indistinguishable from the
    OS CSPRNG, but the seeded PRNG should fail some tests (or at least
    show suspicious statistical regularity).
    """
    import random
    from tests.nist_tests import run_nist_suite, frequency_test, runs_test

    results = {}

    # ── 1. Quantum (raw) ──────────────────────────────────────────────────────
    cfg_raw = QRNGConfig(n_qubits=8, n_shots=256, apply_extractor=False)
    rng_raw = QRNG(cfg_raw)
    q_bits_raw = rng_raw.generate_bits(n_bits)
    results["quantum_raw"] = {
        "source": "Quantum (PennyLane, no extractor)",
        "bits": q_bits_raw,
        "proportion_ones": sum(q_bits_raw) / n_bits,
        "nist": run_nist_suite(q_bits_raw, verbose=False),
    }

    # ── 2. Quantum (with extractor) ───────────────────────────────────────────
    cfg_ext = QRNGConfig(n_qubits=8, n_shots=512, apply_extractor=True)
    rng_ext = QRNG(cfg_ext)
    q_bits_ext = rng_ext.generate_bits(n_bits)
    results["quantum_extracted"] = {
        "source": "Quantum (PennyLane, with Von Neumann extractor)",
        "bits": q_bits_ext,
        "proportion_ones": sum(q_bits_ext) / n_bits,
        "nist": run_nist_suite(q_bits_ext, verbose=False),
    }

    # ── 3. OS CSPRNG (secrets module) ────────────────────────────────────────
    os_bytes = secrets.token_bytes(n_bits // 8)
    os_bits  = [int(b) for byte in os_bytes for b in format(byte, '08b')][:n_bits]
    results["os_csprng"] = {
        "source": "OS CSPRNG (secrets.token_bytes)",
        "bits": os_bits,
        "proportion_ones": sum(os_bits) / n_bits,
        "nist": run_nist_suite(os_bits, verbose=False),
    }

    # ── 4. Seeded PRNG (predictable!) ─────────────────────────────────────────
    rng_pred = random.Random(42)          # fixed seed — fully predictable
    pred_bits = [rng_pred.getrandbits(1) for _ in range(n_bits)]
    results["seeded_prng"] = {
        "source": "Seeded PRNG (random.Random(42)) — PREDICTABLE",
        "bits": pred_bits,
        "proportion_ones": sum(pred_bits) / n_bits,
        "nist": run_nist_suite(pred_bits, verbose=False),
    }

    # ── Print comparison report ───────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  QUANTUM vs CLASSICAL RANDOMNESS COMPARISON")
    print("═" * 72)
    print(f"  {'Source':<42} {'P(1)':>6}  {'NIST Passes':>12}")
    print("  " + "─" * 68)
    for key, data in results.items():
        n_pass = sum(1 for r in data["nist"] if r.passed)
        n_total = len(data["nist"])
        src = data["source"]
        prop = data["proportion_ones"]
        print(f"  {src:<42} {prop:>6.4f}  {n_pass}/{n_total:>10}")
    print("═" * 72 + "\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Speed benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_throughput(rng: QRNG, n_bits: int = 50000) -> dict:
    """
    Measure raw throughput of the QRNG in bits per second.

    On a modern laptop with default.qubit simulator:
      - 8 qubits, 512 shots → ~50,000 bits/s
    Real IBM hardware would be much slower (~100-1000 bits/s) due to
    queue wait times and gate execution latency.

    Compare against secrets.token_bytes: ~1 Gbit/s  (OS is much faster —
    quantum is NOT meant to replace it for bulk encryption.  It seeds a CSPRNG.)
    """
    import time, secrets

    # Quantum benchmark
    t0      = time.perf_counter()
    bits    = rng.generate_bits(n_bits)
    q_time  = time.perf_counter() - t0
    q_rate  = n_bits / q_time

    # Classical comparison
    t0      = time.perf_counter()
    _       = secrets.token_bytes(n_bits // 8)
    c_time  = time.perf_counter() - t0
    c_rate  = n_bits / c_time

    print(f"\n  Throughput comparison ({n_bits:,} bits):")
    print(f"    Quantum QRNG  : {q_rate:>12,.0f} bits/s  ({q_time:.3f} s)")
    print(f"    OS CSPRNG     : {c_rate:>12,.0f} bits/s  ({c_time:.6f} s)")
    print(f"    Classical is {c_rate/q_rate:,.0f}× faster\n")
    print(f"  → Use QRNG to seed a CSPRNG, not for bulk key material.\n")

    return {"quantum_bps": q_rate, "classical_bps": c_rate, "ratio": c_rate / q_rate}
