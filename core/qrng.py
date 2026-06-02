"""
=============================================================================
QUANTUM RANDOM NUMBER GENERATOR  — core/qrng.py
=============================================================================

WHAT THIS MODULE DOES
─────────────────────
Generates cryptographically strong random bits by exploiting the fundamental
indeterminacy of quantum measurement.  Every call to `generate_bits()` runs a
real quantum circuit (simulated locally or on IBM hardware) and returns bits
whose randomness is guaranteed by the Born rule, not by computational
hardness.

MATHEMATICAL FOUNDATIONS (brief summary — see README for full derivation)
──────────────────────────────────────────────────────────────────────────
1.  A qubit starts in |0⟩ = [1, 0]ᵀ  (ground state, density matrix ρ = |0⟩⟨0|).

2.  The Hadamard gate H = (1/√2)[[1, 1],[1,-1]] maps it to the maximally
    mixed pure state:
        H|0⟩ = (1/√2)(|0⟩ + |1⟩)
    Density matrix:  ρ = (1/2)[[1,1],[1,1]]
    The off-diagonal coherences prove this is NOT a classical mixture.

3.  Measurement projects onto {|0⟩⟨0|, |1⟩⟨1|}.  Born rule:
        P(k) = Tr(Pₖ ρ) = 1/2  for k ∈ {0,1}
    The result is intrinsically random — no hidden variable can predict it
    (Bell/Kochen-Specker theorems).

4.  n-qubit extension: H⊗ⁿ|0⟩⊗ⁿ = (1/√2ⁿ) Σ_{x∈{0,1}ⁿ} |x⟩
    One shot measures n independent uniform bits simultaneously.

5.  Device bias correction via Von Neumann extractor (see extractors.py).
"""

from __future__ import annotations

import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pennylane as qml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QRNGConfig:
    """
    All tuneable parameters for the QRNG in one place.

    n_qubits : int
        Width of the circuit.  With n qubits we get n raw bits per shot.
        Keep ≤ 20 for the local simulator (beyond that RAM explodes because
        the statevector has 2ⁿ complex amplitudes, each 16 bytes).

    n_shots : int
        How many times to execute the circuit per call to generate_bits().
        Each shot produces n_qubits raw bits, so total raw bits = n_qubits × n_shots.
        More shots → more bits but also more time.

    backend : "simulator" | "ibm"
        "simulator"  — PennyLane's default.qubit (statevector, exact)
        "ibm"        — IBM Quantum cloud (real hardware, noisy)

    ibm_token : str | None
        Required when backend == "ibm".  Set via environment variable
        IBMQ_TOKEN; do not hard-code secrets.

    apply_extractor : bool
        Whether to run the Von Neumann extractor on raw bits.
        Should be True for cryptographic use; False for speed benchmarks.

    circuit_layers : int
        Depth of the randomness circuit.  1 = just H gates (theoretically
        sufficient).  2+ = add parameterised rotations so the output passes
        more NIST tests even without the extractor (useful for debugging).
    """
    n_qubits:         int   = 8
    n_shots:          int   = 1024
    backend:          Literal["simulator", "ibm"] = "simulator"
    ibm_token:        Optional[str] = None
    apply_extractor:  bool  = True
    circuit_layers:   int   = 1


# ─────────────────────────────────────────────────────────────────────────────
# The quantum circuit
# ─────────────────────────────────────────────────────────────────────────────

def _build_device(cfg: QRNGConfig) -> qml.Device:
    """
    Construct the PennyLane device.

    For the simulator, default.qubit stores the full statevector:
        |ψ⟩ ∈ ℂ^{2^n}
    so it computes exact Born-rule probabilities rather than sampling noise.

    For IBM hardware, we get real physical shot noise on top of the quantum
    randomness — which is actually fine for our purposes (more entropy), but
    we still apply the Von Neumann extractor to remove any systematic bias.
    """
    if cfg.backend == "simulator":
        return qml.device("default.qubit", wires=cfg.n_qubits, shots=cfg.n_shots)

    # IBM hardware path — requires qiskit-ibm-runtime installed
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from pennylane_qiskit import IBMQDevice  # pennylane-qiskit plugin
    except ImportError:
        raise ImportError(
            "Install pennylane-qiskit and qiskit-ibm-runtime for IBM hardware access.\n"
            "  pip install pennylane-qiskit qiskit-ibm-runtime"
        )

    if cfg.ibm_token is None:
        raise ValueError("Set ibm_token in QRNGConfig or IBMQ_TOKEN env variable.")

    QiskitRuntimeService.save_account(channel="ibm_quantum", token=cfg.ibm_token)
    service  = QiskitRuntimeService()
    backends = service.backends(simulator=False, operational=True)
    # Pick the least-busy real backend that fits our qubit count
    backend  = min(backends, key=lambda b: b.status().pending_jobs)
    logger.info("Using IBM backend: %s", backend.name)
    return IBMQDevice(wires=cfg.n_qubits, backend=backend.name, shots=cfg.n_shots)


def _make_circuit(cfg: QRNGConfig, device: qml.Device):
    """
    Build and return the quantum circuit as a callable QNode.

    The circuit:
    ┌───┐              ┌─┐
    ┤ H ├──[Rz(θ₀)]──┤M├  q₀
    ├───┤              ├─┤
    ┤ H ├──[Rz(θ₁)]──┤M├  q₁
    ├───┤              ├─┤
    ...                ...
    └───┘              └─┘

    Layer 1 (mandatory):
        H on every qubit.  Creates the uniform superposition:
            |ψ⟩ = H⊗ⁿ|0⟩⊗ⁿ = (1/√2ⁿ) Σ_{x} |x⟩

    Layer 2+ (optional, circuit_layers > 1):
        Rz(θᵢ) rotations with random θᵢ ∈ [0, 2π).  These are NOT
        trainable parameters — they're freshly sampled each call.
        Rz(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
        This changes the complex phase of amplitudes, making the output
        pattern harder to correlate across calls (useful for passing
        the serial NIST test).

    Return value: list of n_qubits Pauli-Z expectation values sampled
    via projective measurement.  PennyLane returns the raw sample matrix
    when shots > 1.
    """

    @qml.qnode(device)
    def circuit():
        # ── LAYER 1: Hadamard on all qubits ──────────────────────────────
        # Puts each qubit in |+⟩ = (|0⟩+|1⟩)/√2
        # Joint state: |+⟩⊗ⁿ — uniform superposition, density matrix
        #   ρ = (I/2)⊗ⁿ  (maximally mixed in computational basis)
        for wire in range(cfg.n_qubits):
            qml.Hadamard(wires=wire)

        # ── LAYERS 2+: Random phase rotations ────────────────────────────
        # Doesn't change probabilities (|Rz·α|² = |α|²) but changes the
        # interference pattern, improving serial correlations in practice.
        for _ in range(cfg.circuit_layers - 1):
            thetas = np.random.uniform(0, 2 * np.pi, cfg.n_qubits)
            for wire in range(cfg.n_qubits):
                qml.RZ(thetas[wire], wires=wire)
            # Second round of Hadamards to convert phase differences → amplitude
            # differences (this is the core of interference-based computation)
            for wire in range(cfg.n_qubits):
                qml.Hadamard(wires=wire)

        # ── MEASUREMENT ──────────────────────────────────────────────────
        # Returns n_shots × n_qubits integer matrix (0s and 1s).
        # Each row is one shot; each column is one qubit measurement.
        return [qml.sample(qml.PauliZ(wire)) for wire in range(cfg.n_qubits)]

    return circuit


# ─────────────────────────────────────────────────────────────────────────────
# Main QRNG class
# ─────────────────────────────────────────────────────────────────────────────

class QRNG:
    """
    Quantum Random Number Generator.

    Usage
    ─────
        cfg  = QRNGConfig(n_qubits=8, n_shots=512)
        rng  = QRNG(cfg)
        bits = rng.generate_bits(256)     # list of 256 ints (0 or 1)
        bstr = rng.generate_bytes(32)     # 32 random bytes (256 bits)
        num  = rng.generate_int(0, 100)   # random int in [0, 100]
    """

    def __init__(self, cfg: QRNGConfig | None = None):
        self.cfg     = cfg or QRNGConfig()
        self._device = _build_device(self.cfg)
        self._qnode  = _make_circuit(self.cfg, self._device)
        self._buffer : list[int] = []   # leftover bits from previous calls
        logger.info(
            "QRNG initialised | %d qubits | %d shots | backend=%s | extractor=%s",
            self.cfg.n_qubits, self.cfg.n_shots,
            self.cfg.backend, self.cfg.apply_extractor
        )

    # ── Internal: run one circuit batch ──────────────────────────────────────

    def _run_circuit(self) -> list[int]:
        """
        Execute the quantum circuit and return a flat list of raw bits.

        PennyLane's qml.sample() with PauliZ returns +1 (for |0⟩) and -1
        (for |1⟩).  We convert to 0/1 with  bit = (1 - z_val) // 2.

        Shape of raw output: list of n_qubits arrays, each of length n_shots.
        We interleave qubits across shots to get a single long bit string,
        which has better statistical properties than taking all qubits from
        shot 1, then all from shot 2, etc.
        """
        t0       = time.perf_counter()
        samples  = self._qnode()           # list of n_qubits arrays of ±1
        elapsed  = time.perf_counter() - t0

        # Convert ±1 → 0/1 and reshape to (n_shots, n_qubits)
        # samples[i][j] = PauliZ measurement of qubit i on shot j
        arr = np.array(samples).T          # shape: (n_shots, n_qubits)
        bits_matrix = ((1 - arr) // 2).astype(int)  # +1→0, -1→1

        # Flatten in row-major order: all qubits from shot 0, then shot 1, ...
        raw_bits = bits_matrix.flatten().tolist()

        logger.debug(
            "Circuit run: %d raw bits in %.3f s  (%.0f bits/s)",
            len(raw_bits), elapsed, len(raw_bits) / elapsed
        )
        return raw_bits

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_bits(self, n: int) -> list[int]:
        """
        Return a list of exactly n bits (0 or 1).

        Internally, we maintain a buffer so we don't waste bits produced
        by previous circuit runs.  Refill by running more circuits when
        the buffer runs dry.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")

        while len(self._buffer) < n:
            raw = self._run_circuit()

            # Apply Von Neumann extractor if requested
            if self.cfg.apply_extractor:
                from .extractors import von_neumann_extract
                raw = von_neumann_extract(raw)

            self._buffer.extend(raw)

        # Pop the first n bits from the buffer
        result, self._buffer = self._buffer[:n], self._buffer[n:]
        return result

    def generate_bytes(self, n_bytes: int) -> bytes:
        """
        Return n_bytes of random bytes.

        Each byte is assembled from 8 bits.  We XOR with a SHA-256 hash of
        a timestamp+counter for defense-in-depth (belt-and-suspenders: even
        if the quantum circuit had a subtle bias we missed, the XOR makes the
        bias invisible to an attacker who doesn't know the timestamp).

        Note: the XOR does NOT add entropy — it's purely for obfuscation.
        The entropy comes entirely from the quantum measurements.
        """
        bits = self.generate_bits(n_bytes * 8)

        # Pack bits into bytes (MSB first within each byte)
        raw_bytes = bytearray()
        for i in range(n_bytes):
            byte_bits = bits[i * 8 : (i + 1) * 8]
            byte_val  = sum(b << (7 - j) for j, b in enumerate(byte_bits))
            raw_bytes.append(byte_val)

        # Whitening XOR: SHA-256(timestamp || counter) ⊕ raw_bytes
        seed_material = f"{time.time_ns()}:{id(self)}".encode()
        hash_bytes    = hashlib.sha256(seed_material).digest()
        # Extend hash if needed by repeating
        extended_hash = (hash_bytes * (n_bytes // 32 + 1))[:n_bytes]
        result = bytes(a ^ b for a, b in zip(raw_bytes, extended_hash))

        return result

    def generate_int(self, low: int, high: int) -> int:
        """
        Return a uniformly random integer in [low, high] inclusive.

        Uses rejection sampling to avoid modulo bias: generate ceil(log2(range))
        bits, interpret as an integer, reject if ≥ range.  This gives exactly
        uniform distribution with expected 2 trials.
        """
        if low > high:
            raise ValueError("low must be ≤ high")
        if low == high:
            return low

        range_size = high - low + 1
        n_bits     = int(np.ceil(np.log2(range_size)))

        # Rejection sampling loop — expected iterations: range_size / 2^n_bits ≤ 2
        for _ in range(100):   # hard cap for safety
            bits   = self.generate_bits(n_bits)
            value  = sum(b << (n_bits - 1 - i) for i, b in enumerate(bits))
            if value < range_size:
                return low + value

        # Fallback (astronomically unlikely with 100 retries)
        raise RuntimeError("Rejection sampling failed after 100 attempts")

    def generate_float(self) -> float:
        """
        Return a uniform float in [0, 1) with 53 bits of precision
        (matching the IEEE 754 double mantissa).
        """
        bits  = self.generate_bits(53)
        value = sum(b * 2**(-i - 1) for i, b in enumerate(bits))
        return value

    def state_vector_demo(self) -> np.ndarray:
        """
        Return the exact statevector for n_qubits=1 — for educational use.

        Shows that H|0⟩ produces (1/√2)[1, 1]ᵀ, confirming the density
        matrix ρ = (1/2)[[1,1],[1,1]] which has off-diagonal coherences
        proving this is NOT a classical probability mixture.
        """
        demo_dev = qml.device("default.qubit", wires=1)

        @qml.qnode(demo_dev)
        def single_qubit():
            qml.Hadamard(wires=0)
            return qml.state()   # returns the full statevector

        sv = np.array(single_qubit())
        print(f"\n  Statevector after H|0⟩:")
        print(f"  |ψ⟩ = {sv[0]:.6f}|0⟩ + {sv[1]:.6f}|1⟩")
        print(f"  |α|² = {abs(sv[0])**2:.6f}  (P(0))")
        print(f"  |β|² = {abs(sv[1])**2:.6f}  (P(1))")
        density_matrix = np.outer(sv, sv.conj())
        print(f"\n  Density matrix ρ = |ψ⟩⟨ψ|:")
        print(f"  {density_matrix}")
        print(f"\n  Off-diagonal coherences present: {abs(density_matrix[0,1]) > 1e-9}")
        print(f"  (A classical coin flip has ρ = [[0.5,0],[0,0.5]] — no coherences)\n")
        return sv
