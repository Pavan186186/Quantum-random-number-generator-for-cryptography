# Quantum Random Number Generator (QRNG)

True randomness from quantum measurement indeterminacy, using PennyLane.

## Quick Start

```bash
pip install pennylane scipy numpy
python main.py --quick          # fast demo (~30 seconds)
python main.py --bits 100000    # full NIST suite
```

## Project Structure

```
qrng/
├── main.py                   ← Run this
├── requirements.txt
├── core/
│   ├── qrng.py               ← Quantum circuit + QRNG class
│   └── extractors.py         ← Von Neumann + Toeplitz extractors
├── tests/
│   └── nist_tests.py         ← NIST SP 800-22 statistical tests
└── crypto/
    └── keygen.py             ← AES key gen + quantum vs classical comparison
```
## Mathematical Foundation

### Why quantum measurement is fundamentally random

A qubit in superposition has no definite value before measurement.
This is not epistemic uncertainty (we don't know it yet) — it is
ontic indeterminacy (there is no fact of the matter).

Bell's theorem (1964) and the Kochen-Specker theorem (1967) together
rule out hidden variable theories: no pre-determined values exist that
could predict measurement outcomes. The randomness is real.

### The circuit

```
|0⟩ ──[ H ]──[ Rz(θ) ]──[ H ]──┤M├→ bit
```

- `H` (Hadamard): takes |0⟩ to (|0⟩+|1⟩)/√2
- `Rz(θ)`: phase rotation — changes interference pattern
- `M`: projective measurement, collapses superposition

### Key equations

Hadamard gate:
```
H = (1/√2) [[1, 1], [1, -1]]
H|0⟩ = (1/√2)(|0⟩ + |1⟩)
```

Density matrix of superposition (has off-diagonal coherences):
```
ρ = |ψ⟩⟨ψ| = (1/2)[[1, 1], [1, 1]]
```

Born rule (measurement probabilities):
```
P(0) = Tr(|0⟩⟨0| · ρ) = 1/2
P(1) = Tr(|1⟩⟨1| · ρ) = 1/2
```

Von Neumann extractor (debiasing):
```
Input pairs: (0,1)→0    (1,0)→1    (0,0)→discard    (1,1)→discard
P(output 0) = p(1-p) / [2p(1-p)] = 1/2  ✓ (unbiased regardless of p)
```

Toeplitz extractor (Leftover Hash Lemma):
```
y = T·x mod 2   where T is an m×n matrix over GF(2)
If min-entropy(x) ≥ m + 2·security_param:
    ||P_y - Uniform||_1 ≤ 2^{-security_param}
```

## Connecting to Real IBM Quantum Hardware

```python
import os
from core.qrng import QRNG, QRNGConfig

cfg = QRNGConfig(
    backend   = "ibm",
    ibm_token = os.environ["IBMQ_TOKEN"],   # from quantum.ibm.com
    n_qubits  = 5,     # stay small — real hardware has more noise
    n_shots   = 1024,
)
rng = QRNG(cfg)
bits = rng.generate_bits(1000)
```

Get a free IBM Quantum account at https://quantum.ibm.com

## NIST Test Descriptions

| Test | What it checks |
|------|---------------|
| Frequency (Monobit) | Equal number of 0s and 1s overall |
| Block Frequency | Equal 0/1 ratio within every M-bit block |
| Runs | Number of consecutive-identical-bit runs is correct |
| Longest Run | Longest run of 1s matches expected distribution |
| Serial (m=2) | All 2-bit patterns (00,01,10,11) appear equally |
| Serial (m=3) | All 3-bit patterns appear equally |

Pass criterion for each: p-value ≥ 0.01

## Extending This Project

1. **Implement the remaining 9 NIST tests** (see NIST SP 800-22 PDF, free online)
2. **Add noise simulation**: replace `default.qubit` with `default.mixed`
   and add `qml.DepolarizingChannel(p=0.01, wires=i)` after each gate
3. **Benchmark on real IBM hardware**: compare bias p before/after extraction
4. **Quantum advantage proof**: show that after running a seeded PRNG for
   N steps, an adversary can predict all future bits; show the QRNG output
   cannot be predicted even with full knowledge of the circuit

## Why this project matters

Every AES key, TLS session, RSA prime test, and ECDH exchange begins with
a random number.  Classical PRNGs are deterministic.  OS CSPRNGs harvest
physical entropy (timing jitter, interrupt latency) which can be influenced
by a sophisticated attacker who controls the hardware environment.

Quantum measurement entropy cannot be influenced or predicted — not even
by the device manufacturer.  This is the eventual foundation of post-quantum
cryptographic key generation.
