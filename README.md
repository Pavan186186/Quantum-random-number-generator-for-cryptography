# Quantum-random-number-generator-for-cryptography
```python
readme_content = """# True Quantum Random Number Generator (QRNG) Pipeline

## Overview
This project bridges quantum mechanics and classical cryptography by generating true, non-deterministic random numbers. Unlike classical Pseudo-Random Number Generators (PRNGs) like the Mersenne Twister, this pipeline uses the superposition collapse of a qubit to extract physical entropy. 

Because real quantum hardware is noisy and biased, this pipeline includes mathematical post-processing (Randomness Extractors) to produce perfectly uniform, mathematically secure bitstreams that pass the NIST Statistical Test Suite.

## Pipeline Architecture
1. **Entropy Source (Quantum Hardware):** Uses Qiskit to run a single-qubit Hadamard superposition circuit on an IBM Quantum backend.
2. **Min-Entropy Estimation:** Evaluates the raw hardware bias (e.g., T1 relaxation favoring `0` states).
3. **Randomness Extractor:** Squeezes hardware bias out of the raw bitstream using either a **Von Neumann Extractor** (simple, low yield) or a **Toeplitz Matrix Extractor** (strong extraction via Galois Field 2 arithmetic).
4. **NIST Validation:** Runs the extracted stream through the NIST SP 800-22 statistical tests to ensure cryptographic uniformity.
5. **Cryptographic Sink:** Feeds the certified entropy into an HKDF to generate a secure AES-256 key.

## Tech Stack
* **Quantum Backend:** `qiskit`, `qiskit-ibm-runtime`
* **Math & Post-Processing:** `numpy`, `galois` (for GF(2) matrix multiplication)
* **Benchmarking:** `nistrng` 
* **Cryptography:** `cryptography`

## Installation

```

```text
README.md generated successfully.

```bash
git clone [https://github.com/yourusername/qrng-pipeline.git](https://github.com/yourusername/qrng-pipeline.git)
cd qrng-pipeline
pip install -r requirements.txt

```

## Configuration

To run on physical hardware (rather than a local CPU simulator), you need an IBM Quantum API token.

1. Create a free account at [IBM Quantum Platform](https://quantum.ibm.com/).
2. Copy your API token from the dashboard.
3. Export it as an environment variable (or add it to a `.env` file):
```bash
export IBM_QUANTUM_TOKEN="your_api_token_here"

```



## Usage

**Step 1: Generate Raw Entropy**
Execute the quantum circuit to collect raw, biased measurements.

```bash
python src/1_generate_raw_bits.py --shots 20000 --backend hardware

```

**Step 2: Post-Processing**
Debias the raw measurements to achieve a perfect 50/50 distribution.

```bash
python src/2_extractor.py --method toeplitz --input raw_bits.json --output extracted_bits.bin

```

**Step 3: Cryptographic Key Generation**
Validate the uniformity of the bits and derive a secure AES encryption key.

```bash
python src/3_derive_keys.py --input extracted_bits.bin

```

## Disclaimer

This is an educational project demonstrating quantum mechanics and cryptographic post-processing. While the quantum randomness is true, do not use this pipeline in a production environment for securing highly sensitive real-world assets without proper hardware auditing and physical security protocols.