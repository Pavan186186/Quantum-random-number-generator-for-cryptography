from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# ==========================================
# 1. AUTHENTICATION & BACKEND SELECTION
# ==========================================
# Save your account to disk once, then you can remove the token parameter
service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_IBM_API_TOKEN_HERE")

# To test for free without using your quota, change simulator=False to simulator=True
print("Finding the least busy quantum computer...")
backend = service.least_busy(operational=True, simulator=False)
print(f"Routing job to: {backend.name}")

# ==========================================
# 2. BUILD THE QUANTUM CIRCUIT
# ==========================================
# Create a circuit with 1 Qubit and 1 Classical Bit for readout
qc = QuantumCircuit(1, 1)

# Apply the Hadamard Gate to put the qubit in a 50/50 superposition
qc.h(0)

# Measure the qubit (collapses the superposition into a 0 or 1)
qc.measure(0, 0)

# ==========================================
# 3. EXECUTE ON HARDWARE
# ==========================================
# We use the Sampler primitive to run the circuit thousands of times (shots)
# IBM's free tier usually allows up to 10,000 or 20,000 shots per job.
shots_requested = 10000
sampler = Sampler(backend=backend)

print(f"Executing {shots_requested} shots...")
# Submit the job to the quantum hardware
job = sampler.run([qc], shots=shots_requested)
result = job.result()

# ==========================================
# 4. EXTRACT THE RAW BITSTREAM
# ==========================================
# The output is stored as bitstrings (e.g., {"0": 5120, "1": 4880})
# We want to expand this into a flat list of raw integers for post-processing
pub_result = result[0]
counts = pub_result.data.c.get_counts()

raw_bitstream = []
for bit_string, count in counts.items():
    # Append 'count' number of 0s or 1s to our stream
    raw_bitstream.extend([int(bit_string)] * count)

print(f"Generated {len(raw_bitstream)} raw quantum bits.")
print(f"Distribution Bias: {counts}")
print(f"First 50 bits: {raw_bitstream[:50]}")

# You now have your raw entropy array ready for post-processing!