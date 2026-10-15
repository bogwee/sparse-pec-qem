import numpy as np
import torch
from qiskit.quantum_info import PauliList
from src.pauli_utils import generate_pauli_basis, build_symplectic_matrix

class NoisyQuantumHardwareOracle:
    """
    Simulates a noisy quantum gate layer governed by an unknown Sparse Pauli-Lindblad
    noise channel with single-qubit errors and two-qubit ZZ/XX crosstalk.
    """
    def __init__(self, num_qubits: int, seed: int = 42):
        self.num_qubits = num_qubits
        self.rng = np.random.default_rng(seed)
        self.candidate_basis = generate_pauli_basis(num_qubits, max_weight=2)
        self.true_lambdas = self._initialize_true_hardware_rates()

    def _initialize_true_hardware_rates(self) -> torch.Tensor:
        """Assigns realistic sparse error rates to 1-local and 2-local ZZ/XX terms."""
        lambdas = np.zeros(len(self.candidate_basis), dtype=np.float32)
        for idx, pauli in enumerate(self.candidate_basis):
            label = pauli.to_label()
            weight = sum(1 for c in label if c != 'I')
            if weight == 1:
                lambdas[idx] = self.rng.uniform(0.002, 0.008)
            elif weight == 2 and ('ZZ' in label or 'XX' in label):
                # Nearest-neighbor correlated crosstalk
                lambdas[idx] = self.rng.uniform(0.010, 0.022)
        return torch.tensor(lambdas, dtype=torch.float32)

    def measure_cycle_fidelities(self, probe_paulis: PauliList, depths: list[int], shot_noise: float = 0.003) -> torch.Tensor:
        """
        Simulates Cycle Benchmarking measurements: returns measured Pauli fidelities
        F(P_i, d) = exp(-2 * d * sum_k M_{ik} * lambda_k) + Gaussian shot noise.
        """
        M = build_symplectic_matrix(probe_paulis, self.candidate_basis)
        decay_rates = 2.0 * torch.matmul(M, self.true_lambdas) # shape: (num_probes,)
        
        fidelities = []
        for d in depths:
            ideal_f = torch.exp(-d * decay_rates)
            noisy_f = ideal_f + torch.randn_like(ideal_f) * shot_noise
            fidelities.append(torch.clamp(noisy_f, 1e-4, 1.0))
            
        return torch.stack(fidelities, dim=1) # shape: (num_probes, num_depths)