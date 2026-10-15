import itertools
import numpy as np
import torch
from qiskit.quantum_info import Pauli, PauliList

def generate_pauli_basis(num_qubits: int, max_weight: int = 2) -> PauliList:
    """
    Generates all n-qubit Pauli strings up to a specified locality weight
    (e.g., weight <= 2 captures single-qubit and nearest-neighbor crosstalk errors).
    """
    alphabet = ['I', 'X', 'Y', 'Z']
    basis_labels = []
    for prod in itertools.product(alphabet, repeat=num_qubits):
        weight = sum(1 for char in prod if char != 'I')
        if 0 < weight <= max_weight:
            basis_labels.append("".join(prod))
    return PauliList(basis_labels)

def build_symplectic_matrix(pauli_list_row: PauliList, pauli_list_col: PauliList) -> torch.Tensor:
    """
    Builds the binary commutation matrix M where:
    M[i, j] = 0 if Pauli i and Pauli j commute
    M[i, j] = 1 if Pauli i and Pauli j anticommute
    """
    n_rows = len(pauli_list_row)
    n_cols = len(pauli_list_col)
    M = np.zeros((n_rows, n_cols), dtype=np.float32)
    
    for i, p_row in enumerate(pauli_list_row):
        # anti_commutes returns boolean array against all elements in pauli_list_col
        M[i, :] = p_row.anticommutes(pauli_list_col).astype(np.float32)
        
    return torch.tensor(M, dtype=torch.float32)

def build_walsh_hadamard_sign_matrix(pauli_basis: PauliList) -> torch.Tensor:
    """
    Builds the Walsh-Hadamard sign matrix W[i, j] = (-1)^<P_i, P_j>_symplectic
    Used to convert between Pauli error probabilities p_k and eigenvalues f_k.
    """
    M = build_symplectic_matrix(pauli_basis, pauli_basis)
    return 1.0 - 2.0 * M