# SparsePEC: Differentiable Pauli-Lindblad Noise Learning for Overhead-Optimal Probabilistic Error Cancellation

## Overview
**SparsePEC** is an applied mathematics and quantum error mitigation framework that learns hardware-correlated noise channels and computes quasi-probability decompositions for **Probabilistic Error Cancellation (PEC)**. By formulating noise tomography as a differentiable $L_1$-regularized optimization problem in PyTorch, this framework reconstructs sparse Pauli-Lindblad generators and tunes the bias-variance trade-off to drastically reduce Monte Carlo sampling overhead ($\gamma^2$).

## Motivation
Near-term and early fault-tolerant quantum processors suffer from correlated two-qubit crosstalk that violates simple local depolarizing assumptions. While Probabilistic Error Cancellation (PEC) provides an unbiased estimator of noiseless expectation values by inverting the hardware noise channel $\mathcal{N}^{-1}$, it faces two major engineering bottlenecks:
1. **Characterization Complexity:** Full Quantum Process Tomography scales as $\mathcal{O}(4^{2n})$, making it intractable beyond a few qubits.
2. **Exponential Sampling Overhead:** The number of circuit shots required to maintain a fixed variance scales as $\gamma^{2L}$, where $\gamma = \sum_i \vert{}\eta_i\vert{} \ge 1$ is the quasi-probability negativity factor and $L$ is the circuit depth.

This project implements a **Sparse Pauli-Lindblad noise learner** paired with a **regularized quasi-probability optimizer** to compress the active error basis and slash sampling overhead while preserving observable fidelity.

## Core Objectives (Missions)
1. **Correlated Noise Emulation:** Simulate an $n$-qubit superconducting/neutral-atom gate layer subject to local depolarizing noise and non-local $ZZ$ crosstalk.
2. **Differentiable Pauli-Lindblad Tomography:** Learn the sparse model rates $\vec{\lambda}$ of the Lindbladian generator $\mathcal{L}(\rho) = \sum_{k \in \mathcal{K}} \lambda_k (P_k \rho P_k^\dagger - \rho)$ from randomized Pauli fidelity decay data using PyTorch.
3. **Analytic & Regularized Channel Inversion:** Compute the quasi-probability representation $\mathcal{N}^{-1}(\rho) = \sum_k \eta_k P_k \rho P_k^\dagger$ and optimize the bias-variance trade-off to minimize $\gamma^2$.
4. **Monte Carlo Trajectory Mitigation:** Execute importance-sampled noisy circuits on a 6-qubit Trotterized Transverse-Field Ising Model (TFIM) and benchmark against unmitigated and standard Mitiq baselines.

## Tech Stack
* **Quantum Simulation & Mitigation:** `Qiskit`, `Qiskit-Aer`, `Mitiq`
* **Optimization & Autodiff:** `PyTorch`, `SciPy`
* **Scientific Computing & Visualization:** `NumPy`, `Matplotlib`, `PyYAML`