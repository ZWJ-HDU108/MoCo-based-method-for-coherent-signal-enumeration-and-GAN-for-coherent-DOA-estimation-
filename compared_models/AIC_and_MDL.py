import torch
import numpy as np


def add_awgn(signal, snr_dB):
    sig_power = torch.mean(torch.abs(signal) ** 2)
    snr_linear = 10 ** (snr_dB / 10.0)
    noise_power = sig_power / snr_linear
    noise = torch.sqrt(noise_power / 2) * (
        torch.randn_like(signal.real) + 1j * torch.randn_like(signal.real)
    )
    return signal + noise


def aic_estimation(M, T, SNR, theta_deg, dd=0.5):
    """
        M: Number of array elements
        T: Number of snapshots
        SNR: Signal-to-noise ratio (dB)
        theta_deg: List of true source angles (degrees)
        dd: Element spacing (half-wavelength)
    """
    derad = torch.tensor(np.pi / 180)
    twpi = torch.tensor(2 * np.pi)
    k_true = len(theta_deg)
    theta = torch.tensor(theta_deg, dtype=torch.float64)
    d = torch.arange(M, dtype=torch.float64) * dd  # Element positions

    # Array manifold matrix (M, k)
    A = torch.exp(-1j * twpi * d.unsqueeze(1) * torch.sin(theta * derad).unsqueeze(0))

    # Generate signals
    S = torch.randn(k_true, T, dtype=torch.float64)

    # Generate received signals
    X0 = A @ S.to(torch.complex128)
    X = add_awgn(X0, SNR)

    # Covariance matrix
    R = (X @ X.conj().T) / T

    # Eigenvalue decomposition (sorted in descending order)
    EVA = torch.linalg.eigvalsh(R).flip(0)

    # Iterate through all candidate source numbers
    aic_values = torch.zeros(M, dtype=torch.float64)
    for i in range(M):
        noise_eigs = EVA[i:]                                  # The smallest M-i eigenvalues
        n = M - i
        arith_mean = noise_eigs.mean()                        # Arithmetic mean
        geom_mean = noise_eigs.prod() ** (1.0 / n)            # Geometric mean
        aic_values[i] = -2 * T * n * torch.log(geom_mean / arith_mean) + 2 * i * (2 * M - i)

    # Select the minimum AIC value
    idx = torch.argmin(aic_values)
    estimated_k = idx.item()

    print(f"True number of sources: {k_true}")
    print(f"AIC estimated number of sources: {estimated_k}")

    return estimated_k, aic_values


def mdl_estimation(M, T, SNR, theta_deg, dd=0.5):
    """
        M: Number of array elements
        T: Number of snapshots
        SNR: Signal-to-noise ratio (dB)
        theta_deg: List of true source angles (degrees)
        dd: Element spacing (half-wavelength)
    """
    derad = torch.tensor(np.pi / 180)
    twpi = torch.tensor(2 * np.pi)
    k_true = len(theta_deg)
    theta = torch.tensor(theta_deg, dtype=torch.float64)
    d = torch.arange(M, dtype=torch.float64) * dd

    # Array manifold matrix
    A = torch.exp(-1j * twpi * d.unsqueeze(1) * torch.sin(theta * derad).unsqueeze(0))

    # Generate signals
    S = torch.randn(k_true, T, dtype=torch.float64)

    # Generate received signals
    X0 = A @ S.to(torch.complex128)
    X = add_awgn(X0, SNR)

    # Covariance matrix
    R = (X @ X.conj().T) / T

    # Eigenvalue decomposition (sorted in descending order)
    EVA = torch.linalg.eigvalsh(R).flip(0)

    # Iterate through all candidate source numbers
    mdl_values = torch.zeros(M, dtype=torch.float64)
    for i in range(M):
        noise_eigs = EVA[i:]
        n = M - i
        arith_mean = noise_eigs.mean()
        geom_mean = noise_eigs.prod() ** (1.0 / n)
        mdl_values[i] = -T * n * torch.log(geom_mean / arith_mean) + 0.5 * i * (2 * M - i) * np.log(T)

    # Select the minimum MDL value
    idx = torch.argmin(mdl_values)
    estimated_k = idx.item()

    print(f"True number of sources: {k_true}")
    print(f"MDL estimated number of sources: {estimated_k}")

    return estimated_k, mdl_values