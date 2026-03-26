import numpy as np
import h5py
from itertools import combinations
from tqdm import tqdm
import os
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


class Config:
    M = 16  # Number of array elements
    d = 0.5  # Element spacing

    K = 2  # Number of sources

    MAX_DOA = 60  # Field of view range: ±60
    GRID_RES = 1  # Resolution = 1

    SNR_MIN = -10  # Minimum SNR
    SNR_MAX = 20   # Maximum SNR
    SNR_STEP = 2   # SNR step size

    T = 400  # Number of snapshots

    # Output path
    OUTPUT_DIR = 'Switch to your path'
    OUTPUT_FILENAME = 'Switch to your path'


def steering_vector(theta_deg: float, M: int, d: float) -> np.ndarray:
    theta_rad = np.deg2rad(theta_deg)
    n = np.arange(M)
    a = np.exp(1j * 2 * np.pi * d * np.sin(theta_rad) * n)
    return a


def generate_incoherent_signal(
        A: np.ndarray,  # Array manifold matrix
        K: int,         # Number of sources
        T: int,         # Number of snapshots
        noise_power: float
) -> Tuple[np.ndarray, np.ndarray]:
    M = A.shape[0]

    # Generate K independent complex Gaussian source signals
    S = (np.random.randn(K, T) + 1j * np.random.randn(K, T)) / np.sqrt(2)

    # Array received signal
    X = A @ S

    # Generate complex Gaussian white noise
    N = np.sqrt(noise_power / 2) * (np.random.randn(M, T) + 1j * np.random.randn(M, T))

    # Received signal with noise
    Y = X + N

    # Sample covariance matrix
    R_scm = (Y @ Y.conj().T) / T

    return Y, R_scm


def generate_coherent_signal(
        A: np.ndarray,  # Array manifold matrix
        K: int,         # Number of sources
        T: int,         # Number of snapshots
        noise_power: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = A.shape[0]

    # Generate reference signal
    s1 = (np.random.randn(1, T) + 1j * np.random.randn(1, T)) / np.sqrt(2)

    phi_vec = np.random.uniform(0, 2 * np.pi, size=K - 1)  # Generate K-1 coherence coefficients α_k = e^{jφ_k}, φ_k ∈ [0, 2π) random phase
    alpha_vec = np.exp(1j * phi_vec)

    # Construct coherence coefficient vector [1, α_2, α_3, ..., α_K]^T
    coeffs = np.concatenate([[1], alpha_vec]).reshape(K, 1)  # shape: (K, 1)

    # Construct coherent signal matrix S = coeffs * s1
    S = coeffs @ s1  # shape: (K, T), rank=1

    # Array received signal
    X = A @ S

    # Generate complex Gaussian white noise
    N = np.sqrt(noise_power / 2) * (np.random.randn(M, T) + 1j * np.random.randn(M, T))

    # Received signal with noise
    Y = X + N

    # Sample covariance matrix
    R_scm = (Y @ Y.conj().T) / T

    return Y, R_scm, phi_vec


def noise_whitening(R: np.ndarray) -> np.ndarray:
    """Subtract estimated noise power from covariance matrix"""
    M = R.shape[0]

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(R)

    # Estimate noise power using the minimum eigenvalue
    noise_estimate = np.min(eigenvalues)

    # Subtract noise term
    R_whitened = R - noise_estimate * np.eye(M)

    # Ensure positive semi-definite
    eigenvalues_w, eigenvectors = np.linalg.eigh(R_whitened)
    eigenvalues_w = np.maximum(eigenvalues_w, 0)  # Achieved by truncating negative eigenvalues to 0
    R_whitened = eigenvectors @ np.diag(eigenvalues_w) @ eigenvectors.conj().T

    return R_whitened


def normalize_covariance_matrix(R: np.ndarray) -> np.ndarray:
    """Normalize covariance matrix and extract three-channel representation"""
    M = R.shape[0]

    # Find maximum magnitude value and normalize
    max_val = np.max(np.abs(R))
    if max_val > 0:
        R_norm = R / max_val
    else:
        R_norm = R

    # Extract three channels
    R_3ch = np.zeros((3, M, M), dtype=np.float32)

    R_3ch[0, :, :] = np.real(R_norm)
    R_3ch[1, :, :] = np.imag(R_norm)
    phase = np.angle(R_norm)
    R_3ch[2, :, :] = phase / np.pi

    return R_3ch


def generate_single_sample(
        angles: Tuple,  # Source angles, supports any K sources
        snr_db: float,  # Signal-to-noise ratio
        M: int,         # Number of array elements
        d: float,       # Element spacing
        T: int          # Number of snapshots
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single sample pair"""
    K = len(angles)
    noise_power = 10 ** (-snr_db / 10)

    # Construct array manifold matrix
    A = np.zeros((M, K), dtype=np.complex128)
    for k, theta in enumerate(angles):
        A[:, k] = steering_vector(theta, M, d)

    _, R_incoh = generate_incoherent_signal(A, K, T, noise_power)  # Generate incoherent signal covariance matrix (full rank: Target)

    _, R_coh, phi = generate_coherent_signal(A, K, T, noise_power)  # Generate coherent signal covariance matrix (rank deficient: Input)

    # Remove noise term, preserve signal subspace rank characteristics
    R_coh = noise_whitening(R_coh)
    R_incoh = noise_whitening(R_incoh)

    # Normalize and extract three channels
    R_coh_3ch = normalize_covariance_matrix(R_coh)
    R_incoh_3ch = normalize_covariance_matrix(R_incoh)

    return R_coh_3ch, R_incoh_3ch, phi


def process_batch(args):
    """Process a batch of samples"""
    batch_indices, angle_pairs, snr_values, config = args

    results = []
    for idx in batch_indices:
        # Calculate angle pair index and SNR index
        angle_idx = idx // len(snr_values)
        snr_idx = idx % len(snr_values)

        angles = angle_pairs[angle_idx]
        snr_db = snr_values[snr_idx]

        R_coh, R_incoh, phi = generate_single_sample(
            angles, snr_db, config['M'], config['d'], config['T']
        )

        results.append({
            'idx': idx,
            'R_coh': R_coh,
            'R_incoh': R_incoh,
            'angles': np.array(angles),
            'snr': snr_db,
            'phi': phi
        })

    return results


def generate_dataset(config: Config,  # Configuration parameters
                     use_parallel: bool = True,  # Whether to use parallel computation
                     n_workers: Optional[int] = None  # Number of parallel worker processes, defaults to CPU cores - 1
                     ):
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILENAME)

    # Generate angle grid and combinations
    grids = np.arange(-config.MAX_DOA, config.MAX_DOA + config.GRID_RES, config.GRID_RES)
    angle_pairs = list(combinations(grids, config.K))

    # Generate SNR value list
    snr_values = np.arange(config.SNR_MIN, config.SNR_MAX + config.SNR_STEP, config.SNR_STEP)

    # Calculate total number of samples
    n_angle_pairs = len(angle_pairs)
    n_snr_levels = len(snr_values)
    n_samples = n_angle_pairs * n_snr_levels

    # Initialize data arrays
    coherent_data = np.zeros((n_samples, 3, config.M, config.M), dtype=np.float32)
    incoherent_data = np.zeros((n_samples, 3, config.M, config.M), dtype=np.float32)
    angles_data = np.zeros((n_samples, config.K), dtype=np.float32)
    snr_data = np.zeros(n_samples, dtype=np.float32)
    phi_data = np.zeros((n_samples, config.K - 1), dtype=np.float32)  # K-1 coherence coefficient phases

    # Configuration dictionary (for parallel processing)
    config_dict = {
        'M': config.M,
        'd': config.d,
        'T': config.T
    }

    if use_parallel and n_samples > 100:
        # Parallel generation
        if n_workers is None:
            n_workers = max(1, multiprocessing.cpu_count() - 1)

        print(f"\nUsing {n_workers} processes for parallel data generation...")

        # Batch processing
        batch_size = max(1, n_samples // (n_workers * 10))
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_indices = list(range(i, min(i + batch_size, n_samples)))
            batches.append((batch_indices, angle_pairs, snr_values, config_dict))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]

            with tqdm(total=n_samples, desc="Generating samples") as pbar:
                for future in as_completed(futures):
                    results = future.result()
                    for res in results:
                        idx = res['idx']
                        coherent_data[idx] = res['R_coh']
                        incoherent_data[idx] = res['R_incoh']
                        angles_data[idx] = res['angles']
                        snr_data[idx] = res['snr']
                        phi_data[idx] = res['phi']
                    pbar.update(len(results))
    else:
        # Serial generation
        print("\nGenerating data serially...")
        sample_idx = 0

        with tqdm(total=n_samples, desc="Generating samples") as pbar:
            for angle_idx, angles in enumerate(angle_pairs):
                for snr_idx, snr_db in enumerate(snr_values):
                    R_coh, R_incoh, phi = generate_single_sample(
                        angles, snr_db, config.M, config.d, config.T
                    )

                    coherent_data[sample_idx] = R_coh
                    incoherent_data[sample_idx] = R_incoh
                    angles_data[sample_idx] = np.array(angles)
                    snr_data[sample_idx] = snr_db
                    phi_data[sample_idx] = phi

                    sample_idx += 1
                    pbar.update(1)

    # Save to HDF5 file
    print(f"\nSaving data to: {output_path}")

    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(output_path, 'w') as f:
        # Main datasets
        f.create_dataset('coherent_input', data=coherent_data,
                         compression='gzip', compression_opts=4)
        f.create_dataset('incoherent_target', data=incoherent_data,
                         compression='gzip', compression_opts=4)

        # Labels and metadata
        f.create_dataset('angles', data=angles_data)
        f.create_dataset('snr', data=snr_data)
        f.create_dataset('coherent_phase', data=phi_data)

        # Save configuration information as attributes
        f.attrs['ULA_M'] = config.M
        f.attrs['d'] = config.d
        f.attrs['SOURCE_K'] = config.K
        f.attrs['MAX_DOA'] = config.MAX_DOA
        f.attrs['GRID_RES'] = config.GRID_RES
        f.attrs['SNR_MIN'] = config.SNR_MIN
        f.attrs['SNR_MAX'] = config.SNR_MAX
        f.attrs['SNR_STEP'] = config.SNR_STEP
        f.attrs['SNAPSHOTS'] = config.T
        f.attrs['n_samples'] = n_samples
        f.attrs['n_angle_pairs'] = n_angle_pairs
        f.attrs['n_snr_levels'] = n_snr_levels

    return output_path


def verify_rank_deficiency(output_path: str, n_samples: int = 5):
    """Verify the rank deficiency characteristic of coherent signal covariance matrices by comparing singular values of coherent and incoherent covariance matrices"""
    with h5py.File(output_path, 'r') as f:
        # Randomly select a few samples
        total_samples = f['coherent_input'].shape[0]
        indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)

        for idx in indices:
            # Reconstruct complex matrices
            coh_data = f['coherent_input'][idx]
            incoh_data = f['incoherent_target'][idx]
            angles = f['angles'][idx]
            snr = f['snr'][idx]

            # Recover complex matrix from three channels (using only real and imaginary parts)
            R_coh = coh_data[0] + 1j * coh_data[1]
            R_incoh = incoh_data[0] + 1j * incoh_data[1]

            # Compute singular values
            sv_coh = np.linalg.svd(R_coh, compute_uv=False)
            sv_incoh = np.linalg.svd(R_incoh, compute_uv=False)

            # Normalize singular values
            sv_coh_norm = sv_coh / sv_coh[0]
            sv_incoh_norm = sv_incoh / sv_incoh[0]

            # Estimate effective rank (singular values > 0.01 * maximum singular value)
            rank_coh = np.sum(sv_coh_norm > 0.01)
            rank_incoh = np.sum(sv_incoh_norm > 0.01)

            print(f"\nSample {idx}: angles={angles}, SNR={snr:.1f}dB")
            print(f"  Coherent signal - effective rank: {rank_coh}, first 5 normalized singular values: {sv_coh_norm[:5]}")
            print(f"  Incoherent signal - effective rank: {rank_incoh}, first 5 normalized singular values: {sv_incoh_norm[:5]}")


if __name__ == '__main__':
    # Create configuration
    config = Config()

    # Modify configuration as needed
    # config.SNR_MIN = -10
    # config.SNR_MAX = 10
    # config.T = 200

    # Generate dataset
    output_path = generate_dataset(config, use_parallel=False)

    # Verify rank deficiency characteristic
    verify_rank_deficiency(output_path, n_samples=5)