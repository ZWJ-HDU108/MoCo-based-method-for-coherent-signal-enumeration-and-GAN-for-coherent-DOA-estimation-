import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


def generate_steering_matrix(M: int, doa_deg: np.ndarray, d: float = 0.5) -> np.ndarray:
    doa_rad = np.deg2rad(doa_deg)   # (K, 1)
    n = np.arange(M).reshape(-1, 1)    # [0, 1, 2, ..., M-1]^T
    A = np.exp(-1j * 2 * np.pi * d * n * np.sin(doa_rad).reshape(1, -1))  # (M, K)
    return A


def generate_coh_matrix(K: int, rng: np.random.Generator, mode: str = "random") -> np.ndarray:
    """
    K: number of sources
    rng: NumPy random number generator
    mode: coherent mode
        "full": fully coherent
        "partial": partially coherent, with rank randomly distributed in [1, K-1], excluding complete incoherence
        "random": mixed mode, with rank randomized in [1, K], including fully coherent/partially coherent/completely incoherent
        "adaptive": K<=4 fully coherent, K>4 partially coherent
    """

    # If the number of signals is less than or equal to 1, there is no coherence issue, return identity matrix directly
    if K <= 1:
        return np.eye(K, dtype=np.complex128)

    # Initialize empty matrix, will later truncate to actual columns used
    C = np.zeros((K, K), dtype=np.complex128)

    # Determine number of coherent groups based on mode
    if mode == "full":
        n_groups = 1  # Fully coherent: rank is always 1
    elif mode == "partial":
        n_groups = rng.integers(1, K)  # Partially coherent: rank 1 ~ K-1
    elif mode == "adaptive":
        if K <= 4:
            n_groups = 1  # K<=4: fully coherent
        else:
            n_groups = rng.integers(1, K)  # K>4: partially coherent
    else:  # "random"
        n_groups = rng.integers(1, K + 1)  # Mixed: rank 1 ~ K

    # Assign group IDs to each source
    group_ids = np.sort(rng.choice(n_groups, size=K, replace=True))  # Randomly sample K times from n_groups, e.g., n_groups=2, K=3, sample 3 times from [0,1], could be [0,0,1] or [0,1,1], then sort
    unique_groups = np.unique(group_ids)  # Remove duplicates

    ind_col = 0  # Current independent source column index
    for g in unique_groups:
        members = np.where(group_ids == g)[0]  # Return all indices where group_ids == g
        master = members[0]  # Select the first member in the group as the master source

        # Master source
        C[master, ind_col] = 1.0 + 0j

        # Other coherent signals within the group
        for m in members[1:]:
            rho = rng.uniform(0.3, 1.0)            # Coherent amplitude
            phi = rng.uniform(0, 2 * np.pi)         # Coherent phase
            C[m, ind_col] = rho * np.exp(1j * phi)

        ind_col += 1

    # Truncate to actual independent source columns used
    C = C[:, :ind_col]
    return C


def _sample_doas_with_separation(
    K: int,
    doa_range: Tuple[float, float],
    min_sep: float,
    rng: np.random.Generator,
    max_attempts: int = 500,
) -> np.ndarray:
    """
    K: number of angles
    doa_range: (min_angle, max_angle)
    min_sep: minimum angle separation
    rng: random number generator
    max_attempts: maximum number of rejection sampling attempts
    """

    lo, hi = doa_range  # DOA lower and upper bounds

    # Rejection sampling
    for _ in range(max_attempts):
        doas = rng.uniform(lo, hi, size=K)  # Uniform distribution, randomly select K angles
        doas.sort()
        if np.all(np.diff(doas) >= min_sep):
            rng.shuffle(doas)  # Shuffle to eliminate order bias
            return doas

    raise ValueError(f"Can not find proper DOA in {max_attempts} attemps.")


def generate_single_sample(
    M: int,
    K: int,
    rng: np.random.Generator,
    doa_range: Tuple[float, float] = (-60.0, 60.0),
    snapshot_range: Tuple[int, int] = (50, 400),
    snr_range: Tuple[int, int] = (-15, 15),
    min_angle_sep: float = 2.0,
    coh_mode: str = "random",
) -> np.ndarray:
    """
    M: number of array elements
    K: number of sources (class label), K=0 means pure noise
    rng: random number generator
    doa_range: DOA range
    snapshot_range: snapshot count range
    snr_range: signal-to-noise ratio range
    min_angle_sep: minimum angle separation
    """

    # Randomly select snapshot count and SNR
    T = rng.integers(snapshot_range[0], snapshot_range[1] + 1)
    snr_dB = rng.uniform(snr_range[0], snr_range[1])
    snr_linear = 10 ** (snr_dB / 10.0)

    if K == 0:
        # No incident signals
        noise = (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) / np.sqrt(2)
        X = noise
    else:
        # Generate DOA angles with minimum separation
        doas = _sample_doas_with_separation(K, doa_range, min_angle_sep, rng)

        # Array manifold matrix
        A = generate_steering_matrix(M, doas)       # (M, K)

        # Coherent mixing matrix
        C = generate_coh_matrix(K, rng, mode=coh_mode)        # (K, K_ind)
        K_ind = C.shape[1]                           # Actual number of independent sources

        # Effective steering matrix
        A_eff = A @ C                                # (M, K_ind)

        S = (rng.standard_normal((K_ind, T)) + 1j * rng.standard_normal((K_ind, T))) / np.sqrt(2)
        S = np.sqrt(snr_linear) * S

        noise = (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) / np.sqrt(2)

        X = A_eff @ S + noise                        # (M, T)

    R = (X @ X.conj().T) / T                         # (M, M)

    # Normalize by trace to eliminate absolute power
    trace_val = np.abs(np.trace(R))
    if trace_val > 1e-12:
        R_norm = R / trace_val
    else:
        R_norm = R

    return R_norm


def generate_single_sample_with_fixed_doa(
    M: int,
    K: int,
    doas: np.ndarray,
    rng: np.random.Generator,
    snapshot_range: Tuple[int, int] = (50, 400),
    snr_range: Tuple[int, int] = (-15, 15),
    coh_mode: str = "random",
) -> np.ndarray:
    """
    Same as generate_single_sample, but uses externally provided fixed DOA
    Two views share DOA, but SNR, snapshot count, noise, and coherent matrix are all independent
    """

    T = rng.integers(snapshot_range[0], snapshot_range[1] + 1)
    snr_dB = rng.uniform(snr_range[0], snr_range[1])
    snr_linear = 10 ** (snr_dB / 10.0)

    if K == 0:
        noise = (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) / np.sqrt(2)
        X = noise
    else:
        A = generate_steering_matrix(M, doas)        # (M, K)
        C = generate_coh_matrix(K, rng, mode=coh_mode)               # (K, K_ind)
        K_ind = C.shape[1]
        A_eff = A @ C                                 # (M, K_ind)

        S = (rng.standard_normal((K_ind, T)) + 1j * rng.standard_normal((K_ind, T))) / np.sqrt(2)
        S = np.sqrt(snr_linear) * S

        noise = (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) / np.sqrt(2)
        X = A_eff @ S + noise

    R = (X @ X.conj().T) / T

    trace_val = np.abs(np.trace(R))
    if trace_val > 1e-12:
        R_norm = R / trace_val
    else:
        R_norm = R

    return R_norm


def scm_to_3channel(R: np.ndarray) -> np.ndarray:
    real_part = np.real(R)
    imag_part = np.imag(R)
    phase_part = np.angle(R) / np.pi  # Normalize to [-1, 1]

    features = np.stack([real_part, imag_part, phase_part], axis=0)
    return features.astype(np.float32)


class CohSourceDataset(Dataset):
    """
    num_samples_per_class: number of samples per class per epoch
    M: number of array elements (default 16)
    K_max: maximum number of sources (default 10, total 11 classes: 0~10)
    doa_range: DOA angle range
    snapshot_range: snapshot count range
    snr_range: SNR range
    min_angle_sep: minimum angle separation
    feature_mode: (3, M, M)
    seed: random seed
    """

    def __init__(
        self,
        num_samples_per_class: int = 1000,
        M: int = 16,
        K_max: int = 10,
        doa_range: Tuple[float, float] = (-60.0, 60.0),
        snapshot_range: Tuple[int, int] = (20, 400),
        snr_range: Tuple[int, int] = (-15, 15),
        min_angle_sep: float = 2.0,
        coh_mode: str = "random",
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.M = M
        self.K_max = K_max
        self.num_classes = K_max + 1                        # 0, 1, ..., K_max
        self.num_samples_per_class = num_samples_per_class
        self.total_samples = self.num_classes * num_samples_per_class
        self.doa_range = doa_range
        self.snapshot_range = snapshot_range
        self.snr_range = snr_range
        self.min_angle_sep = min_angle_sep
        self.coh_mode = coh_mode
        self.seed = seed

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        K = idx // self.num_samples_per_class

        if self.seed is not None:
            rng1 = np.random.default_rng(self.seed + idx * 2)
            rng2 = np.random.default_rng(self.seed + idx * 2 + 1)
        else:
            rng1 = np.random.default_rng()
            rng2 = np.random.default_rng()

        if K == 0:
            # K=0 pure noise, no DOA, generate independently
            R1 = generate_single_sample(
                M=self.M, K=0, rng=rng1,
                doa_range=self.doa_range,
                snapshot_range=self.snapshot_range,
                snr_range=self.snr_range,
                min_angle_sep=self.min_angle_sep,
                coh_mode=self.coh_mode,
            )
            R2 = generate_single_sample(
                M=self.M, K=0, rng=rng2,
                doa_range=self.doa_range,
                snapshot_range=self.snapshot_range,
                snr_range=self.snr_range,
                min_angle_sep=self.min_angle_sep,
                coh_mode=self.coh_mode,
            )
        else:
            # K>=1: share DOA, independently generate other parameters
            doas = _sample_doas_with_separation(
                K, self.doa_range, self.min_angle_sep, rng1
            )
            R1 = generate_single_sample_with_fixed_doa(
                M=self.M, K=K, doas=doas, rng=rng1,
                snapshot_range=self.snapshot_range,
                snr_range=self.snr_range,
                coh_mode=self.coh_mode,
            )
            R2 = generate_single_sample_with_fixed_doa(
                M=self.M, K=K, doas=doas, rng=rng2,
                snapshot_range=self.snapshot_range,
                snr_range=self.snr_range,
                coh_mode=self.coh_mode,
            )

        feat1 = scm_to_3channel(R1)
        feat2 = scm_to_3channel(R2)

        view1 = torch.from_numpy(feat1)
        view2 = torch.from_numpy(feat2)

        return view1, view2, K


def worker_init_fn(worker_id: int):
    """
    Ensure each worker in DataLoader has independent random state,
    avoid generating duplicate data in multi-process loading
    Usage: DataLoader(..., worker_init_fn=worker_init_fn)
    """

    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed + worker_id)


def create_dataloader(
    dataset: CohSourceDataset,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Note: drop_last=True ensures each batch has consistent size, which is crucial for contrastive learning loss computation.
    """

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

if __name__ == "__main__":
    dataset = CohSourceDataset()
    train_loader = create_dataloader(dataset)

    for view1_batch, view2_batch, K_batch in train_loader:
        print(f"view1_batch shape: {view1_batch.shape}")
        print(f"view2_batch shape: {view2_batch.shape}")
        print(f"K_batch shape: {K_batch.shape}")
        print(f"K_batch value: {K_batch.numpy()}")