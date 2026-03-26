import numpy as np
import h5py
from itertools import combinations
from tqdm import tqdm
import os
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


class Config:
    M = 16  # 阵元数
    d = 0.5  # 阵元间距

    K = 2  # 信源数量

    MAX_DOA = 60  # 视场角范围：±60
    GRID_RES = 1  # 分辨率=1

    SNR_MIN = -10  # 最小SNR
    SNR_MAX = 20  # 最大SNR
    SNR_STEP = 2  # SNR步长

    T = 400  # 快拍数

    # 输出路径
    OUTPUT_DIR = './gan_doa_dataset'
    OUTPUT_FILENAME = 'gan_cov_matrix_dataset.h5'


def steering_vector(theta_deg: float, M: int, d: float) -> np.ndarray:
    theta_rad = np.deg2rad(theta_deg)
    n = np.arange(M)
    a = np.exp(1j * 2 * np.pi * d * np.sin(theta_rad) * n)
    return a


def generate_incoherent_signal(
        A: np.ndarray,  # 阵列流形矩阵
        K: int,  # 信源数量
        T: int,  # 快拍数
        noise_power: float
) -> Tuple[np.ndarray, np.ndarray]:
    M = A.shape[0]

    # 生成K个独立的复高斯信源信号
    S = (np.random.randn(K, T) + 1j * np.random.randn(K, T)) / np.sqrt(2)

    # 阵列接收信号
    X = A @ S

    # 生成复高斯白噪声
    N = np.sqrt(noise_power / 2) * (np.random.randn(M, T) + 1j * np.random.randn(M, T))

    # 含噪声的接收信号
    Y = X + N

    # 采样协方差矩阵
    R_scm = (Y @ Y.conj().T) / T

    return Y, R_scm


def generate_coherent_signal(
        A: np.ndarray,  # 阵列流形矩阵
        K: int,  # 信源数量
        T: int,  # 快拍数
        noise_power: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = A.shape[0]

    # 生成基准信号
    s1 = (np.random.randn(1, T) + 1j * np.random.randn(1, T)) / np.sqrt(2)

    phi_vec = np.random.uniform(0, 2 * np.pi, size=K - 1)  # 生成K-1个相干系数 α_k = e^{jφ_k}，φ_k ∈ [0, 2π) 随机相位
    alpha_vec = np.exp(1j * phi_vec)

    # 构建相干系数向量 [1, α_2, α_3, ..., α_K]^T
    coeffs = np.concatenate([[1], alpha_vec]).reshape(K, 1)  # shape: (K, 1)

    # 构建相干信号矩阵 S = coeffs * s1
    S = coeffs @ s1  # shape: (K, T)，rank=1

    # 阵列接收信号
    X = A @ S

    # 生成复高斯白噪声
    N = np.sqrt(noise_power / 2) * (np.random.randn(M, T) + 1j * np.random.randn(M, T))

    # 含噪声的接收信号
    Y = X + N

    # 采样协方差矩阵
    R_scm = (Y @ Y.conj().T) / T

    return Y, R_scm, phi_vec


def noise_whitening(R: np.ndarray) -> np.ndarray:
    """从协方差矩阵中减去估计的噪声功率"""
    M = R.shape[0]

    # 计算特征值
    eigenvalues = np.linalg.eigvalsh(R)

    # 用最小特征值估计噪声功率
    noise_estimate = np.min(eigenvalues)

    # 减去噪声项
    R_whitened = R - noise_estimate * np.eye(M)

    # 确保半正定
    eigenvalues_w, eigenvectors = np.linalg.eigh(R_whitened)
    eigenvalues_w = np.maximum(eigenvalues_w, 0)  # 通过将小于0的特征值截断为0实现
    R_whitened = eigenvectors @ np.diag(eigenvalues_w) @ eigenvectors.conj().T

    return R_whitened


def normalize_covariance_matrix(R: np.ndarray) -> np.ndarray:
    """归一化协方差矩阵并提取三通道表示"""
    M = R.shape[0]

    # 找到最大模值并归一化
    max_val = np.max(np.abs(R))
    if max_val > 0:
        R_norm = R / max_val
    else:
        R_norm = R

    # 提取三通道
    R_3ch = np.zeros((3, M, M), dtype=np.float32)

    R_3ch[0, :, :] = np.real(R_norm)
    R_3ch[1, :, :] = np.imag(R_norm)
    phase = np.angle(R_norm)
    R_3ch[2, :, :] = phase / np.pi

    return R_3ch


def generate_single_sample(
        angles: Tuple,  # 信源角度，支持任意K个
        snr_db: float,  # 信噪比
        M: int,  # 阵元数
        d: float,  # 阵元间距
        T: int  # 快拍数
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成单个样本对"""
    K = len(angles)
    noise_power = 10 ** (-snr_db / 10)

    # 构建阵列流形矩阵
    A = np.zeros((M, K), dtype=np.complex128)
    for k, theta in enumerate(angles):
        A[:, k] = steering_vector(theta, M, d)

    _, R_incoh = generate_incoherent_signal(A, K, T, noise_power)  # 生成非相干信号协方差矩阵（满秩: Target）

    _, R_coh, phi = generate_coherent_signal(A, K, T, noise_power)  # 生成相干信号协方差矩阵（秩亏缺: Input）

    # 去除噪声项，保留信号子空间秩特性
    R_coh = noise_whitening(R_coh)
    R_incoh = noise_whitening(R_incoh)

    # 归一化并提取三通道
    R_coh_3ch = normalize_covariance_matrix(R_coh)
    R_incoh_3ch = normalize_covariance_matrix(R_incoh)

    return R_coh_3ch, R_incoh_3ch, phi


def process_batch(args):
    """处理一批样本"""
    batch_indices, angle_pairs, snr_values, config = args

    results = []
    for idx in batch_indices:
        # 计算角度对索引和SNR索引
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


def generate_dataset(config: Config,  # 配置参数
                     use_parallel: bool = True,  # 是否使用并行计算
                     n_workers: Optional[int] = None  # 并行工作进程数，默认为CPU核心数-1
                     ):
    # 创建输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILENAME)

    # 生成角度网格和组合
    grids = np.arange(-config.MAX_DOA, config.MAX_DOA + config.GRID_RES, config.GRID_RES)
    angle_pairs = list(combinations(grids, config.K))

    # 生成SNR值列表
    snr_values = np.arange(config.SNR_MIN, config.SNR_MAX + config.SNR_STEP, config.SNR_STEP)

    # 计算总样本数
    n_angle_pairs = len(angle_pairs)
    n_snr_levels = len(snr_values)
    n_samples = n_angle_pairs * n_snr_levels

    # 初始化数据数组
    coherent_data = np.zeros((n_samples, 3, config.M, config.M), dtype=np.float32)
    incoherent_data = np.zeros((n_samples, 3, config.M, config.M), dtype=np.float32)
    angles_data = np.zeros((n_samples, config.K), dtype=np.float32)
    snr_data = np.zeros(n_samples, dtype=np.float32)
    phi_data = np.zeros((n_samples, config.K - 1), dtype=np.float32)  # K-1个相干系数相位

    # 配置字典（用于并行）
    config_dict = {
        'M': config.M,
        'd': config.d,
        'T': config.T
    }

    if use_parallel and n_samples > 100:
        # 并行生成
        if n_workers is None:
            n_workers = max(1, multiprocessing.cpu_count() - 1)

        print(f"\n使用 {n_workers} 个进程并行生成数据...")

        # 分批
        batch_size = max(1, n_samples // (n_workers * 10))
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_indices = list(range(i, min(i + batch_size, n_samples)))
            batches.append((batch_indices, angle_pairs, snr_values, config_dict))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]

            with tqdm(total=n_samples, desc="生成样本") as pbar:
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
        # 串行生成
        print("\n串行生成数据...")
        sample_idx = 0

        with tqdm(total=n_samples, desc="生成样本") as pbar:
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

    # 保存到HDF5文件
    print(f"\n保存数据到: {output_path}")

    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(output_path, 'w') as f:
        # 主要数据集
        f.create_dataset('coherent_input', data=coherent_data,
                         compression='gzip', compression_opts=4)
        f.create_dataset('incoherent_target', data=incoherent_data,
                         compression='gzip', compression_opts=4)

        # 标签和元数据
        f.create_dataset('angles', data=angles_data)
        f.create_dataset('snr', data=snr_data)
        f.create_dataset('coherent_phase', data=phi_data)

        # 保存配置信息作为属性
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
    """验证相干信号协方差矩阵的秩亏缺特性,通过比较相干和非相干协方差矩阵的奇异值来验证"""
    with h5py.File(output_path, 'r') as f:
        # 随机选择几个样本
        total_samples = f['coherent_input'].shape[0]
        indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)

        for idx in indices:
            # 重构复数矩阵
            coh_data = f['coherent_input'][idx]
            incoh_data = f['incoherent_target'][idx]
            angles = f['angles'][idx]
            snr = f['snr'][idx]

            # 从三通道恢复复数矩阵（仅用实部和虚部）
            R_coh = coh_data[0] + 1j * coh_data[1]
            R_incoh = incoh_data[0] + 1j * incoh_data[1]

            # 计算奇异值
            sv_coh = np.linalg.svd(R_coh, compute_uv=False)
            sv_incoh = np.linalg.svd(R_incoh, compute_uv=False)

            # 归一化奇异值
            sv_coh_norm = sv_coh / sv_coh[0]
            sv_incoh_norm = sv_incoh / sv_incoh[0]

            # 估计有效秩（奇异值 > 0.01 * 最大奇异值）
            rank_coh = np.sum(sv_coh_norm > 0.01)
            rank_incoh = np.sum(sv_incoh_norm > 0.01)

            print(f"\n样本 {idx}: 角度={angles}, SNR={snr:.1f}dB")
            print(f"  相干信号 - 有效秩: {rank_coh}, 前5个归一化奇异值: {sv_coh_norm[:5]}")
            print(f"  非相干信号 - 有效秩: {rank_incoh}, 前5个归一化奇异值: {sv_incoh_norm[:5]}")


if __name__ == '__main__':
    # 创建配置
    config = Config()

    # 可以根据需要修改配置
    # config.SNR_MIN = -10
    # config.SNR_MAX = 10
    # config.T = 200

    # 生成数据集
    output_path = generate_dataset(config, use_parallel=False)

    # 验证秩亏缺特性
    verify_rank_deficiency(output_path, n_samples=5)