import numpy as np
import torch

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class AdvancedCovarianceAugmentation:
    def __init__(self,
                 snr_perturbation=True,
                 snr_range=(-3, 3),  # dB
                 diagonal_loading=True,
                 loading_range=(0.001, 0.1),
                 normalize=True):
        self.snr_perturbation = snr_perturbation
        self.snr_range = snr_range
        self.diagonal_loading = diagonal_loading
        self.loading_range = loading_range
        self.normalize = normalize

    def __call__(self, x):
        x = x.copy()
        real_part = x[0]
        imag_part = x[1]
        M = real_part.shape[0]

        # SNR扰动
        if self.snr_perturbation:
            snr_delta = np.random.uniform(self.snr_range[0], self.snr_range[1])
            scale = 10 ** (snr_delta / 20)
            real_part = real_part * scale
            imag_part = imag_part * scale

        # 对角加载
        if self.diagonal_loading:
            loading = np.random.uniform(self.loading_range[0], self.loading_range[1])
            real_part = real_part + loading * np.eye(M)

        # 重新计算相位
        complex_cov = real_part + 1j * imag_part
        phase = np.angle(complex_cov)

        # 归一化
        if self.normalize:
            norm = np.sqrt(np.sum(real_part ** 2) + np.sum(imag_part ** 2))
            if norm > 1e-8:
                real_part = real_part / norm
                imag_part = imag_part / norm

        x = np.stack([real_part, imag_part, phase], axis=0)
        return torch.from_numpy(x.astype(np.float32))