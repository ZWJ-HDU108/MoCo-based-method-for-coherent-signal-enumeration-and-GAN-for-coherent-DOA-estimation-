import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from models.GAN import GAN_generator, GAN_discriminator


class TrainConfig:
    # 数据路径
    DATA_PATH = './gan_doa_dataset/training_data.h5'

    # 训练参数
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.0002
    BETA1 = 0.5  # Adam optimizer beta1
    LAMBDA_L1 = 100  # L1 loss权重

    # 模型保存
    SAVE_DIR = './checkpoints'
    SAVE_INTERVAL = 10  # 每10个epoch保存一次

    # 验证与可视化
    VALIDATE_INTERVAL = 20  # 每20个epoch验证一次
    OUTPUT_DIR = './outputs'

    # 设备
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# 数据集类
class DOADataset(Dataset):
    def __init__(self, h5_path: str, load_to_memory: bool = True):
        self.h5_path = h5_path  # HDF5数据文件路径
        self.load_to_memory = load_to_memory  # 是否将全部数据加载到内存

        with h5py.File(h5_path, 'r') as f:
            self.n_samples = f['coherent_input'].shape[0]

            if load_to_memory:
                # 加载全部数据到内存
                self.coherent_data = torch.from_numpy(f['coherent_input'][:]).float()
                self.incoherent_data = torch.from_numpy(f['incoherent_target'][:]).float()
                self.angles = f['angles'][:]
                self.snr = f['snr'][:]
                print(f"数据已加载到内存: {self.n_samples} 样本")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """返回一对训练数据，即同一信源的相干信号与非相干信号"""
        if self.load_to_memory:
            return self.coherent_data[idx], self.incoherent_data[idx]
        else:
            with h5py.File(self.h5_path, 'r') as f:
                coherent = torch.from_numpy(f['coherent_input'][idx]).float()
                incoherent = torch.from_numpy(f['incoherent_target'][idx]).float()
            return coherent, incoherent


def validate_and_plot(generator, dataloader, epoch, config, device):
    """验证并可视化GAN的秩恢复效果"""
    generator.eval()  # 验证模式
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 获取一个batch的数据
    coherent_batch, incoherent_batch = next(iter(dataloader))
    coherent_batch = coherent_batch.to(device)
    incoherent_batch = incoherent_batch.to(device)

    with torch.no_grad():
        generated_batch = generator(coherent_batch)

    # 选择第一个样本进行分析
    idx = 0
    coherent = coherent_batch[idx].cpu().numpy()      # Input
    generated = generated_batch[idx].cpu().numpy()    # Generated
    incoherent = incoherent_batch[idx].cpu().numpy()  # Target

    # 1. 奇异值分析
    R_input = coherent[0] + 1j * coherent[1]  # Input (相干信号：应该秩亏缺，第2奇异值很小)

    R_generated = generated[0] + 1j * generated[1]  # Generated (GAN输出：期望秩恢复，第2奇异值变大)

    R_target = incoherent[0] + 1j * incoherent[1]   # Target (非相干信号：满秩参考，第2奇异值大)

    # 对三个矩阵分别做SVD分解
    sv_input = np.linalg.svd(R_input, compute_uv=False)
    sv_generated = np.linalg.svd(R_generated, compute_uv=False)
    sv_target = np.linalg.svd(R_target, compute_uv=False)

    # 归一化奇异值
    sv_input_norm = sv_input / sv_input[0]
    sv_generated_norm = sv_generated / sv_generated[0]
    sv_target_norm = sv_target / sv_target[0]

    # 2. 绘图
    fig = plt.figure(figsize=(16, 10))

    # 奇异值分布
    ax_svd = fig.add_subplot(2, 1, 1)
    x = np.arange(1, 17)

    ax_svd.semilogy(x, sv_input_norm, 'ro-', linewidth=2, markersize=8,
                    label=f'Input (Coherent) - σ₂/σ₁={sv_input_norm[1]:.4f}')
    ax_svd.semilogy(x, sv_generated_norm, 'gs-', linewidth=2, markersize=8,
                    label=f'Generated - σ₂/σ₁={sv_generated_norm[1]:.4f}')
    ax_svd.semilogy(x, sv_target_norm, 'b^-', linewidth=2, markersize=8,
                    label=f'Target (Incoherent) - σ₂/σ₁={sv_target_norm[1]:.4f}')

    ax_svd.set_xlabel('Singular Value Index', fontsize=12)
    ax_svd.set_ylabel('Normalized Singular Value (log scale)', fontsize=12)
    ax_svd.set_title(f'SVD Analysis - Rank Restoration Check (Epoch {epoch})', fontsize=14)
    ax_svd.legend(fontsize=11, loc='upper right')
    ax_svd.grid(True, alpha=0.3)
    ax_svd.set_xlim([1, 16])
    ax_svd.set_ylim([1e-4, 2])

    rank_improvement = (sv_generated_norm[1] - sv_input_norm[1]) / (sv_target_norm[1] - sv_input_norm[1] + 1e-8)
    rank_improvement = np.clip(rank_improvement, 0, 1) * 100

    textstr = f'Rank Restoration: {rank_improvement:.1f}%\n'
    textstr += f'(0%=No improvement, 100%=Perfect)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_svd.text(0.02, 0.02, textstr, transform=ax_svd.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)

    # Heatmap对比
    ax1 = fig.add_subplot(2, 3, 4)
    ax2 = fig.add_subplot(2, 3, 5)
    ax3 = fig.add_subplot(2, 3, 6)

    # 统一颜色范围
    vmin, vmax = -1, 1
    cmap = 'RdBu_r'

    # Input (Coherent：秩亏缺)
    im1 = ax1.imshow(coherent[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title('Input (Coherent SCM)\nRank-deficient', fontsize=11)
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Generated (恢复后)
    im2 = ax2.imshow(generated[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title('Generated (Restored SCM)\nExpected: Full-rank', fontsize=11)
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # Target (Incoherent：满秩)
    im3 = ax3.imshow(incoherent[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax3.set_title('Target (Incoherent SCM)\nFull-rank', fontsize=11)
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    plt.suptitle(f'Pix2Pix GAN - Covariance Matrix Rank Restoration\nEpoch {epoch}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(config.OUTPUT_DIR, f'validation_epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [Validation] σ₂/σ₁ - Input: {sv_input_norm[1]:.4f}, "
          f"Generated: {sv_generated_norm[1]:.4f}, Target: {sv_target_norm[1]:.4f}")
    print(f"  [Validation] Rank Restoration: {rank_improvement:.1f}%")
    print(f"  [Validation] Figure saved to: {save_path}")

    generator.train()

    return {
        'sv_input': sv_input_norm[1],
        'sv_generated': sv_generated_norm[1],
        'sv_target': sv_target_norm[1],
        'rank_restoration': rank_improvement
    }


def train(config):
    """主训练函数  Loss = L_GAN(BCE) + λ * L_L1"""
    print("=" * 70)
    print("Pix2Pix GAN Training for DOA Covariance Matrix Rank Restoration")
    print("=" * 70)

    device = torch.device(config.DEVICE)
    print(f"Device: {device}")

    # 创建保存目录
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 加载数据集
    dataset = DOADataset(config.DATA_PATH, load_to_memory=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    # 使用自定义的GAN模型架构
    generator = GAN_generator(
        img_channel=3,
        width=32,
        middle_blk_num_enc=1,
        middle_blk_num_dec=1,
        enc_blk_nums=[1, 2, 2],
        dec_blk_nums=[2, 2, 1],
        dilations=[1, 4, 9],
        extra_depth_wise=True
    ).to(device)

    discriminator = GAN_discriminator().to(device)

    # 损失函数
    criterion_GAN = nn.BCELoss()  # GAN Loss (Binary Cross Entropy)
    criterion_L1 = nn.L1Loss()    # L1 Loss (像素级重建)

    # 优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, 0.999))

    # 训练历史记录
    history = {
        'g_loss': [],
        'd_loss': [],
        'g_gan_loss': [],
        'g_l1_loss': [],
        'sv_input': [],
        'sv_generated': [],
        'sv_target': [],
        'rank_restoration': []
    }

    # 训练循环
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Lambda L1: {config.LAMBDA_L1}")
    print("=" * 70 + "\n")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_g_gan = 0.0
        epoch_g_l1 = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}")

        for batch_idx, (coherent, incoherent) in enumerate(pbar):
            batch_size = coherent.size(0)

            # 将数据移到GPU上
            coherent = coherent.to(device)      # Input
            incoherent = incoherent.to(device)  # Target (Ground Truth)

            # 真假标签 shape: (B, 128, 1, 1)
            real_label = torch.ones(batch_size, 128, 1, 1, device=device)
            fake_label = torch.zeros(batch_size, 128, 1, 1, device=device)

            # 训练判别器 (Discriminator)
            optimizer_D.zero_grad()

            # 判别真实样本对 (Input, Target): 先拼接再传入 shape: [B, 3, 16, 16] -> [B, 6, 16, 16]
            real_pair = torch.cat((coherent, incoherent), dim=1)  # (B, 6, 16, 16)
            pred_real = discriminator(real_pair)
            loss_D_real = criterion_GAN(pred_real, real_label)

            # 生成假样本
            generated = generator(coherent)

            # 判别假样本对 (Input, Generated)
            fake_pair = torch.cat((coherent, generated.detach()), dim=1)  # (B, 6, 16, 16)  detach避免梯度流向G
            pred_fake = discriminator(fake_pair)
            loss_D_fake = criterion_GAN(pred_fake, fake_label)

            # 判别器总损失
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # 训练生成器 (Generator)
            optimizer_G.zero_grad()

            # GAN Loss: 欺骗判别器
            fake_pair = torch.cat((coherent, generated), dim=1)  # (B, 6, 16, 16)
            pred_fake = discriminator(fake_pair)
            loss_G_GAN = criterion_GAN(pred_fake, real_label)

            # L1 Loss: 像素级重建损失
            loss_G_L1 = criterion_L1(generated, incoherent)

            # 生成器总损失: L_total = L_GAN + λ * L_L1
            loss_G = loss_G_GAN + config.LAMBDA_L1 * loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            # 累计损失
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            epoch_g_gan += loss_G_GAN.item()
            epoch_g_l1 += loss_G_L1.item()

            # 更新进度条
            pbar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}',
                'G_L1': f'{loss_G_L1.item():.4f}'
            })

        # 计算epoch平均损失
        n_batches = len(dataloader)
        avg_g_loss = epoch_g_loss / n_batches
        avg_d_loss = epoch_d_loss / n_batches
        avg_g_gan = epoch_g_gan / n_batches
        avg_g_l1 = epoch_g_l1 / n_batches

        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        history['g_gan_loss'].append(avg_g_gan)
        history['g_l1_loss'].append(avg_g_l1)

        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
              f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, "
              f"G_GAN: {avg_g_gan:.4f}, G_L1: {avg_g_l1:.4f}")

        # 验证与可视化
        if epoch % config.VALIDATE_INTERVAL == 0 or epoch == 1:
            print(f"\n[Epoch {epoch}] 执行验证...")
            val_results = validate_and_plot(generator, dataloader, epoch, config, device)
            history['sv_input'].append(val_results['sv_input'])
            history['sv_generated'].append(val_results['sv_generated'])
            history['sv_target'].append(val_results['sv_target'])
            history['rank_restoration'].append(val_results['rank_restoration'])
            print()

        # 保存模型
        if epoch % config.SAVE_INTERVAL == 0:
            save_path_g = os.path.join(config.SAVE_DIR, f'generator_epoch_{epoch:03d}.pth')
            save_path_d = os.path.join(config.SAVE_DIR, f'discriminator_epoch_{epoch:03d}.pth')
            torch.save(generator.state_dict(), save_path_g)
            torch.save(discriminator.state_dict(), save_path_d)
            print(f"[Epoch {epoch}] 模型已保存到 {config.SAVE_DIR}")

    # 保存最终模型
    torch.save(generator.state_dict(), os.path.join(config.SAVE_DIR, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(config.SAVE_DIR, 'discriminator_final.pth'))

    # 绘制训练曲线
    plot_training_history(history, config)

    return generator, discriminator, history


def plot_training_history(history, config):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history['g_loss']) + 1)

    # 生成器和判别器损失
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['g_loss'], 'b-', label='Generator Loss')
    ax1.plot(epochs, history['d_loss'], 'r-', label='Discriminator Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('GAN Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 生成器GAN和L1损失
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['g_gan_loss'], 'g-', label='G_GAN Loss')
    ax2.plot(epochs, history['g_l1_loss'], 'm-', label='G_L1 Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Generator Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 奇异值比值变化
    if history['sv_generated']:
        ax3 = axes[1, 0]
        val_epochs = [config.VALIDATE_INTERVAL * (i+1) if i > 0 else 1
                      for i in range(len(history['sv_generated']))]
        ax3.plot(val_epochs, history['sv_input'], 'ro-', label='Input (Coherent)')
        ax3.plot(val_epochs, history['sv_generated'], 'gs-', label='Generated')
        ax3.plot(val_epochs, history['sv_target'], 'b^-', label='Target (Incoherent)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('σ₂/σ₁')
        ax3.set_title('Singular Value Ratio Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 秩恢复百分比
    if history['rank_restoration']:
        ax4 = axes[1, 1]
        ax4.plot(val_epochs, history['rank_restoration'], 'ko-', linewidth=2, markersize=8)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Rank Restoration (%)')
        ax4.set_title('Rank Restoration Progress')
        ax4.set_ylim([0, 105])
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Perfect')
        ax4.legend()

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Pix2Pix GAN for DOA Covariance Matrix Rank Restoration')
    parser.add_argument('--data', type=str, default='/home/hipeson/zwj/myproject2/data/gan_doa_dataset/gan_cov_matrix_dataset.h5',
                        help='Path to HDF5 training data')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lambda_l1', type=float, default=40, help='L1 loss weight')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/without_sg', help='Model save directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/without_sg', help='Output directory for visualizations')

    args = parser.parse_args()

    # 更新配置
    config = TrainConfig()
    config.DATA_PATH = args.data
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.LAMBDA_L1 = args.lambda_l1
    config.SAVE_DIR = args.save_dir
    config.OUTPUT_DIR = args.output_dir

    # 开始训练
    train(config)


if __name__ == '__main__':
    main()