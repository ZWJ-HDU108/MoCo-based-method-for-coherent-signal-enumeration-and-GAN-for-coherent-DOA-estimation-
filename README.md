# MoCo-based-method-for-coherent-signal-enumeration-and-GAN-for-coherent-DOA-estimation-
This is the code for the paper "Robust Coherent Source Enumeration and DOA Estimation via Momentum Contrast and Adversarial Decoherence"

# MoCo-SourceEstimation: 基于监督对比学习的相干源个数估计

本项目提供了一套基于 **MoCo v2 (Momentum Contrast)** 改进的自监督/监督对比学习框架，专门用于解决阵列信号处理中的**相干源个数估计**问题。通过引入多尺度特征融合网络（MFFNet）和监督对比损失（SupCon Loss），模型能够在低信噪比和强相干环境下实现高精度的信噪源个数检测。

## 🌟 核心特性

* **对比学习架构**：采用改进的 MoCo v2 结构，通过动量更新的队列维持大量的负样本。
* **监督对比损失**：在 `SupLoss.py` 中实现了 `MoCoSupConLoss`，利用标签信息增强特征空间的类内聚合和类间分离。
* **物理驱动的数据增强**：`data_generator.py` 模拟了真实的阵列信号物理模型，支持多种相干模式（完全相干、部分相干、自适应相干）。
* **先进的网络骨干**：
    * **MFFNet**: 包含 FPN（特征金字塔）和 PAN（路径聚合网络）结构，配合 SE（Squeeze-and-Excitation）注意力机制。
    * **ResNetEncoder**: 适配小尺寸协方差矩阵输入（如 16x16）的改进型 ResNet。
* **基准对比**：集成了传统的统计学估计方法（AIC, MDL），便于进行性能对标。

## 📂 项目结构

```text
.
├── MoCov2/
│   ├── builder.py           # MoCo v2 模型构建器 (Query/Key 编码器, 动量更新, 队列管理)
│   └── SupLoss.py           # 异步监督对比学习损失函数实现
├── data/
│   └── data_generator.py    # 阵列信号仿真器：生成协方差矩阵 (SCM) 并转换为 3 通道图像
├── models/
│   ├── MFFNet.py            # 多尺度特征融合网络 (Backbone + FPN + PAN)
│   ├── ECNet.py / ERNet.py  # 用于对比或特定任务的轻量级 MLP 网络
│   └── main_mocov2.py       # 训练主程序，包含数据加载、预热队列及训练循环
├── baseline/
│   └── AIC_and_MDL.py       # 传统源个数估计方法实现
└── README.md
