# MoCo-SourceEstimation: Supervised Contrastive Learning for Coherent Source Enumeration

This project provides a framework based on **MoCo v2 (Momentum Contrast)** and **Supervised Contrastive Learning** to estimate the number of coherent signal sources in array signal processing. The model achieves high accuracy in challenging environments with low SNR and strong source correlation.

## 🌟 Key Features

* **Contrastive Learning Architecture**: Implements MoCo v2 with a query encoder, a momentum-updated key encoder, and a dynamic queue to manage negative samples.
* **Supervised Contrastive Loss**: Includes `MoCoSupConLoss` to leverage label information, pulling samples of the same class together while pushing different classes apart in the embedding space.
* **Physics-Based Data Generation**: A robust simulator for generating Sample Covariance Matrices (SCM) with various coherence modes: `full`, `partial`, and `random`.
* **Advanced Backbones**: 
    * **MFFNet**: A multi-scale fusion network featuring FPN/PAN structures and SE attention modules.
    * **ResNet-based Encoders**: Modified ResNet architectures optimized for small-sized input matrices (e.g., 16x16).
* **Baseline Comparisons**: Built-in implementations of traditional statistical methods like **AIC** and **MDL**.

## 📂 Project Structure

* `MoCov2/builder.py`: Core MoCo model logic (queue management and momentum updates).
* `MoCov2/SupLoss.py`: Implementation of the Asymmetric Supervised Contrastive Loss.
* `data/data_generator.py`: Signal simulation and SCM-to-3-channel image conversion.
* `models/MFFNet.py`: Multi-scale feature fusion network architecture.
* `main_mocov2.py`: The primary training script including queue warmup and the training loop.
* `baseline/AIC_and_MDL.py`: Traditional source enumeration benchmarks.

---

## 🚀 How to Use

### 1. Prerequisites
Install the required Python packages:
```bash
pip install torch torchvision numpy
