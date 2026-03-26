# MoCo-based Coherent Source Enumeartion and CMR-GAN Coherent DOA Estimation

This is the code for the paper "***Robust Coherent Source Enumeration and DOA Estimation via Momentum Contrast and Adversarial Decoherence***". 

This project provides a framework based on **MoCo v2 (Momentum Contrast)** and **Supervised Contrastive Learning** to estimate the number of coherent signal sources and **CMR-GAN for coherent DOA estimation** in array signal processing. The model achieves high accuracy in challenging environments with low SNR and strong source correlation.

## 📂 Project Structure

* `MoCov2/builder.py`: Core MoCo model logic (queue management and momentum updates).
* `MoCov2/SupLoss.py`: Implementation of the **Asymmetric Supervised Contrastive Loss**.
* `data/data_generator.py`: **on-the-fly** data generation for source enumeration.
* `compared_models`: Diferent compared models for source enumeration in the paper.
* `main_mocov2.py`: The primary training script for MoCo including **queue warmup and the training loop**.
* `models`: Code for **CMR-GAN**.
* `models/arch.py`: Code for **EBlock/DBlock, SG and brach**.
* `models/arch_util.py`: Code for some components (LayerNorm2d).
* `models/GAN.py`: The whole architecture of CMR-GAN.
* `gan_data_generator.py`: Data generation for CMR-GAN.
* `train_gan.py`: The primary training script for CMR-GAN.

---
## 🛠️ Requirements

### Pip Packages
| Package | Version |
|---------|---------|
| absl-py | 2.3.1 |
| anyio | 4.11.0 |
| certifi | 2025.11.12 |
| click | 8.3.1 |
| contourpy | 1.3.3 |
| cycler | 0.12.1 |
| einops | 0.8.1 |
| filelock | 3.19.1 |
| fonttools | 4.60.1 |
| fsspec | 2025.9.0 |
| grpcio | 1.76.0 |
| h11 | 0.16.0 |
| h5py | 3.14.0 |
| hf-xet | 1.2.0 |
| httpcore | 1.0.9 |
| httpx | 0.28.1 |
| huggingface-hub | 1.1.5 |
| idna | 3.11 |
| jinja2 | 3.1.6 |
| joblib | 1.5.2 |
| kiwisolver | 1.4.9 |
| markdown | 3.10 |
| markupsafe | 2.1.5 |
| matplotlib | 3.8.4 |
| mpmath | 1.3.0 |
| narwhals | 2.13.0 |
| natten | 0.17.5+torch260cu124 |
| networkx | 3.5 |
| numpy | 1.26.4 |
| nvidia-cublas-cu12 | 12.4.5.8 |
| nvidia-cuda-cupti-cu12 | 12.4.127 |
| nvidia-cuda-nvrtc-cu12 | 12.4.127 |
| nvidia-cuda-runtime-cu12 | 12.4.127 |
| nvidia-cudnn-cu12 | 9.1.0.70 |
| nvidia-cufft-cu12 | 11.2.1.3 |
| nvidia-curand-cu12 | 10.3.5.147 |
| nvidia-cusolver-cu12 | 11.6.1.9 |
| nvidia-cusparse-cu12 | 12.3.1.170 |
| nvidia-cusparselt-cu12 | 0.6.2 |
| nvidia-nccl-cu12 | 2.21.5 |
| nvidia-nvjitlink-cu12 | 12.4.127 |
| nvidia-nvtx-cu12 | 12.4.127 |
| packaging | 25.0 |
| pandas | 2.2.2 |
| pillow | 12.0.0 |
| plotly | 6.5.0 |
| protobuf | 6.33.1 |
| pyparsing | 3.2.5 |
| python-dateutil | 2.9.0.post0 |
| pytz | 2025.2 |
| pyyaml | 6.0.3 |
| safetensors | 0.7.0 |
| scikit-learn | 1.7.1 |
| scipy | 1.16.3 |
| shellingham | 1.5.4 |
| six | 1.17.0 |
| sniffio | 1.3.1 |
| sympy | 1.13.1 |
| tensorboard | 2.16.2 |
| tensorboard-data-server | 0.7.2 |
| thop | 0.1.1-2207130030 |
| threadpoolctl | 3.6.0 |
| timm | 1.0.20 |
| torch | 2.6.0+cu124 |
| torchaudio | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| tqdm | 4.66.2 |
| triton | 3.2.0 |
| typer-slim | 0.20.0 |
| typing-extensions | 4.15.0 |
| tzdata | 2025.2 |
| werkzeug | 3.1.3 |

## Environment Summary
- **Python Version**: 3.12.0
- **PyTorch Version**: 2.6.0+cu124 (CUDA 12.4)
- **CUDA Support**: Yes (cu124)
- **Key Libraries**: torch, torchvision, torchaudio, numpy, scipy, matplotlib, scikit-learn, pandas, h5py, timm

## 🚀 How to Use

**Run Command**:

```bash
python main_mocov2.py --arch resnet34 --mlp --cos --moco-m 0.999 --moco-t 0.07 --epochs 200 --batch-size 256 --learning-rate 0.001
python train_gan.py --epochs 200 --batch_size 64 --lr 0.0001 --lambda_l1 40
```

## 📝 License

[MIT License]
