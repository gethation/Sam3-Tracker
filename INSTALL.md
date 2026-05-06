# Installation

This project uses Ultralytics SAM3 for video tracking and YOLOv8 label export.

## 1. Create a Python Environment

Python 3.12 is known to work.

```powershell
conda create -n sam3-tracker python=3.12
conda activate sam3-tracker
```

## 2. Install PyTorch

Install the PyTorch build that matches your machine.

First check your NVIDIA driver:

```powershell
nvidia-smi
```

Use a PyTorch CUDA wheel whose CUDA version is less than or equal to the
`CUDA Version` shown by `nvidia-smi`.

For example, if `nvidia-smi` shows CUDA 12.7, install a CUDA 12.6 PyTorch wheel:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

If your driver supports CUDA 12.8 or newer, a CUDA 12.8 wheel is also valid:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For other CUDA versions or CPU-only install, use the official selector:

https://pytorch.org/get-started/locally/

## 3. Install Project Requirements

```powershell
pip install -r requirements.txt
```

The scripts disable PyTorch compile/TorchInductor by default to avoid Triton
setup issues on Windows. This is slower than compiled inference but easier to
install and run.

## 4. Download `sam3.pt`

Official SAM3 weights are gated on Hugging Face. Log in, request/accept access to
`facebook/sam3`, then run:

```powershell
hf auth login
hf download facebook/sam3 sam3.pt --local-dir .
```

Place input videos in:

```text
data/source/
```

Run tracking:

```powershell
python scripts/track_sam3_points.py
```
