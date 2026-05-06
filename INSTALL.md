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

For CUDA 12.8:

```powershell
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

For a different CUDA version or CPU-only install, use the selector on:

https://pytorch.org/get-started/locally/

## 3. Install Project Requirements

```powershell
pip install -r requirements.txt
```

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
