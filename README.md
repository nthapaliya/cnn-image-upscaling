# CNN Image Super-Resolution: 4× Upscaling

> Comparing three convolutional neural network architectures for single-image
> super-resolution — upscaling low-resolution images by 4× while maintaining
> visual quality.

Final project for **CSCI E-89: Deep Learning**, Harvard Extension School, Fall 2024.

---

## Overview

Single-image super-resolution (SISR) is the task of recovering a plausible
high-resolution image from a low-resolution input. This project trains and evaluates
three CNN architectures at 4× upscaling on the
[FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) (70,000 high-quality face images),
measuring output quality with PSNR and SSIM.

**Why faces?** Face images provide a structured, well-studied benchmark domain where
quality degradation is perceptually obvious and metrics are well-calibrated.

---

## Architectures compared

| Architecture | Description |
|---|---|
| **SRCNN** | Simple 3-layer CNN; pioneering super-resolution network (Dong et al. 2014) |
| **ESPCN** | Sub-pixel convolution (pixel shuffle) upscaling; efficient and sharp (Shi et al. 2016) |
| **EDSR** | Enhanced Deep SR — removes batch norm for more stable training at depth (Lim et al. 2017) |

---

## Project structure

```
cnn-image-upscaling/
├── notebook.ipynb    # main notebook
├── pyproject.toml    # uv dependency spec
├── data/
│   ├── train/        # 65,000 images (downloaded via Kaggle)
│   └── test/         # 5,000 images (held out randomly from train set)
└── models/           # saved checkpoints (created on first run)
```

---

## Setup

**Requirements:** Python 3.10–3.12 (`uv` handles this automatically), a CUDA-capable GPU.
TensorFlow does not yet support Python 3.13+.

```bash
git clone https://github.com/<you>/cnn-image-upscaling.git
cd cnn-image-upscaling

uv python pin 3.12
uv sync
```

---

## Download the dataset

You need a [Kaggle API key](https://www.kaggle.com/docs/api) configured at
`~/.kaggle/kaggle.json`.

```bash
mkdir -p data/train data/test
cd data/train
kaggle datasets download arnaud58/flickrfaceshq-dataset-ffhq
unzip flickrfaceshq-dataset-ffhq.zip
```

Then create the hold-out test set (run once from the project root):

```python
import glob, shutil, numpy as np

for f in np.random.choice(glob.glob('data/train/*.png'), size=5000, replace=False):
    shutil.move(f, f.replace('train', 'test'))
```

---

## Running the notebook

```bash
uv run jupyter lab
```

Open `e89-Thapaliya-Niraj-Final_Project.ipynb` and run all cells in order. The notebook:

1. Loads and preprocesses images (downsamples to create LR/HR pairs)
2. Defines all three network architectures
3. Trains each model and logs PSNR/SSIM on the validation set
4. Produces comparison visualisations

> **Training time:** approximately 1–2 hours per architecture on a mid-range GPU.
> Pre-trained checkpoints are not included due to file size.

---

## Evaluation metrics

- **PSNR** (Peak Signal-to-Noise Ratio): higher is better; measured in dB.
- **SSIM** (Structural Similarity Index): higher is better; ranges from 0 to 1.

Both are computed on the luminance channel (Y of YCbCr) to match standard practice.

---

## References

- Dong et al. (2014). [SRCNN: Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)
- Shi et al. (2016). [Real-Time Single Image and Video Super-Resolution Using ESPCN](https://arxiv.org/abs/1609.05158)
- Lim et al. (2017). [EDSR: Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)
- Karras et al. (2019). [FFHQ: A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948)
